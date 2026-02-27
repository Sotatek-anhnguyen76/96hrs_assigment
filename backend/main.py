"""
Multi-Modal Chat API
Talks directly to ComfyUI for image generation — no external API dependency.
"""
import logging
import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from config import settings
from chat_service import ChatService
from character_profiles import get_character, CHARACTERS
from cost_logger import get_session_summary
from google_chat import send_generation_result, send_text_only

# AIO testing mode: swap to single-workflow image bridge
if settings.USE_AIO_MODE:
    from image_bridge_aio import ImageBridge, STORAGE_DIR
else:
    from image_bridge import ImageBridge, STORAGE_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(name)s] - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

chat_service = ChatService()
image_bridge = ImageBridge()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Multi-Modal Chat API starting...")
    logger.info(f"ComfyUI: {settings.COMFYUI_SERVER_ADDRESS}")
    yield
    logger.info("Shutting down.")


app = FastAPI(title="Multi-Modal Chat API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve generated images as static files at /images/
app.mount("/images", StaticFiles(directory=STORAGE_DIR), name="images")


# --- Request/Response models ---

class ChatMessage(BaseModel):
    role: str = Field(..., description="'user' or 'assistant'")
    content: str = Field(..., description="Message text")


class PersonaOverride(BaseModel):
    name: str | None = None
    age: int | None = None
    personality: str | None = None
    occupation: str | None = None
    relationship: str | None = None
    ethnicity: str | None = None
    bodyType: str | None = None
    hairStyle: str | None = None
    hairColor: str | None = None
    eyeColor: str | None = None
    style: str | None = None


class ChatRequest(BaseModel):
    character_id: str = Field(..., description="Character profile ID (e.g. 'luna')")
    message: str = Field(..., description="User's message")
    conversation_history: list[ChatMessage] = Field(
        default_factory=list,
        description="Previous messages for context",
    )
    persona_override: PersonaOverride | None = Field(None, description="Override character persona fields")


class ChatResponse(BaseModel):
    message: str = Field(..., description="Character's text response")
    image_url: str | None = Field(None, description="Generated image URL if applicable")
    image_generating: bool = Field(False, description="True if image is being generated")
    image_context: str | None = Field(None, description="What the image depicts")


class CharacterInfo(BaseModel):
    id: str
    name: str
    description: str
    has_ref_image: bool
    avatar_url: str | None = None


class GenerateImageRequest(BaseModel):
    character_id: str = Field(..., description="Character profile ID")
    image_context: str = Field("", description="Scene description / edit prompt")
    pose_description: str | None = Field(None, description="Body pose for xAI image gen")
    outfit_description: str | None = Field(None, description="Outfit change instruction for SAM3 workflow")


def _make_public_url(relative_url: str | None, http_request: Request) -> str | None:
    """Convert a relative /images/... path to a full public URL."""
    if not relative_url:
        return None
    origin = http_request.headers.get("origin", "")
    if origin:
        return f"{origin}{relative_url}"
    scheme = http_request.url.scheme
    host = http_request.headers.get("host", "localhost:8000")
    return f"{scheme}://{host}{relative_url}"


def _detect_workflow_type(chat_result: dict) -> str:
    """Determine which workflow/switch was chosen based on chat intent."""
    if settings.USE_AIO_MODE:
        parts = []
        if chat_result.get("pose_description"):
            parts.append("pose")
        if chat_result.get("outfit_description"):
            parts.append("outfit")
        if chat_result.get("image_context") and not parts:
            parts.append("scene")
        return f"AIO ({' + '.join(parts)})" if parts else "AIO"
    else:
        if chat_result.get("pose_description"):
            return "Pose Workflow"
        return "Outfit Workflow"


def _make_steps_public(steps: list[dict] | None, http_request: Request) -> list[dict] | None:
    """Convert step image_url fields to public URLs."""
    if not steps:
        return steps
    public_steps = []
    for step in steps:
        s = dict(step)
        if s.get("image_url"):
            s["image_url"] = _make_public_url(s["image_url"], http_request)
        public_steps.append(s)
    return public_steps


def _build_system_prompt(character: dict, override: PersonaOverride | None = None) -> str:
    """Build a system prompt from persona fields, applying any overrides."""
    persona = dict(character.get("persona", {}))
    base_name = character["chat_name"]

    if override:
        for field in ["name", "age", "personality", "occupation", "relationship",
                       "ethnicity", "bodyType", "hairStyle", "hairColor", "eyeColor", "style"]:
            val = getattr(override, field, None)
            if val is not None:
                persona[field] = val

    name = persona.get("name", base_name)
    age = persona.get("age", 25)
    occupation = persona.get("occupation", "")
    personality = persona.get("personality", "friendly")
    relationship = persona.get("relationship", "stranger")
    ethnicity = persona.get("ethnicity", "")
    body_type = persona.get("bodyType", "average")
    hair_style = persona.get("hairStyle", "")
    hair_color = persona.get("hairColor", "")
    eye_color = persona.get("eyeColor", "")

    return (
        f"You are {name}, a {age}-year-old {occupation}. "
        f"Your personality is {personality}. "
        f"Your relationship with the user is: {relationship}. "
        f"You speak naturally and casually, like texting someone you know. "
        f"You are a {ethnicity} person with {hair_style} {hair_color} hair, "
        f"{eye_color} eyes, and a {body_type} build."
    )


# --- Endpoints ---

@app.get("/health")
async def health():
    return {"status": "healthy", "version": "1.0.0"}


@app.get("/cost")
async def cost_summary():
    """Current session Grok API cost summary."""
    return get_session_summary()


@app.get("/characters", response_model=list[CharacterInfo])
async def get_characters():
    """List all available characters."""
    result = []
    for char_id, char in CHARACTERS.items():
        has_ref = os.path.exists(char.get("ref_image", ""))
        # Prefer Supabase avatar URL; fall back to local /avatar/ endpoint
        avatar = char.get("avatar_url") or (f"/avatar/{char_id}" if has_ref else None)
        result.append(
            CharacterInfo(
                id=char_id,
                name=char["chat_name"],
                description=char["chat_system_prompt"][:120] + "...",
                has_ref_image=has_ref,
                avatar_url=avatar,
            )
        )
    return result


@app.get("/avatar/{character_id}")
async def get_avatar(character_id: str):
    """Serve character reference image as avatar."""
    character = get_character(character_id)
    if not character:
        raise HTTPException(status_code=404, detail="Character not found")
    ref_path = character.get("ref_image", "")
    if not os.path.exists(ref_path):
        raise HTTPException(status_code=404, detail="Avatar not found")
    return FileResponse(ref_path)


@app.post("/characters/upload-image")
async def upload_character_image(file: UploadFile = File(...)):
    """Upload a custom reference image. Saves it as the 'custom' character's ref image."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    from character_profiles import REF_IMAGE_DIR
    uploads_dir = os.path.join(REF_IMAGE_DIR, "uploads")
    os.makedirs(uploads_dir, exist_ok=True)

    # Save with original extension
    ext = os.path.splitext(file.filename or "image.jpg")[1] or ".jpg"
    save_path = os.path.join(uploads_dir, f"custom{ext}")
    content = await file.read()
    with open(save_path, "wb") as f:
        f.write(content)

    # Update the custom character's ref_image path at runtime
    custom_char = CHARACTERS.get("custom")
    if custom_char:
        custom_char["ref_image"] = save_path

    logger.info(f"Custom image uploaded: {save_path} ({len(content)} bytes)")
    return {"status": "ok", "path": save_path, "size": len(content)}


@app.get("/characters/{character_id}/persona")
async def get_character_persona(character_id: str):
    """Get the full persona config for a character."""
    character = get_character(character_id)
    if not character:
        raise HTTPException(status_code=404, detail="Character not found")
    return {
        "id": character_id,
        "name": character["chat_name"],
        "persona": character.get("persona", {}),
        "system_prompt": character["chat_system_prompt"],
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, http_request: Request):
    """
    Send a message to a character.
    Returns text response + optional image URL.
    """
    character = get_character(request.character_id)
    if not character:
        raise HTTPException(status_code=404, detail=f"Character '{request.character_id}' not found")

    # Use persona override if provided, otherwise use default system prompt
    if request.persona_override:
        system_prompt = _build_system_prompt(character, request.persona_override)
    else:
        system_prompt = character["chat_system_prompt"]

    history = [{"role": m.role, "content": m.content} for m in request.conversation_history]

    chat_result = await chat_service.chat(
        user_message=request.message,
        character_prompt=system_prompt,
        conversation_history=history,
    )

    response = ChatResponse(
        message=chat_result["message"],
        image_url=None,
        image_generating=False,
        image_context=chat_result.get("image_context"),
    )

    # If the character wants to send an image, generate it using the ref image
    has_ref = os.path.exists(character.get("ref_image", ""))
    wants_image = chat_result["send_image"] and (chat_result.get("image_context") or chat_result.get("pose_description") or chat_result.get("outfit_description"))

    if wants_image and not has_ref:
        logger.warning(f"Skipping image gen for '{request.character_id}': no reference image uploaded")

    if wants_image and has_ref:
        response.image_generating = True
        try:
            import time
            t0 = time.monotonic()

            image_result = await image_bridge.generate_image(
                ref_image_path=character["ref_image"],
                prompt=chat_result.get("image_context", ""),
                pose_description=chat_result.get("pose_description"),
                outfit_description=chat_result.get("outfit_description"),
            )

            duration = time.monotonic() - t0

            if image_result["status"] == "succeeded":
                response.image_url = image_result["image_url"]
                response.image_generating = False

                # Send to Google Chat with public URLs
                public_image_url = _make_public_url(response.image_url, http_request)
                public_original_url = _make_public_url(image_result.get("original_image_url"), http_request)
                public_steps = _make_steps_public(image_result.get("steps"), http_request)

                send_generation_result(
                    character_name=character["chat_name"],
                    user_message=request.message,
                    ai_message=chat_result["message"],
                    image_url=public_image_url,
                    original_image_url=public_original_url,
                    steps=public_steps,
                    duration=duration,
                    workflow_type=_detect_workflow_type(chat_result),
                )
            else:
                logger.error(f"Image generation failed: {image_result.get('error')}")
                response.image_generating = False
        except Exception as e:
            logger.error(f"Image bridge error: {e}")
            response.image_generating = False
    else:
        # Text-only message — notify Google Chat
        send_text_only(
            character_name=character["chat_name"],
            user_message=request.message,
            ai_message=chat_result["message"],
        )

    return response


@app.post("/chat/stream")
async def chat_text_only(request: ChatRequest):
    """
    Send a message and get ONLY the text response (no image generation).
    Use this for fast responses, then call /chat/generate-image separately.
    """
    character = get_character(request.character_id)
    if not character:
        raise HTTPException(status_code=404, detail=f"Character '{request.character_id}' not found")

    if request.persona_override:
        system_prompt = _build_system_prompt(character, request.persona_override)
    else:
        system_prompt = character["chat_system_prompt"]

    history = [{"role": m.role, "content": m.content} for m in request.conversation_history]

    chat_result = await chat_service.chat(
        user_message=request.message,
        character_prompt=system_prompt,
        conversation_history=history,
    )

    return {
        "message": chat_result["message"],
        "send_image": chat_result["send_image"],
        "image_context": chat_result.get("image_context"),
    }


@app.post("/chat/generate-image")
async def generate_image_for_chat(request: GenerateImageRequest, http_request: Request):
    """
    Generate an image for a character given a scene context.
    Called separately from chat for async image generation.
    """
    character = get_character(request.character_id)
    if not character:
        raise HTTPException(status_code=404, detail=f"Character '{request.character_id}' not found")

    if not os.path.exists(character.get("ref_image", "")):
        raise HTTPException(status_code=400, detail="No reference image uploaded for this character. Please upload one first.")

    import time
    t0 = time.monotonic()

    image_result = await image_bridge.generate_image(
        ref_image_path=character["ref_image"],
        prompt=request.image_context,
        pose_description=request.pose_description,
        outfit_description=request.outfit_description,
    )

    duration = time.monotonic() - t0

    # Send to Google Chat with public URLs
    if image_result.get("status") == "succeeded" and image_result.get("image_url"):
        public_url = _make_public_url(image_result["image_url"], http_request)
        public_original_url = _make_public_url(image_result.get("original_image_url"), http_request)
        public_steps = _make_steps_public(image_result.get("steps"), http_request)

        # Detect workflow type from the request params
        wf_type = "AIO" if settings.USE_AIO_MODE else ("Pose Workflow" if request.pose_description else "Outfit Workflow")
        if settings.USE_AIO_MODE:
            parts = []
            if request.pose_description:
                parts.append("pose")
            if request.outfit_description:
                parts.append("outfit")
            if request.image_context and not parts:
                parts.append("scene")
            wf_type = f"AIO ({' + '.join(parts)})" if parts else "AIO"

        send_generation_result(
            character_name=character["chat_name"],
            user_message=request.image_context,
            ai_message=f"[Image generated: {request.image_context}]",
            image_url=public_url,
            original_image_url=public_original_url,
            steps=public_steps,
            duration=duration,
            workflow_type=wf_type,
        )

    return image_result


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host=settings.APP_HOST, port=settings.APP_PORT, reload=True)
