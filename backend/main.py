"""
Multi-Modal Chat API
Talks directly to ComfyUI for image generation — no external API dependency.
"""
import logging
import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
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


class ChatRequest(BaseModel):
    character_id: str = Field(..., description="Character profile ID (e.g. 'luna')")
    message: str = Field(..., description="User's message")
    conversation_history: list[ChatMessage] = Field(
        default_factory=list,
        description="Previous messages for context",
    )


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
        result.append(
            CharacterInfo(
                id=char_id,
                name=char["chat_name"],
                description=char["chat_system_prompt"][:120] + "...",
                has_ref_image=has_ref,
                avatar_url=f"/avatar/{char_id}" if has_ref else None,
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


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, http_request: Request):
    """
    Send a message to a character.
    Returns text response + optional image URL.
    """
    character = get_character(request.character_id)
    if not character:
        raise HTTPException(status_code=404, detail=f"Character '{request.character_id}' not found")

    history = [{"role": m.role, "content": m.content} for m in request.conversation_history]

    chat_result = await chat_service.chat(
        user_message=request.message,
        character_prompt=character["chat_system_prompt"],
        conversation_history=history,
    )

    response = ChatResponse(
        message=chat_result["message"],
        image_url=None,
        image_generating=False,
        image_context=chat_result.get("image_context"),
    )

    # If the character wants to send an image, generate it using the ref image
    if chat_result["send_image"] and (chat_result.get("image_context") or chat_result.get("pose_description") or chat_result.get("outfit_description")):
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

    history = [{"role": m.role, "content": m.content} for m in request.conversation_history]

    chat_result = await chat_service.chat(
        user_message=request.message,
        character_prompt=character["chat_system_prompt"],
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
