"""
Multi-Modal Chat API
Talks directly to ComfyUI for image generation — no external API dependency.
"""
import logging
import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from config import settings
from chat_service import ChatService
from image_bridge import ImageBridge, STORAGE_DIR
from character_profiles import get_character, CHARACTERS

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


class GenerateImageRequest(BaseModel):
    character_id: str = Field(..., description="Character profile ID")
    image_context: str = Field("", description="Scene description / edit prompt")
    pose_description: str | None = Field(None, description="Body pose for xAI image gen")
    outfit_description: str | None = Field(None, description="Outfit change instruction for SAM3 workflow")


# --- Endpoints ---

@app.get("/health")
async def health():
    return {"status": "healthy", "version": "1.0.0"}


@app.get("/characters", response_model=list[CharacterInfo])
async def get_characters():
    """List all available characters."""
    result = []
    for char_id, char in CHARACTERS.items():
        result.append(
            CharacterInfo(
                id=char_id,
                name=char["chat_name"],
                description=char["chat_system_prompt"][:120] + "...",
                has_ref_image=os.path.exists(char.get("ref_image", "")),
            )
        )
    return result


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
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
            image_result = await image_bridge.generate_image(
                ref_image_path=character["ref_image"],
                prompt=chat_result.get("image_context", ""),
                pose_description=chat_result.get("pose_description"),
                outfit_description=chat_result.get("outfit_description"),
            )
            if image_result["status"] == "succeeded":
                response.image_url = image_result["image_url"]
                response.image_generating = False
            else:
                logger.error(f"Image generation failed: {image_result.get('error')}")
                response.image_generating = False
        except Exception as e:
            logger.error(f"Image bridge error: {e}")
            response.image_generating = False

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
async def generate_image_for_chat(request: GenerateImageRequest):
    """
    Generate an image for a character given a scene context.
    Called separately from chat for async image generation.
    """
    character = get_character(request.character_id)
    if not character:
        raise HTTPException(status_code=404, detail=f"Character '{request.character_id}' not found")

    image_result = await image_bridge.generate_image(
        ref_image_path=character["ref_image"],
        prompt=request.image_context,
        pose_description=request.pose_description,
        outfit_description=request.outfit_description,
    )

    return image_result


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host=settings.APP_HOST, port=settings.APP_PORT, reload=True)
