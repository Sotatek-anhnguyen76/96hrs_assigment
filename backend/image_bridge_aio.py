"""
AIO TESTING MODE — single workflow handles all image tasks (pose, outfit, scene).
Uses edit_final_AIO.json with IPAdapterFaceID + Face Similarity scoring.

Grok decomposes user requests into sequential steps (max 3).
Each step chains: output of previous step → input of next step.
Face similarity checked after each step; retries if score < 40 (max 2 retries).

Easy to remove: delete this file, set USE_AIO_MODE=false, revert import in main.py.
"""
import copy
import hashlib
import json
import logging
import os
import random
import uuid
from io import BytesIO

import httpx
from PIL import Image

from config import settings
from comfyui_client import ComfyUIClient
from cost_logger import log_chat_call

logger = logging.getLogger(__name__)

_dir = os.path.dirname(__file__)

with open(os.path.join(_dir, "workflows", "edit_final_AIO.json"), "r") as f:
    AIO_WORKFLOW_TEMPLATE = json.load(f)

STORAGE_DIR = os.path.join(_dir, "storage", "images")
os.makedirs(STORAGE_DIR, exist_ok=True)

FACE_SCORE_THRESHOLD = 40
MAX_RETRIES_PER_STEP = 2
MAX_STEPS = 3


def _read_image(path: str) -> tuple[bytes, str, int, int]:
    """Read image file, return (bytes, content_type, width_rounded, height_rounded)."""
    ext = os.path.splitext(path)[1].lower()
    content_type = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png", ".webp": "image/webp",
    }.get(ext, "image/jpeg")

    with open(path, "rb") as f:
        data = f.read()
    with Image.open(BytesIO(data)) as img:
        w = (img.width // 8) * 8
        h = (img.height // 8) * 8
    return data, content_type, w, h


def _save_output(image_bytes: bytes) -> str:
    """Save image bytes to local storage, return /images/ URL path."""
    img_hash = hashlib.md5(image_bytes[:2048]).hexdigest()[:12]
    filename = f"{img_hash}.png"
    path = os.path.join(STORAGE_DIR, filename)
    with open(path, "wb") as f:
        f.write(image_bytes)
    return f"/images/{filename}"


def _prepare_aio_workflow(
    workflow_template: dict,
    input_image_name: str,
    ref_image_name: str,
    prompt: str,
    seed: int | None = None,
    width: int | None = None,
    height: int | None = None,
) -> dict:
    """Inject parameters into the edit_final_AIO workflow template.

    Node 8:  input image (changes per step — current state of the image)
    Node 32: original ref image (stays the same — for face similarity comparison)
    Node 3:  text prompt (the edit instruction for this step)
    Node 2:  KSampler seed
    Node 9:  EmptyLatentImage dimensions
    Node 22: LoRA switch (1=none, 2=PenisLora, 3=multiConceptNSFW)
    """
    wf = copy.deepcopy(workflow_template)

    # Node 8: input image (current image being edited)
    if "8" in wf:
        wf["8"]["inputs"]["image"] = input_image_name

    # Node 32: original ref image (for face similarity)
    if "32" in wf:
        wf["32"]["inputs"]["image"] = ref_image_name

    # Node 3: edit prompt
    if "3" in wf:
        wf["3"]["inputs"]["prompt"] = prompt

    # Node 2: seed
    if seed is not None and "2" in wf:
        wf["2"]["inputs"]["seed"] = seed

    # Node 9: dimensions
    if width and height and "9" in wf:
        wf["9"]["inputs"]["width"] = width
        wf["9"]["inputs"]["height"] = height

    # Node 22: default to no extra LoRA
    if "22" in wf:
        wf["22"]["inputs"]["select"] = 1

    return wf


def _parse_face_score(client: ComfyUIClient) -> float:
    """Extract face similarity score from node 26 (PreviewAny)."""
    raw = client.get_text_output("26")
    if raw is None:
        logger.warning("[AIO] No face score returned, assuming 100")
        return 100.0
    try:
        return float(raw)
    except (ValueError, TypeError):
        logger.warning(f"[AIO] Could not parse face score: '{raw}', assuming 100")
        return 100.0


async def _decompose_request(
    image_context: str | None,
    pose_description: str | None,
    outfit_description: str | None,
) -> list[str]:
    """Call Grok to decompose the user's image request into sequential editing steps.

    Returns a list of 1-3 step prompts, ordered: pose → outfit → scene.
    """
    # Build context from what the chat service returned
    parts = []
    if pose_description:
        parts.append(f"Pose: {pose_description}")
    if outfit_description:
        parts.append(f"Outfit: {outfit_description}")
    if image_context:
        parts.append(f"Scene/context: {image_context}")

    if not parts:
        return ["a natural photo, keep everything the same"]

    user_request = ". ".join(parts)

    system_prompt = f"""You are an image editing director. Given a user's image request, break it into sequential editing steps for an AI image editor.

RULES:
- Maximum {MAX_STEPS} steps
- Each step is a single, focused edit instruction that the AI image editor can execute
- Order: pose changes FIRST, then outfit/clothing, then scene/background/lighting LAST
- Each step must include "Do not change facial features, keep face identity" at the end
- Only include steps that are actually needed. If the request is just about outfit, return 1 step.
- Keep each step prompt under 30 words

Respond with ONLY a JSON array of step prompts, nothing else.

Example input: "wearing bikini, at the mountain, standing with arms raised"
Example output: ["Change pose to standing with arms raised. Do not change facial features, keep face identity", "Change outfit to bikini. Do not change facial features, keep face identity", "Change background to mountain scenery with natural lighting. Do not change facial features, keep face identity"]

Example input: "wearing a red dress"
Example output: ["Change outfit to a red dress. Do not change facial features, keep face identity"]"""

    try:
        async with httpx.AsyncClient() as http:
            resp = await http.post(
                f"{settings.XAI_BASE_URL.rstrip('/')}/chat/completions",
                headers={
                    "Authorization": f"Bearer {settings.XAI_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": settings.XAI_MODEL,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_request},
                    ],
                    "temperature": 0.3,
                    "max_tokens": 500,
                },
                timeout=30.0,
            )
            resp.raise_for_status()
            api_result = resp.json()

            usage = api_result.get("usage", {})
            log_chat_call(
                model=settings.XAI_MODEL,
                endpoint="/chat/completions",
                input_tokens=usage.get("prompt_tokens", 0),
                output_tokens=usage.get("completion_tokens", 0),
                caller="aio_decompose",
            )

            content = api_result["choices"][0]["message"]["content"].strip()

            # Handle markdown code blocks
            if content.startswith("```"):
                content = content.split("\n", 1)[1]
                content = content.rsplit("```", 1)[0].strip()

            steps = json.loads(content)
            if isinstance(steps, list) and steps:
                logger.info(f"[AIO] Grok decomposed into {len(steps)} steps: {steps}")
                return steps[:MAX_STEPS]

    except Exception as e:
        logger.error(f"[AIO] Decompose failed: {e}, falling back to single step")

    # Fallback: single combined prompt
    return [f"{user_request}. Do not change facial features, keep face identity"]


class ImageBridge:
    def __init__(self):
        self.comfyui_address = settings.COMFYUI_SERVER_ADDRESS
        logger.info("AIO MODE ENABLED — using edit_final_AIO.json for all image tasks")

    async def generate_image(
        self,
        ref_image_path: str,
        prompt: str,
        pose_description: str | None = None,
        outfit_description: str | None = None,
    ) -> dict:
        """
        Generate image using the AIO workflow with multi-step chaining.

        1. Grok decomposes request into sequential steps (max 3)
        2. Each step runs edit_final_AIO.json
        3. Output of step N becomes input of step N+1
        4. Face score checked after each step — retry if < 40 (max 2 retries)
        """
        if not os.path.exists(ref_image_path):
            return {"status": "failed", "error": "Reference image not found"}

        # Step 0: Ask Grok to decompose the request
        steps = await _decompose_request(prompt, pose_description, outfit_description)
        logger.info(f"[AIO] Pipeline: {len(steps)} steps to execute")

        client = ComfyUIClient(self.comfyui_address)

        try:
            client.connect()

            # Upload the original ref image (stays constant for face similarity)
            ref_data, ref_ct, w, h = _read_image(ref_image_path)
            ref_filename = os.path.basename(ref_image_path)
            ref_comfyui = client.upload_image(ref_filename, ref_data, ref_ct)

            # Current input starts as the ref image
            current_input_name = ref_comfyui
            final_image_bytes = None

            for step_idx, step_prompt in enumerate(steps):
                logger.info(f"[AIO] Step {step_idx + 1}/{len(steps)}: '{step_prompt[:60]}...'")

                best_image = None
                best_score = -1.0

                for attempt in range(1 + MAX_RETRIES_PER_STEP):
                    seed = random.randint(1, 1_000_000_000)

                    # Use stronger face preservation on retries
                    actual_prompt = step_prompt
                    if attempt > 0:
                        actual_prompt = (
                            f"{step_prompt}. CRITICAL: preserve exact facial features, "
                            f"do not alter face, eyes, nose, mouth, skin tone in any way"
                        )
                        logger.info(f"[AIO] Step {step_idx + 1} retry {attempt}/{MAX_RETRIES_PER_STEP}")

                    workflow = _prepare_aio_workflow(
                        AIO_WORKFLOW_TEMPLATE,
                        input_image_name=current_input_name,
                        ref_image_name=ref_comfyui,
                        prompt=actual_prompt,
                        seed=seed,
                        width=w,
                        height=h,
                    )

                    output_images = client.execute_workflow(workflow)

                    # Get output image from node 6
                    images = (
                        output_images.get("6")
                        or next(iter(output_images.values()), [])
                    )
                    if not images:
                        logger.warning(f"[AIO] Step {step_idx + 1} attempt {attempt + 1}: no output image")
                        continue

                    # Check face similarity score from node 26
                    face_score = _parse_face_score(client)
                    logger.info(f"[AIO] Step {step_idx + 1} attempt {attempt + 1}: face_score={face_score:.1f}")

                    # Track the best result
                    if face_score > best_score:
                        best_score = face_score
                        best_image = images[0]

                    # Good enough — move on
                    if face_score >= FACE_SCORE_THRESHOLD:
                        break

                if best_image is None:
                    return {"status": "failed", "error": f"Step {step_idx + 1} produced no images"}

                if best_score < FACE_SCORE_THRESHOLD:
                    logger.warning(
                        f"[AIO] Step {step_idx + 1}: best face_score={best_score:.1f} "
                        f"(below {FACE_SCORE_THRESHOLD}), using best attempt anyway"
                    )

                final_image_bytes = best_image

                # Upload this step's output as input for the next step
                if step_idx < len(steps) - 1:
                    next_input_name = f"aio_step{step_idx + 1}_{uuid.uuid4().hex[:8]}.png"
                    current_input_name = client.upload_image(
                        next_input_name, best_image, "image/png"
                    )
                    logger.info(f"[AIO] Chained step {step_idx + 1} output → step {step_idx + 2} input")

            # Save final result
            url = _save_output(final_image_bytes)
            logger.info(f"[AIO] Pipeline complete: {len(steps)} steps, final image: {url}")
            return {"status": "succeeded", "image_url": url}

        except Exception as e:
            logger.error(f"[AIO] Pipeline error: {e}", exc_info=True)
            return {"status": "failed", "error": str(e)}
        finally:
            client.disconnect()
