"""
AIO TESTING MODE — single workflow handles all image tasks (pose, outfit, scene).
Uses edit_final_AIO.json with IPAdapterFaceID + Face Similarity scoring.

Grok decomposes user requests into sequential steps (max 3).
Each step chains: output of previous step → input of next step.
Face similarity checked after each step; retries if score < 40 (max 2 retries).

Easy to remove: delete this file, set USE_AIO_MODE=false, revert import in main.py.
"""
import asyncio
import copy
import hashlib
import json
import logging
import os
import random
import uuid
from datetime import datetime, timezone
from io import BytesIO

import httpx
from PIL import Image

from config import settings
from comfyui_client import ComfyUIClient
from cost_logger import log_chat_call

logger = logging.getLogger(__name__)

# --- Local file logger for raw Grok AIO output (shares file with chat_service) ---
_LOG_DIR = os.path.dirname(os.path.abspath(__file__))
_GROK_LOG_PATH = os.path.join(_LOG_DIR, "grok_output.log")

_file_logger = logging.getLogger("grok_output_aio")
_file_logger.setLevel(logging.DEBUG)
_file_logger.propagate = False
if not _file_logger.handlers:
    _fh = logging.FileHandler(_GROK_LOG_PATH, encoding="utf-8")
    _fh.setFormatter(logging.Formatter("%(message)s"))
    _file_logger.addHandler(_fh)

_dir = os.path.dirname(__file__)

with open(os.path.join(_dir, "workflows", "edit_final_AIO.json"), "r") as f:
    AIO_WORKFLOW_TEMPLATE = json.load(f)

STORAGE_DIR = os.path.join(_dir, "storage", "images")
os.makedirs(STORAGE_DIR, exist_ok=True)

FACE_SCORE_THRESHOLD = 40
MAX_RETRIES_PER_STEP = 2
MAX_STEPS = 3
MAX_PIPELINE_RETRIES = 1


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


def _upload_to_supabase(image_bytes: bytes, filename: str) -> str | None:
    """Upload image to Supabase Storage, return public URL or None on failure."""
    if not settings.USE_SUPABASE_STORAGE or not settings.SUPABASE_SERVICE_ROLE_KEY:
        return None
    try:
        url = (
            f"{settings.SUPABASE_URL}/storage/v1/object/"
            f"{settings.SUPABASE_BUCKET_NAME}/generated/{filename}"
        )
        resp = httpx.put(
            url,
            content=image_bytes,
            headers={
                "Authorization": f"Bearer {settings.SUPABASE_SERVICE_ROLE_KEY}",
                "Content-Type": "image/png",
                "x-upsert": "true",
            },
            timeout=30,
        )
        if resp.status_code in (200, 201):
            public_url = (
                f"{settings.SUPABASE_URL}/storage/v1/object/public/"
                f"{settings.SUPABASE_BUCKET_NAME}/generated/{filename}"
            )
            logger.info(f"[Supabase] Uploaded {filename}")
            return public_url
        else:
            logger.warning(f"[Supabase] Upload failed {resp.status_code}: {resp.text[:200]}")
            return None
    except Exception as e:
        logger.error(f"[Supabase] Upload error: {e}")
        return None


def _save_output(image_bytes: bytes) -> dict:
    """Save image bytes to local storage + Supabase. Returns {"local": "/images/...", "supabase": "https://..." | None}."""
    img_hash = hashlib.md5(image_bytes[:2048]).hexdigest()[:12]
    filename = f"{img_hash}.png"
    path = os.path.join(STORAGE_DIR, filename)
    with open(path, "wb") as f:
        f.write(image_bytes)
    local_url = f"/images/{filename}"
    supabase_url = _upload_to_supabase(image_bytes, filename)
    return {"local": local_url, "supabase": supabase_url}


def _prepare_aio_workflow(
    workflow_template: dict,
    input_image_name: str,
    ref_image_name: str,
    prompt: str,
    seed: int | None = None,
    width: int | None = None,
    height: int | None = None,
    lora_switch: int = 1,
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

    # Node 22: LoRA switch (1=none, 2=PenisLora, 3=multiConceptNSFW)
    if "22" in wf:
        if lora_switch not in (1, 2, 3):
            lora_switch = 1
        wf["22"]["inputs"]["select"] = lora_switch
        logger.info(f"[AIO] LoRA switch set to {lora_switch} ({LORA_SWITCH_NAMES.get(lora_switch, '?')})")

    return wf


def _parse_face_score(client: ComfyUIClient) -> float:
    """Extract face similarity score from node 26 (PreviewAny) via WebSocket."""
    raw = client.get_text_output("26")
    if raw is None:
        logger.warning("[AIO] No face score returned from node 26, assuming 100")
        return 100.0
    try:
        score = float(raw)
        # Prominent terminal display
        status = "PASS" if score >= FACE_SCORE_THRESHOLD else "FAIL"
        logger.info(
            f"\n{'='*50}\n"
            f"  FACE SIMILARITY: {score:.2f}%  [{status}]  (threshold: {FACE_SCORE_THRESHOLD})\n"
            f"{'='*50}"
        )
        return score
    except (ValueError, TypeError):
        logger.warning(f"[AIO] Could not parse face score: '{raw}', assuming 100")
        return 100.0


LORA_SWITCH_NAMES = {1: "none", 2: "PenisLora", 3: "multiConceptNSFW"}


async def _decompose_request(
    image_context: str | None,
    pose_description: str | None,
    outfit_description: str | None,
    gender: str = "woman",
    user_message: str | None = None,
) -> list[dict]:
    """Call Grok to decompose the user's image request into sequential editing steps.

    Returns a list of 1-3 step dicts: [{"prompt": "...", "switch": 1|2|3}, ...]
    """
    # Build context from what the chat service returned
    parts = []
    parts.append(f"Subject gender: {gender}")
    if user_message:
        parts.append(f"User said: {user_message}")
    if pose_description:
        parts.append(f"Pose: {pose_description}")
    if outfit_description:
        parts.append(f"Outfit: {outfit_description}")
    if image_context:
        parts.append(f"Scene/context: {image_context}")

    if not parts:
        return [{"prompt": "a natural photo, keep everything the same", "switch": 1}]

    user_request = ". ".join(parts)

    system_prompt = f"""Break the user's image request into 1-{MAX_STEPS} sequential editing steps for an AI image editor. Subject: {gender}.

INPUT FORMAT — the request contains these fields (some may be absent):
- "User said:" — the original chat message from the user
- "Pose:" — specific body pose to apply (from chat intent detection)
- "Outfit:" — clothing or nudity description (from chat intent detection)
- "Scene/context:" — environment, background, or mood description
Use ALL provided fields. If Pose is given, create a pose step. If Outfit is given, create an outfit step. If Scene/context is given, create a background step.

RULES:
- Minimum steps needed. Single-aspect request = 1 step.
- Order: pose FIRST, then outfit, then background LAST.
- No lighting changes unless user explicitly asks.
- Each step prompt: under 30 words, end with "Do not change facial features, keep face identity, normalize any bad anatomy".

LORA SWITCH per step:
- 1 = SFW (outfit, pose, background, normal scenes)
- 2 = PenisLora (prompt MUST contain "dick/cock/penis" and triggerword PENISLORA to activate; use for male nude/naked)
- 3 = multiConceptNSFW (for female NSFW; prompt MUST contain trigger words: nsfw, cum_on_face, blowjob, cowgirlout, creamp1e, penis, l1ck, missionary, nipples, reversecowgirlpov, vagina)
Female NSFW → switch 3. Male penis-only → switch 2.

Respond ONLY with a JSON array: [{{"prompt":"...","switch":N}}, ...]

"""

    ts = datetime.now(timezone.utc).isoformat()
    _file_logger.debug(
        f"\n{'='*72}\n"
        f"[{ts}] AIO DECOMPOSE\n"
        f"MODEL: {settings.XAI_MODEL}\n"
        f"USER REQUEST: {user_request}\n"
        f"SYSTEM PROMPT:\n{system_prompt}\n"
        f"{'-'*72}"
    )

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

            _file_logger.debug(f"RAW RESPONSE:\n{content}\n{'-'*72}")

            # Handle markdown code blocks
            if content.startswith("```"):
                content = content.split("\n", 1)[1]
                content = content.rsplit("```", 1)[0].strip()

            steps = json.loads(content)

            _file_logger.debug(
                f"PARSED ({len(steps) if isinstance(steps, list) else 'NOT A LIST'}):\n"
                f"  {json.dumps(steps, indent=2)}"
            )

            if isinstance(steps, list) and steps:
                # Normalize: accept both old string format and new object format
                normalized = []
                for s in steps[:MAX_STEPS]:
                    if isinstance(s, str):
                        normalized.append({"prompt": s, "switch": 1})
                    elif isinstance(s, dict) and "prompt" in s:
                        sw = s.get("switch", 1)
                        if sw not in (1, 2, 3):
                            sw = 1
                        normalized.append({"prompt": s["prompt"], "switch": sw})

                _file_logger.debug(
                    f"NORMALIZED ({len(normalized)}):\n"
                    f"  {json.dumps(normalized, indent=2)}\n"
                    f"{'='*72}"
                )
                logger.info(f"[AIO] Grok decomposed into {len(normalized)} steps: {normalized}")
                return normalized

            _file_logger.debug(f"UNEXPECTED FORMAT — falling back\n{'='*72}")

    except json.JSONDecodeError as exc:
        _file_logger.debug(
            f"JSON PARSE FAILED [{datetime.now(timezone.utc).isoformat()}]:\n"
            f"  error: {exc}\n"
            f"  raw ({len(content)} chars):\n{content}\n"
            f"{'='*72}"
        )
        logger.error(f"[AIO] Decompose JSON parse failed: {exc}, falling back")

    except Exception as e:
        _file_logger.debug(
            f"DECOMPOSE ERROR [{datetime.now(timezone.utc).isoformat()}]:\n"
            f"  {type(e).__name__}: {e}\n"
            f"{'='*72}"
        )
        logger.error(f"[AIO] Decompose failed: {e}, falling back to single step")

    # Fallback
    fallback = [{"prompt": f"{user_request}. Do not change facial features, keep face identity", "switch": 1}]
    _file_logger.debug(f"FALLBACK: {fallback}\n{'='*72}")
    return fallback


class ImageBridge:
    def __init__(self):
        self.comfyui_address = settings.COMFYUI_SERVER_ADDRESS
        self._queue: asyncio.Queue | None = None
        self._worker_task: asyncio.Task | None = None
        logger.info("AIO MODE ENABLED — using edit_final_AIO.json for all image tasks")

    MAX_QUEUE_SIZE = 10

    def _ensure_worker(self):
        """Start the queue worker if not already running."""
        if self._queue is None:
            self._queue = asyncio.Queue(maxsize=self.MAX_QUEUE_SIZE)
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(self._worker())

    async def _worker(self):
        """Process image generation requests one at a time."""
        while True:
            future, args = await self._queue.get()
            try:
                result = await self._generate_image_internal(**args)
                future.set_result(result)
            except Exception as e:
                future.set_result({"status": "failed", "error": str(e)})
            finally:
                self._queue.task_done()

    async def generate_image(
        self,
        ref_image_path: str,
        prompt: str,
        pose_description: str | None = None,
        outfit_description: str | None = None,
        gender: str = "woman",
        user_message: str | None = None,
    ) -> dict:
        """Queue an image generation request. Returns when it's done."""
        self._ensure_worker()

        future = asyncio.get_event_loop().create_future()
        args = {
            "ref_image_path": ref_image_path,
            "prompt": prompt,
            "pose_description": pose_description,
            "outfit_description": outfit_description,
            "gender": gender,
            "user_message": user_message,
        }

        if self._queue.full():
            logger.warning(f"[QUEUE] Rejected — queue full ({self.MAX_QUEUE_SIZE} jobs)")
            return {"status": "failed", "error": f"Queue full ({self.MAX_QUEUE_SIZE} jobs). Try again later."}

        queue_size = self._queue.qsize()
        if queue_size > 0:
            logger.info(f"[QUEUE] Request queued. Position: {queue_size + 1} (waiting for {queue_size} ahead)")
        else:
            logger.info(f"[QUEUE] Processing immediately (no queue)")

        await self._queue.put((future, args))
        return await future

    async def _generate_image_internal(
        self,
        ref_image_path: str,
        prompt: str,
        pose_description: str | None = None,
        outfit_description: str | None = None,
        gender: str = "woman",
        user_message: str | None = None,
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
        steps = await _decompose_request(
            prompt, pose_description, outfit_description,
            gender=gender, user_message=user_message,
        )
        logger.info(f"[AIO] Pipeline: {len(steps)} steps to execute")

        # Run the blocking ComfyUI pipeline in a thread so the event loop
        # stays free for other requests (chat, health, etc.)
        return await asyncio.to_thread(
            self._run_comfyui_pipeline, ref_image_path, steps,
        )

    def _run_comfyui_pipeline(self, ref_image_path: str, steps: list[dict]) -> dict:
        """Blocking ComfyUI pipeline — runs in a worker thread."""
        import time
        last_error = None

        for pipeline_attempt in range(1, MAX_PIPELINE_RETRIES + 1):
            client = ComfyUIClient(self.comfyui_address)

            try:
                client.connect()

                if pipeline_attempt > 1:
                    logger.info(
                        f"\n{'!'*50}\n"
                        f"  PIPELINE RETRY {pipeline_attempt}/{MAX_PIPELINE_RETRIES} (previous error: {last_error})\n"
                        f"{'!'*50}"
                    )

                # Upload the original ref image (stays constant for face similarity)
                ref_data, ref_ct, w, h = _read_image(ref_image_path)
                ref_filename = os.path.basename(ref_image_path)
                ref_comfyui = client.upload_image(ref_filename, ref_data, ref_ct)

                # Current input starts as the ref image
                current_input_name = ref_comfyui
                final_image_bytes = None
                step_results = []  # Track results per step for reporting

                for step_idx, step_info in enumerate(steps):
                    step_t0 = time.monotonic()
                    step_prompt = step_info["prompt"]
                    step_switch = step_info.get("switch", 1)
                    lora_name = LORA_SWITCH_NAMES.get(step_switch, "unknown")
                    logger.info(
                        f"[AIO] Step {step_idx + 1}/{len(steps)}: '{step_prompt[:60]}...' "
                        f"| LoRA switch={step_switch} ({lora_name})"
                    )

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
                            lora_switch=step_switch,
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
                        raise RuntimeError(f"Step {step_idx + 1} produced no images")

                    if best_score < FACE_SCORE_THRESHOLD:
                        logger.warning(
                            f"[AIO] Step {step_idx + 1}: best face_score={best_score:.1f} "
                            f"(below {FACE_SCORE_THRESHOLD}), using best attempt anyway"
                        )

                    # Save this step's output image so it can be shown in Google Chat
                    step_saved = _save_output(best_image)

                    step_duration = round(time.monotonic() - step_t0, 1)
                    logger.info(f"[AIO] Step {step_idx + 1} completed in {step_duration}s")

                    step_results.append({
                        "prompt": step_prompt[:80],
                        "face_score": best_score,
                        "attempts": attempt + 1,
                        "image_url": step_saved["local"],
                        "supabase_url": step_saved["supabase"],
                        "lora_switch": step_switch,
                        "lora_name": lora_name,
                        "duration": step_duration,
                    })

                    final_image_bytes = best_image

                    # Upload this step's output as input for the next step
                    if step_idx < len(steps) - 1:
                        next_input_name = f"aio_step{step_idx + 1}_{uuid.uuid4().hex[:8]}.png"
                        current_input_name = client.upload_image(
                            next_input_name, best_image, "image/png"
                        )
                        logger.info(f"[AIO] Chained step {step_idx + 1} output → step {step_idx + 2} input")

                # Save original ref image so Google Chat can show it
                original_saved = _save_output(ref_data)

                # Save final result
                final_saved = _save_output(final_image_bytes)
                logger.info(f"[AIO] Pipeline complete: {len(steps)} steps, final image: {final_saved['local']}")
                return {
                    "status": "succeeded",
                    "image_url": final_saved["local"],
                    "supabase_url": final_saved["supabase"],
                    "steps": step_results,
                    "original_image_url": original_saved["local"],
                    "original_supabase_url": original_saved["supabase"],
                }

            except Exception as e:
                last_error = str(e)
                logger.error(f"[AIO] Pipeline attempt {pipeline_attempt}/{MAX_PIPELINE_RETRIES} failed: {e}")
                if pipeline_attempt < MAX_PIPELINE_RETRIES:
                    logger.info(f"[AIO] Will retry entire pipeline...")
                else:
                    logger.error(f"[AIO] All {MAX_PIPELINE_RETRIES} pipeline attempts failed", exc_info=True)
                    return {"status": "failed", "error": last_error}
            finally:
                client.disconnect()

        return {"status": "failed", "error": last_error}
