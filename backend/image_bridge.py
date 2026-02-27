"""
Image generation — talks directly to ComfyUI.
Uses the pose workflow: character ref image + xAI-generated pose reference.
Falls back to the outfit workflow (SAM3 segmentation) if no pose description is provided.
"""
import hashlib
import json
import logging
import os
import random
import uuid
from io import BytesIO

from PIL import Image

from config import settings
from comfyui_client import ComfyUIClient
from pose_generator import generate_pose_image

logger = logging.getLogger(__name__)

# Load workflow templates once at import
_dir = os.path.dirname(__file__)

with open(os.path.join(_dir, "workflows", "Pose.json"), "r") as f:
    POSE_WORKFLOW_TEMPLATE = json.load(f)

with open(os.path.join(_dir, "workflows", "Outfit.json"), "r") as f:
    OUTFIT_WORKFLOW_TEMPLATE = json.load(f)

# Local storage for generated images
STORAGE_DIR = os.path.join(_dir, "storage", "images")
os.makedirs(STORAGE_DIR, exist_ok=True)


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


class ImageBridge:
    def __init__(self):
        self.comfyui_address = settings.COMFYUI_SERVER_ADDRESS

    async def generate_image(
        self,
        ref_image_path: str,
        prompt: str,
        pose_description: str | None = None,
        outfit_description: str | None = None,
    ) -> dict:
        """
        Generate a character image using ComfyUI.

        If pose_description is provided:
            → POSE workflow: xAI generates pose ref → ComfyUI applies pose with identity preservation

        If outfit_description is provided:
            → OUTFIT workflow: SAM3 segments clothing → inpaints new outfit

        If neither:
            → OUTFIT workflow with the image_context prompt as fallback

        Args:
            ref_image_path: Local path to the character's reference image
            prompt: Scene/image context description (fallback for outfit workflow)
            pose_description: Body pose description for xAI image generation
            outfit_description: Outfit change instruction for SAM3 workflow

        Returns:
            {"status": "succeeded", "image_url": str}
            or {"status": "failed", "error": str}
        """
        if not os.path.exists(ref_image_path):
            return {"status": "failed", "error": "Reference image not found"}

        if pose_description:
            return await self._run_pose_workflow(ref_image_path, pose_description)
        else:
            outfit_prompt = outfit_description or prompt
            return await self._run_outfit_workflow(ref_image_path, outfit_prompt)

    async def _run_pose_workflow(self, ref_image_path: str, pose_description: str) -> dict:
        """Generate image using pose workflow: ref image + xAI-generated pose image."""
        client = ComfyUIClient(self.comfyui_address)

        try:
            # Step 1: Generate pose reference image via xAI
            logger.info(f"Generating pose reference: '{pose_description[:60]}...'")
            pose_bytes = await generate_pose_image(pose_description)

            # Step 2: Connect to ComfyUI and upload both images
            client.connect()

            # Upload character reference image
            ref_data, ref_ct, _, _ = _read_image(ref_image_path)
            ref_filename = os.path.basename(ref_image_path)
            ref_comfyui = client.upload_image(ref_filename, ref_data, ref_ct)
            logger.info(f"Uploaded character ref: {ref_comfyui}")

            # Upload generated pose reference image
            pose_filename = f"pose_{uuid.uuid4().hex[:8]}.png"
            pose_comfyui = client.upload_image(pose_filename, pose_bytes, "image/png")
            logger.info(f"Uploaded pose ref: {pose_comfyui}")

            # Step 3: Prepare and run pose workflow
            seed = random.randint(1, 1_000_000_000)
            workflow = ComfyUIClient.prepare_pose_workflow(
                POSE_WORKFLOW_TEMPLATE,
                source_image_name=ref_comfyui,
                pose_image_name=pose_comfyui,
                seed=seed,
            )

            logger.info(f"Running pose workflow: seed={seed}")
            output_images = client.execute_workflow(workflow)

            # Node 164 = PreviewImage output in pose workflow
            images = (
                output_images.get("164")
                or output_images.get("8")
                or next(iter(output_images.values()), [])
            )
            if not images:
                return {"status": "failed", "error": "No images from pose workflow"}

            url = _save_output(images[0])
            original_image_url = _save_output(ref_data)
            logger.info(f"Pose workflow complete: {url}")
            return {"status": "succeeded", "image_url": url, "original_image_url": original_image_url}

        except Exception as e:
            logger.error(f"Pose workflow error: {e}", exc_info=True)
            return {"status": "failed", "error": str(e)}
        finally:
            client.disconnect()

    async def _run_outfit_workflow(self, ref_image_path: str, prompt: str) -> dict:
        """Outfit workflow: SAM3 segmentation-based outfit editing."""
        client = ComfyUIClient(self.comfyui_address)

        try:
            client.connect()

            ref_data, ref_ct, w, h = _read_image(ref_image_path)
            ref_filename = os.path.basename(ref_image_path)
            ref_comfyui = client.upload_image(ref_filename, ref_data, ref_ct)

            seed = random.randint(1, 1_000_000_000)
            workflow = ComfyUIClient.prepare_outfit_workflow(
                OUTFIT_WORKFLOW_TEMPLATE,
                input_image_name=ref_comfyui,
                prompt=prompt,
                seed=seed,
                width=w,
                height=h,
            )

            logger.info(f"Running outfit workflow: prompt='{prompt[:60]}...' seed={seed}")
            output_images = client.execute_workflow(workflow)

            # Node 116 = PreviewImage output in outfit workflow
            images = (
                output_images.get("116")
                or next(iter(output_images.values()), [])
            )
            if not images:
                return {"status": "failed", "error": "No images from outfit workflow"}

            url = _save_output(images[0])
            original_image_url = _save_output(ref_data)
            logger.info(f"Outfit workflow complete: {url}")
            return {"status": "succeeded", "image_url": url, "original_image_url": original_image_url}

        except Exception as e:
            logger.error(f"Outfit workflow error: {e}", exc_info=True)
            return {"status": "failed", "error": str(e)}
        finally:
            client.disconnect()
