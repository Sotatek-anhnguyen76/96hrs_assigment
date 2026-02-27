"""
Generates pose reference images using xAI's image generation API (grok-imagine-image).
These are used as the pose reference input (node 170) in the ComfyUI pose workflow.
"""
import logging
import httpx

from config import settings
from cost_logger import log_image_call

logger = logging.getLogger(__name__)


async def generate_pose_image(pose_description: str) -> bytes:
    """
    Generate a pose reference image from a text description using xAI Aurora.

    Args:
        pose_description: Description of the body pose (e.g. "A person sitting
            casually on a chair, legs crossed, looking at camera")

    Returns:
        Image bytes of the generated pose reference

    Raises:
        RuntimeError: If image generation fails
    """
    # Enhance the pose description to get a clean, usable reference
    prompt = (
        f"A single person in a photo, {pose_description}. "
        "Full body visible, simple clean background, natural lighting, "
        "realistic photograph style."
    )

    logger.info(f"Generating pose reference image: '{prompt[:80]}...'")

    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{settings.XAI_BASE_URL.rstrip('/')}/images/generations",
            headers={
                "Authorization": f"Bearer {settings.XAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": settings.XAI_IMAGE_MODEL,
                "prompt": prompt,
                "n": 1,
            },
            timeout=60.0,
        )
        resp.raise_for_status()
        result = resp.json()

        log_image_call(
            model=settings.XAI_IMAGE_MODEL,
            n_images=1,
            caller="pose_generator",
        )

        # Get the image URL from response
        data = result.get("data", [])
        if not data:
            raise RuntimeError(f"No images returned from xAI: {result}")

        image_url = data[0].get("url")
        if not image_url:
            # Try b64_json fallback
            import base64
            b64 = data[0].get("b64_json")
            if b64:
                logger.info("Got pose image as base64")
                return base64.b64decode(b64)
            raise RuntimeError(f"No image URL or b64 in response: {data[0].keys()}")

        # Download the image
        logger.info(f"Downloading pose image from: {image_url[:60]}...")
        img_resp = await client.get(image_url, timeout=30.0)
        img_resp.raise_for_status()

        logger.info(f"Pose reference image generated: {len(img_resp.content)} bytes")
        return img_resp.content
