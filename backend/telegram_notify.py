"""
Telegram channel — sends image generation results as photo albums.
Posts input image, step images, and final output with captions.
"""
import html
import json
import logging

import requests

from cost_logger import get_session_summary

logger = logging.getLogger(__name__)

TELEGRAM_TOKEN = "8623724838:AAFn2XGr0y5RI190AKD00PUlp8EoyoVzS1Q"
CHANNEL_ID = "-1003854175419"
API_BASE = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"


def _send_media_group(media: list[dict]):
    """Send a media group (photo album) to the Telegram channel."""
    if not TELEGRAM_TOKEN or not CHANNEL_ID:
        return
    try:
        resp = requests.post(
            f"{API_BASE}/sendMediaGroup",
            data={
                "chat_id": CHANNEL_ID,
                "media": json.dumps(media),
            },
            timeout=30,
        )
        if resp.status_code == 200:
            logger.info("[Telegram] Media group sent")
        else:
            logger.warning("[Telegram] Returned %s: %s", resp.status_code, resp.text[:300])
    except Exception as e:
        logger.error("[Telegram] Failed to send: %s", e)


def send_generation_result(
    character_name: str,
    user_message: str,
    ai_message: str,
    image_url: str | None = None,
    original_image_url: str | None = None,
    steps: list[dict] | None = None,
    duration: float = 0.0,
    workflow_type: str = "unknown",
):
    """Send generation result as a Telegram photo album."""
    media = []

    # 1. Input/original image with header caption
    cost = get_session_summary()
    header = (
        f"{character_name} — Image Generated\n"
        f"Duration: {duration:.1f}s\n"
        f"Workflow: {html.escape(workflow_type)}\n"
        f"\n"
        f"User: {html.escape(user_message[:200])}\n"
    )

    if original_image_url:
        media.append({
            "type": "photo",
            "media": original_image_url,
            "caption": header,
        })

    # 2. Step images
    if steps:
        for i, step in enumerate(steps):
            step_img = step.get("image_url")
            if not step_img:
                continue
            score = step.get("face_score")
            score_str = f"{score:.1f}%" if score is not None else "N/A"
            attempts = step.get("attempts", 1)
            lora_name = step.get("lora_name", "none")
            lora_switch = step.get("lora_switch", 1)
            step_duration = step.get("duration", 0)
            prompt = html.escape(step.get("prompt", "N/A"))[:200]

            caption = (
                f"Step {i + 1}/{len(steps)}\n"
                f"Prompt: {prompt}\n"
                f"LoRA: {lora_name} (switch {lora_switch})\n"
                f"Face sim: {score_str}\n"
                f"Attempts: {attempts}\n"
                f"Duration: {step_duration}s"
            )
            media.append({
                "type": "photo",
                "media": step_img,
                "caption": caption,
            })

    # 3. Final output image with cost
    if image_url:
        final_caption = (
            f"Final Output\n"
            f"\n"
            f"Session\n"
            f"API: {cost['total_calls']} calls\n"
            f"Cost: ${cost['total_cost_usd']:.4f}\n"
            f"Tokens: {cost['total_input_tokens']:,} in / {cost['total_output_tokens']:,} out"
        )
        media.append({
            "type": "photo",
            "media": image_url,
            "caption": final_caption,
        })

    if not media:
        return

    # Telegram sendMediaGroup requires 2-10 photos
    if len(media) < 2:
        # Only one image — send as single photo
        _send_single_photo(media[0])
        return

    # Telegram limits to 10 per group
    _send_media_group(media[:10])


def _send_single_photo(photo: dict):
    """Fallback: send a single photo message."""
    if not TELEGRAM_TOKEN or not CHANNEL_ID:
        return
    try:
        resp = requests.post(
            f"{API_BASE}/sendPhoto",
            data={
                "chat_id": CHANNEL_ID,
                "photo": photo["media"],
                "caption": photo.get("caption", ""),
            },
            timeout=30,
        )
        if resp.status_code == 200:
            logger.info("[Telegram] Single photo sent")
        else:
            logger.warning("[Telegram] Returned %s: %s", resp.status_code, resp.text[:300])
    except Exception as e:
        logger.error("[Telegram] Failed to send: %s", e)
