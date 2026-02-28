"""
Google Chat webhook — sends image generation results as rich cards.
Posts source image, output image, prompt, face similarity, duration, and cost info.
"""
import logging
import uuid

import requests

from config import settings
from cost_logger import get_session_summary

logger = logging.getLogger(__name__)

RESPONSE_WEBHOOK = settings.GOOGLE_CHAT_RESPONSE_WEBHOOK_URL
PAYLOAD_WEBHOOK = settings.GOOGLE_CHAT_PAYLOAD_WEBHOOK_URL


def _post(webhook_url: str, payload: dict):
    """Post to Google Chat webhook, log errors silently."""
    if not webhook_url:
        return
    try:
        resp = requests.post(webhook_url, json=payload, timeout=10)
        if resp.status_code == 200:
            logger.info("[GChat] Message sent")
        else:
            logger.warning("[GChat] Returned %s: %s", resp.status_code, resp.text[:200])
    except Exception as e:
        logger.error("[GChat] Failed to send: %s", e)


def send_generation_result(
    character_name: str,
    user_message: str,
    ai_message: str,
    image_url: str | None = None,
    original_image_url: str | None = None,
    steps: list[dict] | None = None,
    face_scores: list[float] | None = None,
    duration: float = 0.0,
    workflow_type: str = "unknown",
    webhook_url: str | None = None,
):
    """Send a generation result card to Google Chat.

    Args:
        character_name: Name of the character (e.g. "Luna")
        user_message: What the user said
        ai_message: Character's text reply
        image_url: Public URL of the final generated image (or None)
        original_image_url: Public URL of the original reference image
        steps: List of step dicts [{"prompt": "...", "face_score": 85.2, "image_url": "..."}, ...]
        face_scores: Face similarity scores per step
        duration: Total generation time in seconds
        workflow_type: Which workflow/switch was chosen (e.g. "pose", "outfit", "aio")
        webhook_url: Override webhook URL
    """
    url = webhook_url or RESPONSE_WEBHOOK
    if not url:
        return

    # Build sections
    sections = []

    # Chat section with workflow type
    sections.append({
        "header": "Chat",
        "widgets": [
            {"decoratedText": {"topLabel": "Workflow", "text": f"Switch: {workflow_type}", "wrapText": True}},
            {"decoratedText": {"topLabel": "User", "text": user_message, "wrapText": True}},
            {"decoratedText": {"topLabel": character_name, "text": ai_message, "wrapText": True}},
        ],
    })

    # Original reference image
    if original_image_url:
        sections.append({
            "header": "Original Image",
            "widgets": [
                {"image": {"imageUrl": original_image_url, "altText": f"{character_name} original"}},
                {
                    "buttonList": {
                        "buttons": [{
                            "text": "Open Original",
                            "onClick": {"openLink": {"url": original_image_url}},
                        }]
                    }
                },
            ],
        })

    # Steps section with images (if multi-step AIO)
    if steps:
        for i, step in enumerate(steps):
            score = step.get("face_score")
            score_str = f"{score:.1f}%" if score is not None else "N/A"
            attempts = step.get("attempts", 1)
            lora_switch = step.get("lora_switch", 1)
            lora_name = step.get("lora_name", "none")
            step_widgets = [
                {
                    "decoratedText": {
                        "topLabel": f"Step {i + 1} Prompt",
                        "text": f"{step.get('prompt', 'N/A')}",
                        "wrapText": True,
                    }
                },
                {
                    "decoratedText": {
                        "topLabel": "Details",
                        "text": f"LoRA: {lora_name} (switch {lora_switch}) | Face sim: {score_str} | Attempts: {attempts}",
                        "wrapText": True,
                    }
                },
            ]
            # Include step image + open button if available
            step_img = step.get("image_url")
            if step_img:
                step_widgets.append(
                    {"image": {"imageUrl": step_img, "altText": f"Step {i + 1} result"}},
                )
                step_widgets.append({
                    "buttonList": {
                        "buttons": [{
                            "text": f"Open Step {i + 1}",
                            "onClick": {"openLink": {"url": step_img}},
                        }]
                    }
                })
            sections.append({
                "header": f"Step {i + 1} / {len(steps)}",
                "widgets": step_widgets,
            })

    # Final output image
    if image_url:
        sections.append({
            "header": "Final Generated Image",
            "widgets": [
                {"image": {"imageUrl": image_url, "altText": f"{character_name} photo"}},
                {
                    "buttonList": {
                        "buttons": [{
                            "text": "Open Image",
                            "onClick": {"openLink": {"url": image_url}},
                        }]
                    }
                },
            ],
        })

    # Cost section
    cost = get_session_summary()
    sections.append({
        "header": "Session Cost",
        "widgets": [
            {
                "decoratedText": {
                    "topLabel": "API Calls / Cost",
                    "text": f"{cost['total_calls']} calls | ${cost['total_cost_usd']:.4f}",
                }
            },
            {
                "decoratedText": {
                    "topLabel": "Tokens",
                    "text": f"In: {cost['total_input_tokens']:,} | Out: {cost['total_output_tokens']:,}",
                }
            },
        ],
    })

    card = {
        "cardsV2": [{
            "cardId": f"nectar-{uuid.uuid4().hex[:8]}",
            "card": {
                "header": {
                    "title": f"{character_name} — Image Generated",
                    "subtitle": f"Duration: {duration:.1f}s",
                },
                "sections": sections,
            },
        }]
    }

    _post(url, card)


def send_text_only(
    character_name: str,
    user_message: str,
    ai_message: str,
    webhook_url: str | None = None,
):
    """Send a simple text-only chat message to Google Chat."""
    url = webhook_url or PAYLOAD_WEBHOOK
    if not url:
        return

    msg = {
        "text": (
            f"*{character_name}*\n\n"
            f"*User:* {user_message}\n"
            f"*{character_name}:* {ai_message}"
        )
    }
    _post(url, msg)


def send_error(
    error_message: str,
    context: str = "",
    webhook_url: str | None = None,
):
    """Send an error notification to Google Chat."""
    url = webhook_url or RESPONSE_WEBHOOK
    if not url:
        return

    msg = {
        "text": f"*ERROR*\n\n*Context:* {context}\n*Error:* {error_message}"
    }
    _post(url, msg)
