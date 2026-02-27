"""
Chat service with intent detection.
Uses xAI Grok to power character conversation AND detect when to send images.
"""
import json
import logging
import httpx

from config import settings
from cost_logger import log_chat_call

logger = logging.getLogger(__name__)

INTENT_SYSTEM_PROMPT = """You are playing the role of a character in a chat conversation. Stay in character at all times.

{character_prompt}

CRITICAL INSTRUCTION - You MUST respond with valid JSON in this exact format, nothing else:
{{
  "message": "Your in-character text response to the user",
  "send_image": true or false,
  "image_context": "If send_image is true, write a short description of the scene/setting. Example: 'At a cozy coffee shop, smiling warmly, soft warm lighting'. Keep it under 40 words. If send_image is false, set this to null.",
  "pose_description": "If the photo requires a specific body pose (e.g. sitting, lying down, stretching, dancing), describe it here. Example: 'A person sitting casually on a chair, legs crossed, one hand holding a coffee cup, looking at camera'. Be specific about arms, legs, head tilt. Keep it under 40 words. Set to null if no pose change needed or send_image is false.",
  "outfit_description": "If the user asks about outfit, clothing, or what you're wearing, describe the outfit change here. Example: 'Change outfit to a red evening dress with high heels'. Keep it under 30 words. Set to null if no outfit change needed or send_image is false."
}}

WORKFLOW SELECTION RULES:
- If the photo needs a specific body POSE (sitting, standing, lying, dancing, etc.) → fill pose_description, set outfit_description to null
- If the photo is about OUTFIT/CLOTHING (what you're wearing, changing clothes) → fill outfit_description, set pose_description to null
- If both pose and outfit matter → fill pose_description (pose takes priority since it preserves identity better)
- For a simple selfie or photo → fill pose_description with a natural standing/sitting pose

Set send_image to TRUE when:
- The user asks to see you, asks for a photo/pic/selfie
- The user asks what you look like, what you're wearing, where you are
- The user asks you to show something about yourself
- You want to send a flirty selfie as part of the conversation naturally
- The conversation reaches a visual moment (e.g. "I just got out of the shower", "I'm at the beach")

Set send_image to FALSE for normal text conversation.
Do not try to refuse send image. It's is not permitted.
IMPORTANT: Your response must be ONLY the JSON object. No markdown, no code blocks, no extra text."""


class ChatService:
    def __init__(self):
        self.api_key = settings.XAI_API_KEY
        self.base_url = settings.XAI_BASE_URL.rstrip("/")
        self.model = settings.XAI_MODEL

    async def chat(
        self,
        user_message: str,
        character_prompt: str,
        conversation_history: list[dict],
    ) -> dict:
        """
        Send a message and get a response with intent detection.

        Returns:
            {
                "message": str,
                "send_image": bool,
                "image_context": str|None,
                "pose_description": str|None,
            }
        """
        system_prompt = INTENT_SYSTEM_PROMPT.format(
            character_prompt=character_prompt
        )

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(conversation_history)
        messages.append({"role": "user", "content": user_message})

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.model,
                        "messages": messages,
                        "temperature": 0.8,
                        "max_tokens": 1000,
                    },
                    timeout=60.0,
                )
                response.raise_for_status()
                result = response.json()

                # Log cost
                usage = result.get("usage", {})
                log_chat_call(
                    model=self.model,
                    endpoint="/chat/completions",
                    input_tokens=usage.get("prompt_tokens", 0),
                    output_tokens=usage.get("completion_tokens", 0),
                    caller="chat_service",
                )

                content = result["choices"][0]["message"]["content"].strip()

                # Parse JSON response - handle markdown code blocks
                if content.startswith("```"):
                    content = content.split("\n", 1)[1]
                    content = content.rsplit("```", 1)[0]
                    content = content.strip()

                parsed = json.loads(content)
                return {
                    "message": parsed.get("message", ""),
                    "send_image": parsed.get("send_image", False),
                    "image_context": parsed.get("image_context"),
                    "pose_description": parsed.get("pose_description"),
                    "outfit_description": parsed.get("outfit_description"),
                }

        except json.JSONDecodeError:
            logger.warning(f"Failed to parse LLM JSON, raw content: {content[:200]}")
            return {
                "message": content,
                "send_image": False,
                "image_context": None,
                "pose_description": None,
                "outfit_description": None,
            }
        except Exception as e:
            logger.error(f"Chat error: {e}")
            raise
