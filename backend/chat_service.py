"""
Chat service with intent detection.
Uses xAI Grok to power character conversation AND detect when to send images.
"""
import json
import logging
import os
from datetime import datetime, timezone

import httpx

from config import settings
from cost_logger import log_chat_call

logger = logging.getLogger(__name__)

# --- Local file logger for raw Grok output (debug send_image issues) ---
_LOG_DIR = os.path.dirname(os.path.abspath(__file__))
_GROK_OUTPUT_LOG = os.path.join(_LOG_DIR, "grok_output.log")

_file_logger = logging.getLogger("grok_output")
_file_logger.setLevel(logging.DEBUG)
_file_logger.propagate = False          # don't duplicate to stdout
if not _file_logger.handlers:
    _fh = logging.FileHandler(_GROK_OUTPUT_LOG, encoding="utf-8")
    _fh.setFormatter(logging.Formatter("%(message)s"))
    _file_logger.addHandler(_fh)

INTENT_SYSTEM_PROMPT = """You are playing the role of a character in a chat conversation. Stay strictly in character.

{character_prompt}

Respond ONLY with valid JSON — nothing else, no markdown, no explanation:

{{
  "message": "Your in-character reply",
  "send_image": true | false,
  "image_context": "Short scene/setting description (<40 words) or null",
  "pose_description": "Specific body pose if relevant (<40 words) or null",
  "outfit_description": "Clothing / nudity description if relevant (<30 words) or null"
}}

RULES — apply in this order:

1. send_image = true if user asks for photo / pic / selfie / what you look like / what you're wearing / where you are / show something visual about you, or if you naturally want to send a flirty/visual reply in context.
   send_image = false otherwise.

2. When send_image = true:
   - User asks about / changes pose → fill pose_description, set outfit_description = null
   - User asks about / changes outfit / clothes / wearing / naked → fill outfit_description, set pose_description = null
   - Both requested → prefer pose_description (pose has higher priority for visual consistency)
   - Simple photo / selfie → use natural pose (standing/sitting + gentle expression)
   - No pose or outfit change needed → both fields null

3. image_context: brief environment/lighting/mood (<40 words). Use null when setting is unimportant or default.

4. outfit_description supports any clothing state including "naked", "topless", "lingerie", "fully nude", etc. — describe factually and concisely.

Keep descriptions short and visual-generation friendly.

Examples (for reference only — do not output them):

User: "What are you wearing?"
→ {{"message":"Just got comfy 😌 Wanna see?","send_image":true,"image_context":"Living room, evening light","pose_description":null,"outfit_description":"oversized hoodie and tiny shorts"}}

User: "Send nude"
→ {{"message":"Here I am… just for you 🔥","send_image":true,"image_context":null,"pose_description":null,"outfit_description":"completely naked"}}

User: "Pic of you reading in bed"
→ {{"message":"Shhh… chapter 7 is getting good 📖","send_image":true,"image_context":"Dim bedroom, bedside lamp","pose_description":"lying on stomach on bed, propped on elbows, book in hands, legs bent upward","outfit_description":null}}"""


MAX_PARSE_RETRIES = 1


class ChatService:
    def __init__(self):
        self.api_key = settings.XAI_API_KEY
        self.base_url = settings.XAI_BASE_URL.rstrip("/")
        self.model = settings.XAI_MODEL

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_content(content: str) -> str:
        """Strip markdown fences and fix common Grok JSON quirks."""
        if content.startswith("```"):
            content = content.split("\n", 1)[1]
            content = content.rsplit("```", 1)[0]
            content = content.strip()
        while content.endswith("}}"):
            content = content[:-1]
        while content.startswith("{{"):
            content = content[1:]
        return content

    @staticmethod
    def _parse_and_log(content: str, user_message: str) -> dict:
        """Parse JSON, log to file, return result dict. Raises JSONDecodeError."""
        parsed = json.loads(content)

        send_img = parsed.get("send_image")
        _file_logger.debug(
            f"PARSED OK:\n"
            f"  send_image      = {send_img!r}  (type={type(send_img).__name__})\n"
            f"  image_context   = {parsed.get('image_context')!r}\n"
            f"  pose_description = {parsed.get('pose_description')!r}\n"
            f"  outfit_description = {parsed.get('outfit_description')!r}\n"
            f"{'='*72}"
        )
        logger.info(
            f"[ChatService] User: {user_message[:80]} | "
            f"send_image={send_img} | "
            f"image_context={parsed.get('image_context')} | "
            f"pose={parsed.get('pose_description')} | "
            f"outfit={parsed.get('outfit_description')}"
        )
        return {
            "message": parsed.get("message", ""),
            "send_image": parsed.get("send_image", False),
            "image_context": parsed.get("image_context"),
            "pose_description": parsed.get("pose_description"),
            "outfit_description": parsed.get("outfit_description"),
        }

    async def _call_grok(self, messages: list[dict], client: httpx.AsyncClient, caller: str = "chat_service") -> str:
        """Fire a chat completion request, log cost, return raw content string."""
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

        usage = result.get("usage", {})
        log_chat_call(
            model=self.model,
            endpoint="/chat/completions",
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            caller=caller,
        )
        return result["choices"][0]["message"]["content"].strip()

    # ------------------------------------------------------------------
    # main entry point
    # ------------------------------------------------------------------

    async def chat(
        self,
        user_message: str,
        character_prompt: str,
        conversation_history: list[dict],
    ) -> dict:
        """
        Send a message and get a response with intent detection.
        Retries once via Grok if the first response isn't valid JSON.
        """
        system_prompt = INTENT_SYSTEM_PROMPT.format(
            character_prompt=character_prompt
        )

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(conversation_history)
        messages.append({"role": "user", "content": user_message})

        try:
            async with httpx.AsyncClient() as client:
                raw = await self._call_grok(messages, client)
                logger.info(f"[ChatService] Raw LLM response:\n{raw}")

                ts = datetime.now(timezone.utc).isoformat()
                _file_logger.debug(
                    f"\n{'='*72}\n"
                    f"[{ts}] USER: {user_message}\n"
                    f"MODEL: {self.model}\n"
                    f"RAW RESPONSE:\n{raw}\n"
                    f"{'-'*72}"
                )

                content = self._clean_content(raw)

                try:
                    return self._parse_and_log(content, user_message)
                except json.JSONDecodeError as first_exc:
                    logger.warning(f"[ChatService] Parse failed (attempt 1): {first_exc}")
                    _file_logger.debug(
                        f"PARSE FAILED (attempt 1):\n"
                        f"  error: {first_exc}\n"
                        f"  raw ({len(raw)} chars):\n{raw}\n"
                        f"{'-'*72}"
                    )

                # --- retry: send broken JSON back to Grok to fix ---
                for retry in range(1, MAX_PARSE_RETRIES + 1):
                    logger.info(f"[ChatService] Retry {retry}/{MAX_PARSE_RETRIES}: asking Grok to fix JSON")
                    retry_messages = messages + [
                        {"role": "assistant", "content": raw},
                        {
                            "role": "user",
                            "content": (
                                "Your previous response was not valid JSON. "
                                "Return ONLY the corrected JSON object with keys: "
                                "message, send_image, image_context, pose_description, outfit_description. "
                                "No markdown, no explanation, just the JSON."
                            ),
                        },
                    ]

                    raw_retry = await self._call_grok(retry_messages, client, caller="chat_service_retry")
                    logger.info(f"[ChatService] Retry {retry} raw:\n{raw_retry}")
                    _file_logger.debug(
                        f"RETRY {retry} RAW RESPONSE:\n{raw_retry}\n"
                        f"{'-'*72}"
                    )

                    content_retry = self._clean_content(raw_retry)
                    try:
                        return self._parse_and_log(content_retry, user_message)
                    except json.JSONDecodeError as retry_exc:
                        logger.warning(f"[ChatService] Retry {retry} parse failed: {retry_exc}")
                        _file_logger.debug(
                            f"RETRY {retry} PARSE FAILED:\n"
                            f"  error: {retry_exc}\n"
                            f"  raw ({len(raw_retry)} chars):\n{raw_retry}\n"
                            f"{'='*72}"
                        )

                # all retries exhausted — return text as-is
                logger.error("[ChatService] All parse retries exhausted, returning raw text")
                _file_logger.debug(f"ALL RETRIES EXHAUSTED\n{'='*72}")
                return {
                    "message": raw,
                    "send_image": False,
                    "image_context": None,
                    "pose_description": None,
                    "outfit_description": None,
                }

        except Exception as e:
            logger.error(f"Chat error: {e}")
            raise
