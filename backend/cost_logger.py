"""
Grok API cost tracker — logs all API calls with token usage and cost to file + terminal.

Pricing (grok-4-1-fast-reasoning):
  Input:  $0.20 / 1M tokens
  Output: $0.50 / 1M tokens

Rate limits:
  Context: 2,000,000 tokens
  TPM: 4M
  RPM: 480
"""
import json
import logging
import os
from datetime import datetime, timezone

logger = logging.getLogger("cost_logger")

_dir = os.path.dirname(__file__)
COST_LOG_FILE = os.path.join(_dir, "grok_cost.log")

# Pricing per 1M tokens
PRICING = {
    "grok-4-1-fast-reasoning": {"input": 0.20, "output": 0.50},
    "grok-4-fast-reasoning": {"input": 0.20, "output": 0.50},
    "grok-imagine-image": {"per_image": 0.10},  # estimate
}

# Running session totals
_session = {
    "total_input_tokens": 0,
    "total_output_tokens": 0,
    "total_cost_usd": 0.0,
    "total_calls": 0,
    "total_images_generated": 0,
}


def _get_pricing(model: str) -> dict:
    """Get pricing for a model, fallback to default."""
    for key in PRICING:
        if key in model:
            return PRICING[key]
    return PRICING["grok-4-1-fast-reasoning"]


def log_chat_call(
    model: str,
    endpoint: str,
    input_tokens: int,
    output_tokens: int,
    caller: str = "",
):
    """Log a chat/completions API call with token usage and cost."""
    pricing = _get_pricing(model)
    input_cost = (input_tokens / 1_000_000) * pricing.get("input", 0.20)
    output_cost = (output_tokens / 1_000_000) * pricing.get("output", 0.50)
    call_cost = input_cost + output_cost

    _session["total_input_tokens"] += input_tokens
    _session["total_output_tokens"] += output_tokens
    _session["total_cost_usd"] += call_cost
    _session["total_calls"] += 1

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "type": "chat",
        "model": model,
        "endpoint": endpoint,
        "caller": caller,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "input_cost_usd": round(input_cost, 6),
        "output_cost_usd": round(output_cost, 6),
        "call_cost_usd": round(call_cost, 6),
        "session_total_usd": round(_session["total_cost_usd"], 6),
    }

    # Log to terminal
    logger.info(
        f"[COST] {caller or 'chat'} | model={model} | "
        f"in={input_tokens} out={output_tokens} tokens | "
        f"cost=${call_cost:.4f} | session_total=${_session['total_cost_usd']:.4f}"
    )

    # Append to file
    with open(COST_LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


def log_image_call(
    model: str,
    n_images: int = 1,
    caller: str = "",
):
    """Log an image generation API call."""
    pricing = _get_pricing(model)
    call_cost = n_images * pricing.get("per_image", 0.10)

    _session["total_cost_usd"] += call_cost
    _session["total_calls"] += 1
    _session["total_images_generated"] += n_images

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "type": "image_generation",
        "model": model,
        "caller": caller,
        "n_images": n_images,
        "call_cost_usd": round(call_cost, 6),
        "session_total_usd": round(_session["total_cost_usd"], 6),
    }

    logger.info(
        f"[COST] {caller or 'image'} | model={model} | "
        f"images={n_images} | cost=${call_cost:.4f} | "
        f"session_total=${_session['total_cost_usd']:.4f}"
    )

    with open(COST_LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")


def get_session_summary() -> dict:
    """Return current session cost summary."""
    return {
        "total_calls": _session["total_calls"],
        "total_input_tokens": _session["total_input_tokens"],
        "total_output_tokens": _session["total_output_tokens"],
        "total_images_generated": _session["total_images_generated"],
        "total_cost_usd": round(_session["total_cost_usd"], 6),
    }
