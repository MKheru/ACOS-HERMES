#!/usr/bin/env python3
"""
MiniMax Token Plan quota inspection tool.

Calls https://www.minimax.io/v1/token_plan/remains and returns the parsed
per-model breakdown (used / total / reset window) for the active MiniMax
credential. Lets the agent self-monitor its 5h-rolling and weekly budgets
before deciding to launch heavy tasks.

API reference: https://github.com/MiniMax-AI/MiniMax-M2/issues/99
The endpoint returns 11 model rows; we surface the ones the user actually
has quota on (interval_total > 0) plus a roll-up summary.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from agent.credential_pool import load_pool
from tools.registry import registry, tool_error

logger = logging.getLogger(__name__)

ENDPOINT = "https://www.minimax.io/v1/token_plan/remains"
TIMEOUT_S = 10

# Friendly name + dashboard panel for each MiniMax model the API reports.
_MODEL_LABELS = {
    "MiniMax-M*": ("Text Generation (M*)", "5h"),
    "speech-hd": ("Text-to-Speech HD", "24h"),
    "MiniMax-Hailuo-2.3-Fast-6s-768p": ("Video (Hailuo Fast)", "24h"),
    "MiniMax-Hailuo-2.3-6s-768p": ("Video (Hailuo)", "24h"),
    "music-2.5": ("Music 2.5", "24h"),
    "music-2.6": ("Music 2.6", "24h"),
    "music-cover": ("Music Cover", "24h"),
    "lyrics_generation": ("Lyrics Generation", "24h"),
    "image-01": ("Image Gen", "24h"),
    "coding-plan-vlm": ("Coding (VLM)", "5h"),
    "coding-plan-search": ("Coding (Search)", "5h"),
}


def _resolve_api_key() -> Optional[str]:
    """Pick the active MiniMax credential's API key, or None if none configured."""
    try:
        pool = load_pool("minimax")
    except Exception as exc:
        logger.warning("minimax_quota: load_pool failed: %s", exc)
        return None
    if pool is None:
        return None
    cred = pool.select()
    if cred is None:
        return None
    return cred.runtime_api_key or None


def _format_reset(remains_ms: int) -> str:
    """Format remaining-time milliseconds as 'XhYYm' or 'YYm'."""
    secs = max(0, int(remains_ms) // 1000)
    h, rem = divmod(secs, 3600)
    m = rem // 60
    if h:
        return f"{h}h{m:02d}m"
    return f"{m}m"


def _summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Reduce 11 raw rows to the active subset + flags."""
    summary: List[Dict[str, Any]] = []
    warnings: List[str] = []

    for row in rows:
        total_5h = row.get("current_interval_total_count", 0) or 0
        total_w = row.get("current_weekly_total_count", 0) or 0
        if total_5h == 0 and total_w == 0:
            continue  # model not part of this plan

        used_5h = row.get("current_interval_usage_count", 0) or 0
        used_w = row.get("current_weekly_usage_count", 0) or 0
        model_name = row.get("model_name", "?")
        label, window = _MODEL_LABELS.get(model_name, (model_name, "?"))

        pct_5h = round(100 * used_5h / total_5h, 1) if total_5h else None
        pct_w = round(100 * used_w / total_w, 1) if total_w else None

        item = {
            "model": model_name,
            "label": label,
            "interval_window": window,
            "interval_used": used_5h,
            "interval_total": total_5h,
            "interval_used_pct": pct_5h,
            "interval_resets_in": _format_reset(row.get("remains_time", 0)),
            "weekly_used": used_w,
            "weekly_total": total_w,
            "weekly_used_pct": pct_w,
            "weekly_resets_in": _format_reset(row.get("weekly_remains_time", 0)),
        }
        summary.append(item)

        # Warn at >=80% of any window
        if pct_5h is not None and pct_5h >= 80:
            warnings.append(
                f"{label} {pct_5h}% of {window} budget used "
                f"({used_5h}/{total_5h}, resets in {item['interval_resets_in']})"
            )
        if pct_w is not None and pct_w >= 80:
            warnings.append(
                f"{label} {pct_w}% of weekly budget used "
                f"({used_w}/{total_w}, resets in {item['weekly_resets_in']})"
            )

    return {"models": summary, "warnings": warnings}


def get_minimax_quota() -> str:
    """Fetch + parse the MiniMax Token Plan remains endpoint. Returns JSON string."""
    api_key = _resolve_api_key()
    if not api_key:
        return tool_error(
            "MiniMax credential not found. Run `hermes login --provider minimax` "
            "or check ~/.hermes/auth.json."
        )

    req = urllib.request.Request(
        ENDPOINT,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            # www.minimax.io is behind Cloudflare; the default urllib UA is
            # blocked (Cloudflare error 1010 / browser_signature_banned).
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        },
        method="GET",
    )

    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
            body = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")[:500] if e.fp else ""
        return tool_error(f"MiniMax API HTTP {e.code}: {body}")
    except urllib.error.URLError as e:
        return tool_error(f"MiniMax API network error: {e.reason}")
    except Exception as e:
        return tool_error(f"MiniMax API call failed: {e}")

    try:
        payload = json.loads(body)
    except json.JSONDecodeError as e:
        return tool_error(f"MiniMax API returned non-JSON: {e}")

    base = payload.get("base_resp") or {}
    if base.get("status_code", 0) != 0:
        return tool_error(
            f"MiniMax API business error {base.get('status_code')}: {base.get('status_msg')}"
        )

    rows = payload.get("model_remains") or []
    out = _summarize(rows)
    return json.dumps(out, ensure_ascii=False, indent=2)


def check_minimax_quota_requirements() -> bool:
    """Available iff a MiniMax credential is in the pool."""
    return _resolve_api_key() is not None


MINIMAX_QUOTA_SCHEMA = {
    "name": "minimax_quota",
    "description": (
        "Inspect the current MiniMax Token Plan budget — how many requests "
        "you've used and how many remain in the rolling 5h window and the "
        "weekly window, per model (Text Generation, TTS HD, Image, Video, "
        "Music). Use this BEFORE launching a heavy task (long agent loop, "
        "batch TTS, image gen) to make sure you have enough headroom, or to "
        "give the user a budget status. Returns JSON with per-model rows and "
        "a `warnings` array listing any quota >= 80% used. No parameters."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}


registry.register(
    name="minimax_quota",
    toolset="tts",  # bundled with text_to_speech (same API key, related budget)
    schema=MINIMAX_QUOTA_SCHEMA,
    handler=lambda args, **kw: get_minimax_quota(),
    check_fn=check_minimax_quota_requirements,
    emoji="📊",
)
