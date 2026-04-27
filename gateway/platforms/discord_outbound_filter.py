"""Defensive outbound filter for Discord — ACOS-HERMES Patch 2.

This is the final line of defence before AH writes to a Discord
channel. Two checks:

  1. Channel whitelist: if the DISCORD_HERMES_CHANNEL_ID env var is
     set, the target channel must match it exactly. Threads that
     belong to the whitelisted channel pass via the chat_id check
     even when sending into the thread (the underlying chat_id is
     still the parent channel).

  2. Defence-in-depth redaction: every outbound payload passes
     through agent.redact.redact_sensitive_text. Any difference
     between input and output is logged at WARNING level so
     operators can investigate.

Both checks are no-ops when their env-var/config triggers are
unset, so this module is safe to import in environments that
don't use the Hermes channel scoping (e.g. multi-channel bots).
"""

from __future__ import annotations

import logging
import os
from typing import Optional, Tuple

from agent.redact import redact_sensitive_text

logger = logging.getLogger(__name__)


def _whitelisted_channel_id() -> Optional[str]:
    """Return the configured allowed channel ID, or None if disabled."""
    value = os.getenv("DISCORD_HERMES_CHANNEL_ID", "").strip()
    return value or None


def filter_outbound(
    content: str,
    chat_id: str,
    thread_id: Optional[str] = None,
) -> Tuple[bool, str, Optional[str]]:
    """Filter an outbound Discord message.

    Args:
        content: the message text the agent wants to send.
        chat_id: the Discord channel ID being targeted (string).
        thread_id: optional thread ID inside chat_id — informational
            only; the whitelist is keyed on chat_id (the thread's
            parent), so threads in the allowed channel pass.

    Returns:
        (allowed, sanitized_content, block_reason)
        - allowed=True  → caller may send sanitized_content
        - allowed=False → caller must NOT send; block_reason explains why
    """
    if content is None:
        return True, content, None

    allowed_id = _whitelisted_channel_id()
    if allowed_id is not None and str(chat_id) != allowed_id:
        logger.critical(
            "BLOCKED outbound Discord message to non-whitelisted channel "
            "(target=%s, allowed=%s, thread_id=%s, content_len=%d)",
            chat_id, allowed_id, thread_id,
            len(content) if isinstance(content, str) else 0,
        )
        return (
            False,
            "",
            f"channel {chat_id} not in whitelist (allowed={allowed_id})",
        )

    sanitized = redact_sensitive_text(content)
    if sanitized != content:
        logger.warning(
            "Outbound Discord message had secrets redacted before send "
            "(channel=%s, thread_id=%s, original_len=%d, sanitized_len=%d)",
            chat_id, thread_id,
            len(content) if isinstance(content, str) else 0,
            len(sanitized) if isinstance(sanitized, str) else 0,
        )

    return True, sanitized, None
