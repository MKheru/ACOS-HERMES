"""AFK state — Patch 13.3b.

Persistent AFK state for hermes-agent + tool blacklist for the AFK
publish-restriction rule.

State file: ``~/.hermes/afk_state.json``. Survives across hermes-agent
restarts so a SIGSYS / OOM / sysctl reboot doesn't drop AH out of an
ongoing autonomous-night session.

Schema:
    {
      "mode": "normal" | "afk_manual" | "afk_auto" | "stand_by",
      "entered_at": ISO8601 UTC | null,
      "last_user_msg_at": ISO8601 UTC | null,
      "tokens_minimax_5h": int,
      "tokens_minimax_5h_reset_at": ISO8601 UTC | null,
      "tokens_minimax_7d": int,
      "tokens_minimax_7d_reset_at": ISO8601 UTC | null,
      "heartbeats_sent": int,           # 0..3 — patch 13.4
      "last_heartbeat_at": ISO8601 UTC | null,
      "cooldown_level": int,            # 0..3 — patch 13.4
      "auth_token_expires_at": ISO8601 UTC | null,  # informational only
    }

This module is intentionally narrow: it manages state + the publish
blacklist. The auto-AFK transition logic (20:00 GMT-3 + idle 30min)
and the heartbeat cycle (J3, J7, J14) live in agent.afk_scheduler
(Patch 13.3c) and agent.afk_heartbeat (Patch 13.4).
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Default state-file location — mirrors hermes_constants.get_hermes_home()
# but with a hard fallback so this module is import-safe even outside the
# normal hermes-agent runtime (e.g. CLI scripts, tests).
def _default_state_path() -> Path:
    try:
        from hermes_constants import get_hermes_home  # type: ignore
        return Path(get_hermes_home()) / "afk_state.json"
    except Exception:
        return Path(os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes"))) / "afk_state.json"


# ---------------------------------------------------------------------------
# Mode constants
# ---------------------------------------------------------------------------

MODE_NORMAL = "normal"
MODE_AFK_MANUAL = "afk_manual"
MODE_AFK_AUTO = "afk_auto"
MODE_STAND_BY = "stand_by"  # Patch 13.4: after 3 heartbeats with no user reply

ALL_MODES = (MODE_NORMAL, MODE_AFK_MANUAL, MODE_AFK_AUTO, MODE_STAND_BY)
AFK_MODES = (MODE_AFK_MANUAL, MODE_AFK_AUTO)


# ---------------------------------------------------------------------------
# State dataclass
# ---------------------------------------------------------------------------

@dataclass
class AFKState:
    mode: str = MODE_NORMAL
    entered_at: Optional[str] = None
    last_user_msg_at: Optional[str] = None
    tokens_minimax_5h: int = 0
    tokens_minimax_5h_reset_at: Optional[str] = None
    tokens_minimax_7d: int = 0
    tokens_minimax_7d_reset_at: Optional[str] = None
    heartbeats_sent: int = 0
    last_heartbeat_at: Optional[str] = None
    cooldown_level: int = 0
    auth_token_expires_at: Optional[str] = None

    def is_afk(self) -> bool:
        return self.mode in AFK_MODES

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "AFKState":
        valid = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore
        cleaned = {k: v for k, v in d.items() if k in valid}
        return cls(**cleaned)


# ---------------------------------------------------------------------------
# Atomic load/save
# ---------------------------------------------------------------------------

def load_state(path: Optional[Path] = None) -> AFKState:
    """Read the AFK state file. Returns default state if file missing/corrupt."""
    p = path or _default_state_path()
    if not p.is_file():
        return AFKState()
    try:
        with p.open(encoding="utf-8") as f:
            data = json.load(f)
        return AFKState.from_dict(data)
    except (json.JSONDecodeError, OSError, ValueError) as e:
        logger.warning("AFK state file %s corrupt or unreadable, resetting: %s", p, e)
        return AFKState()


def save_state(state: AFKState, path: Optional[Path] = None) -> None:
    """Write the AFK state atomically (temp + rename)."""
    p = path or _default_state_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=str(p.parent),
        prefix=p.name + ".tmp-",
        delete=False,
    ) as tmp:
        json.dump(state.to_dict(), tmp, indent=2, sort_keys=True)
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, p)


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Publish-action blacklist
# ---------------------------------------------------------------------------
#
# When AFK is active (manual or auto), the following tool name patterns are
# refused at the tool-execution layer. The intent is: AH may build, test,
# refactor, commit locally, and post on the private #acos-hermes channel,
# but cannot publish anything externally without explicit user review.
#
# The match is on the FULL tool name (after MCP server prefix). Patterns
# are compiled as regex with re.IGNORECASE, anchored to start of name.
# Use `is_publish_action(tool_name)` to test.

# Exact tool names that are publish actions (after stripping any
# ``mcp_<server>_`` prefix from the full tool name). Match is case-sensitive.
_PUBLISH_ACTION_EXACT_NAMES: frozenset[str] = frozenset({
    # Git push variants
    "git_push",
    "git_force_push",
    # GitHub CLI write actions
    "gh_pr_create",
    "gh_pr_merge",
    "gh_pr_close",
    "gh_pr_edit",
    "gh_pr_review",
    "gh_issue_create",
    "gh_issue_close",
    "gh_issue_comment",
    "gh_release_create",
    "gh_repo_edit",
    "gh_workflow_run",
    # Discord cross-channel posting (only #acos-hermes is allowed)
    "send_message",
    "send_dm",
    "send",
    # Force / destructive
    "force_delete",
    "delete_branch",
})

# Regex patterns for full tool names (more flexible matches: messaging
# platforms by server-name prefix, env-file writes, branch protection,
# gh_api write methods).
_PUBLISH_ACTION_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        # Whole MCP servers that are external messaging
        r"^mcp_slack_.*",
        r"^mcp_telegram_.*",
        r"^mcp_signal_.*",
        r"^mcp_email_.*",
        r"^mcp_whatsapp_.*",
        r"^mcp_webhook_.*",
        # Branch-protection edits (PAT could in theory edit them via gh local)
        r".*branches.*protection",
        # Critical config / env modifications
        r".*write_file.*env\.list",
        r".*write_file.*HERMES\.md",
        r".*write_file.*SOUL\.md",
        # gh_api with write methods (PUT / POST / PATCH / DELETE)
        r".*gh_api.*(PUT|POST|PATCH|DELETE).*",
    )
)


def _strip_mcp_prefix(tool_name: str) -> str:
    """Strip ``mcp_<server>_`` prefix from *tool_name* if present.

    Examples:
        mcp_github_gh_pr_create  -> gh_pr_create
        mcp_discord_send_message -> send_message
        git_push                 -> git_push (already bare)
    """
    if not tool_name.startswith("mcp_"):
        return tool_name
    # mcp_<server>_<rest>
    parts = tool_name.split("_", 2)
    return parts[2] if len(parts) >= 3 else tool_name


def is_publish_action(tool_name: str) -> bool:
    """Return True if *tool_name* is on the AFK publish blacklist."""
    if not tool_name:
        return False
    # Pass 1: exact-name match after stripping mcp_<server>_ prefix
    bare = _strip_mcp_prefix(tool_name)
    if bare in _PUBLISH_ACTION_EXACT_NAMES:
        return True
    # Pass 2: regex on full tool name (messaging-server prefix, write_file, etc.)
    return any(p.match(tool_name) for p in _PUBLISH_ACTION_PATTERNS)


def check_publish_blocked_in_afk(tool_name: str, state: AFKState) -> tuple[bool, str]:
    """Combined check: returns (blocked, reason).

    Blocks if:
      - state is AFK (manual / auto) AND tool_name matches blacklist.

    Allows everything else (including all tools when in normal mode).
    """
    if not state.is_afk():
        return False, ""
    if not is_publish_action(tool_name):
        return False, ""
    return True, (
        f"tool '{tool_name}' is a publish action and AH is currently in "
        f"{state.mode}; deferred for user review on return"
    )


# ---------------------------------------------------------------------------
# Convenience: detect AFK in OpenAI-style messages list (mirrors
# agent.provenance.is_afk_mode but works on raw messages without a
# dependency on the tagged Message type).
# ---------------------------------------------------------------------------

def is_afk_from_messages(messages: list[dict]) -> bool:
    """Detect AFK trigger in the LATEST user message of an OpenAI-format
    messages list.

    Mirrors agent.provenance.is_afk_mode but consumes raw messages so
    callers in mcp_tool_scope / discord adapter / tool dispatchers don't
    need to first build a tagged trace.
    """
    # Lazy import to avoid circular dep
    try:
        from agent.provenance import _USER_AFK_TOKENS
    except Exception:
        # Hardcoded fallback — keep in sync with provenance.py
        _USER_AFK_TOKENS = (
            "je vais me coucher", "bonne nuit", "à demain", "a demain",
            "je sors", "je m'absente", "je me deconnecte", "je me déconnecte",
            "afk", "je vais bouffer", "pause", "je m'en vais", "je pars",
            "à plus", "a plus",
            "good night", "good evening", "i'm afk", "i am afk", "i'm away",
            "i am away", "i'm out", "i am out", "see you tomorrow",
            "talk to you tomorrow", "going to sleep", "going offline",
            "logging off",
        )

    last_user = None
    for msg in messages:
        if msg.get("role") == "user":
            last_user = msg
    if last_user is None:
        return False
    content = last_user.get("content") or ""
    if isinstance(content, list):
        # multi-modal — flatten text parts
        content = " ".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return any(token in content.lower() for token in _USER_AFK_TOKENS)
