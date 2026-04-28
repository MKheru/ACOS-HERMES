"""SMCP G2 — Tool scope enforcement.

A compromised or malicious MCP must not be able to chain a tool call into
*another* MCP's tool surface. Without this gate, an MCP "jina" could return
a payload designed to make the agent invoke "filesystem.write" (owned by a
different MCP). With this gate, AH refuses cross-MCP chains unless the user
has explicitly authorised it in their latest message.

The check is invoked from ``AIAgent._invoke_tool`` before any tool dispatch.
On block: a HIGH-severity incident is recorded against the offending MCP via
``mcp_reputation``, the event is logged at CRITICAL, and the tool call is
short-circuited with an explicit refusal string that the LLM can see.

Built-in tools (those *not* prefixed ``mcp_<server>_*``) bypass the check
entirely — they are agent-internal, not driven by external MCPs.

See ``project_smcp_chantiers.md`` (Chantier 2) for design rationale.
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Owner registry. Populated at MCP tool-schema build time
# (tools/mcp_tool.py builds ``prefixed_name`` and immediately calls
# ``register_mcp_tool``). Lookups are constant-time.
# ---------------------------------------------------------------------------

_TOOL_OWNERS: dict[str, str] = {}
_TOOL_OWNERS_LOCK = threading.RLock()


def register_mcp_tool(prefixed_name: str, server_name: str) -> None:
    """Record that *prefixed_name* belongs to MCP *server_name*."""
    with _TOOL_OWNERS_LOCK:
        _TOOL_OWNERS[prefixed_name] = server_name


def unregister_tools_for_server(server_name: str) -> None:
    """Drop all tool entries owned by *server_name* (e.g. on MCP shutdown)."""
    with _TOOL_OWNERS_LOCK:
        for name in [n for n, s in _TOOL_OWNERS.items() if s == server_name]:
            _TOOL_OWNERS.pop(name, None)


def get_owning_mcp(prefixed_name: str) -> Optional[str]:
    """Return the MCP server name owning *prefixed_name*, or None for builtins."""
    with _TOOL_OWNERS_LOCK:
        return _TOOL_OWNERS.get(prefixed_name)


# ---------------------------------------------------------------------------
# User authorisation vocabulary — sourced from agent.provenance for parity
# with the Chantier 3 (G3) provenance gate. Imported lazily so a circular
# import between provenance and this module remains impossible.
# ---------------------------------------------------------------------------


def _user_auth_tokens() -> tuple[str, ...]:
    try:
        from agent.provenance import _USER_AUTH_TOKENS
        return _USER_AUTH_TOKENS
    except Exception:
        return (
            "yes", "ok", "okay", "go ahead", "proceed", "allow",
            "do it", "approved", "authorise", "authorize", "confirmed", "confirm",
        )


# ---------------------------------------------------------------------------
# Trace inspection helpers. ``messages`` here is the raw OpenAI-style message
# list flowing through AIAgent (each item is a dict: role, content, tool_calls,
# tool_call_id, ...). We don't depend on the tagged ``Message`` trace from
# provenance.py because that tagging is not yet wired (Patch 13 / Chantier 3).
# ---------------------------------------------------------------------------


def last_mcp_source_in(messages: list) -> Optional[str]:
    """Return the MCP server whose tool result is the most recent in ``messages``.

    Scans backwards looking for the latest ``role: 'tool'`` entry, then resolves
    its ``tool_call_id`` against the assistant message that produced it. If the
    invoked function name was ``mcp_<server>_*`` and ``<server>`` is registered,
    returns ``<server>``. Otherwise returns None (built-in tool, or no recent
    tool turn).
    """
    if not messages:
        return None

    # 1. Find latest tool message and its tool_call_id.
    tool_call_id = None
    for msg in reversed(messages):
        role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", None)
        if role == "tool":
            tool_call_id = (
                msg.get("tool_call_id") if isinstance(msg, dict)
                else getattr(msg, "tool_call_id", None)
            )
            break
    if not tool_call_id:
        return None

    # 2. Resolve tool_call_id back to the function name in some assistant message.
    for msg in messages:
        role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", None)
        if role != "assistant":
            continue
        tool_calls = (
            msg.get("tool_calls") if isinstance(msg, dict)
            else getattr(msg, "tool_calls", None)
        ) or []
        for tc in tool_calls:
            tc_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
            if tc_id != tool_call_id:
                continue
            fn = tc.get("function") if isinstance(tc, dict) else getattr(tc, "function", None)
            if fn is None:
                continue
            fn_name = fn.get("name") if isinstance(fn, dict) else getattr(fn, "name", None)
            if not fn_name:
                continue
            return get_owning_mcp(fn_name)

    return None


def user_message_authorises_chain(messages: list) -> bool:
    """True if the latest user message contains an explicit auth token.

    Mirrors ``agent.provenance._user_authorised`` but operates on the raw
    OpenAI-style message list rather than the (not-yet-wired) tagged trace.
    """
    if not messages:
        return False
    tokens = _user_auth_tokens()
    for msg in reversed(messages):
        role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", None)
        if role != "user":
            continue
        content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", "")
        if not isinstance(content, str):
            return False
        lowered = content.lower()
        return any(tok in lowered for tok in tokens)
    return False


# ---------------------------------------------------------------------------
# Enforcement entry point. Returns None to allow, or a refusal string to
# short-circuit the tool call. Caller (AIAgent._invoke_tool) returns the
# refusal as the tool result so the LLM sees the block and can ask the user
# for an explicit "ok" / "go ahead" to proceed.
# ---------------------------------------------------------------------------


def enforce_tool_scope(tool_name: str, messages: list) -> Optional[str]:
    """Decide whether the pending tool call is allowed under SMCP G2.

    Args:
        tool_name: the name the LLM is invoking (e.g. ``mcp_filesystem_write``).
        messages: the running OpenAI-style message list at dispatch time.

    Returns:
        None if the call is allowed, otherwise a ``[BLOCKED: ...]`` refusal
        string suitable for use as the tool's response payload.
    """
    tool_owner = get_owning_mcp(tool_name)
    if tool_owner is None:
        return None  # built-in, always allowed

    last_source = last_mcp_source_in(messages)
    if last_source is None:
        return None  # no active MCP source — fresh chain or user-driven
    if last_source == tool_owner:
        return None  # same-MCP chain, allowed

    # Cross-MCP attempt — last_source != tool_owner.
    if user_message_authorises_chain(messages):
        logger.info(
            "cross-MCP allowed by user override: %s -> %s",
            last_source, tool_name,
        )
        return None

    # BLOCK + record incident.
    logger.critical(
        "cross-MCP block (G2): %s tried to chain to %s (owned by %s) — denied",
        last_source, tool_name, tool_owner,
    )
    try:
        from agent.mcp_reputation import _get_reputation_registry, IncidentSeverity
        registry = _get_reputation_registry()
        registry.record_incident(
            uuid=f"server:{last_source}",
            severity=IncidentSeverity.HIGH,
            category="cross_mcp_call_attempt",
            detector="tool_scope_enforcer",
            sample=f"{last_source} -> {tool_name}",
        )
    except Exception as e:
        logger.warning("failed to record cross-MCP incident: %r", e)

    return (
        f"[BLOCKED: cross-MCP call refused (SMCP G2). "
        f"MCP '{last_source}' cannot trigger tool '{tool_name}' "
        f"(owned by MCP '{tool_owner}'). "
        f"If this chain is legitimate, ask the user to authorise with "
        f"'ok', 'go ahead', or 'proceed' in their next message.]"
    )
