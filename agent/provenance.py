"""Provenance-based trust boundary for LLM calls — SMCP run #4 (v1).

Adds a conditional auth parser to detect scope-limited authorizations
(e.g., "yes, but only read-only"). If a user grants permission with a
read-only constraint, tool intents (which imply write/action) are blocked,
while sampling intents (passive generation) are allowed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Optional


class ProvenanceSource(str, Enum):
    """Where a message originated. The string values match log schemas."""
    USER = "user"
    SYSTEM = "system"
    ASSISTANT = "assistant"
    MCP_TOOL = "mcp_tool"
    FILE_READ = "file_read"
    WEB_FETCH = "web_fetch"
    UNKNOWN = "unknown"


_INHERENTLY_TRUSTED: frozenset[ProvenanceSource] = frozenset({
    ProvenanceSource.USER,
    ProvenanceSource.SYSTEM,
})

_INHERENTLY_UNTRUSTED: frozenset[ProvenanceSource] = frozenset({
    ProvenanceSource.MCP_TOOL,
    ProvenanceSource.FILE_READ,
    ProvenanceSource.WEB_FETCH,
    ProvenanceSource.UNKNOWN,
})

_USER_AUTH_TOKENS: tuple[str, ...] = (
    "yes",
    "ok",
    "okay",
    "go ahead",
    "proceed",
    "allow",
    "do it",
    "approved",
    "authorise",
    "authorize",
    "confirmed",
    "confirm",
)

# Markers that introduce a constraint on the authorization
_CONDITIONAL_MARKERS: tuple[str, ...] = (
    "but only",
    "only for",
    "except",
    "never",
    "just",
)

# Keywords that imply a "read-only" or "no-modify" constraint
_READ_ONLY_KEYWORDS: tuple[str, ...] = (
    "read",
    "view",
    "look",
    "see",
    "fetch",
)

# Keywords that imply modification (used with negative markers like "never")
_MODIFY_KEYWORDS: tuple[str, ...] = (
    "write",
    "modify",
    "delete",
    "change",
    "edit",
)

MAX_DEPTH = 2

_TOOL_STEP_SOURCES = frozenset({
    ProvenanceSource.MCP_TOOL,
    ProvenanceSource.FILE_READ,
    ProvenanceSource.WEB_FETCH,
})


@dataclass(frozen=True)
class Tag:
    """Provenance tag attached to a single message."""
    source: ProvenanceSource
    server_name: Optional[str] = None
    tool_name: Optional[str] = None
    parent: Optional["Tag"] = None

    def is_trusted(self) -> bool:
        """True iff this tag (and every ancestor) is inherently trusted."""
        if self.source in _INHERENTLY_UNTRUSTED:
            return False
        if self.parent is not None and not self.parent.is_trusted():
            return False
        return self.source in _INHERENTLY_TRUSTED or (
            self.source == ProvenanceSource.ASSISTANT
            and (self.parent is None or self.parent.is_trusted())
        )


@dataclass(frozen=True)
class Message:
    """A single message in a conversation trace."""
    role: str
    content: str
    tag: Tag


def _extract_read_only_constraint(content: str) -> bool:
    """Parse the user message to detect a 'read-only' conditional authorization.

    Returns True if the user authorized an action but explicitly restricted
    it to read-only operations (e.g., "yes, but only read", "never write").
    """
    content_lower = content.lower()

    # Fast path: check for explicit "read-only" phrase
    if "read-only" in content_lower or "read only" in content_lower:
        return True

    # Structural scan: [auth_token] [conditional_marker] [constraint_keywords]
    for token in _USER_AUTH_TOKENS:
        token_idx = content_lower.find(token)
        if token_idx == -1:
            continue

        # Look for markers after the token
        remainder = content_lower[token_idx + len(token):]
        for marker in _CONDITIONAL_MARKERS:
            marker_idx = remainder.find(marker)
            if marker_idx == -1:
                continue

            # Check the text following the marker for constraints
            # We look at a window of 40 chars to keep it tight
            constraint_window = remainder[marker_idx + len(marker):marker_idx + len(marker) + 40]
            
            # Check for positive read-only indicators
            if any(kw in constraint_window for kw in _READ_ONLY_KEYWORDS):
                return True
            
            # Check for negative modification indicators (e.g. "never write")
            if marker in ("never", "except"):
                if any(kw in constraint_window for kw in _MODIFY_KEYWORDS):
                    return True

    return False


def _user_authorised(trace: Iterable[Message]) -> bool:
    """Return True if the LATEST user message contains an auth token."""
    user_msgs = [m for m in trace if m.tag.source == ProvenanceSource.USER]
    if not user_msgs:
        return False
    last = user_msgs[-1].content.lower()
    return any(token in last for token in _USER_AUTH_TOKENS)


def _count_tool_steps_since_last_user_auth(trace: Iterable[Message]) -> int:
    """Count how many untrusted tool steps occurred since the last user auth message."""
    trace_list = list(trace)
    latest_auth_idx = -1
    for i in range(len(trace_list)-1, -1, -1):
        msg = trace_list[i]
        if msg.tag.source == ProvenanceSource.USER:
            user_content_lower = msg.content.lower()
            if any(token in user_content_lower for token in _USER_AUTH_TOKENS):
                latest_auth_idx = i
                break
    
    count = 0
    start_idx = max(latest_auth_idx, 0)
    for i in range(start_idx, len(trace_list)):
        msg = trace_list[i]
        if msg.tag.source in _TOOL_STEP_SOURCES:
            count += 1
    
    return count


def _untrusted_messages(trace: Iterable[Message]) -> list[Message]:
    return [m for m in trace if not m.tag.is_trusted()]


def should_block_llm_call(
    trace: list[Message],
    intent: str = "sampling",
) -> tuple[bool, str]:
    """Decide whether to BLOCK a fresh LLM call from the host agent."""
    if not trace:
        return False, ""

    untrusted = _untrusted_messages(trace)
    if not untrusted:
        return False, ""

    # Depth check
    tool_depth = _count_tool_steps_since_last_user_auth(trace)
    if tool_depth > MAX_DEPTH:
        return True, (
            f"tool-step depth {tool_depth} exceeds maximum {MAX_DEPTH} since last user auth. "
            f"Requires fresh user authorisation."
        )

    # Helper to check constraints if authorized
    def check_constraints() -> tuple[bool, str]:
        # Find last user message
        last_user_msg = next(
            (m for m in reversed(trace) if m.tag.source == ProvenanceSource.USER),
            None
        )
        if not last_user_msg:
            return False, ""
        
        # Check for read-only constraint
        if _extract_read_only_constraint(last_user_msg.content):
            if intent == "tool":
                return True, (
                    "User authorized read-only scope, blocking tool action (intent='tool')."
                )
            # If intent is 'sampling', we allow it (passive generation)
        
        return False, ""

    # Sampling intent
    if intent == "sampling":
        if _user_authorised(trace):
            blocked, reason = check_constraints()
            if blocked:
                return True, reason
            return False, ""
        sources = sorted({m.tag.source.value for m in untrusted})
        return True, (
            f"sampling/createMessage blocked: trace contains untrusted "
            f"sources {sources} and no user authorisation in latest turn"
        )

    # Tool-chain intent
    if intent == "tool":
        if _user_authorised(trace):
            blocked, reason = check_constraints()
            if blocked:
                return True, reason
            return False, ""
        sources = sorted({m.tag.source.value for m in untrusted})
        return True, (
            f"tool-chain blocked: untrusted sources {sources} without "
            f"user authorisation"
        )

    # Direct user-driven continuation
    if intent == "user_chat":
        last_msg = trace[-1]
        if last_msg.tag.source == ProvenanceSource.USER:
            return False, ""
        return True, (
            f"user_chat blocked: last message is from "
            f"{last_msg.tag.source.value}, not user"
        )

    return True, f"unknown intent {intent!r} — failing closed"


def tag_user(content: str) -> Tag:
    return Tag(source=ProvenanceSource.USER)


def tag_system(content: str) -> Tag:
    return Tag(source=ProvenanceSource.SYSTEM)


def tag_assistant(content: str, parent: Optional[Tag] = None) -> Tag:
    return Tag(source=ProvenanceSource.ASSISTANT, parent=parent)


def tag_mcp_tool(content: str, server_name: str,
                 tool_name: Optional[str] = None) -> Tag:
    return Tag(source=ProvenanceSource.MCP_TOOL,
               server_name=server_name, tool_name=tool_name)


def tag_file_read(content: str, path: str) -> Tag:
    return Tag(source=ProvenanceSource.FILE_READ, server_name=path)


def tag_web_fetch(content: str, url: str) -> Tag:
    return Tag(source=ProvenanceSource.WEB_FETCH, server_name=url)
