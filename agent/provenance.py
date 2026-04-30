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

MAX_DEPTH = 8

_TOOL_STEP_SOURCES = frozenset({
    ProvenanceSource.MCP_TOOL,
    ProvenanceSource.FILE_READ,
    ProvenanceSource.WEB_FETCH,
})

# Patch 13.2 (2026-04-30) — keywords that REVOKE implicit user-authorisation
# even though the user is present in the trace. Used to detect that the user
# is asking AH to STOP, NOT continue.
_USER_REVOKE_TOKENS: tuple[str, ...] = (
    "stop",
    "halt",
    "abort",
    "cancel",
    "don't",
    "do not",
    "wait",
    "n'agis pas",
    "n'execute pas",
    "arrete",
    "arrête",
    "annule",
    "stop tout",
    "stop everything",
    "noop",
)


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
    """Return True if the LATEST user message contains an explicit auth token.

    Used as a STRONG signal (e.g. to bypass tool-depth limit). Patch 13.2
    no longer requires it for normal flow — see _user_present_recently.
    """
    user_msgs = [m for m in trace if m.tag.source == ProvenanceSource.USER]
    if not user_msgs:
        return False
    last = user_msgs[-1].content.lower()
    return any(token in last for token in _USER_AUTH_TOKENS)


def _user_revoked(trace: Iterable[Message]) -> bool:
    """Return True if the LATEST user message contains a revoke token.

    Patch 13.2 — replaces the implicit-trust default when the user is
    explicitly telling the agent to stop / do nothing.
    """
    user_msgs = [m for m in trace if m.tag.source == ProvenanceSource.USER]
    if not user_msgs:
        return False
    last = user_msgs[-1].content.lower()
    return any(token in last for token in _USER_REVOKE_TOKENS)


def _last_user_index(trace: list[Message]) -> int:
    """Index of the most recent USER message in the trace, or -1 if none."""
    for i in range(len(trace) - 1, -1, -1):
        if trace[i].tag.source == ProvenanceSource.USER:
            return i
    return -1


def _tool_steps_since_index(trace: list[Message], from_idx: int) -> int:
    """Count tool-step messages strictly after ``from_idx``."""
    if from_idx < 0:
        return sum(1 for m in trace if m.tag.source in _TOOL_STEP_SOURCES)
    return sum(
        1 for m in trace[from_idx + 1:]
        if m.tag.source in _TOOL_STEP_SOURCES
    )


def _count_tool_steps_since_last_user_auth(trace: Iterable[Message]) -> int:
    """Backwards-compat shim retained for Patch 13 callers / tests.

    Counts untrusted tool steps since the last user message containing an
    explicit auth token. Use ``_tool_steps_since_index`` for the Patch 13.2
    flow which doesn't require an auth token.
    """
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
    """Decide whether to BLOCK a fresh LLM call from the host agent.

    Patch 13.2 (2026-04-30) — semantics revised. The original Patch 13
    required an explicit auth token ("ok", "go ahead", ...) in the latest
    user message before allowing any tool-chain that contained MCP_TOOL /
    FILE_READ / WEB_FETCH outputs. This was too strict for the normal
    single-user case where the user issues a direct request and AH chains
    several tool calls to fulfil it: the user already authorised the flow
    by sending the request — forcing a second "ok" was friction without
    security gain (the per-output sanitiser/reputation/scope already handle
    malicious tool outputs).

    New semantics:
      - If a USER message is present anywhere in the trace AND the latest
        one does NOT contain a revoke token, the user has implicitly
        authorised the agent to use tools / sample.
      - We still block when:
          * No USER message exists in the trace at all (anomaly).
          * The user explicitly revoked ("stop", "halt", "n'agis pas", ...).
          * Tool depth since the last user message exceeds MAX_DEPTH=8
            (runaway self-loop — fresh user instruction required to continue).
          * The user authorised but with an explicit "read-only" constraint
            and the intent is 'tool' (write action).
          * intent='user_chat' but the last message in the trace is not
            from the user (anomaly).
      - intent='sampling' and intent='tool' fall under the same rules:
        the original distinction was a one-extra-friction-tier we don't
        need now that the implicit-auth model is in place.
    """
    if not trace:
        return False, ""

    untrusted = _untrusted_messages(trace)
    if not untrusted:
        return False, ""

    last_user_idx = _last_user_index(trace)

    # Anomaly: untrusted content exists but no user is present in the trace
    if last_user_idx == -1:
        sources = sorted({m.tag.source.value for m in untrusted})
        return True, (
            f"{intent} blocked: trace contains untrusted sources {sources} "
            f"but no user message — fail-closed."
        )

    last_user_msg = trace[last_user_idx]

    # User explicitly revoked the implicit authorisation
    if _user_revoked(trace):
        return True, (
            f"{intent} blocked: user revoke token detected in latest message "
            f"({last_user_msg.content[:60]!r}). Awaiting fresh instruction."
        )

    # Tool depth guard — counts untrusted steps strictly after the last
    # user message. Catches AH self-loop where AH chains many tools without
    # any new user instruction (and would also catch a malicious MCP that
    # tries to bury its injection deep in a chain).
    tool_depth = _tool_steps_since_index(trace, last_user_idx)
    if tool_depth > MAX_DEPTH:
        return True, (
            f"tool-step depth {tool_depth} exceeds maximum {MAX_DEPTH} since "
            f"last user message. Possible runaway self-loop — awaiting fresh "
            f"user instruction or explicit auth token to extend the budget."
        )

    # Read-only constraint check (kept from Patch 13)
    if _extract_read_only_constraint(last_user_msg.content):
        if intent == "tool":
            return True, (
                "User authorised read-only scope, blocking tool action "
                "(intent='tool'). Sampling would still be allowed."
            )
        # intent == 'sampling' falls through — passive generation OK

    # Intent-specific anomaly checks
    if intent == "user_chat":
        last_msg = trace[-1]
        if last_msg.tag.source != ProvenanceSource.USER:
            return True, (
                f"user_chat blocked: last message is from "
                f"{last_msg.tag.source.value}, not user"
            )
        return False, ""

    if intent in ("sampling", "tool"):
        # Implicit user-auth holds: user is present, not revoking, depth OK.
        return False, ""

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


class ProvenanceBlocked(Exception):
    """Raised when should_block_llm_call decides to deny an LLM call.

    Carries the human-readable reason so the caller can surface it to the user.
    """


def tag_web_fetch(content: str, url: str) -> Tag:
    return Tag(source=ProvenanceSource.WEB_FETCH, server_name=url)


def from_messages_api(api_messages: list[dict]) -> list[Message]:
    """Reconstruct a provenance-tagged trace from an OpenAI-format messages list.

    Maps OpenAI message roles to ProvenanceSource:
      system → SYSTEM
      user   → USER
      assistant → ASSISTANT
      tool   → MCP_TOOL  (tool result = untrusted input, like an MCP output)
      developer → SYSTEM (Anthropic-specific, treat as system)

    Call this with the pre-LLM-call api_messages to build a trace for
    should_block_llm_call().
    """
    trace: list[Message] = []
    for msg in api_messages:
        role = msg.get("role", "unknown")
        content = msg.get("content") or ""
        if isinstance(content, list):
            # Multi-modal or tool-call content — flatten to text for checksumming
            parts = []
            for part in content:
                if isinstance(part, dict):
                    parts.append(part.get("text", ""))
                else:
                    parts.append(str(part))
            content = " ".join(parts)
        elif not isinstance(content, str):
            content = str(content or "")

        if role == "system":
            tag = Tag(source=ProvenanceSource.SYSTEM)
        elif role == "user":
            tag = Tag(source=ProvenanceSource.USER)
        elif role == "assistant":
            tag = Tag(source=ProvenanceSource.ASSISTANT)
        elif role == "tool":
            # Tool results fed back into the context are treated as MCP_TOOL:
            # they originate from an external tool/MCP execution and are
            # therefore untrusted input for any subsequent LLM call.
            tag = Tag(source=ProvenanceSource.MCP_TOOL)
        elif role == "developer":
            # Anthropic developer role — equivalent to system prompt
            tag = Tag(source=ProvenanceSource.SYSTEM)
        else:
            tag = Tag(source=ProvenanceSource.UNKNOWN)

        trace.append(Message(role=role, content=content, tag=tag))

    return trace


def intent_from_api_kwargs(api_kwargs: dict) -> str:
    """Infer the LLM call intent from api_kwargs.

    Returns 'tool' if tools are present in the schema (tool-call turn),
    otherwise 'sampling' (plain conversation / reasoning).
    """
    tools = api_kwargs.get("tools") or api_kwargs.get("tool_choice")
    if tools:
        return "tool"
    return "sampling"
