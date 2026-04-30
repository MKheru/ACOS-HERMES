"""Tests for ACOS-HERMES Patch 13 — SMCP G3 provenance gate activation.

Covers: from_messages_api, intent_from_api_kwargs, should_block_llm_call,
and ProvenanceBlocked.
"""

import pytest
from agent.provenance import (
    ProvenanceBlocked,
    ProvenanceSource,
    Tag,
    Message,
    from_messages_api,
    intent_from_api_kwargs,
    should_block_llm_call,
    tag_user,
    tag_mcp_tool,
    tag_system,
)


# ─── from_messages_api ─────────────────────────────────────────────────────────

class TestFromMessagesApi:
    def test_system_message_tagged_system(self):
        msgs = [{"role": "system", "content": "You are a helpful assistant."}]
        trace = from_messages_api(msgs)
        assert len(trace) == 1
        assert trace[0].tag.source == ProvenanceSource.SYSTEM

    def test_user_message_tagged_user(self):
        msgs = [{"role": "user", "content": "Hello"}]
        trace = from_messages_api(msgs)
        assert trace[0].tag.source == ProvenanceSource.USER

    def test_assistant_message_tagged_assistant(self):
        msgs = [{"role": "assistant", "content": "Hi there"}]
        trace = from_messages_api(msgs)
        assert trace[0].tag.source == ProvenanceSource.ASSISTANT

    def test_tool_message_tagged_mcp_tool(self):
        msgs = [{"role": "tool", "content": '{"result": "42"}', "tool_call_id": "call_123"}]
        trace = from_messages_api(msgs)
        assert trace[0].tag.source == ProvenanceSource.MCP_TOOL

    def test_developer_role_tagged_system(self):
        msgs = [{"role": "developer", "content": "system prompt"}]
        trace = from_messages_api(msgs)
        assert trace[0].tag.source == ProvenanceSource.SYSTEM

    def test_unknown_role_tagged_unknown(self):
        msgs = [{"role": "unknown_role", "content": "data"}]
        trace = from_messages_api(msgs)
        assert trace[0].tag.source == ProvenanceSource.UNKNOWN

    def test_multimodal_content_list_flattened(self):
        msgs = [{"role": "user", "content": [
            {"type": "text", "text": "Hello "},
            {"type": "text", "text": "world"},
        ]}]
        trace = from_messages_api(msgs)
        assert "Hello" in trace[0].content
        assert "world" in trace[0].content

    def test_none_content_handled(self):
        msgs = [{"role": "user", "content": None}]
        trace = from_messages_api(msgs)
        assert trace[0].content == ""  # None → ""

    def test_missing_content_field_handled(self):
        msgs = [{"role": "user"}]  # no "content" key
        trace = from_messages_api(msgs)
        assert trace[0].content == ""

    def test_empty_messages_returns_empty_trace(self):
        assert from_messages_api([]) == []

    def test_full_conversation_trace(self):
        msgs = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What's in memory?"},
            {"role": "assistant", "content": "I checked memory."},
            {"role": "tool", "content": '{"total": 8192, "used": 4096}', "tool_call_id": "call_1"},
            {"role": "assistant", "content": "Memory usage: 50%."},
        ]
        trace = from_messages_api(msgs)
        assert [t.tag.source for t in trace] == [
            ProvenanceSource.SYSTEM,
            ProvenanceSource.USER,
            ProvenanceSource.ASSISTANT,
            ProvenanceSource.MCP_TOOL,
            ProvenanceSource.ASSISTANT,
        ]


# ─── intent_from_api_kwargs ───────────────────────────────────────────────────

class TestIntentFromApiKwargs:
    def test_no_tools_is_sampling(self):
        kwargs = {"model": "gpt-4", "messages": []}
        assert intent_from_api_kwargs(kwargs) == "sampling"

    def test_tools_present_is_tool(self):
        kwargs = {
            "model": "gpt-4",
            "messages": [],
            "tools": [{"type": "function", "function": {"name": "search"}}],
        }
        assert intent_from_api_kwargs(kwargs) == "tool"

    def test_tool_choice_present_is_tool(self):
        kwargs = {
            "model": "gpt-4",
            "messages": [],
            "tool_choice": {"type": "function", "function": {"name": "search"}},
        }
        assert intent_from_api_kwargs(kwargs) == "tool"

    def test_empty_tools_list_is_sampling(self):
        kwargs = {"model": "gpt-4", "messages": [], "tools": []}
        assert intent_from_api_kwargs(kwargs) == "sampling"


# ─── should_block_llm_call ────────────────────────────────────────────────────

class TestShouldBlockLlmCall:
    def test_empty_trace_allowed(self):
        blocked, _ = should_block_llm_call([], "sampling")
        assert blocked is False

    def test_trusted_user_only_allowed(self):
        trace = [
            Message(role="user", content="Hello", tag=tag_user("Hello")),
        ]
        blocked, _ = should_block_llm_call(trace, "sampling")
        assert blocked is False

    def test_trusted_system_only_allowed(self):
        trace = [
            Message(role="system", content="You are an agent.", tag=tag_system("You are an agent.")),
        ]
        blocked, _ = should_block_llm_call(trace, "sampling")
        assert blocked is False

    def test_untrusted_mcp_tool_with_user_present_allows_sampling(self):
        """Patch 13.2: a USER message in the trace = implicit auth.

        Replaces the old Patch 13 behaviour where the user had to type
        an explicit "ok"/"go ahead" token to allow tool-chains.
        """
        trace = [
            Message(role="user", content="Hello", tag=tag_user("Hello")),
            Message(role="tool", content='{"result": "data"}', tag=tag_mcp_tool("result", "search", "web_search")),
            Message(role="assistant", content="Let me check...", tag=Tag(source=ProvenanceSource.ASSISTANT)),
        ]
        blocked, _ = should_block_llm_call(trace, "sampling")
        assert blocked is False

    def test_untrusted_mcp_tool_with_user_present_allows_tool(self):
        """Patch 13.2: same as above for intent='tool'."""
        trace = [
            Message(role="user", content="search the web for me please", tag=tag_user("search the web for me please")),
            Message(role="tool", content='{"result": "data"}', tag=tag_mcp_tool("result", "search", "web_search")),
        ]
        blocked, _ = should_block_llm_call(trace, "tool")
        assert blocked is False

    def test_untrusted_no_user_in_trace_blocked(self):
        """Patch 13.2: trace with untrusted content but no user is anomalous."""
        trace = [
            Message(role="system", content="You are an agent.", tag=tag_system("You are an agent.")),
            Message(role="tool", content='{"result": "data"}', tag=tag_mcp_tool("result", "search", "web_search")),
        ]
        blocked, reason = should_block_llm_call(trace, "sampling")
        assert blocked is True
        assert "no user message" in reason.lower()

    def test_user_auth_token_allows_chain(self):
        """Explicit auth token still works (not required, but should not break)."""
        trace = [
            Message(role="user", content="yes, go ahead", tag=tag_user("yes, go ahead")),
            Message(role="tool", content='{"result": "42"}', tag=tag_mcp_tool("result", "search", "web_search")),
        ]
        blocked, _ = should_block_llm_call(trace, "sampling")
        assert blocked is False

    def test_user_revoke_blocks_subsequent_tool_chain(self):
        """Patch 13.2: user revoke token blocks tool-chain."""
        trace = [
            Message(role="user", content="search the web", tag=tag_user("search the web")),
            Message(role="tool", content='{"result": "ok"}', tag=tag_mcp_tool("ok", "s", "t")),
            Message(role="user", content="stop, don't continue", tag=tag_user("stop, don't continue")),
            Message(role="tool", content='{"result": "more"}', tag=tag_mcp_tool("more", "s", "t")),
        ]
        blocked, reason = should_block_llm_call(trace, "tool")
        assert blocked is True
        assert "revoke" in reason.lower() or "stop" in reason.lower()

    def test_user_revoke_french_blocks(self):
        """Patch 13.2: French revoke tokens also blocked."""
        trace = [
            Message(role="user", content="Arrête maintenant", tag=tag_user("Arrête maintenant")),
            Message(role="tool", content='{"x": 1}', tag=tag_mcp_tool("x", "s", "t")),
        ]
        blocked, reason = should_block_llm_call(trace, "tool")
        assert blocked is True
        assert "revoke" in reason.lower()

    def test_read_only_constraint_blocks_tool_intent(self):
        trace = [
            Message(role="user", content="yes, but only read", tag=tag_user("yes, but only read")),
            Message(role="tool", content='{"result": "42"}', tag=tag_mcp_tool("result", "search", "web_search")),
        ]
        blocked, reason = should_block_llm_call(trace, "tool")
        assert blocked is True
        assert "read-only" in reason.lower()

    def test_read_only_constraint_allows_sampling_intent(self):
        trace = [
            Message(role="user", content="yes, but only read", tag=tag_user("yes, but only read")),
            Message(role="tool", content='{"result": "42"}', tag=tag_mcp_tool("result", "search", "web_search")),
        ]
        blocked, _ = should_block_llm_call(trace, "sampling")
        assert blocked is False

    def test_tool_depth_exceeds_max_blocks(self):
        """Patch 13.2: MAX_DEPTH = 8. A chain of 9 tool steps triggers block."""
        trace = [
            Message(role="user", content="search and analyze", tag=tag_user("search and analyze")),
        ] + [
            Message(role="tool", content=f'{{"result": "{i}"}}', tag=tag_mcp_tool(str(i), f"s{i}", f"t{i}"))
            for i in range(9)  # 9 tool steps > MAX_DEPTH=8
        ]
        blocked, reason = should_block_llm_call(trace, "tool")
        assert blocked is True
        assert "depth" in reason.lower()

    def test_tool_depth_at_max_allowed(self):
        """Patch 13.2: 8 tool steps is the boundary (allowed)."""
        trace = [
            Message(role="user", content="search and analyze", tag=tag_user("search and analyze")),
        ] + [
            Message(role="tool", content=f'{{"result": "{i}"}}', tag=tag_mcp_tool(str(i), f"s{i}", f"t{i}"))
            for i in range(8)  # exactly MAX_DEPTH
        ]
        blocked, _ = should_block_llm_call(trace, "tool")
        assert blocked is False

    def test_user_chat_with_last_user_allowed(self):
        trace = [
            Message(role="user", content="Hello", tag=tag_user("Hello")),
            Message(role="assistant", content="Hi", tag=Tag(source=ProvenanceSource.ASSISTANT)),
        ]
        blocked, _ = should_block_llm_call(trace, "user_chat")
        assert blocked is False

    def test_user_chat_with_last_not_user_blocked(self):
        trace = [
            Message(role="assistant", content="Hi", tag=Tag(source=ProvenanceSource.ASSISTANT)),
            Message(role="tool", content='{"result": "42"}', tag=tag_mcp_tool("42", "s", "t")),
        ]
        blocked, _ = should_block_llm_call(trace, "user_chat")
        assert blocked is True

    def test_unknown_intent_fails_closed(self):
        # Need untrusted content so we reach the unknown-intent branch
        trace = [
            Message(role="user", content="hello", tag=tag_user("hello")),
            Message(role="tool", content='{"x": 1}', tag=tag_mcp_tool("x", "s", "t")),
        ]
        blocked, reason = should_block_llm_call(trace, "unknown_intent")
        assert blocked is True
        assert "unknown intent" in reason.lower()


# ─── ProvenanceBlocked ────────────────────────────────────────────────────────

class TestProvenanceBlocked:
    def test_carries_reason(self):
        exc = ProvenanceBlocked("test reason string")
        assert str(exc) == "test reason string"

    def test_is_exception(self):
        assert issubclass(ProvenanceBlocked, Exception)
