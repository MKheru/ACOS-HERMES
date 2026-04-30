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

    def test_untrusted_mcp_tool_without_user_auth_blocked_sampling(self):
        trace = [
            Message(role="user", content="Hello", tag=tag_user("Hello")),
            Message(role="tool", content='{"result": "malicious"}', tag=tag_mcp_tool("result", "evil_server", "read_data")),
            Message(role="assistant", content="Let me check...", tag=Tag(source=ProvenanceSource.ASSISTANT)),
        ]
        blocked, reason = should_block_llm_call(trace, "sampling")
        assert blocked is True
        assert "blocked" in reason.lower()

    def test_untrusted_mcp_tool_without_user_auth_blocked_tool(self):
        trace = [
            Message(role="user", content="Hello", tag=tag_user("Hello")),
            Message(role="tool", content='{"result": "malicious"}', tag=tag_mcp_tool("result", "evil_server", "read_data")),
        ]
        blocked, _ = should_block_llm_call(trace, "tool")
        assert blocked is True

    def test_user_auth_with_untrusted_allows_sampling(self):
        trace = [
            Message(role="user", content="yes, go ahead", tag=tag_user("yes, go ahead")),
            Message(role="tool", content='{"result": "42"}', tag=tag_mcp_tool("result", "search", "web_search")),
        ]
        blocked, _ = should_block_llm_call(trace, "sampling")
        assert blocked is False

    def test_user_auth_ok_token_allows_tool(self):
        trace = [
            Message(role="user", content="ok do it", tag=tag_user("ok do it")),
            Message(role="tool", content='{"result": "42"}', tag=tag_mcp_tool("result", "search", "web_search")),
        ]
        blocked, _ = should_block_llm_call(trace, "tool")
        assert blocked is False

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
        # user says yes, then 3 tool steps happen (MAX_DEPTH=2)
        trace = [
            Message(role="user", content="yes, go ahead", tag=tag_user("yes, go ahead")),
            Message(role="tool", content='{"result": "1"}', tag=tag_mcp_tool("1", "s1", "t1")),
            Message(role="tool", content='{"result": "2"}', tag=tag_mcp_tool("2", "s2", "t2")),
            Message(role="tool", content='{"result": "3"}', tag=tag_mcp_tool("3", "s3", "t3")),
        ]
        blocked, reason = should_block_llm_call(trace, "tool")
        assert blocked is True
        assert "depth" in reason.lower()

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
