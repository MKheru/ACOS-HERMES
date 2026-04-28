"""Tests for agent/mcp_tool_scope.py — SMCP G2 cross-MCP enforcement."""

import logging
import pytest

from agent import mcp_tool_scope
from agent.mcp_tool_scope import (
    enforce_tool_scope,
    get_owning_mcp,
    last_mcp_source_in,
    register_mcp_tool,
    unregister_tools_for_server,
    user_message_authorises_chain,
)


@pytest.fixture(autouse=True)
def _clean_registry():
    """Ensure each test starts with a clean owner registry."""
    mcp_tool_scope._TOOL_OWNERS.clear()
    yield
    mcp_tool_scope._TOOL_OWNERS.clear()


# ---------------------------------------------------------------------------
# Owner registry
# ---------------------------------------------------------------------------


class TestOwnerRegistry:
    def test_register_then_lookup(self):
        register_mcp_tool("mcp_jina_read_url", "jina")
        assert get_owning_mcp("mcp_jina_read_url") == "jina"

    def test_unknown_tool_returns_none(self):
        assert get_owning_mcp("file_read") is None

    def test_unregister_drops_all_for_server(self):
        register_mcp_tool("mcp_jina_read_url", "jina")
        register_mcp_tool("mcp_jina_search_web", "jina")
        register_mcp_tool("mcp_filesystem_write", "filesystem")
        unregister_tools_for_server("jina")
        assert get_owning_mcp("mcp_jina_read_url") is None
        assert get_owning_mcp("mcp_filesystem_write") == "filesystem"


# ---------------------------------------------------------------------------
# last_mcp_source_in — trace inspection
# ---------------------------------------------------------------------------


class TestLastMcpSource:
    def test_empty_messages(self):
        assert last_mcp_source_in([]) is None

    def test_no_tool_messages(self):
        msgs = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        assert last_mcp_source_in(msgs) is None

    def test_resolves_mcp_source_via_tool_call_id(self):
        register_mcp_tool("mcp_jina_read_url", "jina")
        msgs = [
            {"role": "user", "content": "fetch X"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "call_1", "function": {"name": "mcp_jina_read_url"}}
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "page content"},
        ]
        assert last_mcp_source_in(msgs) == "jina"

    def test_returns_none_when_last_tool_was_builtin(self):
        msgs = [
            {"role": "user", "content": "save a note"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "call_1", "function": {"name": "memory"}}
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "saved"},
        ]
        assert last_mcp_source_in(msgs) is None


# ---------------------------------------------------------------------------
# user_message_authorises_chain
# ---------------------------------------------------------------------------


class TestUserAuthorisesChain:
    def test_no_user_message(self):
        assert user_message_authorises_chain([]) is False

    def test_user_says_ok(self):
        msgs = [{"role": "user", "content": "ok, go ahead"}]
        assert user_message_authorises_chain(msgs) is True

    def test_user_says_proceed(self):
        msgs = [{"role": "user", "content": "Proceed with that"}]
        assert user_message_authorises_chain(msgs) is True

    def test_user_says_no_token(self):
        msgs = [{"role": "user", "content": "fetch the URL and write it"}]
        assert user_message_authorises_chain(msgs) is False

    def test_only_latest_user_message_counts(self):
        msgs = [
            {"role": "user", "content": "ok"},
            {"role": "assistant", "content": "done"},
            {"role": "user", "content": "now do something else"},
        ]
        assert user_message_authorises_chain(msgs) is False


# ---------------------------------------------------------------------------
# enforce_tool_scope — the orchestration entry point
# ---------------------------------------------------------------------------


class TestEnforceToolScope:
    def test_builtin_tool_always_allowed(self):
        # No registration; builtin not in owner map.
        assert enforce_tool_scope("file_read", [{"role": "user", "content": "go"}]) is None

    def test_no_active_mcp_source_allowed(self):
        register_mcp_tool("mcp_filesystem_write", "filesystem")
        msgs = [{"role": "user", "content": "write a file"}]
        assert enforce_tool_scope("mcp_filesystem_write", msgs) is None

    def test_same_mcp_chain_allowed(self):
        register_mcp_tool("mcp_jina_read_url", "jina")
        register_mcp_tool("mcp_jina_search_web", "jina")
        msgs = [
            {"role": "user", "content": "research X"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c1", "function": {"name": "mcp_jina_search_web"}}
            ]},
            {"role": "tool", "tool_call_id": "c1", "content": "results"},
        ]
        assert enforce_tool_scope("mcp_jina_read_url", msgs) is None

    def test_cross_mcp_without_auth_blocked(self):
        register_mcp_tool("mcp_jina_read_url", "jina")
        register_mcp_tool("mcp_filesystem_write", "filesystem")
        msgs = [
            {"role": "user", "content": "fetch and write"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c1", "function": {"name": "mcp_jina_read_url"}}
            ]},
            {"role": "tool", "tool_call_id": "c1", "content": "page content"},
        ]
        result = enforce_tool_scope("mcp_filesystem_write", msgs)
        assert result is not None
        assert "BLOCKED" in result
        assert "jina" in result and "filesystem" in result

    def test_cross_mcp_with_user_auth_allowed(self):
        register_mcp_tool("mcp_jina_read_url", "jina")
        register_mcp_tool("mcp_filesystem_write", "filesystem")
        msgs = [
            {"role": "user", "content": "fetch the URL"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c1", "function": {"name": "mcp_jina_read_url"}}
            ]},
            {"role": "tool", "tool_call_id": "c1", "content": "page content"},
            {"role": "assistant", "content": "Should I write it to /tmp/foo.txt?"},
            {"role": "user", "content": "ok, go ahead"},
        ]
        assert enforce_tool_scope("mcp_filesystem_write", msgs) is None

    def test_cross_mcp_block_emits_critical_log(self, caplog):
        register_mcp_tool("mcp_jina_read_url", "jina")
        register_mcp_tool("mcp_filesystem_write", "filesystem")
        msgs = [
            {"role": "user", "content": "x"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c1", "function": {"name": "mcp_jina_read_url"}}
            ]},
            {"role": "tool", "tool_call_id": "c1", "content": "..."},
        ]
        with caplog.at_level(logging.CRITICAL, logger="agent.mcp_tool_scope"):
            enforce_tool_scope("mcp_filesystem_write", msgs)
        assert any(
            "cross-MCP block" in r.message and "jina" in r.message
            for r in caplog.records
        )
