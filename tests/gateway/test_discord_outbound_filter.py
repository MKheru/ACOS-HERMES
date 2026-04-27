"""Tests for ACOS-HERMES Patch 2 Discord outbound filter."""

import os
from unittest.mock import patch

from gateway.platforms.discord_outbound_filter import filter_outbound


class TestChannelWhitelist:
    def test_allowed_when_env_unset(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("DISCORD_HERMES_CHANNEL_ID", None)
            allowed, sanitized, reason = filter_outbound("hello", chat_id="123")
        assert allowed is True
        assert reason is None

    def test_allowed_when_env_empty(self):
        with patch.dict(os.environ, {"DISCORD_HERMES_CHANNEL_ID": ""}, clear=False):
            allowed, sanitized, reason = filter_outbound("hello", chat_id="123")
        assert allowed is True
        assert reason is None

    def test_allowed_on_match(self):
        with patch.dict(os.environ, {"DISCORD_HERMES_CHANNEL_ID": "999"}, clear=False):
            allowed, sanitized, reason = filter_outbound("hi", chat_id="999")
        assert allowed is True
        assert reason is None

    def test_blocked_on_mismatch(self):
        with patch.dict(os.environ, {"DISCORD_HERMES_CHANNEL_ID": "999"}, clear=False):
            allowed, sanitized, reason = filter_outbound("hi", chat_id="123")
        assert allowed is False
        assert reason is not None
        assert "999" in reason and "123" in reason
        assert sanitized == ""

    def test_chat_id_string_coercion(self):
        # int chat_id from caller should still match a string env value.
        with patch.dict(os.environ, {"DISCORD_HERMES_CHANNEL_ID": "999"}, clear=False):
            allowed, _, _ = filter_outbound("hi", chat_id=999)
        assert allowed is True

    def test_thread_id_does_not_bypass_whitelist(self):
        """Threads in the allowed channel still pass through chat_id, not thread_id."""
        with patch.dict(os.environ, {"DISCORD_HERMES_CHANNEL_ID": "999"}, clear=False):
            # chat_id mismatch — block even with a benign-looking thread_id
            allowed, _, _ = filter_outbound("hi", chat_id="000", thread_id="999")
        assert allowed is False


class TestRedaction:
    def test_benign_text_unchanged(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("DISCORD_HERMES_CHANNEL_ID", None)
            allowed, sanitized, _ = filter_outbound(
                "All systems nominal.", chat_id="1",
            )
        assert allowed is True
        assert sanitized == "All systems nominal."

    def test_secret_in_text_is_redacted(self):
        # Use a synthetic prefixed token that the redactor recognises.
        # sk- prefix is in the redactor's known prefix set.
        leaky = "Here is the key: sk-proj-AAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("DISCORD_HERMES_CHANNEL_ID", None)
            allowed, sanitized, _ = filter_outbound(leaky, chat_id="1")
        assert allowed is True
        # The full token must not survive verbatim.
        assert "sk-proj-AAAAAAAAAAAAAAAAAAAAAAAAAAAA" not in sanitized

    def test_none_content_passthrough(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("DISCORD_HERMES_CHANNEL_ID", None)
            allowed, sanitized, _ = filter_outbound(None, chat_id="1")
        assert allowed is True
        assert sanitized is None
