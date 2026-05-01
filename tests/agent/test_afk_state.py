"""Tests for agent.afk_state — Patch 13.3b."""

import json
from pathlib import Path

import pytest

from agent.afk_state import (
    AFKState,
    MODE_AFK_AUTO,
    MODE_AFK_MANUAL,
    MODE_NORMAL,
    MODE_STAND_BY,
    check_publish_blocked_in_afk,
    is_afk_from_messages,
    is_publish_action,
    load_state,
    save_state,
)


# ─── AFKState dataclass ──────────────────────────────────────────────────────

class TestAFKState:
    def test_default_is_normal(self):
        s = AFKState()
        assert s.mode == MODE_NORMAL
        assert s.is_afk() is False

    def test_afk_manual_is_afk(self):
        s = AFKState(mode=MODE_AFK_MANUAL)
        assert s.is_afk() is True

    def test_afk_auto_is_afk(self):
        s = AFKState(mode=MODE_AFK_AUTO)
        assert s.is_afk() is True

    def test_stand_by_not_afk_for_publish(self):
        # Standby is post-AFK after 3 heartbeats; publish blacklist still
        # applies via check_publish_blocked_in_afk only if mode IS AFK.
        s = AFKState(mode=MODE_STAND_BY)
        assert s.is_afk() is False

    def test_to_from_dict_roundtrip(self):
        s = AFKState(mode=MODE_AFK_MANUAL, entered_at="2026-04-30T20:00:00Z",
                    tokens_minimax_5h=42)
        d = s.to_dict()
        s2 = AFKState.from_dict(d)
        assert s2 == s

    def test_from_dict_drops_unknown_fields(self):
        d = {"mode": MODE_AFK_AUTO, "rogue_field": "ignored"}
        s = AFKState.from_dict(d)
        assert s.mode == MODE_AFK_AUTO


# ─── load / save ──────────────────────────────────────────────────────────────

class TestLoadSave:
    def test_load_missing_returns_default(self, tmp_path: Path):
        p = tmp_path / "afk_state.json"
        s = load_state(p)
        assert s.mode == MODE_NORMAL

    def test_save_then_load_roundtrip(self, tmp_path: Path):
        p = tmp_path / "afk_state.json"
        original = AFKState(mode=MODE_AFK_MANUAL, entered_at="2026-04-30T20:00:00Z",
                           heartbeats_sent=1)
        save_state(original, p)
        assert p.is_file()
        loaded = load_state(p)
        assert loaded == original

    def test_save_atomic(self, tmp_path: Path):
        p = tmp_path / "afk_state.json"
        s1 = AFKState(mode=MODE_AFK_MANUAL)
        save_state(s1, p)
        s2 = AFKState(mode=MODE_AFK_AUTO)
        save_state(s2, p)
        loaded = load_state(p)
        assert loaded.mode == MODE_AFK_AUTO

    def test_load_corrupt_returns_default(self, tmp_path: Path):
        p = tmp_path / "afk_state.json"
        p.write_text("{ this is not json")
        s = load_state(p)
        assert s.mode == MODE_NORMAL


# ─── publish blacklist ────────────────────────────────────────────────────────

class TestPublishBlacklist:
    def test_git_push_blocked(self):
        assert is_publish_action("mcp_git_git_push") is True
        assert is_publish_action("git_push") is True
        assert is_publish_action("mcp_github_git_force_push") is True

    def test_gh_pr_create_blocked(self):
        assert is_publish_action("mcp_github_gh_pr_create") is True
        assert is_publish_action("gh_pr_merge") is True
        assert is_publish_action("gh_pr_close") is True

    def test_gh_issue_create_blocked(self):
        assert is_publish_action("mcp_github_gh_issue_create") is True
        assert is_publish_action("gh_issue_comment") is True

    def test_discord_send_blocked(self):
        assert is_publish_action("mcp_discord_send_message") is True
        assert is_publish_action("mcp_discord_send_dm") is True

    def test_other_messaging_blocked(self):
        assert is_publish_action("mcp_slack_send") is True
        assert is_publish_action("mcp_telegram_send") is True
        assert is_publish_action("mcp_signal_send") is True
        assert is_publish_action("mcp_email_send") is True

    def test_webhook_blocked(self):
        assert is_publish_action("mcp_webhook_post") is True

    def test_normal_tools_allowed(self):
        # These should NOT match the blacklist
        assert is_publish_action("mcp_filesystem_read_file") is False
        assert is_publish_action("mcp_git_git_commit") is False
        assert is_publish_action("mcp_git_git_status") is False
        assert is_publish_action("mcp_acos_builder_cargo_build") is False
        assert is_publish_action("mcp_jina_search_web") is False
        assert is_publish_action("git_diff") is False

    def test_empty_or_none_tool_name(self):
        assert is_publish_action("") is False
        assert is_publish_action(None) is False  # type: ignore


class TestCheckPublishBlockedInAfk:
    def test_normal_mode_allows_publish(self):
        s = AFKState(mode=MODE_NORMAL)
        blocked, _ = check_publish_blocked_in_afk("mcp_git_git_push", s)
        assert blocked is False

    def test_afk_manual_blocks_publish(self):
        s = AFKState(mode=MODE_AFK_MANUAL)
        blocked, reason = check_publish_blocked_in_afk("mcp_git_git_push", s)
        assert blocked is True
        assert "afk_manual" in reason

    def test_afk_auto_blocks_publish(self):
        s = AFKState(mode=MODE_AFK_AUTO)
        blocked, reason = check_publish_blocked_in_afk("mcp_github_gh_pr_create", s)
        assert blocked is True
        assert "afk_auto" in reason

    def test_afk_allows_non_publish(self):
        s = AFKState(mode=MODE_AFK_AUTO)
        blocked, _ = check_publish_blocked_in_afk("mcp_git_git_commit", s)
        assert blocked is False

    def test_stand_by_treats_as_not_afk(self):
        # is_afk() is False in stand_by — publish blacklist doesn't fire here.
        # The runtime layer will refuse new tool calls anyway when in stand_by,
        # so no double-enforcement needed.
        s = AFKState(mode=MODE_STAND_BY)
        blocked, _ = check_publish_blocked_in_afk("mcp_git_git_push", s)
        assert blocked is False


# ─── is_afk_from_messages ────────────────────────────────────────────────────

class TestIsAfkFromMessages:
    def test_no_messages(self):
        assert is_afk_from_messages([]) is False

    def test_no_user_messages(self):
        msgs = [{"role": "system", "content": "you are AH"}]
        assert is_afk_from_messages(msgs) is False

    def test_french_trigger(self):
        msgs = [{"role": "user", "content": "Bonne nuit, je vais me coucher"}]
        assert is_afk_from_messages(msgs) is True

    def test_english_trigger(self):
        msgs = [{"role": "user", "content": "good night, talk to you tomorrow"}]
        assert is_afk_from_messages(msgs) is True

    def test_only_latest_user_msg_counts(self):
        # AFK trigger in earlier msg, normal text in latest → NOT AFK
        msgs = [
            {"role": "user", "content": "je vais me coucher"},
            {"role": "tool", "content": "result"},
            {"role": "user", "content": "ok je suis revenu, fais X"},
        ]
        assert is_afk_from_messages(msgs) is False

    def test_multimodal_content_flattened(self):
        msgs = [{
            "role": "user",
            "content": [{"type": "text", "text": "good night"}, {"type": "text", "text": "see you"}],
        }]
        assert is_afk_from_messages(msgs) is True
