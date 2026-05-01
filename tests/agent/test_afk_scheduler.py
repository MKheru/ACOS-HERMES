"""Tests for agent.afk_scheduler — Patch 13.3c."""

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from agent.afk_state import (
    AFKState,
    MODE_AFK_AUTO,
    MODE_AFK_MANUAL,
    MODE_NORMAL,
    save_state,
)


@pytest.fixture
def afk_state_file(tmp_path, monkeypatch):
    """Redirect afk_state.json to a tmp dir for the duration of the test."""
    state_path = tmp_path / "afk_state.json"
    # Patch the default state path resolver
    import agent.afk_state as afk_state
    monkeypatch.setattr(afk_state, "_default_state_path", lambda: state_path)
    return state_path


# ─── process_user_message ─────────────────────────────────────────────────────

class TestProcessUserMessage:
    def test_normal_msg_keeps_normal_mode(self, afk_state_file):
        from agent.afk_scheduler import process_user_message
        state, confirm = process_user_message("Continue le WS14")
        assert state.mode == MODE_NORMAL
        assert confirm is None
        assert state.last_user_msg_at is not None

    def test_afk_trigger_flips_to_manual(self, afk_state_file):
        from agent.afk_scheduler import process_user_message
        state, confirm = process_user_message("Bonne nuit, je vais me coucher")
        assert state.mode == MODE_AFK_MANUAL
        assert confirm is not None
        assert "AFK manuel" in confirm
        assert state.entered_at is not None

    def test_afk_trigger_english(self, afk_state_file):
        from agent.afk_scheduler import process_user_message
        state, confirm = process_user_message("Good night, see you tomorrow")
        assert state.mode == MODE_AFK_MANUAL
        assert confirm is not None

    def test_revoke_in_afk_flips_to_normal(self, afk_state_file):
        from agent.afk_scheduler import process_user_message
        # Setup: AH already in AFK
        save_state(AFKState(mode=MODE_AFK_MANUAL, entered_at="2026-04-30T20:00:00Z"))
        state, confirm = process_user_message("stop, j'ai changé d'avis")
        assert state.mode == MODE_NORMAL
        assert confirm is not None
        assert "désactivé" in confirm

    def test_revoke_in_normal_no_op(self, afk_state_file):
        from agent.afk_scheduler import process_user_message
        state, confirm = process_user_message("stop")  # already in normal
        assert state.mode == MODE_NORMAL
        assert confirm is None  # no transition, no message

    def test_afk_trigger_when_already_afk_no_op(self, afk_state_file):
        from agent.afk_scheduler import process_user_message
        save_state(AFKState(mode=MODE_AFK_MANUAL, entered_at="2026-04-30T20:00:00Z"))
        state, confirm = process_user_message("je vais me coucher (encore)")
        assert state.mode == MODE_AFK_MANUAL
        assert confirm is None  # no transition

    def test_user_msg_during_afk_auto_returns_to_normal(self, afk_state_file):
        from agent.afk_scheduler import process_user_message
        save_state(AFKState(mode=MODE_AFK_AUTO, entered_at="2026-04-30T23:00:00Z"))
        state, confirm = process_user_message("salut, je suis revenu, fais X")
        assert state.mode == MODE_NORMAL
        assert confirm is not None
        assert "retour" in confirm

    def test_last_user_msg_at_always_updated(self, afk_state_file):
        from agent.afk_scheduler import process_user_message
        state, _ = process_user_message("first msg")
        first_ts = state.last_user_msg_at
        assert first_ts is not None

        # Even normal-msg-no-transition should update timestamp
        state2, _ = process_user_message("second msg")
        assert state2.last_user_msg_at is not None


# ─── check_auto_transitions ──────────────────────────────────────────────────

class TestCheckAutoTransitions:
    def test_normal_mode_at_23utc_idle_long_flips_to_auto(self, afk_state_file):
        from agent.afk_scheduler import check_auto_transitions
        # Setup: idle since 22:00 UTC (1h ago at 23:00)
        save_state(AFKState(mode=MODE_NORMAL, last_user_msg_at="2026-04-30T22:00:00Z"))
        now = datetime(2026, 4, 30, 23, 5, 0, tzinfo=timezone.utc)
        state, notif = check_auto_transitions(now)
        assert state.mode == MODE_AFK_AUTO
        assert notif is not None
        assert "AFK auto" in notif

    def test_normal_mode_at_22utc_no_trigger(self, afk_state_file):
        from agent.afk_scheduler import check_auto_transitions
        save_state(AFKState(mode=MODE_NORMAL, last_user_msg_at="2026-04-30T20:00:00Z"))
        now = datetime(2026, 4, 30, 22, 0, 0, tzinfo=timezone.utc)
        state, notif = check_auto_transitions(now)
        assert state.mode == MODE_NORMAL
        assert notif is None

    def test_normal_mode_at_23utc_recent_user_no_trigger(self, afk_state_file):
        from agent.afk_scheduler import check_auto_transitions
        # User msg 5 min ago — not idle long enough
        save_state(AFKState(mode=MODE_NORMAL, last_user_msg_at="2026-04-30T22:55:00Z"))
        now = datetime(2026, 4, 30, 23, 0, 0, tzinfo=timezone.utc)
        state, notif = check_auto_transitions(now)
        assert state.mode == MODE_NORMAL
        assert notif is None

    def test_afk_manual_at_23utc_no_override(self, afk_state_file):
        # Already in AFK manual — clock shouldn't flip the mode
        from agent.afk_scheduler import check_auto_transitions
        save_state(AFKState(mode=MODE_AFK_MANUAL, entered_at="2026-04-30T20:00:00Z",
                           last_user_msg_at="2026-04-30T20:00:00Z"))
        now = datetime(2026, 4, 30, 23, 5, 0, tzinfo=timezone.utc)
        state, notif = check_auto_transitions(now)
        assert state.mode == MODE_AFK_MANUAL  # unchanged
        assert notif is None

    def test_no_last_user_msg_at_treated_as_idle(self, afk_state_file):
        from agent.afk_scheduler import check_auto_transitions
        save_state(AFKState(mode=MODE_NORMAL, last_user_msg_at=None))
        now = datetime(2026, 4, 30, 23, 5, 0, tzinfo=timezone.utc)
        state, notif = check_auto_transitions(now)
        assert state.mode == MODE_AFK_AUTO  # treats unknown idle as ∞
        assert notif is not None
