"""Tests for agent.afk_heartbeat — Patch 13.4."""

from datetime import datetime, timedelta, timezone

import pytest

from agent.afk_state import (
    AFKState,
    MODE_AFK_AUTO,
    MODE_AFK_MANUAL,
    MODE_NORMAL,
    MODE_STAND_BY,
    save_state,
)


@pytest.fixture
def afk_state_file(tmp_path, monkeypatch):
    """Redirect afk_state.json to a tmp dir for the duration of the test."""
    state_path = tmp_path / "afk_state.json"
    import agent.afk_state as afk_state
    monkeypatch.setattr(afk_state, "_default_state_path", lambda: state_path)
    return state_path


class TestEvaluateHeartbeat:
    def test_normal_mode_no_hb(self, afk_state_file):
        from agent.afk_heartbeat import evaluate_heartbeat
        save_state(AFKState(mode=MODE_NORMAL))
        state, msg = evaluate_heartbeat()
        assert state.mode == MODE_NORMAL
        assert msg is None

    def test_afk_just_entered_no_hb_yet(self, afk_state_file):
        from agent.afk_heartbeat import evaluate_heartbeat
        save_state(AFKState(mode=MODE_AFK_MANUAL,
                           entered_at="2026-04-30T20:00:00Z"))
        # Same day
        now = datetime(2026, 4, 30, 22, 0, 0, tzinfo=timezone.utc)
        state, msg = evaluate_heartbeat(now)
        assert state.heartbeats_sent == 0
        assert msg is None

    def test_hb1_fires_at_j3(self, afk_state_file):
        from agent.afk_heartbeat import evaluate_heartbeat
        save_state(AFKState(mode=MODE_AFK_MANUAL,
                           entered_at="2026-04-30T20:00:00Z"))
        # J+3 days + 1h
        now = datetime(2026, 5, 3, 21, 0, 0, tzinfo=timezone.utc)
        state, msg = evaluate_heartbeat(now)
        assert state.heartbeats_sent == 1
        assert msg is not None
        assert "Heartbeat 1/3" in msg

    def test_hb1_doesnt_fire_twice(self, afk_state_file):
        from agent.afk_heartbeat import evaluate_heartbeat
        save_state(AFKState(mode=MODE_AFK_MANUAL,
                           entered_at="2026-04-30T20:00:00Z",
                           heartbeats_sent=1,
                           last_heartbeat_at="2026-05-03T20:00:00Z"))
        # Still J+3, should not re-fire
        now = datetime(2026, 5, 4, 12, 0, 0, tzinfo=timezone.utc)
        state, msg = evaluate_heartbeat(now)
        assert state.heartbeats_sent == 1
        assert msg is None  # waiting for J7

    def test_hb2_fires_at_j7(self, afk_state_file):
        from agent.afk_heartbeat import evaluate_heartbeat
        save_state(AFKState(mode=MODE_AFK_MANUAL,
                           entered_at="2026-04-30T20:00:00Z",
                           heartbeats_sent=1,
                           last_heartbeat_at="2026-05-03T20:00:00Z"))
        now = datetime(2026, 5, 7, 21, 0, 0, tzinfo=timezone.utc)
        state, msg = evaluate_heartbeat(now)
        assert state.heartbeats_sent == 2
        assert msg is not None
        assert "Heartbeat 2/3" in msg

    def test_hb3_fires_at_j14(self, afk_state_file):
        from agent.afk_heartbeat import evaluate_heartbeat
        save_state(AFKState(mode=MODE_AFK_MANUAL,
                           entered_at="2026-04-30T20:00:00Z",
                           heartbeats_sent=2,
                           last_heartbeat_at="2026-05-07T20:00:00Z"))
        now = datetime(2026, 5, 14, 21, 0, 0, tzinfo=timezone.utc)
        state, msg = evaluate_heartbeat(now)
        assert state.heartbeats_sent == 3
        assert msg is not None
        assert "Heartbeat 3/3" in msg
        assert "DERNIER PING" in msg

    def test_stand_by_after_hb3_plus_24h(self, afk_state_file):
        from agent.afk_heartbeat import evaluate_heartbeat
        save_state(AFKState(mode=MODE_AFK_MANUAL,
                           entered_at="2026-04-30T20:00:00Z",
                           heartbeats_sent=3,
                           last_heartbeat_at="2026-05-14T20:00:00Z"))
        # 25h after HB3
        now = datetime(2026, 5, 15, 21, 0, 0, tzinfo=timezone.utc)
        state, msg = evaluate_heartbeat(now)
        assert state.mode == MODE_STAND_BY
        assert msg is not None
        assert "stand_by" in msg.lower()

    def test_no_stand_by_within_24h_after_hb3(self, afk_state_file):
        from agent.afk_heartbeat import evaluate_heartbeat
        save_state(AFKState(mode=MODE_AFK_MANUAL,
                           entered_at="2026-04-30T20:00:00Z",
                           heartbeats_sent=3,
                           last_heartbeat_at="2026-05-14T20:00:00Z"))
        # Only 12h after HB3
        now = datetime(2026, 5, 15, 8, 0, 0, tzinfo=timezone.utc)
        state, msg = evaluate_heartbeat(now)
        assert state.mode == MODE_AFK_MANUAL  # unchanged
        assert msg is None

    def test_afk_auto_also_gets_hb(self, afk_state_file):
        from agent.afk_heartbeat import evaluate_heartbeat
        save_state(AFKState(mode=MODE_AFK_AUTO,
                           entered_at="2026-04-30T23:00:00Z"))
        now = datetime(2026, 5, 4, 0, 0, 0, tzinfo=timezone.utc)
        state, msg = evaluate_heartbeat(now)
        assert state.heartbeats_sent == 1
        assert msg is not None

    def test_no_entered_at_self_heals(self, afk_state_file):
        from agent.afk_heartbeat import evaluate_heartbeat
        save_state(AFKState(mode=MODE_AFK_MANUAL, entered_at=None))
        state, msg = evaluate_heartbeat()
        # entered_at gets set, no msg this round
        assert state.entered_at is not None
        assert msg is None

    def test_stand_by_mode_returns_no_action(self, afk_state_file):
        from agent.afk_heartbeat import evaluate_heartbeat
        save_state(AFKState(mode=MODE_STAND_BY,
                           entered_at="2026-04-30T20:00:00Z",
                           heartbeats_sent=3))
        now = datetime(2026, 6, 1, 0, 0, 0, tzinfo=timezone.utc)
        state, msg = evaluate_heartbeat(now)
        assert state.mode == MODE_STAND_BY  # unchanged
        assert msg is None
