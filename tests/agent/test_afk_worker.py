"""Tests for agent.afk_worker — WS-AUTO-002.

Spec reference: ~/Documents/Projects/ACOS-HERMES/.workspace/WS_AFK_WORK_LOOP_SPEC.md §11
"""

import json
import threading
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from agent.afk_state import (
    AFKState,
    MODE_AFK_AUTO,
    MODE_AFK_MANUAL,
    MODE_NORMAL,
    MODE_STAND_BY,
)
from agent import afk_worker as W
from agent.afk_worker import (
    AFKWorker,
    CycleResult,
    TaskCandidate,
    _render_jsonl_obj,
    _render_md_block,
    install_parent_agent_getter,
    purge_picked_index,
)


# ─── helpers ─────────────────────────────────────────────────────────────────


def _afk_state_file(tmp_path: Path, monkeypatch, mode=MODE_AFK_AUTO, **fields):
    """Set up an isolated AFK state file for the duration of one test."""
    state_path = tmp_path / "afk_state.json"
    monkeypatch.setattr(
        "agent.afk_state._default_state_path", lambda: state_path
    )
    state = AFKState(mode=mode, **fields)
    if state.entered_at is None and mode in (MODE_AFK_AUTO, MODE_AFK_MANUAL):
        state.entered_at = "2026-05-03T00:00:00Z"
    from agent.afk_state import save_state
    save_state(state)
    return state_path


def _make_worker(tmp_path: Path, monkeypatch, adapters=None, loop=None) -> AFKWorker:
    stop = threading.Event()
    return AFKWorker(stop_event=stop, adapters=adapters, loop=loop)


def _cfg(tmp_path: Path, **overrides) -> dict:
    """Default cfg dict used in tests."""
    base = {
        "enabled": True,
        "interval_minutes": 30,
        "status_files": [],
        "log_path": str(tmp_path / "AFK_LOG.md"),
        "log_jsonl_path": str(tmp_path / "afk_log.jsonl"),
        "picked_index_path": str(tmp_path / "picked.json"),
        "quota": {"minimax_skip_threshold_pct": 80},
        "stand_by_behavior": "stop",
        "model_routing": {
            "code_review": {
                "provider": "openrouter",
                "model": "anthropic/claude-sonnet-4.6",
                "base_url": "https://openrouter.ai/api/v1",
                "api_key_env": "OPENROUTER_API_KEY",
                "toolsets": ["terminal", "file", "todo"],
            },
            "research": {
                "provider": "minimax",
                "model": "MiniMax-M2.7",
                "base_url": "https://api.minimax.io/anthropic",
                "api_key_env": "",
                "toolsets": ["jina", "file", "todo"],
            },
        },
    }
    base.update(overrides)
    return base


# ═══ §11.1 unit tests ══════════════════════════════════════════════════════


class TestPickTask:
    def _write_status(self, tmp_path: Path, name: str, lines: list[str]) -> Path:
        p = tmp_path / name
        p.write_text("\n".join(lines), encoding="utf-8")
        return p

    def test_priority_order_p0_beats_p1_p2(self, tmp_path, monkeypatch):
        f = self._write_status(
            tmp_path, "S.md",
            [
                "## Section A",
                "- [ ] Low task [priority:P2] [afk:research]",
                "- [ ] Critical task [priority:P0] [afk:research]",
                "- [ ] Medium task [priority:P1] [afk:research]",
            ],
        )
        w = _make_worker(tmp_path, monkeypatch)
        cfg = _cfg(tmp_path, status_files=[str(f)])
        task = w._pick_task(cfg)
        assert task is not None
        assert "Critical task" in task.title
        assert task.priority == "P0"

    def test_excludes_afk_no(self, tmp_path, monkeypatch):
        f = self._write_status(
            tmp_path, "S.md",
            [
                "## Section",
                "- [ ] Khéri-only task [priority:P0] [afk:no]",
                "- [ ] Worker-fine task [priority:P1] [afk:research]",
            ],
        )
        w = _make_worker(tmp_path, monkeypatch)
        cfg = _cfg(tmp_path, status_files=[str(f)])
        task = w._pick_task(cfg)
        assert task is not None
        assert "Worker-fine" in task.title

    def test_excludes_blocked(self, tmp_path, monkeypatch):
        f = self._write_status(
            tmp_path, "S.md",
            [
                "## Section",
                "- [ ] Waiting task [priority:P0] [blocked:waiting-PR] [afk:research]",
                "- [ ] Free task [priority:P1] [afk:research]",
            ],
        )
        w = _make_worker(tmp_path, monkeypatch)
        cfg = _cfg(tmp_path, status_files=[str(f)])
        task = w._pick_task(cfg)
        assert task is not None
        assert "Free task" in task.title

    def test_no_tags_defaults_p2_research(self, tmp_path, monkeypatch):
        f = self._write_status(
            tmp_path, "S.md",
            [
                "## Section",
                "- [ ] Bare task with no tags",
            ],
        )
        w = _make_worker(tmp_path, monkeypatch)
        cfg = _cfg(tmp_path, status_files=[str(f)])
        task = w._pick_task(cfg)
        assert task is not None
        assert task.priority == "P2"
        assert task.afk_type == "research"

    def test_dedupe_via_picked_index_completed(self, tmp_path, monkeypatch):
        f = self._write_status(
            tmp_path, "S.md",
            [
                "## Section",
                "- [ ] Already done [priority:P0] [afk:research]",
                "- [ ] Still open [priority:P1] [afk:research]",
            ],
        )
        w = _make_worker(tmp_path, monkeypatch)
        cfg = _cfg(tmp_path, status_files=[str(f)])
        # Mark first task as completed in picked-index
        first = w._scan_status_file(f)[0]
        idx = {first.task_key(): {"last_cycle_status": "completed", "n_cycles": 1}}
        w._save_picked_index(idx, cfg)
        task = w._pick_task(cfg)
        assert task is not None
        assert "Still open" in task.title

    def test_error_cooldown_skips_then_retries(self, tmp_path, monkeypatch):
        f = self._write_status(
            tmp_path, "S.md",
            ["## Section", "- [ ] Errored task [priority:P0] [afk:research]"],
        )
        w = _make_worker(tmp_path, monkeypatch)
        cfg = _cfg(tmp_path, status_files=[str(f)])
        first = w._scan_status_file(f)[0]
        idx = {
            first.task_key(): {
                "last_cycle_status": "error",
                "n_cycles": 1,
                "error_skips_remaining": 2,
            }
        }
        w._save_picked_index(idx, cfg)
        # First call: still in cooldown, skipped, counter decremented
        assert w._pick_task(cfg) is None
        idx2 = w._load_picked_index(cfg)
        assert idx2[first.task_key()]["error_skips_remaining"] == 1
        # Decrement to 0
        assert w._pick_task(cfg) is None
        idx3 = w._load_picked_index(cfg)
        assert idx3[first.task_key()]["error_skips_remaining"] == 0
        # Now eligible again
        task = w._pick_task(cfg)
        assert task is not None
        assert "Errored task" in task.title

    def test_missing_status_file_logs_warning_skip(self, tmp_path, monkeypatch, caplog):
        w = _make_worker(tmp_path, monkeypatch)
        cfg = _cfg(tmp_path, status_files=[str(tmp_path / "does-not-exist.md")])
        with caplog.at_level("WARNING"):
            assert w._pick_task(cfg) is None
        assert any("status file not found" in m for m in caplog.messages)

    def test_in_progress_marker_eligible(self, tmp_path, monkeypatch):
        f = self._write_status(
            tmp_path, "S.md",
            ["## Section", "- [/] In progress task [priority:P0] [afk:research]"],
        )
        w = _make_worker(tmp_path, monkeypatch)
        cfg = _cfg(tmp_path, status_files=[str(f)])
        task = w._pick_task(cfg)
        assert task is not None
        assert "In progress" in task.title

    def test_done_marker_skipped(self, tmp_path, monkeypatch):
        f = self._write_status(
            tmp_path, "S.md",
            [
                "## Section",
                "- [x] Done task [priority:P0] [afk:research]",
                "- [ ] Open task [priority:P2] [afk:research]",
            ],
        )
        w = _make_worker(tmp_path, monkeypatch)
        cfg = _cfg(tmp_path, status_files=[str(f)])
        task = w._pick_task(cfg)
        assert task is not None
        assert "Open task" in task.title


class TestPickModel:
    def test_matrix_match_code_review(self, tmp_path, monkeypatch):
        w = _make_worker(tmp_path, monkeypatch)
        cfg = _cfg(tmp_path)
        t = TaskCandidate(
            source_file="x", section="y", line_number=1, raw_line="",
            title="t", priority="P1", afk_type="code_review",
        )
        m = w._pick_model(t, cfg)
        assert m is not None
        assert m["provider"] == "openrouter"
        assert m["model"] == "anthropic/claude-sonnet-4.6"

    def test_decision_structurelle_blocked(self, tmp_path, monkeypatch):
        w = _make_worker(tmp_path, monkeypatch)
        cfg = _cfg(tmp_path)
        t = TaskCandidate(
            source_file="x", section="y", line_number=1, raw_line="",
            title="t", priority="P0", afk_type="decision_structurelle",
        )
        assert w._pick_model(t, cfg) is None

    def test_unmapped_type_blocked(self, tmp_path, monkeypatch):
        w = _make_worker(tmp_path, monkeypatch)
        cfg = _cfg(tmp_path)
        t = TaskCandidate(
            source_file="x", section="y", line_number=1, raw_line="",
            title="t", priority="P0", afk_type="exotic_unknown",
        )
        assert w._pick_model(t, cfg) is None


class TestQuota:
    def test_or_logic_5h_high(self):
        q = {"minimax_5h_pct": 85.0, "minimax_weekly_pct": 5.0}
        assert AFKWorker._quota_exceeds(q, 80) is True

    def test_or_logic_weekly_high(self):
        q = {"minimax_5h_pct": 5.0, "minimax_weekly_pct": 90.0}
        assert AFKWorker._quota_exceeds(q, 80) is True

    def test_below_threshold(self):
        q = {"minimax_5h_pct": 30.0, "minimax_weekly_pct": 10.0}
        assert AFKWorker._quota_exceeds(q, 80) is False

    def test_unknown_does_not_block(self):
        q = {"minimax_5h_pct": None, "minimax_weekly_pct": None}
        assert AFKWorker._quota_exceeds(q, 80) is False


class TestRendering:
    def test_jsonl_completed_cycle(self):
        cycle = CycleResult(
            tick_id="afk-2026-05-15-2230",
            started_at="2026-05-15T22:30:00Z",
            completed_at="2026-05-15T22:33:04Z",
            afk_mode="afk_auto",
            status="completed",
            task=TaskCandidate(
                source_file="/h/SMCP_STATUS.md", section="§3.4",
                line_number=42, raw_line="", title="Patch 16",
                priority="P1", afk_type="code_refactor",
            ),
            model_used={"provider": "openrouter", "model": "anthropic/claude-sonnet-4.6"},
            delegation_summary="did the refactor",
            delegation_api_calls=12,
            delegation_duration_s=184.3,
            delegation_status="completed",
            quota_before={"minimax_5h_pct": 22.1, "minimax_weekly_pct": 4.0},
            quota_after={"minimax_5h_pct": 22.1, "minimax_weekly_pct": 4.0},
        )
        obj = _render_jsonl_obj(cycle)
        assert obj["status"] == "completed"
        assert obj["task"]["title"] == "Patch 16"
        assert obj["delegation"]["api_calls"] == 12
        # JSON-serialisable
        assert json.dumps(obj)

    def test_md_block_contains_key_fields(self):
        cycle = CycleResult(
            tick_id="afk-x",
            started_at="2026-05-15T22:30:00Z",
            afk_mode="afk_auto",
            status="completed",
            task=TaskCandidate(
                source_file="/x", section="§3", line_number=1, raw_line="",
                title="My Task", priority="P0", afk_type="research",
            ),
            model_used={"provider": "minimax", "model": "MiniMax-M2.7"},
        )
        block = _render_md_block(cycle)
        assert "completed" in block
        assert "My Task" in block
        assert "minimax" in block
        assert block.rstrip().endswith("---")


class TestLogAppend:
    def test_md_appends_block(self, tmp_path, monkeypatch):
        w = _make_worker(tmp_path, monkeypatch)
        cfg = _cfg(tmp_path)
        path = Path(cfg["log_path"])
        path.write_text("EXISTING\n", encoding="utf-8")  # pre-existing content
        cycle = CycleResult(
            tick_id="t1", started_at="2026-05-15T22:30:00Z",
            afk_mode="afk_auto", status="skipped_no_task",
        )
        w._log_cycle(cycle, cfg)
        content = path.read_text(encoding="utf-8")
        assert content.startswith("EXISTING\n")
        assert "skipped_no_task" in content

    def test_jsonl_one_line_per_cycle(self, tmp_path, monkeypatch):
        w = _make_worker(tmp_path, monkeypatch)
        cfg = _cfg(tmp_path)
        path = Path(cfg["log_jsonl_path"])
        for i in range(3):
            cycle = CycleResult(
                tick_id=f"t{i}", started_at=f"2026-05-15T22:3{i}:00Z",
                afk_mode="afk_auto", status="skipped_no_task",
            )
            w._log_cycle(cycle, cfg)
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 3
        for ln in lines:
            obj = json.loads(ln)  # all valid JSON
            assert obj["status"] == "skipped_no_task"


class TestDynamicRecap:
    def test_no_jsonl_returns_fallback(self, tmp_path, monkeypatch):
        from agent.afk_heartbeat import _build_dynamic_recap
        state = AFKState(mode=MODE_AFK_AUTO, entered_at="2026-05-03T00:00:00Z")
        cfg = {"log_jsonl_path": str(tmp_path / "missing.jsonl")}
        recap = _build_dynamic_recap(state, cfg=cfg)
        assert "Aucun cycle" in recap or "absent" in recap

    def test_filters_pre_entered_at(self, tmp_path, monkeypatch):
        from agent.afk_heartbeat import _build_dynamic_recap
        # Two cycles: one before AFK entered, one after
        path = tmp_path / "afk_log.jsonl"
        with path.open("w", encoding="utf-8") as f:
            json.dump({
                "tick_id": "before", "started_at": "2026-05-01T00:00:00Z",
                "status": "completed", "task": {"title": "OLD"},
            }, f); f.write("\n")
            json.dump({
                "tick_id": "after", "started_at": "2026-05-04T00:00:00Z",
                "status": "completed", "task": {"title": "NEW"},
            }, f); f.write("\n")
        state = AFKState(mode=MODE_AFK_AUTO, entered_at="2026-05-03T00:00:00Z")
        cfg = {"log_jsonl_path": str(path)}
        recap = _build_dynamic_recap(state, cfg=cfg)
        assert "NEW" in recap
        assert "OLD" not in recap


class TestPurgePickedIndex:
    def test_purge_removes_existing(self, tmp_path, monkeypatch):
        p = tmp_path / "picked.json"
        p.write_text("{}", encoding="utf-8")
        cfg = {"picked_index_path": str(p)}
        assert purge_picked_index(cfg) is True
        assert not p.exists()

    def test_purge_missing_no_op(self, tmp_path, monkeypatch):
        cfg = {"picked_index_path": str(tmp_path / "ghost.json")}
        # Should not raise
        result = purge_picked_index(cfg)
        assert result is False


# ═══ §11.2 integration tests ══════════════════════════════════════════════


class TestTickGates:
    def test_normal_mode_silent_skip(self, tmp_path, monkeypatch):
        _afk_state_file(tmp_path, monkeypatch, mode=MODE_NORMAL)
        w = _make_worker(tmp_path, monkeypatch)
        cfg = _cfg(tmp_path)
        # No log file should be created (silent skip)
        w._tick(cfg)
        assert not Path(cfg["log_jsonl_path"]).exists()

    def test_stand_by_silent_skip(self, tmp_path, monkeypatch):
        # Q8 stand_by behavior = stop. is_afk() returns False for stand_by
        # (AFK_MODES = manual + auto only), so the worker silent-skips just
        # like in normal mode. No log spam every 30 min for the duration of
        # an indefinite stand_by.
        _afk_state_file(tmp_path, monkeypatch, mode=MODE_STAND_BY)
        w = _make_worker(tmp_path, monkeypatch)
        cfg = _cfg(tmp_path)
        w._tick(cfg)
        assert not Path(cfg["log_jsonl_path"]).exists()

    def test_quota_high_logs_skipped_quota(self, tmp_path, monkeypatch):
        _afk_state_file(tmp_path, monkeypatch, mode=MODE_AFK_AUTO)
        w = _make_worker(tmp_path, monkeypatch)
        cfg = _cfg(tmp_path)
        with patch.object(w, "_read_quota", return_value={
            "minimax_5h_pct": 85.0, "minimax_weekly_pct": 10.0,
        }):
            w._tick(cfg)
        obj = json.loads(
            Path(cfg["log_jsonl_path"]).read_text(encoding="utf-8").splitlines()[0]
        )
        assert obj["status"] == "skipped_quota"

    def test_no_eligible_task_logs_skipped_no_task(self, tmp_path, monkeypatch):
        _afk_state_file(tmp_path, monkeypatch, mode=MODE_AFK_AUTO)
        w = _make_worker(tmp_path, monkeypatch)
        cfg = _cfg(tmp_path)  # status_files=[] by default
        with patch.object(w, "_read_quota", return_value={
            "minimax_5h_pct": 5.0, "minimax_weekly_pct": 3.0,
        }):
            w._tick(cfg)
        obj = json.loads(
            Path(cfg["log_jsonl_path"]).read_text(encoding="utf-8").splitlines()[0]
        )
        assert obj["status"] == "skipped_no_task"


class TestTickFullCycle:
    def test_decision_blocked_posts_discord(self, tmp_path, monkeypatch):
        _afk_state_file(tmp_path, monkeypatch, mode=MODE_AFK_AUTO)
        f = tmp_path / "S.md"
        f.write_text(
            "## Section\n- [ ] X [priority:P0] [afk:decision_structurelle]\n",
            encoding="utf-8",
        )
        w = _make_worker(tmp_path, monkeypatch)
        cfg = _cfg(tmp_path, status_files=[str(f)])
        with patch.object(w, "_read_quota", return_value={
            "minimax_5h_pct": 5.0, "minimax_weekly_pct": 3.0,
        }), patch.object(w, "_post_discord") as post_mock:
            w._tick(cfg)
        # Discord posted with the block message
        assert post_mock.call_count == 1
        msg = post_mock.call_args[0][0]
        assert "décision structurelle" in msg
        # Cycle logged with the right status
        obj = json.loads(
            Path(cfg["log_jsonl_path"]).read_text(encoding="utf-8").splitlines()[0]
        )
        assert obj["status"] == "skipped_decision_blocked"

    def test_completed_cycle_logs_full(self, tmp_path, monkeypatch):
        _afk_state_file(tmp_path, monkeypatch, mode=MODE_AFK_AUTO)
        f = tmp_path / "S.md"
        f.write_text(
            "## Section A\n- [ ] Refactor X [priority:P1] [afk:code_review]\n",
            encoding="utf-8",
        )
        w = _make_worker(tmp_path, monkeypatch)
        cfg = _cfg(tmp_path, status_files=[str(f)])
        # Stub the parent_agent getter and delegate_task
        install_parent_agent_getter(lambda: object())
        fake_payload = json.dumps({
            "results": [{
                "status": "completed",
                "summary": "OK done",
                "api_calls": 5,
                "duration_seconds": 12.3,
            }]
        })
        with patch.object(w, "_read_quota", return_value={
            "minimax_5h_pct": 5.0, "minimax_weekly_pct": 3.0,
        }), patch("tools.delegate_tool.delegate_task", return_value=fake_payload):
            w._tick(cfg)
        # Log shows completed
        obj = json.loads(
            Path(cfg["log_jsonl_path"]).read_text(encoding="utf-8").splitlines()[0]
        )
        assert obj["status"] == "completed"
        assert obj["delegation"]["api_calls"] == 5
        assert obj["model_used"]["provider"] == "openrouter"
        # Picked-index updated
        idx = json.loads(Path(cfg["picked_index_path"]).read_text(encoding="utf-8"))
        assert any(
            entry.get("last_cycle_status") == "completed"
            for entry in idx.values()
        )
        # Cleanup
        install_parent_agent_getter(None)


# ═══ §11.4 security ═══════════════════════════════════════════════════════


class TestSecurity:
    def test_worker_does_not_modify_status_files(self, tmp_path, monkeypatch):
        """The worker must only READ status files, never write to them."""
        _afk_state_file(tmp_path, monkeypatch, mode=MODE_AFK_AUTO)
        f = tmp_path / "S.md"
        original = "## Section\n- [ ] T [priority:P0] [afk:research]\n"
        f.write_text(original, encoding="utf-8")
        w = _make_worker(tmp_path, monkeypatch)
        cfg = _cfg(tmp_path, status_files=[str(f)])
        install_parent_agent_getter(lambda: object())
        fake_payload = json.dumps({
            "results": [{
                "status": "completed", "summary": "done",
                "api_calls": 1, "duration_seconds": 1.0,
            }]
        })
        with patch.object(w, "_read_quota", return_value={
            "minimax_5h_pct": 5.0, "minimax_weekly_pct": 3.0,
        }), patch("tools.delegate_tool.delegate_task", return_value=fake_payload):
            w._tick(cfg)
        # Status file content unchanged
        assert f.read_text(encoding="utf-8") == original
        install_parent_agent_getter(None)

    def test_disabled_via_config_thread_exits(self, tmp_path, monkeypatch):
        """enabled: false → run() returns immediately without ticking."""
        w = _make_worker(tmp_path, monkeypatch)
        with patch.object(w, "_load_cfg", return_value={"enabled": False}), \
             patch.object(w, "_tick") as tick_mock:
            w.run()
            tick_mock.assert_not_called()

    def test_api_key_value_never_in_log(self, tmp_path, monkeypatch):
        """The log files must never contain the resolved api_key value."""
        _afk_state_file(tmp_path, monkeypatch, mode=MODE_AFK_AUTO)
        f = tmp_path / "S.md"
        f.write_text(
            "## S\n- [ ] Code thing [priority:P0] [afk:code_review]\n",
            encoding="utf-8",
        )
        secret = "sk-or-v1-FAKE-SECRET-VALUE-12345"
        monkeypatch.setenv("OPENROUTER_API_KEY", secret)
        w = _make_worker(tmp_path, monkeypatch)
        cfg = _cfg(tmp_path, status_files=[str(f)])
        install_parent_agent_getter(lambda: object())
        fake_payload = json.dumps({
            "results": [{
                "status": "completed", "summary": "done",
                "api_calls": 1, "duration_seconds": 1.0,
            }]
        })
        with patch.object(w, "_read_quota", return_value={
            "minimax_5h_pct": 5.0, "minimax_weekly_pct": 3.0,
        }), patch("tools.delegate_tool.delegate_task", return_value=fake_payload):
            w._tick(cfg)
        for path_key in ("log_path", "log_jsonl_path"):
            content = Path(cfg[path_key]).read_text(encoding="utf-8")
            assert secret not in content, f"secret leaked into {path_key}"
        install_parent_agent_getter(None)
