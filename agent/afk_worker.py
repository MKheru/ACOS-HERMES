"""AFK Work Loop — WS-AUTO-002.

Background thread that, while AH is in afk_manual or afk_auto, periodically:
  1. Checks MiniMax quota headroom
  2. Picks a tagged in_progress task from configured status files
  3. Routes to a model based on the task type (matrix from config.yaml)
  4. Delegates execution to a leaf sub-agent via delegate_task
  5. Appends the cycle to ~/AFK_LOG.md + ~/.hermes/afk_log.jsonl

Driven by gateway/run.py at service startup; stops on stop_event.

Spec reference: ~/Documents/Projects/ACOS-HERMES/.workspace/WS_AFK_WORK_LOOP_SPEC.md
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import re
import tempfile
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from agent.afk_state import (
    AFKState,
    MODE_STAND_BY,
    MODE_NORMAL,
    AFK_MODES,
    load_state,
    save_state,
    now_iso,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults — all overridable via ~/.hermes/config.yaml afk_worker.*
# ---------------------------------------------------------------------------

DEFAULT_INTERVAL_MIN = 30
DEFAULT_QUOTA_THRESHOLD_PCT = 80
DEFAULT_LOG_MD = "~/AFK_LOG.md"
DEFAULT_LOG_JSONL = "~/.hermes/afk_log.jsonl"
DEFAULT_PICKED_INDEX = "~/.hermes/afk_worker_picked.json"
DEFAULT_STAND_BY_BEHAVIOR = "stop"  # Q8

# Cooldown (§3.5)
COOLDOWN_ERROR_THRESHOLD = 3
COOLDOWN_LEVEL_TICK_SKIPS: dict[int, float] = {1: 2, 2: 4, 3: float("inf")}
ERROR_COOLDOWN_TICKS = 3  # ticks to skip a task that errored before retry

# Tag regexes (§3.4)
RE_PRIORITY = re.compile(r"\[priority:(P0|P1|P2)\]", re.IGNORECASE)
RE_AFK_TYPE = re.compile(r"\[afk:([a-z_]+)\]", re.IGNORECASE)
RE_BLOCKED = re.compile(r"\[blocked:[^\]]+\]", re.IGNORECASE)
RE_AFK_NO = re.compile(r"\[afk:no\]", re.IGNORECASE)
RE_TAG_ANY = re.compile(r"\[[a-z_]+:[^\]]+\]", re.IGNORECASE)

# Markdown task line — captures the checkbox marker and the rest.
# Eligible markers: ' ' (todo) and '/' (in_progress).
RE_TASK_LINE = re.compile(r"^\s*-\s*\[([ /])\]\s+(.+?)\s*$")
RE_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")

# Priority sort key (lower = higher priority)
_PRIORITY_RANK = {"P0": 0, "P1": 1, "P2": 2}


# ---------------------------------------------------------------------------
# Module-level parent-agent getter (set by gateway/run.py at startup)
# ---------------------------------------------------------------------------

_RUNTIME_PARENT_AGENT_GETTER = None


def install_parent_agent_getter(fn) -> None:
    """Register a callable that returns the active parent_agent.

    Called by gateway/run.py during startup. Lets the worker thread reach
    into the gateway's session-keyed agent map without holding a reference
    cycle.
    """
    global _RUNTIME_PARENT_AGENT_GETTER
    _RUNTIME_PARENT_AGENT_GETTER = fn


def _get_runtime_parent_agent():
    if _RUNTIME_PARENT_AGENT_GETTER is None:
        return None
    try:
        return _RUNTIME_PARENT_AGENT_GETTER()
    except Exception:
        logger.exception("afk-worker: parent_agent getter raised")
        return None


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class TaskCandidate:
    source_file: str
    section: str
    line_number: int
    raw_line: str
    title: str
    priority: str  # "P0" | "P1" | "P2"
    afk_type: str
    is_excluded: bool = False
    exclusion_reason: Optional[str] = None

    def task_key(self) -> str:
        return f"{self.source_file}::{self.section}::{self.title}"

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "source_file": self.source_file,
            "section": self.section,
            "line_number": self.line_number,
            "title": self.title,
            "priority": self.priority,
            "afk_type": self.afk_type,
        }


@dataclass
class CycleResult:
    tick_id: str
    started_at: str
    completed_at: Optional[str] = None
    afk_mode: Optional[str] = None
    status: str = "pending"
    task: Optional[TaskCandidate] = None
    model_used: Optional[dict[str, str]] = None
    delegation_summary: Optional[str] = None
    delegation_api_calls: Optional[int] = None
    delegation_duration_s: Optional[float] = None
    delegation_status: Optional[str] = None
    delegation_error: Optional[str] = None
    quota_before: Optional[dict] = None
    quota_after: Optional[dict] = None
    notes: list[str] = field(default_factory=list)


CYCLE_STATUSES = (
    "completed",
    "skipped_quota",
    "skipped_no_task",
    "skipped_decision_blocked",
    "skipped_cooldown",
    "error",
    "interrupted",
)


# ---------------------------------------------------------------------------
# AFKWorker
# ---------------------------------------------------------------------------


class AFKWorker:
    """Background thread driving the AFK Work Loop."""

    def __init__(
        self,
        stop_event: threading.Event,
        adapters=None,
        loop=None,
    ):
        self.stop_event = stop_event
        self.adapters = adapters
        self.loop = loop
        self._delegation_lock = threading.Lock()  # Q6: 1 delegation at a time
        self._cooldown_skip_remaining = 0
        self._consecutive_errors = 0

    # ───────────────────────── thread entry-point ─────────────────────

    def run(self) -> None:
        """Loop until stop_event is set."""
        cfg = self._load_cfg()
        if not cfg.get("enabled", True):
            logger.info("afk-worker disabled via config; thread exits")
            return

        interval_s = self._interval_seconds(cfg)
        logger.info("afk-worker thread started (interval=%ds)", interval_s)

        # Initial sleep so the worker doesn't stampede right after a service
        # restart (heartbeat ticker is doing its own thing).
        if self.stop_event.wait(interval_s):
            return

        while not self.stop_event.is_set():
            try:
                self._tick(cfg)
            except Exception:
                logger.exception("afk-worker tick raised; will retry next interval")
                self._consecutive_errors += 1
                if self._consecutive_errors >= COOLDOWN_ERROR_THRESHOLD:
                    self._raise_cooldown()

            # Re-load config each tick so live edits apply without restart
            cfg = self._load_cfg()
            if not cfg.get("enabled", True):
                logger.info("afk-worker disabled via config; thread exits")
                return
            interval_s = self._interval_seconds(cfg)

            if self.stop_event.wait(interval_s):
                return
        logger.info("afk-worker thread stopped")

    # ───────────────────────── single tick ────────────────────────────

    def _tick(self, cfg: dict) -> None:
        cycle = CycleResult(
            tick_id=self._build_tick_id(),
            started_at=now_iso(),
        )

        state = load_state()
        cycle.afk_mode = state.mode

        # Gate 1: not in AFK → silent skip (no log entry).
        # This also catches MODE_STAND_BY (Q8 stop): is_afk() only returns
        # True for afk_manual/afk_auto, so stand_by silent-skips here too.
        # Silent rather than "skipped_stand_by" log entries because stand_by
        # can last indefinitely and we don't want 48 log lines/day of noise.
        if not state.is_afk():
            return

        # Gate 2: cooldown
        if self._cooldown_skip_remaining > 0:
            self._cooldown_skip_remaining -= 1
            cycle.status = "skipped_cooldown"
            cycle.notes.append(
                f"cooldown_level={state.cooldown_level} "
                f"skips_remaining={self._cooldown_skip_remaining}"
            )
            cycle.completed_at = now_iso()
            self._log_cycle(cycle, cfg)
            return

        # Gate 4: MiniMax quota
        threshold = float(
            (cfg.get("quota") or {}).get(
                "minimax_skip_threshold_pct", DEFAULT_QUOTA_THRESHOLD_PCT
            )
        )
        cycle.quota_before = self._read_quota()
        if self._quota_exceeds(cycle.quota_before, threshold):
            cycle.status = "skipped_quota"
            cycle.notes.append(f"quota>={threshold}% skip")
            cycle.completed_at = now_iso()
            self._log_cycle(cycle, cfg)
            return

        # Pick task
        task = self._pick_task(cfg)
        if task is None:
            cycle.status = "skipped_no_task"
            cycle.completed_at = now_iso()
            self._log_cycle(cycle, cfg)
            return
        cycle.task = task

        # Pick model
        model_cfg = self._pick_model(task, cfg)
        if model_cfg is None:
            cycle.status = "skipped_decision_blocked"
            cycle.completed_at = now_iso()
            self._post_discord_block(task)
            self._log_cycle(cycle, cfg)
            return
        cycle.model_used = {
            "provider": str(model_cfg.get("provider", "")),
            "model": str(model_cfg.get("model", "")),
        }

        # Verbose pre-delegation Discord post (default ON — see config.verbose)
        if self._verbose(cfg):
            self._post_discord_cycle_starting(task, model_cfg)

        # Run delegation (mutually-exclusive with itself)
        with self._delegation_lock:
            result = self._run_delegation(task, model_cfg)
        cycle.delegation_summary = (result.get("summary") or "")[:500] or None
        cycle.delegation_api_calls = result.get("api_calls")
        cycle.delegation_duration_s = result.get("duration_seconds")
        cycle.delegation_status = result.get("status")
        cycle.delegation_error = result.get("error")

        # Mark in picked-index for dedupe
        self._mark_picked(task, last_cycle_status=cycle.delegation_status or "unknown", cfg=cfg)

        # Quota delta after
        cycle.quota_after = self._read_quota()

        # Verbose post-delegation Discord post (default ON)
        if self._verbose(cfg):
            self._post_discord_cycle_done(cycle)

        # Cycle status & cooldown reset / increment
        if cycle.delegation_status == "completed":
            cycle.status = "completed"
            self._consecutive_errors = 0
            st = load_state()
            if (st.cooldown_level or 0) > 0:
                st.cooldown_level = 0
                save_state(st)
        else:
            cycle.status = "error"
            self._consecutive_errors += 1
            if self._consecutive_errors >= COOLDOWN_ERROR_THRESHOLD:
                self._raise_cooldown()

        cycle.completed_at = now_iso()
        self._log_cycle(cycle, cfg)

    # ───────────────────────── config helpers ─────────────────────────

    def _load_cfg(self) -> dict:
        try:
            from hermes_cli.config import load_config
            full = load_config() or {}
        except Exception:
            logger.debug("afk-worker: load_config failed, treating as disabled")
            return {"enabled": False}
        return (full.get("afk_worker") or {})

    def _interval_seconds(self, cfg: dict) -> int:
        try:
            mins = int(cfg.get("interval_minutes", DEFAULT_INTERVAL_MIN))
        except (TypeError, ValueError):
            mins = DEFAULT_INTERVAL_MIN
        return max(60, mins * 60)

    def _build_tick_id(self) -> str:
        return datetime.now(timezone.utc).strftime("afk-%Y-%m-%d-%H%M")

    # ───────────────────────── quota ──────────────────────────────────

    def _read_quota(self) -> dict:
        """Inspect MiniMax Token Plan via the existing tool. Returns a
        small dict with the two windows we care about. Unknown values are
        None (treated as 'don't block')."""
        try:
            from tools.minimax_quota import get_minimax_quota
            payload = json.loads(get_minimax_quota())
        except Exception as e:
            logger.warning("afk-worker: quota read failed: %s", e)
            return {"minimax_5h_pct": None, "minimax_weekly_pct": None}
        for row in payload.get("models") or []:
            if row.get("model") == "MiniMax-M*":
                return {
                    "minimax_5h_pct": row.get("interval_used_pct"),
                    "minimax_weekly_pct": row.get("weekly_used_pct"),
                }
        return {"minimax_5h_pct": None, "minimax_weekly_pct": None}

    @staticmethod
    def _quota_exceeds(quota: dict, threshold_pct: float) -> bool:
        for v in (quota.get("minimax_5h_pct"), quota.get("minimax_weekly_pct")):
            if v is None:
                continue
            try:
                if float(v) >= threshold_pct:
                    return True
            except (TypeError, ValueError):
                continue
        return False

    # ───────────────────────── pick task ──────────────────────────────

    def _pick_task(self, cfg: dict) -> Optional[TaskCandidate]:
        files = cfg.get("status_files") or []
        candidates: list[TaskCandidate] = []
        for raw_path in files:
            path = Path(os.path.expanduser(str(raw_path)))
            if not path.is_file():
                logger.warning("afk-worker: status file not found: %s (OQ5 skip)", path)
                continue
            try:
                candidates.extend(self._scan_status_file(path))
            except OSError as e:
                logger.warning("afk-worker: cannot read %s: %s", path, e)
                continue

        # Filter excluded + already-completed-in-picked-index
        picked = self._load_picked_index(cfg)
        eligible = []
        for c in candidates:
            if c.is_excluded:
                continue
            entry = picked.get(c.task_key())
            if entry is None:
                eligible.append(c)
                continue
            last_status = entry.get("last_cycle_status")
            if last_status == "completed":
                continue  # already done — wait for status file edit
            if last_status == "error":
                # Cooldown N ticks before retry
                n_cycles = entry.get("n_cycles", 0)
                err_skips = entry.get("error_skips_remaining", 0)
                if err_skips > 0:
                    # Decrement (we observed it again this tick — not picking)
                    entry["error_skips_remaining"] = err_skips - 1
                    self._save_picked_index(picked, cfg)
                    continue
                # Cooldown expired → eligible again
                eligible.append(c)
                continue
            # Any other status → eligible
            eligible.append(c)

        if not eligible:
            return None

        # Sort: P0 < P1 < P2; tie-break by source-file mtime ascending
        def _key(c: TaskCandidate):
            try:
                mtime = os.path.getmtime(c.source_file)
            except OSError:
                mtime = 0.0
            return (_PRIORITY_RANK.get(c.priority, 9), mtime)

        eligible.sort(key=_key)
        return eligible[0]

    def _scan_status_file(self, path: Path) -> list[TaskCandidate]:
        out: list[TaskCandidate] = []
        current_section = "(top)"
        with path.open(encoding="utf-8") as f:
            for lineno, raw in enumerate(f, start=1):
                line = raw.rstrip("\n")
                m_h = RE_HEADING.match(line)
                if m_h:
                    current_section = m_h.group(2).strip()
                    continue
                m_t = RE_TASK_LINE.match(line)
                if not m_t:
                    continue
                marker = m_t.group(1)
                title_full = m_t.group(2)

                # Parse tags
                m_pri = RE_PRIORITY.search(title_full)
                priority = m_pri.group(1).upper() if m_pri else "P2"

                m_type = RE_AFK_TYPE.search(title_full)
                afk_type_raw = m_type.group(1).lower() if m_type else "research"

                excluded = False
                reason = None
                if RE_AFK_NO.search(title_full):
                    excluded = True
                    reason = "[afk:no] tag"
                elif RE_BLOCKED.search(title_full):
                    excluded = True
                    reason = "[blocked:*] tag"

                # Strip all known tags from title for cleanliness
                title_clean = RE_TAG_ANY.sub("", title_full).strip()
                # Strip trailing markdown emphasis tokens left over
                title_clean = re.sub(r"\s+", " ", title_clean).strip()

                out.append(
                    TaskCandidate(
                        source_file=str(path),
                        section=current_section,
                        line_number=lineno,
                        raw_line=line,
                        title=title_clean or title_full.strip(),
                        priority=priority,
                        afk_type=afk_type_raw,
                        is_excluded=excluded,
                        exclusion_reason=reason,
                    )
                )
        return out

    # ───────────────────────── pick model ─────────────────────────────

    def _pick_model(self, task: TaskCandidate, cfg: dict) -> Optional[dict]:
        """Lookup task.afk_type in matrix. Returns None for unmapped types
        (caller posts Discord block). decision_structurelle is the explicit
        sentinel — never mapped."""
        if task.afk_type == "decision_structurelle":
            return None
        routing = cfg.get("model_routing") or {}
        return routing.get(task.afk_type)

    # ───────────────────────── delegation ─────────────────────────────

    def _run_delegation(self, task: TaskCandidate, model_cfg: dict) -> dict:
        from tools.delegate_tool import delegate_task

        goal = (
            f"AFK autonomous tick — work on the following task pulled from "
            f"AH's backlog:\n\n"
            f"  Source: {task.source_file} ({task.section})\n"
            f"  Title:  {task.title}\n"
            f"  AFK type: {task.afk_type} | Priority: {task.priority}\n\n"
            f"Constraints:\n"
            f"- §9.1 of HERMES.md: do NOT modify hermes-agent code, configs, "
            f"or skills. Read-only on those paths.\n"
            f"- Publish actions blocked (already enforced at tool layer).\n"
            f"- Output a structured summary of what you did, what's left, "
            f"and any blockers.\n"
        )

        parent_agent = _get_runtime_parent_agent()
        if parent_agent is None:
            return {
                "status": "error",
                "summary": None,
                "error": "no parent_agent available at runtime",
                "api_calls": 0,
                "duration_seconds": 0.0,
            }

        # Resolve api_key from env (we do NOT load the value into the log).
        api_key_env_name = model_cfg.get("api_key_env") or ""
        api_key = os.environ.get(api_key_env_name) if api_key_env_name else None
        api_key = api_key or None  # empty string → None

        # OQ3 — toolsets from matrix; fallback to safe minimal
        toolsets = model_cfg.get("toolsets") or ["file", "todo"]

        # WS-AUTO-002 Bug 1 debug — trace what we're about to pass to
        # delegate_task. Logs key length only, never the value (§1).
        logger.info(
            "afk-worker delegation overrides: provider=%s model=%s "
            "base_url=%s api_key_env=%r api_key_len=%d toolsets=%s",
            model_cfg.get("provider"),
            model_cfg.get("model"),
            model_cfg.get("base_url"),
            api_key_env_name,
            len(api_key) if api_key else 0,
            toolsets,
        )

        try:
            result_str = delegate_task(
                goal=goal,
                context=None,
                toolsets=list(toolsets),
                role="leaf",
                parent_agent=parent_agent,
                model_override=model_cfg.get("model") or None,
                provider_override=model_cfg.get("provider") or None,
                base_url_override=model_cfg.get("base_url") or None,
                api_key_override=api_key,
            )
        except TypeError as e:
            # Likely the extended signature isn't deployed yet — fall back
            # silently so the worker keeps running on the default route.
            logger.warning(
                "afk-worker: delegate_task does not accept overrides (%s); "
                "falling back to default credentials",
                e,
            )
            result_str = delegate_task(
                goal=goal,
                context=None,
                toolsets=list(toolsets),
                role="leaf",
                parent_agent=parent_agent,
            )
        except Exception as e:
            return {
                "status": "error",
                "summary": None,
                "error": f"delegate_task raised: {e}",
                "api_calls": 0,
                "duration_seconds": 0.0,
            }

        try:
            payload = json.loads(result_str)
        except Exception as e:
            return {
                "status": "error",
                "summary": None,
                "error": f"delegate_task returned non-JSON: {e}",
                "api_calls": 0,
                "duration_seconds": 0.0,
            }

        if isinstance(payload, dict) and payload.get("error"):
            return {
                "status": "error",
                "summary": None,
                "error": str(payload.get("error")),
                "api_calls": 0,
                "duration_seconds": 0.0,
            }

        results = (payload.get("results") if isinstance(payload, dict) else None) or []
        if not results:
            return {
                "status": "error",
                "summary": None,
                "error": "no results array in delegate_task payload",
                "api_calls": 0,
                "duration_seconds": 0.0,
            }
        r = results[0]
        status = r.get("status")
        summary = r.get("summary") or ""
        error = r.get("error")

        # Bug 2 fix — delegate_task returns status=completed even when the
        # child agent's LLM call exhausted retries without ever responding.
        # Detect the pattern in the summary and promote to status=error so
        # the picked-index marks the task for retry instead of "done".
        if status == "completed" and summary.lstrip().lower().startswith(
            "api call failed"
        ):
            logger.warning(
                "afk-worker: child reports completed but summary indicates "
                "API failure; promoting to status=error"
            )
            status = "error"
            error = error or summary.strip().splitlines()[0]

        return {
            "status": status,
            "summary": summary or None,
            "error": error,
            "api_calls": r.get("api_calls", 0),
            "duration_seconds": r.get("duration_seconds", 0.0),
        }

    # ───────────────────────── picked-index dedupe ────────────────────

    def _picked_path(self, cfg: dict) -> Path:
        return Path(
            os.path.expanduser(cfg.get("picked_index_path") or DEFAULT_PICKED_INDEX)
        )

    def _load_picked_index(self, cfg: dict) -> dict:
        p = self._picked_path(cfg)
        if not p.is_file():
            return {}
        try:
            with p.open(encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("afk-worker: picked-index unreadable (%s) — resetting", e)
        return {}

    def _save_picked_index(self, idx: dict, cfg: dict) -> None:
        p = self._picked_path(cfg)
        p.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write via tempfile + os.replace
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(p.parent),
            prefix=p.name + ".tmp-",
            delete=False,
        ) as tmp:
            json.dump(idx, tmp, indent=2, sort_keys=True)
            tmp_path = Path(tmp.name)
        os.replace(tmp_path, p)

    def _mark_picked(self, task: TaskCandidate, last_cycle_status: str, cfg: dict) -> None:
        idx = self._load_picked_index(cfg)
        key = task.task_key()
        entry = idx.get(key) or {}
        now = now_iso()
        entry.setdefault("first_picked_at", now)
        entry["last_picked_at"] = now
        entry["n_cycles"] = int(entry.get("n_cycles", 0)) + 1
        entry["last_cycle_status"] = last_cycle_status
        if last_cycle_status == "error":
            entry["error_skips_remaining"] = ERROR_COOLDOWN_TICKS
        else:
            entry["error_skips_remaining"] = 0
        idx[key] = entry
        self._save_picked_index(idx, cfg)

    # ───────────────────────── cooldown ───────────────────────────────

    def _raise_cooldown(self) -> None:
        st = load_state()
        st.cooldown_level = min(3, (st.cooldown_level or 0) + 1)
        save_state(st)
        skips = COOLDOWN_LEVEL_TICK_SKIPS.get(st.cooldown_level, 0)
        if skips == float("inf"):
            self._cooldown_skip_remaining = 10**9
            self._post_discord_cooldown_halt(st.cooldown_level)
        else:
            self._cooldown_skip_remaining = int(skips)
            if st.cooldown_level >= 2:
                self._post_discord_cooldown_warning(st.cooldown_level, int(skips))
        # Cooldown absorbs the consecutive-error counter
        self._consecutive_errors = 0

    # ───────────────────────── Discord posts ──────────────────────────

    def _select_discord_adapter(self):
        if not self.adapters:
            return None
        _iter = self.adapters.values() if isinstance(self.adapters, dict) else self.adapters
        return next(
            (a for a in _iter if getattr(a, "name", "").lower() == "discord"),
            None,
        )

    def _post_discord(self, text: str) -> None:
        """Send a message to the home channel from this background thread.

        Mirrors gateway/run.py:_post_afk_notif_to_discord pattern."""
        if self.loop is None:
            logger.debug("afk-worker: no asyncio loop, dropping Discord msg")
            return
        adapter = self._select_discord_adapter()
        if adapter is None:
            logger.debug("afk-worker: no Discord adapter")
            return
        channel_id = os.environ.get("DISCORD_HERMES_CHANNEL_ID")
        if not channel_id:
            logger.debug("afk-worker: DISCORD_HERMES_CHANNEL_ID empty")
            return
        try:
            import asyncio
            fut = asyncio.run_coroutine_threadsafe(
                adapter.send(channel_id, text), self.loop
            )
            fut.result(timeout=15)
        except Exception:
            logger.exception("afk-worker: Discord post failed")

    @staticmethod
    def _verbose(cfg: dict) -> bool:
        """Return True if afk_worker.verbose is enabled (default True).

        When True, the worker posts 2 Discord messages per delegated cycle
        (pre-delegation + post-delegation) on top of the existing
        block/cooldown notifications. Set false for silent operation.
        """
        v = cfg.get("verbose")
        if v is None:
            return True
        return bool(v)

    def _post_discord_cycle_starting(self, task: TaskCandidate, model_cfg: dict) -> None:
        toolsets = model_cfg.get("toolsets") or ["file", "todo"]
        provider = model_cfg.get("provider", "?")
        model = model_cfg.get("model", "?")
        # Truncate long titles so Discord doesn't fail on > 2000 chars
        title = task.title if len(task.title) <= 200 else (task.title[:197] + "...")
        self._post_discord(
            f"🔄 **AFK cycle starting**\n"
            f"• Tâche : `{title}`\n"
            f"• Source : `{task.source_file}` ({task.section}, l.{task.line_number})\n"
            f"• Type AFK : `{task.afk_type}` | Priority : `{task.priority}`\n"
            f"• Modèle : `{provider} / {model}`\n"
            f"• Toolsets : {', '.join(f'`{t}`' for t in toolsets)}\n"
            f"⏱ Délégation lancée — `delegation.child_timeout_seconds` = 600s max."
        )

    def _post_discord_cycle_done(self, cycle: CycleResult) -> None:
        status = cycle.delegation_status or "unknown"
        emoji = "✅" if status == "completed" else ("❌" if status == "error" else "⚠️")
        title = (cycle.task.title if cycle.task else "?")
        if len(title) > 200:
            title = title[:197] + "..."
        # Build a result block — summary truncated to ~500 chars (already
        # capped at cycle build time, but defensive here too).
        summary = cycle.delegation_summary or "(no summary)"
        if len(summary) > 500:
            summary = summary[:497] + "..."
        duration = cycle.delegation_duration_s
        duration_str = f"{duration:.1f}s" if isinstance(duration, (int, float)) else "?"
        api_calls = cycle.delegation_api_calls if cycle.delegation_api_calls is not None else "?"

        msg = (
            f"{emoji} **AFK cycle done — `{status}`**\n"
            f"• Tâche : `{title}`\n"
            f"• API calls : {api_calls} | Duration : {duration_str}\n"
        )
        if cycle.delegation_error:
            msg += f"• Error : `{cycle.delegation_error}`\n"
        if status == "completed" and summary != "(no summary)":
            msg += f"\n**Résumé sub-agent :**\n> {summary.replace(chr(10), chr(10) + '> ')}\n"
        # Quota delta info
        qa = cycle.quota_after or {}
        if qa.get("minimax_5h_pct") is not None or qa.get("minimax_weekly_pct") is not None:
            msg += (
                f"\n📊 Quota MiniMax : "
                f"{qa.get('minimax_5h_pct', '?')}% (5h) / "
                f"{qa.get('minimax_weekly_pct', '?')}% (semaine)"
            )
        msg += f"\n_(détails : `~/AFK_LOG.md`)_"
        self._post_discord(msg)

    def _post_discord_block(self, task: TaskCandidate) -> None:
        self._post_discord(
            f"🚫 **AFK skip — décision structurelle**\n"
            f"Tâche : `{task.title}`\n"
            f"Source : `{task.source_file}` ({task.section})\n"
            f"Type AFK : `{task.afk_type}` (non mappé dans la matrice).\n"
            f"§9.1 HERMES.md — j'attends ton arbitrage avant d'avancer."
        )

    def _post_discord_cooldown_warning(self, level: int, skip_ticks: int) -> None:
        self._post_discord(
            f"⚠️ **AFK worker cooldown level {level}**\n"
            f"3 erreurs consécutives sur les délégations. Je skip les "
            f"{skip_ticks} prochains ticks et je réessaie ensuite.\n"
            f"Détails : `~/AFK_LOG.md`."
        )

    def _post_discord_cooldown_halt(self, level: int) -> None:
        self._post_discord(
            f"🛑 **AFK worker halt — cooldown level {level}**\n"
            f"Trop d'erreurs consécutives. Le worker est arrêté jusqu'au "
            f"prochain redémarrage du service ou retour en mode normal.\n"
            f"Détails : `~/AFK_LOG.md`."
        )

    # ───────────────────────── logging (md + jsonl) ───────────────────

    def _log_cycle(self, cycle: CycleResult, cfg: dict) -> None:
        md_path = Path(os.path.expanduser(cfg.get("log_path") or DEFAULT_LOG_MD))
        jsonl_path = Path(
            os.path.expanduser(cfg.get("log_jsonl_path") or DEFAULT_LOG_JSONL)
        )
        md_path.parent.mkdir(parents=True, exist_ok=True)
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._append_md(md_path, cycle)
        except OSError as e:
            logger.warning("afk-worker: cannot append %s: %s", md_path, e)
        try:
            self._append_jsonl(jsonl_path, cycle)
        except OSError as e:
            logger.warning("afk-worker: cannot append %s: %s", jsonl_path, e)

    @staticmethod
    def _append_md(path: Path, cycle: CycleResult) -> None:
        block = _render_md_block(cycle)
        with path.open("a", encoding="utf-8") as f:
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                f.write(block)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    @staticmethod
    def _append_jsonl(path: Path, cycle: CycleResult) -> None:
        line = json.dumps(_render_jsonl_obj(cycle), ensure_ascii=False)
        with path.open("a", encoding="utf-8") as f:
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                f.write(line + "\n")
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)


# ---------------------------------------------------------------------------
# Renderers (separate from the class so they're trivially testable)
# ---------------------------------------------------------------------------


def _render_jsonl_obj(cycle: CycleResult) -> dict[str, Any]:
    obj: dict[str, Any] = {
        "tick_id": cycle.tick_id,
        "started_at": cycle.started_at,
        "completed_at": cycle.completed_at,
        "afk_mode": cycle.afk_mode,
        "status": cycle.status,
        "task": cycle.task.to_jsonable() if cycle.task else None,
        "model_used": cycle.model_used,
        "delegation": (
            {
                "summary": cycle.delegation_summary,
                "api_calls": cycle.delegation_api_calls,
                "duration_seconds": cycle.delegation_duration_s,
                "status": cycle.delegation_status,
                "error": cycle.delegation_error,
            }
            if cycle.delegation_status is not None
            else None
        ),
        "quota_before": cycle.quota_before,
        "quota_after": cycle.quota_after,
        "notes": list(cycle.notes),
    }
    return obj


def _render_md_block(cycle: CycleResult) -> str:
    lines: list[str] = []
    lines.append(
        f"## {cycle.started_at} — {cycle.tick_id} — {cycle.afk_mode or '?'}"
    )
    lines.append("")
    lines.append(f"**Status** : {cycle.status}")
    if cycle.task:
        lines.append(
            f"**Tâche pickée** : `{cycle.task.source_file}` "
            f"({cycle.task.section}) — {cycle.task.title}"
        )
        lines.append(
            f"**Type AFK** : {cycle.task.afk_type} | "
            f"**Priority** : {cycle.task.priority}"
        )
    if cycle.model_used:
        lines.append(
            f"**Modèle utilisé** : "
            f"{cycle.model_used.get('provider', '?')} / "
            f"{cycle.model_used.get('model', '?')}"
        )
    if cycle.delegation_status is not None:
        lines.append("")
        lines.append("**Délégation** :")
        lines.append(f"- subagent status : {cycle.delegation_status}")
        lines.append(f"- API calls : {cycle.delegation_api_calls}")
        if cycle.delegation_duration_s is not None:
            lines.append(f"- duration : {cycle.delegation_duration_s:.1f}s")
        if cycle.delegation_error:
            lines.append(f"- error : `{cycle.delegation_error}`")
    if cycle.delegation_summary:
        lines.append("")
        lines.append("**Résumé du sub-agent** (≤ 500 chars) :")
        lines.append("> " + cycle.delegation_summary.replace("\n", "\n> "))
    if cycle.quota_before or cycle.quota_after:
        lines.append("")
        lines.append("**Quota MiniMax** :")
        if cycle.quota_before:
            lines.append(
                f"- avant : "
                f"{cycle.quota_before.get('minimax_5h_pct')}% (5h) / "
                f"{cycle.quota_before.get('minimax_weekly_pct')}% (semaine)"
            )
        if cycle.quota_after:
            lines.append(
                f"- après : "
                f"{cycle.quota_after.get('minimax_5h_pct')}% (5h) / "
                f"{cycle.quota_after.get('minimax_weekly_pct')}% (semaine)"
            )
    if cycle.notes:
        lines.append("")
        lines.append(f"**Notes** : {' | '.join(cycle.notes)}")
    lines.append("")
    lines.append("---")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers usable by other modules (afk_scheduler purge hook — OQ4)
# ---------------------------------------------------------------------------


def purge_picked_index(cfg: Optional[dict] = None) -> bool:
    """Delete the picked-index file if present. Called by afk_scheduler
    when AH transitions out of AFK to normal mode (OQ4)."""
    if cfg is None:
        try:
            from hermes_cli.config import load_config
            cfg = (load_config() or {}).get("afk_worker") or {}
        except Exception:
            cfg = {}
    p = Path(os.path.expanduser(cfg.get("picked_index_path") or DEFAULT_PICKED_INDEX))
    try:
        if p.is_file():
            p.unlink()
            logger.info("afk-worker: picked-index purged (mode flip → normal)")
            return True
    except OSError as e:
        logger.warning("afk-worker: could not purge picked-index: %s", e)
    return False
