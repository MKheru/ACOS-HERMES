"""MCP Reputation Registry — SMCP Run #5 baseline (v0).

This is a behavioural defence layer that complements the provenance gate
(Run #4) and the regex sanitiser (Patches 1+7+8). The insight: even
identical content from two different MCP servers carries different risk.
Source identity matters. Track it.

Each MCP has a stable UUID, a declared capability manifest, and a
reputation score that evolves with observed incidents. Decisions to
allow or block tool calls / sampling are modulated by the score,
producing band-based behaviour (BANNED / HIGH_SUSPICION / PROBATION /
TRUSTED / VETERAN).

This module is consumed by tools/mcp_tool.py at three points:
  1. registration on stdio connect
  2. pre-call gate before invoking a tool
  3. SamplingHandler.__call__ to refuse low-reputation servers

For the lab harness it is exercised standalone via run_reputation_lab.py
against mcp_reputation_corpus.jsonl. The corpus replays incident history
on a fresh registry then asks for a verdict on a proposed action.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from enum import IntEnum
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Severity ladder + band thresholds
# ---------------------------------------------------------------------------

class IncidentSeverity(IntEnum):
    LOW = 1        # weak signal: anomalous formatting, single regex match
    MEDIUM = 5     # confirmed injection-like content
    HIGH = 15      # capability violation, attempted sampling without auth
    CRITICAL = 30  # exfil attempt, sensitive-path write, multi-incident cluster


# Score impact by severity (negative — these are penalties).
_SEVERITY_PENALTY: dict[IncidentSeverity, float] = {
    IncidentSeverity.LOW: -1.0,
    IncidentSeverity.MEDIUM: -5.0,
    IncidentSeverity.HIGH: -15.0,
    IncidentSeverity.CRITICAL: -30.0,
}

# Quarantine durations by severity (seconds).
_QUARANTINE_SECS: dict[IncidentSeverity, float] = {
    IncidentSeverity.LOW: 0.0,
    IncidentSeverity.MEDIUM: 0.0,
    IncidentSeverity.HIGH: 60 * 60.0,           # 1h
    IncidentSeverity.CRITICAL: 24 * 60 * 60.0,  # 24h
}

# Cluster detection configuration
_CLUSTER_THRESHOLD_COUNT: int = 3
_CLUSTER_WINDOW_SECONDS: float = 60 * 60.0  # 1 hour
_CLUSTER_QUARANTINE_SECONDS: float = 60 * 60.0  # 1 hour partial quarantine

# Score bands (open intervals, lower-inclusive).
class ReputationBand(IntEnum):
    BANNED = 0           # 0..30
    HIGH_SUSPICION = 1   # 30..60
    PROBATION = 2        # 60..80
    TRUSTED = 3          # 80..95
    VETERAN = 4          # 95..100


def _band_for(score: float) -> ReputationBand:
    if score < 30:
        return ReputationBand.BANNED
    if score < 60:
        return ReputationBand.HIGH_SUSPICION
    if score < 80:
        return ReputationBand.PROBATION
    if score < 95:
        return ReputationBand.TRUSTED
    return ReputationBand.VETERAN


# Default initial score for a new MCP — entered on first registration.
INITIAL_SCORE: float = 70.0
SCORE_FLOOR: float = 0.0
SCORE_CEIL: float = 100.0
DAILY_RECOVERY: float = 0.1   # +0.1 per day with zero incidents over 7d window
RECOVERY_WINDOW_DAYS: int = 7
SCHEMA_VERSION: int = 1


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MCPIdentity:
    """Stable identity of an MCP server. UUID is canonical; the rest is
    metadata captured at registration."""
    uuid: str
    name: str
    install_origin: str = ""
    declared_capabilities: frozenset[str] = frozenset()

    def to_json(self) -> dict:
        return {
            "uuid": self.uuid,
            "name": self.name,
            "install_origin": self.install_origin,
            "declared_capabilities": sorted(self.declared_capabilities),
        }

    @classmethod
    def from_json(cls, d: dict) -> "MCPIdentity":
        return cls(
            uuid=d["uuid"],
            name=d.get("name", ""),
            install_origin=d.get("install_origin", ""),
            declared_capabilities=frozenset(d.get("declared_capabilities", [])),
        )


@dataclass
class IncidentRecord:
    timestamp: float
    severity: IncidentSeverity
    category: str        # "injection_pattern", "capability_violation", ...
    detector: str        # "regex_sanitiser", "provenance_gate", ...
    sample: str          # short redacted excerpt for human review

    def to_json(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "severity": int(self.severity),
            "category": self.category,
            "detector": self.detector,
            "sample": self.sample,
        }

    @classmethod
    def from_json(cls, d: dict) -> "IncidentRecord":
        return cls(
            timestamp=float(d["timestamp"]),
            severity=IncidentSeverity(int(d["severity"])),
            category=d.get("category", "unknown"),
            detector=d.get("detector", "unknown"),
            sample=d.get("sample", ""),
        )


@dataclass
class ReputationState:
    identity: MCPIdentity
    score: float
    incidents: list[IncidentRecord] = field(default_factory=list)
    first_seen: float = 0.0
    last_updated: float = 0.0
    quarantined_until: float = 0.0    # epoch; 0 = no quarantine

    def band(self) -> ReputationBand:
        return _band_for(self.score)

    def in_quarantine(self, now: float) -> bool:
        return now < self.quarantined_until

    def to_json(self) -> dict:
        return {
            "identity": self.identity.to_json(),
            "score": self.score,
            "incidents": [i.to_json() for i in self.incidents],
            "first_seen": self.first_seen,
            "last_updated": self.last_updated,
            "quarantined_until": self.quarantined_until,
        }

    @classmethod
    def from_json(cls, d: dict) -> "ReputationState":
        return cls(
            identity=MCPIdentity.from_json(d["identity"]),
            score=float(d["score"]),
            incidents=[IncidentRecord.from_json(x) for x in d.get("incidents", [])],
            first_seen=float(d.get("first_seen", 0.0)),
            last_updated=float(d.get("last_updated", 0.0)),
            quarantined_until=float(d.get("quarantined_until", 0.0)),
        )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class MCPReputationRegistry:
    """In-memory registry of MCP reputation states. Persistence is
    handled by to_json / from_json — the host is responsible for atomic
    writes to disk (tools/mcp_tool.py uses tempfile+rename in the v1
    integration).
    """

    def __init__(self, now_fn=time.time) -> None:
        self._now = now_fn
        self._states: dict[str, ReputationState] = {}

    # -- Registration ------------------------------------------------------

    def register(self, identity: MCPIdentity) -> ReputationState:
        """Create or refresh the state for an MCP. New MCP starts at
        INITIAL_SCORE on probation."""
        now = self._now()
        existing = self._states.get(identity.uuid)
        if existing is None:
            state = ReputationState(
                identity=identity,
                score=INITIAL_SCORE,
                incidents=[],
                first_seen=now,
                last_updated=now,
                quarantined_until=0.0,
            )
            self._states[identity.uuid] = state
            return state
        # Refresh metadata (capabilities may have widened on upgrade).
        existing.identity = identity
        existing.last_updated = now
        return existing

    # -- Incident handling -------------------------------------------------

    def record_incident(
        self,
        uuid: str,
        severity: IncidentSeverity,
        category: str,
        detector: str,
        sample: str = "",
    ) -> ReputationState:
        """Apply an incident: append record, decrement score, possibly
        impose quarantine. Triggers cluster detection logic."""
        now = self._now()
        state = self._states.get(uuid)
        if state is None:
            # Unknown MCP — register a placeholder identity so the
            # incident isn't lost. Caller should have registered first.
            stub = MCPIdentity(uuid=uuid, name=f"unknown:{uuid[:8]}")
            state = self.register(stub)

        # 1. Append the new incident first.
        record = IncidentRecord(
            timestamp=now,
            severity=severity,
            category=category,
            detector=detector,
            sample=sample[:200],   # cap retained sample for size
        )
        state.incidents.append(record)
        # Cap retained incident history.
        if len(state.incidents) > 200:
            state.incidents = state.incidents[-200:]

        # 2. Apply standard penalty.
        delta = _SEVERITY_PENALTY[severity]
        state.score = max(SCORE_FLOOR, min(SCORE_CEIL, state.score + delta))

        # 3. Apply standard quarantine if applicable.
        q = _QUARANTINE_SECS[severity]
        if q > 0:
            new_q_until = now + q
            if new_q_until > state.quarantined_until:
                state.quarantined_until = new_q_until

        # 4. Cluster detection: check for N+ MEDIUM/HIGH incidents within window.
        # We only check if the current incident is MEDIUM or higher to avoid noise.
        if severity >= IncidentSeverity.MEDIUM:
            window_start = now - _CLUSTER_WINDOW_SECONDS
            # Count incidents of MEDIUM or higher severity within the window.
            # Note: this includes the incident we just added.
            count = sum(
                1
                for i in state.incidents
                if i.timestamp >= window_start and i.severity >= IncidentSeverity.MEDIUM
            )

            if count >= _CLUSTER_THRESHOLD_COUNT:
                # Threshold met. Synthesize a HIGH incident.
                # We do this by adding a synthetic record and applying its penalty/quarantine.
                # To avoid infinite recursion or double counting, we manually apply the
                # HIGH logic here without calling record_incident again.
                synthetic_record = IncidentRecord(
                    timestamp=now,
                    severity=IncidentSeverity.HIGH,
                    category="cluster_escalation",
                    detector="reputation_registry",
                    sample=f"Cluster escalation: {count} incidents in {_CLUSTER_WINDOW_SECONDS/60:.0f}m",
                )
                state.incidents.append(synthetic_record)
                
                # Apply HIGH penalty
                high_delta = _SEVERITY_PENALTY[IncidentSeverity.HIGH]
                state.score = max(SCORE_FLOOR, min(SCORE_CEIL, state.score + high_delta))

                # Apply Cluster-specific quarantine (1h partial)
                # If existing quarantine is longer, preserve it.
                cluster_q_until = now + _CLUSTER_QUARANTINE_SECONDS
                if cluster_q_until > state.quarantined_until:
                    state.quarantined_until = cluster_q_until

        state.last_updated = now
        return state

    # -- Recovery ----------------------------------------------------------

    def apply_recovery(self, uuid: str) -> ReputationState:
        """Apply daily recovery if no incidents in the recovery window."""
        now = self._now()
        state = self._states.get(uuid)
        if state is None:
            raise KeyError(uuid)

        cutoff = now - RECOVERY_WINDOW_DAYS * 86400.0
        recent_incidents = [i for i in state.incidents if i.timestamp >= cutoff]
        if recent_incidents:
            return state

        days_since_update = max(0.0, (now - state.last_updated) / 86400.0)
        if days_since_update <= 0:
            return state
        gain = DAILY_RECOVERY * days_since_update
        state.score = min(SCORE_CEIL, state.score + gain)
        state.last_updated = now
        return state

    # -- Decision API ------------------------------------------------------

    def evaluate_action(
        self,
        uuid: str,
        action: str,
        context: Optional[dict] = None,
    ) -> tuple[bool, str, dict]:
        """Decide whether to allow an action by an MCP.

        action ∈ {"tool_call", "sampling", "structured_response"}
        context may include:
          - "requested_capability": str, the capability the action needs
          - "user_authorised": bool, fed by the provenance gate
          - "depth": int, tool-chain depth so far

        Returns (allow, reason, metadata).
        metadata always contains: score, band, quarantined.
        """
        ctx = context or {}
        now = self._now()
        state = self._states.get(uuid)
        if state is None:
            return (
                False,
                f"unknown MCP uuid {uuid!r} — register before evaluating",
                {"score": None, "band": None, "quarantined": False},
            )

        meta = {
            "score": state.score,
            "band": state.band().name,
            "quarantined": state.in_quarantine(now),
            "incident_count": len(state.incidents),
        }

        if state.in_quarantine(now):
            return (
                False,
                f"MCP {state.identity.name} in quarantine until "
                f"{state.quarantined_until:.0f}",
                meta,
            )

        band = state.band()

        if band == ReputationBand.BANNED:
            return (False, "MCP banned by reputation (score < 30)", meta)

        if band == ReputationBand.HIGH_SUSPICION:
            if action == "sampling":
                return (False, "sampling refused: MCP in HIGH_SUSPICION", meta)
            if action == "tool_call":
                cap = ctx.get("requested_capability", "")
                if cap and cap not in state.identity.declared_capabilities:
                    return (
                        False,
                        f"capability {cap!r} not declared by MCP "
                        f"{state.identity.name}",
                        meta,
                    )
                # Read-only-ish whitelist: limit to declared caps that
                # don't include write/exec.
                if any(w in cap.lower() for w in ("write", "delete", "exec",
                                                   "post", "send")):
                    return (
                        False,
                        f"action {cap!r} blocked: HIGH_SUSPICION allows "
                        "read-only only",
                        meta,
                    )
                return (True, "tool_call allowed (HIGH_SUSPICION read-only)", meta)
            return (True, "structured_response allowed", meta)

        if band == ReputationBand.PROBATION:
            if action == "sampling" and not ctx.get("user_authorised", False):
                return (
                    False,
                    "sampling requires user authorisation in PROBATION",
                    meta,
                )
            cap = ctx.get("requested_capability", "")
            if cap and state.identity.declared_capabilities and \
                    cap not in state.identity.declared_capabilities:
                return (
                    False,
                    f"capability {cap!r} outside declared manifest",
                    meta,
                )
            return (True, "allowed (PROBATION default)", meta)

        # TRUSTED + VETERAN: base allow, only check capability manifest.
        cap = ctx.get("requested_capability", "")
        if cap and state.identity.declared_capabilities and \
                cap not in state.identity.declared_capabilities:
            return (
                False,
                f"capability {cap!r} outside declared manifest "
                f"(even for trusted MCP)",
                meta,
            )
        return (True, f"allowed ({band.name})", meta)

    # -- Inspection / persistence -----------------------------------------

    def get_state(self, uuid: str) -> Optional[ReputationState]:
        return self._states.get(uuid)

    def to_json(self) -> str:
        payload = {
            "_schema": SCHEMA_VERSION,
            "states": {u: s.to_json() for u, s in self._states.items()},
        }
        return json.dumps(payload, indent=2, ensure_ascii=False)

    @classmethod
    def from_json(cls, s: str, now_fn=time.time) -> "MCPReputationRegistry":
        try:
            payload = json.loads(s)
        except json.JSONDecodeError:
            # Fail-closed: return empty registry rather than propagate
            # corruption upstream.
            return cls(now_fn=now_fn)
        reg = cls(now_fn=now_fn)
        for u, raw in payload.get("states", {}).items():
            try:
                reg._states[u] = ReputationState.from_json(raw)
            except (KeyError, ValueError, TypeError):
                continue
        return reg


# ---------------------------------------------------------------------------
# Process-level singleton + atomic disk persistence
# ---------------------------------------------------------------------------
# tools/mcp_tool.py imports _get_reputation_registry() at sampling time
# to gate calls and at tool-result time to record incidents. The registry
# state is persisted at HERMES_HOME/mcp_reputation.json with a
# tempfile+rename pattern for atomicity.

import os
from pathlib import Path
import logging

_log = logging.getLogger(__name__)
_registry_singleton: Optional[MCPReputationRegistry] = None


def _reputation_path() -> Path:
    home = os.environ.get("HERMES_HOME", os.path.expanduser("~/.hermes"))
    return Path(home) / "mcp_reputation.json"


def _get_reputation_registry() -> MCPReputationRegistry:
    """Return the process-level registry, loading from disk on first call."""
    global _registry_singleton
    if _registry_singleton is not None:
        return _registry_singleton
    path = _reputation_path()
    if path.exists():
        try:
            _registry_singleton = MCPReputationRegistry.from_json(
                path.read_text(encoding="utf-8")
            )
        except Exception as exc:
            _log.warning(
                "Failed to load reputation registry from %s: %s — "
                "starting with fresh state.", path, exc,
            )
            _registry_singleton = MCPReputationRegistry()
    else:
        _registry_singleton = MCPReputationRegistry()
    return _registry_singleton


def _save_reputation_registry() -> None:
    """Atomic-write the registry to disk. Best-effort — log + continue
    on failure rather than crash the agent."""
    if _registry_singleton is None:
        return
    path = _reputation_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(_registry_singleton.to_json(), encoding="utf-8")
        os.replace(str(tmp), str(path))
    except Exception as exc:
        _log.warning("Failed to save reputation registry to %s: %s",
                     path, exc)
