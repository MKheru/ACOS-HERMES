"""AFK scheduler — Patch 13.3c.

Drives AFK state transitions:

  process_user_message(content, now_utc) -> (state, confirm_msg_or_None)
      Called from the Discord adapter on every user message. Detects AFK
      triggers, revoke tokens, or auto-AFK exit (any user msg during
      afk_auto). Updates last_user_msg_at. Returns a Discord-postable
      confirmation when the mode flipped.

  check_auto_transitions(now_utc) -> (state, notification_or_None)
      Called from the gateway cron ticker every 60s. Flips state to
      afk_auto if 20:00 GMT-3 (= 23:00 UTC) reached AND the user has been
      idle for ≥30 min AND the current mode is normal.

Both functions persist the resulting state to ~/.hermes/afk_state.json
via agent.afk_state.save_state.
"""

from __future__ import annotations

import logging
from datetime import datetime, time, timedelta, timezone
from typing import Optional

from agent.afk_state import (
    AFKState,
    MODE_AFK_AUTO,
    MODE_AFK_MANUAL,
    MODE_NORMAL,
    is_afk_from_messages,
    load_state,
    now_iso,
    save_state,
)

logger = logging.getLogger(__name__)

# Auto-AFK trigger window: 20:00 GMT-3 = 23:00 UTC.
# Configurable via afk_auto_trigger_hour_utc env-style override later.
AFK_AUTO_TRIGGER_HOUR_UTC = 23  # 20:00 GMT-3
IDLE_THRESHOLD_MINUTES = 30


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def process_user_message(content: str, now_utc: Optional[datetime] = None) -> tuple[AFKState, Optional[str]]:
    """Update AFK state on every incoming user message.

    Returns (new_state, confirmation_message_or_None). The caller (Discord
    adapter) should post the confirmation to #acos-hermes if non-None.
    """
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)
    now_str = now_utc.strftime("%Y-%m-%dT%H:%M:%SZ")

    state = load_state()
    old_mode = state.mode
    state.last_user_msg_at = now_str

    confirm: Optional[str] = None
    content_lower = (content or "").lower()

    # Lazy import to avoid circular ref; tokens are the source of truth.
    from agent.provenance import _USER_AFK_TOKENS, _USER_REVOKE_TOKENS

    has_afk_trigger = any(token in content_lower for token in _USER_AFK_TOKENS)
    has_revoke = any(token in content_lower for token in _USER_REVOKE_TOKENS)

    # 1. Revoke takes precedence: if user says "stop" while in AFK, exit AFK.
    if has_revoke and state.is_afk():
        state.mode = MODE_NORMAL
        state.entered_at = None
        state.heartbeats_sent = 0
        state.cooldown_level = 0
        confirm = (
            "🔔 Mode AFK désactivé (revoke détecté). Je reviens en mode normal — "
            "publish externe ré-autorisé, MAX_DEPTH redescend à 8."
        )

    # 2. AFK manual trigger (only if not already in AFK).
    elif has_afk_trigger and not state.is_afk():
        state.mode = MODE_AFK_MANUAL
        state.entered_at = now_str
        confirm = (
            "🌙 **Mode AFK manuel activé.**\n"
            "• MAX_DEPTH levé à 50 → je peux chainer 50 tool calls "
            "sans te redemander confirmation\n"
            "• Publish externe désactivé : git push, gh pr create, "
            "gh issue create, slack/telegram/email, branch protection, "
            "writes sur env.list / HERMES.md / SOUL.md → bloqués\n"
            "• Travail local autorisé : cargo build/test, qemu, commits "
            "locaux, posts sur ce salon\n"
            "• Tu peux désactiver à tout moment avec : `stop`, `arrête`, "
            "`cancel`, ou simplement un nouveau message qui n'a pas de "
            "trigger AFK.\n\n"
            "💤 Bonne nuit Khéri. Je continue selon ta direction."
        )

    # 3. Auto-AFK exit: any user msg flips state back to normal (option C
    # default per Khéri's spec — option A "termine la tâche en cours puis
    # répond" is signalled separately by the agent runtime, this hook
    # only updates persistent state).
    elif state.mode == MODE_AFK_AUTO and not has_afk_trigger and not has_revoke:
        state.mode = MODE_NORMAL
        state.entered_at = None
        state.heartbeats_sent = 0
        state.cooldown_level = 0
        confirm = (
            "👋 Je vois que tu es de retour. Je sors du mode AFK auto et "
            "reprends le mode normal. Je termine la tâche en cours puis "
            "je te réponds — dis-moi `stop` si tu veux que j'interrompe "
            "tout maintenant."
        )

    # Persist if anything changed (mode flip OR last_user_msg_at update).
    save_state(state)

    if state.mode != old_mode:
        logger.info("AFK state: %s -> %s (user msg: %r)", old_mode, state.mode, content[:80])

    return state, confirm


def check_auto_transitions(now_utc: Optional[datetime] = None) -> tuple[AFKState, Optional[str]]:
    """Evaluate auto-AFK trigger: 20:00 GMT-3 + idle ≥ 30min + state normal.

    Called by the gateway cron ticker every 60s. Returns (state, notification
    or None). Notification is a Discord-postable string when auto-AFK fires.
    """
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)

    state = load_state()

    # Only flip to afk_auto from normal mode. AFK manual / standby are
    # explicit user states and shouldn't be overridden by the clock.
    if state.mode != MODE_NORMAL:
        return state, None

    # Time-of-day trigger: 20:00 GMT-3 == 23:00 UTC. Allow a 1-hour window
    # (23:00-23:59 UTC) so the cron tick that lands at 23:01 still triggers.
    if now_utc.hour != AFK_AUTO_TRIGGER_HOUR_UTC:
        return state, None

    # Idle check: last_user_msg_at older than IDLE_THRESHOLD_MINUTES.
    if state.last_user_msg_at is None:
        # No record of any user msg — assume idle long enough to trigger.
        idle_minutes = float("inf")
    else:
        try:
            last = datetime.strptime(state.last_user_msg_at, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
            idle_minutes = (now_utc - last).total_seconds() / 60.0
        except (ValueError, TypeError):
            idle_minutes = float("inf")

    if idle_minutes < IDLE_THRESHOLD_MINUTES:
        return state, None

    # Conditions met — flip to afk_auto.
    state.mode = MODE_AFK_AUTO
    state.entered_at = now_iso()
    save_state(state)
    logger.info(
        "AFK auto triggered at %s (idle %.1f min)",
        now_utc.isoformat(), idle_minutes,
    )
    notification = (
        "🌙 **Mode AFK auto activé** (20:00 GMT-3 + idle ≥ 30 min).\n"
        "• MAX_DEPTH levé à 50, publish externe désactivé\n"
        "• Je vais chercher des tâches in_progress dans ~/SMCP_STATUS.md, "
        "~/WS*_STATUS.md, ~/acos/docs/ROADMAP.md\n"
        "• Je te ping en heartbeat à J+3, J+7, J+14 (option B)\n"
        "• Tu peux annuler à tout moment avec `stop` / `arrête` / nouveau msg\n\n"
        "💤 Bonne soirée Khéri."
    )
    return state, notification
