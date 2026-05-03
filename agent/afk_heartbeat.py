"""AFK heartbeat — Patch 13.4.

Implements the 3-cycle heartbeat schedule (option B per Khéri's choice):
  HB1 at entered_at + 3 days
  HB2 at entered_at + 7 days
  HB3 at entered_at + 14 days
  STAND_BY at HB3 + 24h with no user reply

Each heartbeat asks Khéri for orientation (continue / pivot / stop).
If three heartbeats pass without a user message arriving (which would
flip the state out of AFK via afk_scheduler.process_user_message), AH
flips to MODE_STAND_BY: stops new commits, just monitors and posts a
final summary daily.

Called from gateway/run.py:_start_cron_ticker every 60s. Uses the
Discord adapter to post if available.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from agent.afk_state import (
    AFKState,
    AFK_MODES,
    MODE_AFK_AUTO,
    MODE_AFK_MANUAL,
    MODE_NORMAL,
    MODE_STAND_BY,
    load_state,
    now_iso,
    save_state,
)

logger = logging.getLogger(__name__)

# Option B heartbeat schedule (Khéri's choice)


HB_DAYS = (3, 7, 14)
STAND_BY_DELAY_HOURS_AFTER_HB3 = 24


def _parse_iso(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    try:
        return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None


def _format_hb_message(hb_cycle: int, days_in_afk: int, state: AFKState) -> str:
    """Build the Discord message for a heartbeat at the given cycle."""
    common_tail = (
        "\n\n**Que veux-tu que je fasse ?**\n"
        "• `continue` — je poursuis sur la même direction\n"
        "• `change` puis ta nouvelle consigne — je pivote\n"
        "• `stop` ou `arrête` — j'arrête tout, mode normal\n"
        "• Pas de réponse — je continue selon mon jugement et je te ping au prochain heartbeat"
    )

    intervals_str = ", ".join(f"J{int(d)}" if d == int(d) else f"J{d}" for d in HB_DAYS)

    if hb_cycle == 1:
        return (
            f"💓 **Heartbeat 1/{len(HB_DAYS)} — J+{days_in_afk}** "
            f"(intervals: {intervals_str}).\n\n"
            f"Ça fait {days_in_afk} jours que tu es en AFK ({state.mode}). "
            f"J'espère que tu vas bien.\n\n"
            f"📊 État de mon travail (à compléter par AH lui-même au prochain "
            f"démarrage de session — ce ping est juste l'alerte schedule).\n"
            f"Voir ~/AFK_LOG.md pour le récap détaillé."
        ) + common_tail

    if hb_cycle == 2:
        return (
            f"💓 **Heartbeat 2/{len(HB_DAYS)} — J+{days_in_afk}**.\n\n"
            f"Toujours pas de nouvelles depuis le HB1. Je continue mon travail "
            f"selon les missions whitelist.\n\n"
            f"📊 Récap à voir dans ~/AFK_LOG.md (mis à jour par AH à chaque "
            f"changement d'état)."
        ) + common_tail

    if hb_cycle == 3:
        sb_h = STAND_BY_DELAY_HOURS_AFTER_HB3
        sb_str = f"{int(sb_h)}h" if sb_h == int(sb_h) else f"{sb_h}h"
        return (
            f"💓 **Heartbeat {len(HB_DAYS)}/{len(HB_DAYS)} — J+{days_in_afk} — DERNIER PING**.\n\n"
            f"Khéri, ça fait {days_in_afk} jours sans nouvelles. Si tu ne réponds "
            f"pas dans les **{sb_str}**, je passe en mode **stand_by** complet :\n"
            f"• Plus de nouveaux commits / missions\n"
            f"• Lecture / monitoring uniquement\n"
            f"• Récap final posté ici\n\n"
            f"J'espère que tout va bien de ton côté."
        ) + common_tail

    return f"💓 Heartbeat #{hb_cycle} (unexpected cycle)"


def evaluate_heartbeat(now_utc: Optional[datetime] = None) -> tuple[AFKState, Optional[str]]:
    """Decide whether to send a heartbeat or flip to stand_by.

    Returns (new_state, message_to_post_or_None).

    State machine:
      mode in (afk_manual, afk_auto), heartbeats_sent=0:
        if entered_at + 3d <= now → send HB1, heartbeats_sent=1
      heartbeats_sent=1:
        if entered_at + 7d <= now → send HB2, heartbeats_sent=2
      heartbeats_sent=2:
        if entered_at + 14d <= now → send HB3, heartbeats_sent=3,
        last_heartbeat_at = now
      heartbeats_sent=3:
        if last_heartbeat_at + 24h <= now → flip to stand_by, send final notice
    """
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)

    state = load_state()

    # Only act when in AFK modes (manual or auto)
    if state.mode not in AFK_MODES:
        return state, None

    entered = _parse_iso(state.entered_at)
    if entered is None:
        # No entered_at recorded — set it now to avoid runaway HB triggers
        state.entered_at = now_iso()
        save_state(state)
        return state, None

    days_in_afk = (now_utc - entered).total_seconds() / 86400.0

    # Determine which heartbeat is due
    sent = state.heartbeats_sent
    next_cycle = sent + 1  # cycles are 1-indexed in messages

    if next_cycle <= len(HB_DAYS):
        target_days = HB_DAYS[next_cycle - 1]
        if days_in_afk < target_days:
            # Not yet time for the next HB
            return state, None

        # Fire heartbeat next_cycle
        state.heartbeats_sent = next_cycle
        state.last_heartbeat_at = now_iso()
        save_state(state)
        msg = _format_hb_message(next_cycle, int(days_in_afk), state)
        logger.info(
            "AFK heartbeat %d/%d fired at J+%.1f (state=%s)",
            next_cycle, len(HB_DAYS), days_in_afk, state.mode,
        )
        return state, msg

    # Already sent 3 heartbeats — check stand_by transition
    last_hb = _parse_iso(state.last_heartbeat_at)
    if last_hb is None:
        return state, None
    hours_since_hb3 = (now_utc - last_hb).total_seconds() / 3600.0
    if hours_since_hb3 < STAND_BY_DELAY_HOURS_AFTER_HB3:
        return state, None

    # Flip to stand_by
    old_mode = state.mode
    state.mode = MODE_STAND_BY
    save_state(state)
    final = (
        f"🛑 **Bascule en mode `stand_by`** après 3 heartbeats sans réponse.\n\n"
        f"AH passe en mode lecture/monitoring uniquement :\n"
        f"• Plus de nouveaux commits / missions / labs\n"
        f"• Récap final disponible dans ~/AFK_LOG.md\n"
        f"• Tu peux me ré-activer à tout moment avec un nouveau message\n\n"
        f"Ça fait {(now_utc - entered).days} jours en AFK ({old_mode}). "
        f"À très bientôt Khéri."
    )
    logger.info(
        "AFK stand_by triggered: 3 heartbeats sent, last %.1fh ago, no user reply",
        hours_since_hb3,
    )
    return state, final
