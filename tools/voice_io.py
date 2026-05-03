#!/usr/bin/env python3
"""
voice_io.py — Local TTS/STT loop for hermes-agent via whisper.cpp + Piper services.

Services:
    whisper.service  127.0.0.1:8765   POST /inference         multipart, returns {"text": ...}
    tts.service      127.0.0.1:8766   POST /synthesize        JSON in, WAV out
                                      POST /synthesize-discord JSON in/out (Discord-ready bundle)
                                      GET  /health             JSON status

Design rule: this module **never** spawns subprocesses. hermes-agent.service runs under
strict SECCOMP — fork+exec of ffmpeg/ffprobe triggers SIGSYS and kills the process.
All audio transformation (WAV→Opus, waveform RMS) is offloaded to tts.service via
the /synthesize-discord endpoint, which runs under a relaxed sandbox.

Format Discord native voice message:
    flags    = 8192  (IS_VOICE_MESSAGE = 1 << 13)
    waveform = 256 RMS samples (byte array), base64-encoded
    audio    = OGG Opus 48 kHz mono
    https://github.com/discord/discord-api-docs/issues/4406
"""

from __future__ import annotations

import base64
import json
import logging
import os
from typing import Tuple

import aiohttp

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

STT_URL = "http://127.0.0.1:8765/inference"
TTS_URL = "http://127.0.0.1:8766/synthesize"
TTS_DISCORD_URL = "http://127.0.0.1:8766/synthesize-discord"
TTS_HEALTH_URL = "http://127.0.0.1:8766/health"


# ---------------------------------------------------------------------------
# STT — Whisper service
# ---------------------------------------------------------------------------

async def stt(audio_bytes: bytes, language: str = "fr") -> str:
    """Transcribe audio via the local whisper.cpp service.

    Args:
        audio_bytes: raw audio (any format ffmpeg can decode; whisper.cpp uses
                     --convert and re-encodes to 16 kHz mono internally).
        language: ISO-639-1 code, or "auto".

    Returns:
        Transcribed text, stripped.

    Raises:
        aiohttp.ClientError: on transport / HTTP error.
    """
    data = aiohttp.FormData()
    data.add_field("file", audio_bytes, filename="audio.ogg", content_type="audio/ogg")
    data.add_field("response_format", "json")
    data.add_field("language", language)

    async with aiohttp.ClientSession() as session:
        async with session.post(
            STT_URL,
            data=data,
            timeout=aiohttp.ClientTimeout(total=300),
        ) as resp:
            resp.raise_for_status()
            payload = await resp.json()
            return payload.get("text", "").strip()


# ---------------------------------------------------------------------------
# TTS — Piper service (raw WAV)
# ---------------------------------------------------------------------------

async def tts(text: str, voice: str = "fr_FR-upmc-medium") -> bytes:
    """Synthesize text to WAV bytes (PCM 22050 Hz mono 16-bit).

    Use this when you just need a WAV (e.g. local playback). For Discord
    native voice messages, prefer ``tts_discord`` which returns OGG Opus
    + pre-computed waveform without forcing the caller to fork ffmpeg.
    """
    async with aiohttp.ClientSession() as session:
        async with session.post(
            TTS_URL,
            json={"text": text, "voice": voice},
            timeout=aiohttp.ClientTimeout(total=30),
        ) as resp:
            resp.raise_for_status()
            return await resp.read()


async def tts_discord(text: str, voice: str = "fr_FR-upmc-medium") -> Tuple[bytes, str, float]:
    """Synthesize text to a Discord-ready bundle.

    All heavy lifting (WAV→OGG Opus + RMS waveform) happens server-side in
    tts.service, so hermes-agent does not spawn any subprocess.

    Returns:
        (ogg_bytes, waveform_base64, duration_seconds)

    Raises:
        aiohttp.ClientError: on transport / HTTP error.
    """
    async with aiohttp.ClientSession() as session:
        async with session.post(
            TTS_DISCORD_URL,
            json={"text": text, "voice": voice},
            timeout=aiohttp.ClientTimeout(total=60),
        ) as resp:
            resp.raise_for_status()
            payload = await resp.json()

    ogg_bytes = base64.b64decode(payload["ogg_b64"])
    waveform_b64 = payload["waveform_b64"]
    duration = float(payload["duration_secs"])
    return ogg_bytes, waveform_b64, duration


async def tts_health_check() -> dict:
    """Return tts.service /health payload, or a fallback dict on error."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                TTS_HEALTH_URL,
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                if resp.status == 200:
                    return await resp.json()
                return {"status": "unavailable", "code": resp.status}
    except Exception as e:
        return {"status": "unreachable", "error": str(e)[:200]}


# ---------------------------------------------------------------------------
# Native voice message detection (Discord input)
# ---------------------------------------------------------------------------

DISCORD_VOICE_MESSAGE_FLAG = 1 << 13  # IS_VOICE_MESSAGE = 8192


def is_native_voice_message(message) -> bool:
    """True if ``message`` is a Discord native voice message.

    Detection: ``flags & 8192`` AND a single audio attachment.
    """
    flags_val = getattr(message, "flags", None)
    if flags_val is None:
        return False
    flags_int = getattr(flags_val, "value", flags_val)
    if not (flags_int & DISCORD_VOICE_MESSAGE_FLAG):
        return False

    attachments = getattr(message, "attachments", None) or []
    if len(attachments) != 1:
        return False

    content_type = getattr(attachments[0], "content_type", "") or ""
    return content_type.startswith("audio/")
