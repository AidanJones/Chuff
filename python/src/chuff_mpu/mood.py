"""Mood state machine for the UNO Q MPU side.

Mirrors the esp32 track's (not yet implemented) mood.py. Consumes
novelty/surprise scores from memory.py and dream.py and produces a
Mood, which ipc.py then ships to the MCU side as JSON for the LED
matrix / Modulino Pixels to render per shared/mood_grammar.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from chuff_mpu import config


class Mood(str, Enum):
    """Culture colour grammar states — see shared/mood_grammar.md."""

    DREAMING = "dreaming"
    MILD_NOVELTY = "mild_novelty"
    STRONG_NOVELTY = "strong_novelty"
    SONG_RECOGNISED = "song_recognised"
    NEW_SOUND_LEARNED = "new_sound_learned"
    BORED = "bored"
    SURPRISE = "surprise"
    ACKNOWLEDGEMENT = "acknowledgement"
    CONTENTMENT = "contentment"


@dataclass
class MoodState:
    """Current mood plus the scalar energy sent alongside it over IPC."""

    mood: Mood
    energy: float  # 0.0-1.0, drives e.g. blink/animation speed on the MCU side


def update_mood(
    current: MoodState,
    novelty: float | None,
    surprise: float | None,
    idle_s: float,
    ack_received: bool = False,
) -> MoodState:
    """Advance the mood state machine given the latest scores.

    Precedence (highest first): an acknowledgement just heard over the
    comms channel is always the most immediate reactive event; then a
    dream-loop surprise comparison that just concluded; then idle-time
    states (bored overrides dreaming — it's the same "no new sounds"
    condition extended much further, see shared/mood_grammar.md);
    finally the novelty score from the latest capture. If none of these
    fired this tick (nothing new happened), the mood holds.

    One simplification worth stating plainly: this codebase's
    convention (see sim/agent.py) is that a sound only gets committed to
    memory when its novelty score is already >= NOVELTY_THRESHOLD_HIGH —
    the same condition the mood grammar calls "strong novelty". So
    "strong novelty" and "new sound learned" are, in practice, the same
    event here; NEW_SOUND_LEARNED is emitted (the bigger of the two
    reactions — a small celebration, not just a flash) rather than
    returning both.

    `ack_received` isn't derivable from novelty/surprise/idle_s alone —
    it's an optional 5th parameter (default False, so existing 4-arg
    callers are unaffected) rather than a signal this function invents
    out of nothing.

    Args:
        current: The mood state before this update.
        novelty: Latest novelty score from memory.novelty_score(), or None
            if no audio was captured this tick.
        surprise: Latest dream-loop surprise score from dream.surprise_score(),
            or None if not waking from a dream this tick.
        idle_s: Seconds since the last new sound was heard.
        ack_received: Whether an acknowledgement tone was just heard over
            the comms channel this tick (shared/comms_protocol.md).

    Returns:
        The updated MoodState.
    """
    if ack_received:
        return MoodState(Mood.ACKNOWLEDGEMENT, energy=0.6)

    if surprise is not None:
        if surprise >= config.DREAM_SURPRISE_THRESHOLD:
            return MoodState(Mood.SURPRISE, energy=min(1.0, 0.7 + 0.3 * surprise))
        return MoodState(Mood.CONTENTMENT, energy=max(0.1, 0.3 * (1.0 - surprise)))

    if idle_s >= config.BORED_TIMEOUT_S:
        return MoodState(Mood.BORED, energy=0.1)
    if idle_s >= config.IDLE_TIMEOUT_S:
        return MoodState(Mood.DREAMING, energy=0.2)

    if novelty is not None:
        if novelty < config.RECOGNITION_THRESHOLD:
            return MoodState(Mood.SONG_RECOGNISED, energy=0.5)
        if novelty >= config.NOVELTY_THRESHOLD_HIGH:
            return MoodState(Mood.NEW_SOUND_LEARNED, energy=1.0)
        if novelty >= config.NOVELTY_THRESHOLD_LOW:
            span = config.NOVELTY_THRESHOLD_HIGH - config.NOVELTY_THRESHOLD_LOW
            frac = (novelty - config.NOVELTY_THRESHOLD_LOW) / span if span else 0.0
            return MoodState(Mood.MILD_NOVELTY, energy=0.4 + 0.3 * frac)
        return MoodState(Mood.CONTENTMENT, energy=0.3)

    return current
