"""Mood state machine for the UNO Q MPU side.

Mirrors esp32/mood.py. Consumes novelty/surprise scores from memory.py and
dream.py and produces a Mood, which ipc.py then ships to the MCU side as
JSON for the LED matrix / Modulino Pixels to render per
shared/mood_grammar.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


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
) -> MoodState:
    """Advance the mood state machine given the latest scores.

    Args:
        current: The mood state before this update.
        novelty: Latest novelty score from memory.novelty_score(), or None
            if no audio was captured this tick.
        surprise: Latest dream-loop surprise score from dream.surprise_score(),
            or None if not waking from a dream this tick.
        idle_s: Seconds since the last new sound was heard.

    Returns:
        The updated MoodState.
    """
    raise NotImplementedError
