"""Foreground-only capture controller, with no legacy voice/dream feature mixing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from chuff_mpu import audio_safety, config
from chuff_mpu.attention import ForegroundLearner, Observation
from chuff_mpu.mood import Mood, MoodState


class MoodSink(Protocol):
    def send_mood(self, mood: str, energy: float) -> None: ...


@dataclass(frozen=True)
class SelectiveConfig:
    bored_after_s: float = config.BORED_TIMEOUT_S
    candidate_energy: float = 0.5
    familiar_energy: float = 0.25
    cue_frequency_hz: float = 900.0
    cue_duration_s: float = 0.12
    cue_amplitude: float = 0.15
    cue_guard_s: float = 0.25

    def __post_init__(self) -> None:
        if not np.isfinite(self.bored_after_s) or self.bored_after_s <= 0:
            raise ValueError("bored timeout must be finite and positive")
        if not 0 <= self.candidate_energy <= 1 or not 0 <= self.familiar_energy <= 1:
            raise ValueError("display energies must be in [0, 1]")
        if (not np.isfinite(self.cue_frequency_hz) or self.cue_frequency_hz <= 0
                or not 0 < self.cue_duration_s <= 1
                or not 0 < self.cue_amplitude <= config.SAFE_PEAK_AMPLITUDE
                or not 0 < self.cue_guard_s <= 2):
            raise ValueError("invalid learning cue settings")


@dataclass(frozen=True)
class SelectiveUpdate:
    observation: Observation
    mood: MoodState
    idle_s: float
    cue: np.ndarray | None

    def as_dict(self) -> dict:
        return {"status": self.observation.status, "novelty": self.observation.novelty,
                "salience": self.observation.salience, "prototype_id": self.observation.prototype_id,
                "mood": self.mood.mood.value, "energy": self.mood.energy,
                "idle_s": self.idle_s, "cue_samples": 0 if self.cue is None else len(self.cue)}


class SelectiveController:
    """Map foreground outcomes to display messages; cue only newly admitted patterns.

    This path owns only a ForegroundLearner and does not call Repertoire,
    EmbeddingStore, or the legacy dream routine. Background/uncertain input
    returns to calm (or boredom), never latching the previous learning cue.
    Nothing is played by this class: the optional cue is returned to the caller.
    """

    def __init__(self, listener: ForegroundLearner | None = None,
                 settings: SelectiveConfig | None = None, *, sink: MoodSink | None = None,
                 learning_cue: bool = False) -> None:
        if listener is not None and not isinstance(listener, ForegroundLearner):
            raise TypeError("selective controller requires foreground memory, not legacy MFCC memory")
        self.listener = listener if listener is not None else ForegroundLearner()
        self.settings = settings or SelectiveConfig()
        if self.settings.cue_frequency_hz >= self.listener.settings.sample_rate / 2:
            raise ValueError("learning cue frequency must be below Nyquist")
        self.sink = sink
        self.learning_cue = learning_cue
        self.idle_s = 0.0

    def process(self, clip: np.ndarray) -> SelectiveUpdate:
        result = self.listener.observe(clip)
        foreground = result.status in ("candidate", "learned", "recognised")
        self.idle_s = 0.0 if foreground else self.idle_s + len(clip) / self.listener.settings.sample_rate
        if result.status == "learned":
            mood = MoodState(Mood.NEW_SOUND_LEARNED, 1.0)
        elif result.status == "candidate":
            mood = MoodState(Mood.MILD_NOVELTY, self.settings.candidate_energy)
        elif result.status == "recognised":
            mood = MoodState(Mood.SONG_RECOGNISED, self.settings.familiar_energy)
        elif result.status != "calibrating" and self.idle_s >= self.settings.bored_after_s:
            mood = MoodState(Mood.BORED, 0.1)
        else:
            mood = MoodState(Mood.CONTENTMENT, 0.15)
        cue = None
        if self.learning_cue and result.status == "learned":
            c = self.settings
            rate = self.listener.settings.sample_rate
            t = np.arange(int(c.cue_duration_s * rate)) / rate
            cue = audio_safety.limit_audio(
                c.cue_amplitude * np.sin(2 * np.pi * c.cue_frequency_hz * t) * np.hanning(len(t)), rate)
        if self.sink is not None:
            self.sink.send_mood(mood.mood.value, mood.energy)
        return SelectiveUpdate(result, mood, self.idle_s, cue)
