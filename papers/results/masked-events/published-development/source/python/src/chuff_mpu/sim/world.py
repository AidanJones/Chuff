"""Synthetic acoustic environment standing in for a room + microphone.

A small fixed library of distinct "real-world sounds" (plain tones and
chirps — not real-world recordings, just enough acoustic variety to be
distinguishable), each presented with a little frequency jitter so
repeats of "the same sound" aren't bit-identical, plus background noise
mixed in to stand in for an untreated room. Chuff's own played-back
calls get mixed with fresh noise the same way via `hear()`, so "hearing
its own output mixed in with external noise" is just another call to
the same method.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, lfilter

from chuff_mpu.sim import tones

# label -> synthesis spec. Frequencies deliberately span and partly
# overlap config.CALL_FREQ_MIN_HZ..CALL_FREQ_MAX_HZ's range so some of
# Chuff's own repertoire calls land acoustically close to real sounds.
SOUND_LIBRARY: dict[str, dict] = {
    "doorbell": {"kind": "tone", "freq": 1200},
    "kettle_whistle": {"kind": "tone", "freq": 2600},
    "phone_ring": {"kind": "tone", "freq": 1800},
    "car_horn": {"kind": "tone", "freq": 420},
    "dog_bark": {"kind": "chirp", "freq": 350, "freq_end": 520},
    "bird_call": {"kind": "chirp", "freq": 3200, "freq_end": 2600},
}


def lowpass_filter(signal: np.ndarray, sample_rate: int, cutoff_hz: float, order: int = 4) -> np.ndarray:
    """4th-order Butterworth low-pass, standing in for frequency-dependent
    loss over distance/reverberation — real atmospheric absorption and
    reverberant energy loss both increase with frequency. See
    shared/research_backlog.md Theme 1.2 (the Acoustic Adaptation
    Hypothesis): for a pure tone, a cutoff below its frequency doesn't
    distort it so much as crush its amplitude toward the noise floor,
    which is exactly the failure mode that should make a too-high
    repertoire call unrecognisable over a lossy channel.

    A single one-pole filter (6 dB/octave) turned out too gentle to ever
    push a repertoire-range tone below the noise floor within one
    octave of the cutoff — empirically confirmed before settling on a
    steeper order, not assumed.
    """
    nyquist = sample_rate / 2
    b, a = butter(order, cutoff_hz / nyquist, btype="low")
    return lfilter(b, a, signal).astype(np.float32)


class Environment:
    """Presents sounds and models what the microphone actually picks up."""

    def __init__(
        self,
        sample_rate: int,
        clip_duration_s: float,
        seed: int = 0,
        background_noise_amp: float = 0.03,
    ) -> None:
        self.sample_rate = sample_rate
        self.clip_duration_s = clip_duration_s
        self.rng = np.random.default_rng(seed)
        self.background_noise_amp = background_noise_amp
        self.labels = list(SOUND_LIBRARY)

    def present(self, label: str | None = None, jitter_hz: float = 15.0) -> tuple[str, np.ndarray]:
        """Synthesise one instance of a sound (random label if none given)."""
        label = label or str(self.rng.choice(self.labels))
        spec = SOUND_LIBRARY[label]
        freq = max(20.0, spec["freq"] + self.rng.normal(scale=jitter_hz))
        if spec["kind"] == "tone":
            sig = tones.sine_tone(freq, self.clip_duration_s, self.sample_rate)
        else:
            freq_end = max(20.0, spec["freq_end"] + self.rng.normal(scale=jitter_hz))
            sig = tones.chirp(freq, freq_end, self.clip_duration_s, self.sample_rate)
        return label, sig

    def hear(self, signal: np.ndarray) -> np.ndarray:
        """What the mic actually picks up: `signal` plus fresh ambient noise,
        trimmed/padded to one clip's length — used both for real-world
        sounds and for Chuff hearing its own played-back call."""
        n = int(self.clip_duration_s * self.sample_rate)
        noise = tones.white_noise(self.clip_duration_s, self.sample_rate, self.background_noise_amp, self.rng)
        mixed = tones.mix(tones.fit_length(signal, n), noise)
        return tones.fit_length(mixed, n)

    def transmit(self, signal: np.ndarray, attenuation: float = 1.0, cutoff_hz: float | None = None) -> np.ndarray:
        """What a *different* unit's mic picks up from `signal` after it
        has travelled some distance: optional frequency-dependent loss
        (`cutoff_hz`, see `lowpass_filter`), then amplitude attenuation,
        then the same fresh-noise mixing as `hear()`. Distinct from
        `hear()`, which models "my own mic, right here" — no distance,
        so no attenuation or filtering, only room noise.
        """
        out = signal if cutoff_hz is None else lowpass_filter(signal, self.sample_rate, cutoff_hz)
        return self.hear(out * attenuation)
