"""Minimal audio synthesis building blocks shared by world.py and agent.py.

Deliberately simple (pure sine tones and linear chirps, no envelopes or
harmonics) — this is a test of the listen/remember/repeat *architecture*,
not an attempt at realistic sound design.
"""

from __future__ import annotations

import numpy as np


def sine_tone(freq_hz: float, duration_s: float, sample_rate: int, amplitude: float = 0.3) -> np.ndarray:
    """A plain sine wave at a fixed frequency."""
    n = int(duration_s * sample_rate)
    t = np.linspace(0, duration_s, n, endpoint=False)
    return (amplitude * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)


def chirp(
    freq_start_hz: float, freq_end_hz: float, duration_s: float, sample_rate: int, amplitude: float = 0.3
) -> np.ndarray:
    """A linear frequency sweep from freq_start_hz to freq_end_hz."""
    n = int(duration_s * sample_rate)
    t = np.linspace(0, duration_s, n, endpoint=False)
    freq_t = np.linspace(freq_start_hz, freq_end_hz, n)
    phase = 2 * np.pi * np.cumsum(freq_t) / sample_rate
    return (amplitude * np.sin(phase)).astype(np.float32)


def white_noise(
    duration_s: float, sample_rate: int, amplitude: float = 0.05, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Background/ambient noise standing in for "the room"."""
    rng = rng or np.random.default_rng()
    n = int(duration_s * sample_rate)
    return (amplitude * rng.standard_normal(n)).astype(np.float32)


def mix(*signals: np.ndarray) -> np.ndarray:
    """Sum signals of possibly-different lengths, zero-padding the shorter ones."""
    length = max(len(s) for s in signals)
    out = np.zeros(length, dtype=np.float32)
    for s in signals:
        out[: len(s)] += s
    return out


def fit_length(signal: np.ndarray, n: int) -> np.ndarray:
    """Truncate or zero-pad `signal` to exactly `n` samples."""
    if len(signal) > n:
        return signal[:n]
    if len(signal) < n:
        return np.pad(signal, (0, n - len(signal)))
    return signal
