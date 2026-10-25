"""Audio output safety guardrails.

Every sample Chuff plays through a real speaker MUST pass through this
module first (`comms.py`'s `Repertoire.play_call()` and `sim/`'s agent
both do). It's the single choke point between "a numpy array some piece
of code computed" and "actual voltage on a speaker" — nothing downstream
of `limit_audio()` / `safe_play()` is trusted to have gotten the
amplitude right, however it was computed upstream.

Three independent protections, applied in order, each handling a
different way a bug could otherwise blow a speaker or amp:

  1. Sanitise — replace any NaN/Inf sample with silence. A runaway
     computation (divide-by-zero in a synthesis formula, etc.) must
     become silence, never an arbitrarily loud pop.
  2. Duration cap — truncate anything longer than SAFE_MAX_DURATION_S.
     A bug that generates an endless or much-longer-than-intended tone
     must not drive the speaker for an unbounded time.
  3. Soft-knee limiter + hard ceiling — gently compress as the signal
     approaches SAFE_PEAK_AMPLITUDE, then hard-clip anything that still
     exceeds it. The soft knee avoids harsh digital clipping artifacts
     for signals just over the line; the hard clip after it is the
     actual guarantee — speakers/amps care about peak voltage, not
     about whether the clipping sounds musical.

All three thresholds live in config.py (SAFE_*), not here, per the
project-wide "tunables live in one config module" rule.
"""

from __future__ import annotations

import numpy as np

from chuff_mpu import config


def limit_audio(
    samples: np.ndarray,
    sample_rate: int,
    max_peak: float = config.SAFE_PEAK_AMPLITUDE,
    max_duration_s: float = config.SAFE_MAX_DURATION_S,
    knee_fraction: float = config.SAFE_LIMITER_KNEE,
) -> np.ndarray:
    """Return a version of `samples` guaranteed safe to send to a speaker.

    Args:
        samples: 1-D audio samples, nominally in [-1, 1].
        sample_rate: Sample rate in Hz (needed to apply the duration cap).
        max_peak: Hard ceiling on |sample| in the output.
        max_duration_s: Hard ceiling on output length, in seconds.
        knee_fraction: Where (as a fraction of max_peak) the soft-knee
            compression starts; above this point the signal is
            compressed, not passed through unchanged.

    Returns:
        A new float32 array, no longer than `max_duration_s` worth of
        samples, with every value's magnitude <= max_peak.
    """
    samples = np.asarray(samples, dtype=np.float32)

    # 1. Sanitise: NaN/Inf -> silence, never an uncontrolled spike.
    if not np.all(np.isfinite(samples)):
        samples = samples.copy()
        samples[~np.isfinite(samples)] = 0.0

    # 2. Duration cap.
    max_frames = int(max_duration_s * sample_rate)
    if samples.shape[0] > max_frames:
        samples = samples[:max_frames]

    # 3. Soft-knee limiter, then an unconditional hard clip as the real
    # guarantee (the tanh compression above is a courtesy for sound
    # quality, not something to rely on for the actual amplitude bound).
    knee = max_peak * knee_fraction
    abs_samples = np.abs(samples)
    over_knee = abs_samples > knee
    if np.any(over_knee):
        span = max(max_peak - knee, 1e-9)
        excess = abs_samples[over_knee] - knee
        compressed = knee + span * np.tanh(excess / span)
        samples = samples.copy()
        samples[over_knee] = np.sign(samples[over_knee]) * compressed

    return np.clip(samples, -max_peak, max_peak).astype(np.float32)


def safe_play(samples: np.ndarray, sample_rate: int, device: int | str | None = None) -> np.ndarray:
    """Limit `samples` and best-effort play them through an output device.

    Always returns the limited array — what was (or would have been)
    played — even when no audio output device is available, so callers
    that only need the guaranteed-safe signal (e.g. sim/'s "hear my own
    echo" feedback loop) don't need real speaker hardware to run.
    """
    limited = limit_audio(samples, sample_rate)
    try:
        import sounddevice as sd

        sd.play(limited, samplerate=sample_rate, device=device)
        sd.wait()
    except Exception:
        # No output device / no audio backend / headless CI, etc. The
        # safety guarantee above already happened regardless of whether
        # playback itself succeeds.
        pass
    return limited
