"""Audio capture and MFCC feature extraction for the UNO Q MPU side.

Full-fat equivalent of the esp32 track's (not yet implemented) audio
pipeline, but running on Debian Linux with librosa/numpy instead of
emlearn-micropython. Captures from a USB mic via sounddevice and extracts
MFCC vectors for the KNN novelty/memory pipeline in memory.py.

See sim/ for a from-the-ground-up demonstration of this feeding into
memory.py + comms.py with no real microphone/speaker involved.
"""

from __future__ import annotations

import librosa
import numpy as np
import sounddevice as sd


def capture_clip(duration_s: float, sample_rate: int) -> np.ndarray:
    """Record one mono audio clip from the default input device.

    Args:
        duration_s: Clip length in seconds.
        sample_rate: Sample rate in Hz.

    Returns:
        1-D float32 PCM array of shape (duration_s * sample_rate,).
    """
    frames = int(duration_s * sample_rate)
    clip = sd.rec(frames, samplerate=sample_rate, channels=1, dtype="float32")
    sd.wait()
    return clip.reshape(-1)


def extract_mfcc(clip: np.ndarray, sample_rate: int, n_mfcc: int) -> np.ndarray:
    """Extract an MFCC feature vector from a raw audio clip.

    Drops the 0th coefficient (overall log-energy) before averaging:
    empirically, with cosine distance as the comparison metric (see
    memory.py), c0 dominates the vector's magnitude and makes every clip
    look alike regardless of pitch/timbre — two different tones at
    similar loudness come out closer to each other than two repeats of
    the same tone at different loudness. Dropping it is standard practice
    for pitch/timbre comparison and is what makes
    config.RECOGNITION_THRESHOLD / NOVELTY_THRESHOLD_* actually separate
    same-sound-again from a genuinely different sound (verified in
    eval/tasks/sound_memory_and_voice/ and sim/).

    Args:
        clip: 1-D float32 PCM samples.
        sample_rate: Sample rate in Hz the clip was captured at.
        n_mfcc: Number of MFCC coefficients to return (c0 is computed and
            discarded in addition to these, so n_mfcc+1 are computed).

    Returns:
        1-D float32 vector of length n_mfcc (time-averaged across the clip).
    """
    clip = np.asarray(clip, dtype=np.float32)
    coeffs = librosa.feature.mfcc(y=clip, sr=sample_rate, n_mfcc=n_mfcc + 1)
    return coeffs[1:].mean(axis=1).astype(np.float32)
