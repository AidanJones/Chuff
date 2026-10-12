"""Audio capture and MFCC feature extraction for the UNO Q MPU side.

Full-fat equivalent of esp32/audio.py, but running on Debian Linux with
librosa/numpy instead of emlearn-micropython. Captures from a USB mic via
sounddevice and extracts MFCC vectors for the KNN novelty/memory pipeline
in memory.py.

Not yet implemented — see python/src/chuff_mpu/mvp_demo.py for a working
(but much simpler) RMS-energy stand-in used for the current MVP.
"""

from __future__ import annotations

import numpy as np


def capture_clip(duration_s: float, sample_rate: int) -> np.ndarray:
    """Record one mono audio clip from the default input device.

    Args:
        duration_s: Clip length in seconds.
        sample_rate: Sample rate in Hz.

    Returns:
        1-D float32 PCM array of shape (duration_s * sample_rate,).
    """
    raise NotImplementedError


def extract_mfcc(clip: np.ndarray, sample_rate: int, n_mfcc: int) -> np.ndarray:
    """Extract an MFCC feature vector from a raw audio clip.

    Args:
        clip: 1-D float32 PCM samples.
        sample_rate: Sample rate in Hz the clip was captured at.
        n_mfcc: Number of MFCC coefficients to compute.

    Returns:
        1-D float32 vector of length n_mfcc (time-averaged across the clip).
    """
    raise NotImplementedError
