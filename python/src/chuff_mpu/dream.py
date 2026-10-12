"""Dream loop: idle-time prediction and surprise scoring.

Mirrors esp32/dream.py. After IDLE_TIMEOUT_S of no new audio, interpolates
between two stored embeddings to form a prediction; compares the next real
observation against that prediction to produce a surprise score that drives
mood.py's "dreaming" / "surprise" states.
"""

from __future__ import annotations

import numpy as np

from chuff_mpu.memory import EmbeddingStore


def predict_next(store: EmbeddingStore, weight: float = 0.5) -> np.ndarray:
    """Interpolate between two randomly sampled stored embeddings.

    Args:
        store: The embedding store to sample from.
        weight: Interpolation weight in [0, 1] applied to the first sample.

    Returns:
        Predicted embedding vector.
    """
    raise NotImplementedError


def surprise_score(observed: np.ndarray, predicted: np.ndarray) -> float:
    """Cosine distance between an observed embedding and a prediction.

    Args:
        observed: The newly captured MFCC embedding.
        predicted: The embedding predicted by predict_next().

    Returns:
        Cosine distance in [0, 2]; higher means more surprising.
    """
    raise NotImplementedError
