"""Dream loop: idle-time prediction and surprise scoring.

Mirrors the esp32 track's (not yet implemented) dream loop. After
IDLE_TIMEOUT_S of no new audio, interpolates between two stored
embeddings to form a prediction; compares the next real observation
against that prediction to produce a surprise score that drives
mood.py's "dreaming" / "surprise" states.
"""

from __future__ import annotations

import numpy as np

from chuff_mpu.memory import EmbeddingStore


def predict_next(store: EmbeddingStore, weight: float = 0.5,
                 *, rng: np.random.Generator | None = None) -> np.ndarray:
    """Interpolate between two randomly sampled stored embeddings.

    Args:
        store: The embedding store to sample from.
        weight: Interpolation weight in [0, 1] applied to the first sample.
        rng: Caller-owned sampling stream. Supply this for reproducible dreams.

    Returns:
        Predicted embedding vector.
    """
    if not np.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError("interpolation weight must be finite and in [0, 1]")
    a, b = store.sample_pair(rng=rng)
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return (weight * a + (1.0 - weight) * b).astype(np.float32)


def surprise_score(observed: np.ndarray, predicted: np.ndarray) -> float:
    """Cosine distance between an observed embedding and a prediction.

    Same metric convention as memory.EmbeddingStore.novelty_score() and
    comms.Repertoire's relevance gate — see memory.py's docstring for why
    cosine distance (not Euclidean) is used consistently throughout.

    Args:
        observed: The newly captured MFCC embedding.
        predicted: The embedding predicted by predict_next().

    Returns:
        Cosine distance in [0, 2]; higher means more surprising.
    """
    observed = np.asarray(observed, dtype=np.float32)
    predicted = np.asarray(predicted, dtype=np.float32)
    denom = (np.linalg.norm(observed) * np.linalg.norm(predicted)) or 1e-9
    return float(1.0 - np.dot(observed, predicted) / denom)
