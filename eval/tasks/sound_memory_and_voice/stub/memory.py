"""KNN embedding store for the UNO Q MPU side.

Equivalent role to the esp32 track's (not yet implemented) PSRAM-backed
KNN store, but backed by scikit-learn's NearestNeighbors and ordinary
process memory (MAX_EMBEDDINGS in config.py is far higher here than the
ESP32's PSRAM ceiling allows).
"""

from __future__ import annotations

import numpy as np


class EmbeddingStore:
    """Grows-at-runtime nearest-neighbour store of MFCC embeddings."""

    def __init__(self, max_embeddings: int) -> None:
        """Create an empty store with a fixed capacity.

        Args:
            max_embeddings: Maximum number of embeddings to retain; oldest
                entries are evicted once the store is full.
        """
        raise NotImplementedError

    def add(self, embedding: np.ndarray) -> None:
        """Add a new MFCC embedding to the store.

        Args:
            embedding: 1-D feature vector, e.g. from audio.extract_mfcc.
        """
        raise NotImplementedError

    def novelty_score(self, embedding: np.ndarray) -> float:
        """Return the distance from `embedding` to its nearest stored entry.

        Use cosine distance (1 - cosine similarity), not raw Euclidean
        distance — it's what the dream loop's surprise scoring uses
        elsewhere in this codebase, and config.py's thresholds
        (NOVELTY_THRESHOLD_LOW/HIGH, RECOGNITION_THRESHOLD) are calibrated
        assuming the same metric throughout.

        Args:
            embedding: 1-D feature vector to score.

        Returns:
            Distance to nearest neighbour, or 1.0 (maximally novel) if the
            store is empty.
        """
        raise NotImplementedError

    def sample_pair(self) -> tuple[np.ndarray, np.ndarray]:
        """Sample two stored embeddings at random, for the dream loop.

        Returns:
            A pair of distinct stored embeddings.

        Raises:
            ValueError: If fewer than two embeddings are stored.
        """
        raise NotImplementedError

    def save(self, path: str) -> None:
        """Persist the store to disk (JSON/npz) so it survives a restart."""
        raise NotImplementedError

    def load(self, path: str) -> None:
        """Load a previously persisted store from disk."""
        raise NotImplementedError

    def __len__(self) -> int:
        """Number of embeddings currently stored (<= max_embeddings)."""
        raise NotImplementedError
