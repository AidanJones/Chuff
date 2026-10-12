"""KNN embedding store for the UNO Q MPU side.

Equivalent role to the esp32 track's (not yet implemented) PSRAM-backed
KNN store, but backed by scikit-learn's NearestNeighbors and ordinary
process memory (MAX_EMBEDDINGS in config.py is far higher here than the
ESP32's PSRAM ceiling allows).

Validated two ways: eval/tasks/sound_memory_and_voice/grader.py scores
this implementation directly, and sim/ exercises it end-to-end against
real synthesised audio (not just abstract feature vectors).
"""

from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors


class EmbeddingStore:
    """Grows-at-runtime nearest-neighbour store of MFCC embeddings."""

    def __init__(self, max_embeddings: int) -> None:
        """Create an empty store with a fixed capacity.

        Args:
            max_embeddings: Maximum number of embeddings to retain; oldest
                entries are evicted once the store is full.
        """
        self.max_embeddings = max_embeddings
        self._items: list[np.ndarray] = []

    def add(self, embedding: np.ndarray) -> None:
        """Add a new MFCC embedding to the store.

        Args:
            embedding: 1-D feature vector, e.g. from audio.extract_mfcc.
        """
        self._items.append(np.asarray(embedding, dtype=np.float32))
        if len(self._items) > self.max_embeddings:
            self._items.pop(0)  # evict oldest

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
        if not self._items:
            return 1.0
        nn = NearestNeighbors(n_neighbors=1, metric="cosine")
        nn.fit(np.stack(self._items))
        query = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
        distance, _ = nn.kneighbors(query, n_neighbors=1)
        return float(distance[0, 0])

    def sample_pair(self) -> tuple[np.ndarray, np.ndarray]:
        """Sample two stored embeddings at random, for the dream loop.

        Returns:
            A pair of distinct stored embeddings.

        Raises:
            ValueError: If fewer than two embeddings are stored.
        """
        if len(self._items) < 2:
            raise ValueError("need at least two stored embeddings to sample a pair")
        rng = np.random.default_rng()
        i, j = rng.choice(len(self._items), size=2, replace=False)
        return self._items[i], self._items[j]

    def save(self, path: str) -> None:
        """Persist the store to disk (JSON/npz) so it survives a restart."""
        np.savez(path, *self._items)

    def load(self, path: str) -> None:
        """Load a previously persisted store from disk."""
        data = np.load(path)
        self._items = [data[key] for key in data.files]

    def __len__(self) -> int:
        """Number of embeddings currently stored (<= max_embeddings)."""
        return len(self._items)
