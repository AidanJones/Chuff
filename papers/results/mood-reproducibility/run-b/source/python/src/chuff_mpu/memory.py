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
    """Grows-at-runtime nearest-neighbour store of MFCC embeddings.

    Eviction policy is swappable (`eviction="fifo"` or `"recency"`),
    defaulting to the original FIFO behaviour so existing callers/tests
    are unaffected. See shared/research_backlog.md Theme 2.2 and
    sim/memory_eviction_experiment.py for why: pure FIFO discards a
    rarely-but-regularly-heard sound exactly as readily as a genuine
    one-off, with no regard for whether it keeps turning out to matter.
    "recency" instead evicts whichever stored entry has gone *longest
    without being the nearest neighbour to anything* — i.e. whichever
    memory novelty_score() hasn't found relevant in the longest time —
    so a sound that keeps recurring (and so keeps getting matched,
    whether or not it's re-added) stays protected even under capacity
    pressure from many never-repeated sounds.
    """

    def __init__(self, max_embeddings: int, eviction: str = "fifo") -> None:
        """Create an empty store with a fixed capacity.

        Args:
            max_embeddings: Maximum number of embeddings to retain; an
                entry is evicted once the store is full, per `eviction`.
            eviction: "fifo" (evict oldest-added, the original/default
                behaviour) or "recency" (evict whichever entry has gone
                longest without being matched by novelty_score()).
        """
        if eviction not in ("fifo", "recency"):
            raise ValueError(f"unknown eviction policy: {eviction!r}")
        self.max_embeddings = max_embeddings
        self.eviction = eviction
        self._items: list[np.ndarray] = []
        self._last_matched: list[int] = []  # only meaningful for eviction="recency"
        self._tick = 0

    def add(self, embedding: np.ndarray) -> None:
        """Add a new MFCC embedding to the store.

        Args:
            embedding: 1-D feature vector, e.g. from audio.extract_mfcc.
        """
        self._tick += 1
        self._items.append(np.asarray(embedding, dtype=np.float32))
        self._last_matched.append(self._tick)
        if len(self._items) > self.max_embeddings:
            evict_idx = 0 if self.eviction == "fifo" else int(np.argmin(self._last_matched))
            self._items.pop(evict_idx)
            self._last_matched.pop(evict_idx)

    def novelty_score(self, embedding: np.ndarray) -> float:
        """Return the distance from `embedding` to its nearest stored entry.

        Use cosine distance (1 - cosine similarity), not raw Euclidean
        distance — it's what the dream loop's surprise scoring uses
        elsewhere in this codebase, and config.py's thresholds
        (NOVELTY_THRESHOLD_LOW/HIGH, RECOGNITION_THRESHOLD) are calibrated
        assuming the same metric throughout.

        Under eviction="recency", calling this also refreshes the
        matched entry's protection from eviction — querying "do I
        recognise this" is itself what keeps a memory alive, independent
        of whether the caller goes on to call add() for it.

        Args:
            embedding: 1-D feature vector to score.

        Returns:
            Distance to nearest neighbour, or 1.0 (maximally novel) if the
            store is empty.
        """
        self._tick += 1
        if not self._items:
            return 1.0
        nn = NearestNeighbors(n_neighbors=1, metric="cosine")
        nn.fit(np.stack(self._items))
        query = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
        distance, index = nn.kneighbors(query, n_neighbors=1)
        if self.eviction == "recency":
            self._last_matched[int(index[0, 0])] = self._tick
        return float(distance[0, 0])

    def sample_pair(self, rng: np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Sample two stored embeddings at random, for the dream loop.

        Args:
            rng: Optional caller-owned generator for reproducible replay.
                Omission retains nondeterministic interactive sampling.

        Returns:
            A pair of distinct stored embeddings.

        Raises:
            ValueError: If fewer than two embeddings are stored.
        """
        if len(self._items) < 2:
            raise ValueError("need at least two stored embeddings to sample a pair")
        rng = rng if rng is not None else np.random.default_rng()
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
