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

import json
import os
from pathlib import Path
import tempfile

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
        self._items.append(np.array(embedding, dtype=np.float32, copy=True))
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
            Independent copies of two distinct stored entries. Editing the
            returned arrays cannot change live memory or eviction history.

        Raises:
            ValueError: If fewer than two embeddings are stored.
        """
        if len(self._items) < 2:
            raise ValueError("need at least two stored embeddings to sample a pair")
        rng = rng if rng is not None else np.random.default_rng()
        i, j = rng.choice(len(self._items), size=2, replace=False)
        return self._items[i].copy(), self._items[j].copy()

    SNAPSHOT_SCHEMA = "chuff.legacy.embedding-store.v1"

    def _validated_vectors(self, vectors):
        if len(vectors) > self.max_embeddings:
            raise ValueError("snapshot exceeds configured capacity")
        items, dimensions = [], None
        for vector in vectors:
            vector = np.asarray(vector)
            if vector.ndim != 1 or not vector.size or vector.dtype.kind not in "fiu":
                raise ValueError("embeddings must be nonempty real numeric vectors")
            if dimensions is None:
                dimensions = vector.shape
            if vector.shape != dimensions or not np.isfinite(vector).all():
                raise ValueError("inconsistent or nonfinite embeddings")
            with np.errstate(over="ignore", invalid="ignore"):
                vector = np.array(vector, dtype=np.float32, copy=True)
            if not np.isfinite(vector).all():
                raise ValueError("embedding is out of float32 range")
            items.append(vector)
        return items

    @staticmethod
    def _validated_history(history, count, tick):
        if type(tick) is not int or not 0 <= tick <= np.iinfo(np.int64).max:
            raise ValueError("invalid snapshot clock")
        history = np.asarray(history)
        if history.shape != (count,) or history.dtype.kind not in "iu":
            raise ValueError("invalid recency history")
        if (history < 0).any() or (history > tick).any():
            raise ValueError("recency timestamp outside snapshot clock")
        if count and max(history.tolist()) > np.iinfo(np.int64).max:
            raise ValueError("recency timestamp outside supported range")
        return history.astype(np.int64).tolist()

    def save(self, path: str) -> None:
        """Atomically replace a versioned NPZ, including eviction history.

        Caller-owned sampling RNGs are not part of this store and are not
        serialized. Like np.savez, a missing .npz suffix is appended.
        """
        items = self._validated_vectors(self._items)
        history = np.asarray(self._last_matched) if self._last_matched else np.empty(0, dtype=np.int64)
        history = self._validated_history(history, len(items), self._tick)
        if type(self.max_embeddings) is not int or self.max_embeddings < 0:
            raise ValueError("snapshot capacity must be a nonnegative integer")
        metadata = json.dumps(dict(schema=self.SNAPSHOT_SCHEMA, version=1, eviction=self.eviction,
                                   tick=self._tick, capacity=self.max_embeddings), allow_nan=False)
        destination = Path(path)
        if not str(destination).endswith(".npz"):
            destination = destination.with_name(destination.name+".npz")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".npz", delete=False) as handle:
                temporary = Path(handle.name)
                np.savez_compressed(handle, metadata=np.array(metadata),
                    features=np.stack(items) if items else np.empty((0, 0), dtype=np.float32),
                    last_matched=np.asarray(history, dtype=np.int64))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def load(self, path: str) -> None:
        """Validate a versioned snapshot or legacy NPZ transactionally.

        Versioned snapshots restore the clock/history and require the same
        eviction policy. The receiving store's configured capacity remains
        authoritative. Legacy positional arrays reset recency to insertion
        order. All validation finishes before live state is replaced.
        """
        archive = np.load(path, allow_pickle=False)
        if not hasattr(archive, "files"):
            raise ValueError("expected an NPZ embedding snapshot")
        with archive as data:
            if len(data.files) == 3 and set(data.files) == {"metadata", "features", "last_matched"}:
                metadata = json.loads(str(data["metadata"].item()))
                expected_keys = {"schema", "version", "eviction", "tick", "capacity"}
                if not isinstance(metadata, dict) or set(metadata) != expected_keys:
                    raise ValueError("invalid snapshot metadata")
                if metadata["schema"] != self.SNAPSHOT_SCHEMA or type(metadata["version"]) is not int or metadata["version"] != 1:
                    raise ValueError("unsupported embedding snapshot schema/version")
                if metadata["eviction"] != self.eviction:
                    raise ValueError("snapshot eviction policy differs from receiving store")
                features = data["features"]
                if (features.ndim != 2 or features.dtype.kind not in "fiu"
                        or (len(features) == 0 and features.shape[1] != 0)):
                    raise ValueError("invalid snapshot feature matrix")
                if type(metadata["capacity"]) is not int or metadata["capacity"] < len(features):
                    raise ValueError("invalid saved capacity")
                items = self._validated_vectors(features)
                tick = metadata["tick"]
                history = self._validated_history(data["last_matched"], len(items), tick)
            else:
                count = len(data.files)
                expected = [f"arr_{i}" for i in range(count)]
                if set(data.files) != set(expected):
                    raise ValueError("invalid legacy memory keys")
                items = self._validated_vectors([data[key] for key in expected])
                history, tick = list(range(1, len(items)+1)), len(items)
        # Commit only after the file has closed and every vector has passed.
        self._items = items
        self._last_matched = history
        self._tick = tick

    def __len__(self) -> int:
        """Number of embeddings currently stored (<= max_embeddings)."""
        return len(self._items)
