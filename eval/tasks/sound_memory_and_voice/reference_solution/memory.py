"""Reference solution for memory.py — used only to validate the grader
itself (it should score close to 1.0 on the sound_memory dimension).
Not part of the Chuff application; not imported by anything except
harness/agents/reference_agent.py.
"""

from __future__ import annotations

import numpy as np


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1e-9
    return float(1.0 - np.dot(a, b) / denom)


class EmbeddingStore:
    def __init__(self, max_embeddings: int) -> None:
        self.max_embeddings = max_embeddings
        self._items: list[np.ndarray] = []

    def add(self, embedding: np.ndarray) -> None:
        self._items.append(np.asarray(embedding, dtype=np.float32))
        if len(self._items) > self.max_embeddings:
            self._items.pop(0)

    def novelty_score(self, embedding: np.ndarray) -> float:
        if not self._items:
            return 1.0
        return min(_cosine_distance(embedding, item) for item in self._items)

    def sample_pair(self) -> tuple[np.ndarray, np.ndarray]:
        if len(self._items) < 2:
            raise ValueError("need at least two stored embeddings")
        rng = np.random.default_rng()
        i, j = rng.choice(len(self._items), size=2, replace=False)
        return self._items[i], self._items[j]

    def save(self, path: str) -> None:
        np.savez(path, *self._items)

    def load(self, path: str) -> None:
        data = np.load(path)
        self._items = [data[key] for key in data.files]

    def __len__(self) -> int:
        return len(self._items)
