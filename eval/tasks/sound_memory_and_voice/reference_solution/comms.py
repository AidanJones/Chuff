"""Reference solution for comms.py — used only to validate the grader
itself (it should score close to 1.0 on the communication dimension).
Not part of the Chuff application; not imported by anything except
harness/agents/reference_agent.py.
"""

from __future__ import annotations

import numpy as np

# How similar a new context must be (cosine similarity) to a previously
# learned context before that learning is allowed to influence the score
# for the new context at all. Below this, the exemplar is irrelevant.
_RELEVANCE_SIM_THRESHOLD = 0.5
# How close (cosine distance) a context must be to an existing exemplar's
# context before it's treated as "the same context again" for updating.
_SAME_CONTEXT_DISTANCE = 0.1


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1e-9
    return float(np.dot(a, b) / denom)


class Repertoire:
    def __init__(self, size: int, freq_min_hz: float, freq_max_hz: float, seed: int) -> None:
        rng = np.random.default_rng(seed)
        self.frequencies = np.sort(rng.uniform(freq_min_hz, freq_max_hz, size=size))
        # Per call: list of [context_vector, weight] exemplars.
        self._exemplars: list[list] = [[] for _ in range(size)]

    def play_call(self, call_index: int, duration_ms: int, sample_rate: int) -> None:
        # No real audio in the reference solution / grader context.
        return None

    def _base_default(self, call_index: int) -> float:
        # Tiny deterministic tie-break so an untried repertoire still picks
        # consistently (lower index very slightly favoured) rather than
        # relying on hidden randomness.
        return 0.5 - 0.001 * call_index

    def closest_call(self, context_embedding: np.ndarray) -> int:
        best_idx = 0
        best_score = float("-inf")
        for idx in range(len(self.frequencies)):
            score = self._base_default(idx)
            for ctx, weight in self._exemplars[idx]:
                sim = _cosine_similarity(context_embedding, ctx)
                if sim > _RELEVANCE_SIM_THRESHOLD:
                    score += weight * sim
            if score > best_score:
                best_score = score
                best_idx = idx
        return best_idx

    def _update(self, call_index: int, context_embedding: np.ndarray, delta: float) -> None:
        context_embedding = np.asarray(context_embedding, dtype=np.float32)
        for pair in self._exemplars[call_index]:
            ctx, weight = pair
            if 1 - _cosine_similarity(context_embedding, ctx) < _SAME_CONTEXT_DISTANCE:
                pair[1] = weight + delta * (1 - weight) if delta >= 0 else weight + delta * (1 + weight)
                return
        self._exemplars[call_index].append([context_embedding, float(np.clip(delta, -1.0, 1.0))])

    def reinforce(self, call_index: int, context_embedding: np.ndarray, rate: float) -> None:
        self._update(call_index, context_embedding, rate)

    def decay(self, call_index: int, context_embedding: np.ndarray, rate: float) -> None:
        self._update(call_index, context_embedding, -rate)


def log_event(path: str, event: dict) -> None:
    import json

    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")
