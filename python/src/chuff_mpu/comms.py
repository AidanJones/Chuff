"""Two-unit emergent acoustic comms (phase 2). Mirrors the esp32 track's
(not yet implemented) comms module.

Acoustic-only channel between two Chuff units: synthesises a repertoire of
tones, plays the closest-matching call for a novel sound, listens for an
acknowledgement tone from the other unit, and reinforces/decays calls
based on whether acknowledgement was heard. See
shared/comms_protocol.md for the full design.

`Repertoire` is deliberately a stateful class (not free functions) so a
single unit's call-to-context weighting can be exercised and graded
without needing real audio hardware — see eval/tasks/sound_memory_and_voice/
(which validated this exact design) and sim/ (which exercises it against
real synthesised audio).

`play_call()` always routes through audio_safety.safe_play() — see that
module's docstring for why every speaker-bound sample goes through one
choke point.
"""

from __future__ import annotations

import json

import numpy as np

from chuff_mpu import audio_safety

# How similar a new context must be (cosine similarity) to a previously
# learned context before that learning is allowed to influence the score
# for the new context at all. Below this, the exemplar is irrelevant —
# without this gate, strong reinforcement/decay learned for one context
# can leak into an unrelated context's preference just because both
# contexts share the same handful of calls to choose from.
_RELEVANCE_SIM_THRESHOLD = 0.5
# How close (cosine distance) a context must be to an existing exemplar's
# context before an update is treated as "the same context again"
# (update it) rather than "a new context for this call" (add a new
# exemplar).
_SAME_CONTEXT_DISTANCE = 0.1


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1e-9
    return float(np.dot(a, b) / denom)


class Repertoire:
    """A single unit's call repertoire plus its learned call-to-context weights."""

    def __init__(
        self,
        size: int,
        freq_min_hz: float,
        freq_max_hz: float,
        seed: int,
    ) -> None:
        """Generate a fixed-seed starting repertoire of call frequencies.

        After construction, `self.frequencies` holds a NumPy array of
        `size` call frequencies in Hz, each within
        [freq_min_hz, freq_max_hz] — this is the public contract other
        code (and the grader) relies on, independent of how weights are
        stored internally.

        Args:
            size: Number of calls in the repertoire.
            freq_min_hz: Lower bound of the frequency range.
            freq_max_hz: Upper bound of the frequency range.
            seed: RNG seed so two units start with overlapping-but-not-
                identical repertoires.
        """
        rng = np.random.default_rng(seed)
        # Deliberately NOT sorted by frequency: closest_call()'s tie-break
        # favours low call_index when nothing's been learned yet (see
        # _base_default), so index order correlating with frequency order
        # would silently bias every "no learning yet" choice toward one
        # end of the frequency range — a confound discovered while
        # testing the Acoustic Adaptation Hypothesis in sim/, see
        # shared/research_backlog.md Theme 1.2.
        self.frequencies = rng.uniform(freq_min_hz, freq_max_hz, size=size)
        # Per call: list of [context_vector, weight] exemplars, learned at runtime.
        self._exemplars: list[list] = [[] for _ in range(size)]

    def play_call(self, call_index: int, duration_ms: int, sample_rate: int) -> None:
        """Synthesise and play one repertoire call through the speaker.

        Args:
            call_index: Index into this repertoire's frequencies.
            duration_ms: Tone duration in milliseconds.
            sample_rate: Output sample rate in Hz.
        """
        freq = float(self.frequencies[call_index])
        n = int(sample_rate * duration_ms / 1000)
        t = np.linspace(0, duration_ms / 1000, n, endpoint=False)
        tone = 0.6 * np.sin(2 * np.pi * freq * t).astype(np.float32)
        audio_safety.safe_play(tone, sample_rate)  # safety-limited regardless of amplitude above

    def _base_default(self, call_index: int) -> float:
        # Tiny deterministic tie-break so an untried repertoire still
        # picks consistently (lower index very slightly favoured) rather
        # than relying on hidden randomness.
        return 0.5 - 0.001 * call_index

    def closest_call(self, context_embedding: np.ndarray) -> int:
        """Pick the call this unit currently prefers for a given context.

        Deterministic when the learned weighting for `context_embedding`
        hasn't changed since the last call. Calls previously
        `reinforce()`d for a similar context keep winning for it; calls
        previously `decay()`ed for a similar context eventually lose to
        something else. Learning tied to one context doesn't leak into a
        dissimilar one, via the relevance gate in `_score()`.

        Args:
            context_embedding: MFCC embedding of the sound that triggered
                this call.

        Returns:
            Index into this repertoire of the call to play.
        """
        best_idx, best_score = 0, float("-inf")
        for idx in range(len(self.frequencies)):
            score = self._score(idx, context_embedding)
            if score > best_score:
                best_idx, best_score = idx, score
        return best_idx

    def _score(self, call_index: int, context_embedding: np.ndarray) -> float:
        score = self._base_default(call_index)
        for ctx, weight in self._exemplars[call_index]:
            sim = _cosine_similarity(context_embedding, ctx)
            if sim > _RELEVANCE_SIM_THRESHOLD:
                score += weight * sim
        return score

    def _update(self, call_index: int, context_embedding: np.ndarray, delta: float) -> None:
        context_embedding = np.asarray(context_embedding, dtype=np.float32)
        for pair in self._exemplars[call_index]:
            ctx, weight = pair
            if 1 - _cosine_similarity(context_embedding, ctx) < _SAME_CONTEXT_DISTANCE:
                pair[1] = weight + delta * (1 - weight) if delta >= 0 else weight + delta * (1 + weight)
                return
        self._exemplars[call_index].append([context_embedding, float(np.clip(delta, -1.0, 1.0))])

    def reinforce(self, call_index: int, context_embedding: np.ndarray, rate: float) -> None:
        """Increase the weight of a call for a given context after acknowledgement."""
        self._update(call_index, context_embedding, rate)

    def decay(self, call_index: int, context_embedding: np.ndarray, rate: float) -> None:
        """Decrease the weight of a call for a given context after silence."""
        self._update(call_index, context_embedding, -rate)


def log_event(path: str, event: dict) -> None:
    """Append one JSONL event (call played/ack received/etc.) to the log file.

    Args:
        path: Path to the JSONL log file.
        event: JSON-serialisable event dict.
    """
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")
