"""Two-unit emergent acoustic comms (phase 2). Mirrors esp32/comms.py.

Acoustic-only channel between two Chuff units: synthesises a repertoire of
tones, plays the closest-matching call for a novel sound, listens for an
acknowledgement tone from the other unit, and reinforces/decays calls
based on whether acknowledgement was heard. See
shared/comms_protocol.md for the full design.

`Repertoire` is deliberately a stateful class (not free functions) so a
single unit's call-to-context weighting can be exercised and graded
without needing real audio hardware — see eval/tasks/sound_memory_and_voice/.
"""

from __future__ import annotations

import numpy as np


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

        After construction, `self.frequencies` must hold a NumPy array of
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
        raise NotImplementedError

    def play_call(self, call_index: int, duration_ms: int, sample_rate: int) -> None:
        """Synthesise and play one repertoire call through the speaker.

        Args:
            call_index: Index into this repertoire's frequencies.
            duration_ms: Tone duration in milliseconds.
            sample_rate: Output sample rate in Hz.
        """
        raise NotImplementedError

    def closest_call(self, context_embedding: np.ndarray) -> int:
        """Pick the call this unit currently prefers for a given context.

        Should be deterministic when the learned weighting for
        `context_embedding` hasn't changed since the last call (repeated
        calls with an unchanged repertoire return the same index), and
        should be more likely to favour calls previously reinforced for a
        similar context, less likely to favour calls previously decayed
        for a similar context.

        Args:
            context_embedding: MFCC embedding of the sound that triggered
                this call.

        Returns:
            Index into this repertoire of the call to play.
        """
        raise NotImplementedError

    def reinforce(self, call_index: int, context_embedding: np.ndarray, rate: float) -> None:
        """Increase the weight of a call for a given context after acknowledgement."""
        raise NotImplementedError

    def decay(self, call_index: int, context_embedding: np.ndarray, rate: float) -> None:
        """Decrease the weight of a call for a given context after silence."""
        raise NotImplementedError


def log_event(path: str, event: dict) -> None:
    """Append one JSONL event (call played/ack received/etc.) to the log file.

    Args:
        path: Path to the JSONL log file.
        event: JSON-serialisable event dict.
    """
    raise NotImplementedError
