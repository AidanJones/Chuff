"""ChuffAgent: the listen -> remember -> repeat loop, end to end.

Wraps the real (not stub) audio.py / memory.py / comms.py modules —
this is the same production code path a real UNO Q would run, just fed
synthetic audio from sim/world.py instead of a microphone.

Every waveform this agent "says" passes through audio_safety before
anything else touches it, whether or not real playback is enabled.
"""

from __future__ import annotations

import numpy as np

from chuff_mpu import audio, audio_safety, comms, config, memory
from chuff_mpu.sim import tones


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1e-9
    return float(1.0 - np.dot(a, b) / denom)


class ChuffAgent:
    def __init__(self, seed: int = 1) -> None:
        self.memory = memory.EmbeddingStore(max_embeddings=config.MAX_EMBEDDINGS)
        self.voice = comms.Repertoire(
            size=config.REPERTOIRE_SIZE,
            freq_min_hz=config.CALL_FREQ_MIN_HZ,
            freq_max_hz=config.CALL_FREQ_MAX_HZ,
            seed=seed,
        )
        # Dry (noise-free) reference embedding per call, computed lazily the
        # first time each call is used — what "hearing that call cleanly"
        # looks like, so the agent has something to compare a noisy echo of
        # its own voice against.
        self._call_reference: dict[int, np.ndarray] = {}

    def hear(self, clip: np.ndarray) -> tuple[np.ndarray, float, bool]:
        """Process one heard clip end to end: extract -> score -> maybe learn.

        Returns:
            (embedding, novelty_score, learned) — `learned` is True iff
            this clip was novel enough to be committed to memory.
        """
        embedding = audio.extract_mfcc(clip, config.SAMPLE_RATE, config.N_MFCC)
        score = self.memory.novelty_score(embedding)
        learned = score >= config.NOVELTY_THRESHOLD_HIGH
        if learned:
            self.memory.add(embedding)
        return embedding, score, learned

    def respond(self, context_embedding: np.ndarray) -> tuple[int, float, np.ndarray]:
        """Pick the closest call in its own voice for this context.

        Returns (call_index, frequency_hz, safety-limited waveform) — the
        waveform is always guardrail-limited, whether or not it's
        actually played through a real speaker (see `speak()`).
        """
        call_index = self.voice.closest_call(context_embedding)
        freq = float(self.voice.frequencies[call_index])
        duration_s = config.CALL_DURATION_MS / 1000
        raw = tones.sine_tone(freq, duration_s, config.SAMPLE_RATE, amplitude=0.6)
        safe_wave = audio_safety.limit_audio(raw, config.SAMPLE_RATE)
        if call_index not in self._call_reference:
            self._call_reference[call_index] = audio.extract_mfcc(
                safe_wave, config.SAMPLE_RATE, config.N_MFCC
            )
        return call_index, freq, safe_wave

    def babble(self, rate: float | None = None) -> None:
        """Practice every repertoire call once, with no external trigger,
        before any real sound is ever heard (ch-53).

        Motivated by the 2024 zebra finch finding that active babbling —
        not passive listening — opens the critical window for template
        formation (shared/research_backlog.md Theme 1.1). A naive
        "play each call once" babble phase would do nothing observable
        here, because `respond()` already lazily caches a call's
        self-reference the moment it's first used in a real round — so
        this method also *self-reinforces* each call against its own
        clean signature, giving every call a weak "home" association
        before any real context competes for it. That's the mechanism
        that can actually change later behaviour: a real sound that
        happens to be acoustically close to one of these self-anchors
        gets a head start toward that call, rather than every first
        encounter being decided purely by index-order tie-breaking.
        """
        rate = config.BABBLE_REINFORCEMENT_RATE if rate is None else rate
        for call_index in range(len(self.voice.frequencies)):
            freq = float(self.voice.frequencies[call_index])
            duration_s = config.CALL_DURATION_MS / 1000
            raw = tones.sine_tone(freq, duration_s, config.SAMPLE_RATE, amplitude=0.6)
            safe_wave = audio_safety.limit_audio(raw, config.SAMPLE_RATE)
            reference = audio.extract_mfcc(safe_wave, config.SAMPLE_RATE, config.N_MFCC)
            self._call_reference[call_index] = reference
            self.voice.reinforce(call_index, reference, rate)

    def speak(self, waveform: np.ndarray, play: bool = False) -> np.ndarray:
        """Re-apply the safety limiter (cheap, idempotent) and optionally
        actually play `waveform`. Always returns the limited waveform."""
        if play:
            return audio_safety.safe_play(waveform, config.SAMPLE_RATE)
        return audio_safety.limit_audio(waveform, config.SAMPLE_RATE)

    def recognise_own_echo(self, call_index: int, echo_clip: np.ndarray, threshold: float | None = None) -> bool:
        """Did the agent's own call survive the noise well enough to still
        sound like itself? Stands in for a second unit's acknowledgement
        (shared/comms_protocol.md) until a real two-unit test exists —
        see TASKS.md CH-23/CH-36."""
        threshold = config.RECOGNITION_THRESHOLD if threshold is None else threshold
        echo_embedding = audio.extract_mfcc(echo_clip, config.SAMPLE_RATE, config.N_MFCC)
        reference = self._call_reference.get(call_index)
        if reference is None:
            return False
        return _cosine_distance(echo_embedding, reference) < threshold

    def reinforce_or_decay(self, call_index: int, context_embedding: np.ndarray, recognised: bool) -> None:
        """Nudge this call's weight for this context based on self-recognition."""
        if recognised:
            self.voice.reinforce(call_index, context_embedding, config.REINFORCEMENT_RATE)
        else:
            self.voice.decay(call_index, context_embedding, config.DECAY_RATE)
