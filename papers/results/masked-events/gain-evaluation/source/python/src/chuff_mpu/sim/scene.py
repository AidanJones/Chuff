"""Seeded continuous-room simulation, with truth kept separate from audio.

Procedural events are not real-world recordings. Room echoes and microphone
colouration are signal-processing approximations, not calibrated acoustics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, fftconvolve, lfilter


EVENTS = ("double_chime", "rising_call", "falling_call")


@dataclass(frozen=True)
class SceneConfig:
    sample_rate: int = 16000
    duration_s: float = 1.0
    rounds: int = 120
    warmup: int = 6
    event_probability: float = 0.35
    nuisance_probability: float = 0.12
    snr_db: float = 6.0
    background_rms: float = 0.025
    room_decay_s: float = 0.12
    background_change: bool = True
    structured_distractors: bool = False

    def __post_init__(self) -> None:
        if self.sample_rate < 8000 or self.duration_s < 0.8:
            raise ValueError("scene needs >=8 kHz audio and >=0.8 second windows")
        if self.rounds < 1 or not 0 <= self.warmup < self.rounds:
            raise ValueError("warmup must be shorter than a positive run length")
        if not 0 <= self.event_probability <= 1 or not 0 <= self.nuisance_probability <= 1:
            raise ValueError("event probabilities must be in [0, 1]")
        if self.background_rms <= 0 or not 0 <= self.room_decay_s <= 1:
            raise ValueError("invalid background level or room decay")
        if not np.isfinite(self.snr_db) or not -30 <= self.snr_db <= 30:
            raise ValueError("SNR must be finite and between -30 and 30 dB")


@dataclass(frozen=True)
class SceneFrame:
    audio: np.ndarray
    label: str | None
    nuisance: bool
    background_changed: bool
    active_snr_db: float | None
    event_start_s: float | None = None
    event_stop_s: float | None = None
    previous_source_tail_s: float = 0.0


def _rms(signal: np.ndarray) -> float:
    return float(np.sqrt(np.mean(signal ** 2)))


class AcousticScene:
    """Independent seeded RNGs for schedule, source, background and channel.

    Sampling backgrounds or rendering a different event never changes future
    event scheduling. All learners in a comparison receive the same frames.
    """

    def __init__(self, seed: int = 0, settings: SceneConfig | None = None) -> None:
        self.settings = settings or SceneConfig()
        self.event_labels = EVENTS
        children = np.random.SeedSequence(seed).spawn(4)
        self.schedule_rng, self.source_rng, self.background_rng, channel_rng = [
            np.random.default_rng(child) for child in children
        ]
        c = self.settings
        self._step = 0
        self._source_tail_until_s = 0.0
        self._noise_state = np.zeros(1)
        self._mic_b, self._mic_a = butter(2, [100, 0.40 * c.sample_rate],
                                         btype="bandpass", fs=c.sample_rate)
        self._mic_state = np.zeros(max(len(self._mic_a), len(self._mic_b)) - 1)
        length = max(1, int(c.room_decay_s * c.sample_rate))
        self._ir = np.zeros(length)
        self._ir[0] = 1.0
        for delay, amplitude in [(0.018, 0.3), (0.043, -0.18), (0.079, 0.1)]:
            index = int(delay * c.sample_rate)
            if index < length:
                self._ir[index] += amplitude * channel_rng.uniform(0.8, 1.2)
        # Sparse exponentially decaying late reflections.
        for index in range(int(0.025 * c.sample_rate), length, max(1, c.sample_rate // 300)):
            self._ir[index] += channel_rng.normal() * 0.06 * np.exp(-6.9 * index / length)
        self._tail = np.zeros(length - 1)
        self._hum_phase = float(channel_rng.uniform(0, 2 * np.pi))

    def _background(self, n: int) -> np.ndarray:
        c = self.settings
        t = (self._step * n + np.arange(n)) / c.sample_rate
        white = self.background_rng.normal(size=n)
        coloured, self._noise_state = lfilter([0.12], [1, -0.88], white, zi=self._noise_state)
        noise = 0.65 * coloured / 0.253 + 0.35 * white
        hum = 0.32 * np.sin(2 * np.pi * 100 * t + self._hum_phase)
        hum += 0.12 * np.sin(2 * np.pi * 200 * t + self._hum_phase)
        drift = 1 + 0.2 * np.sin(2 * np.pi * 0.035 * t + self._hum_phase)
        background = (noise + hum) * drift
        if c.background_change and self._step >= c.rounds // 2:
            background += 0.7 * np.sin(2 * np.pi * 340 * t + self._hum_phase)
            background *= 1.35
        return background * c.background_rms

    def _event(self, label: str) -> np.ndarray:
        c = self.settings
        rng = self.source_rng
        duration = rng.uniform(0.28, 0.42)
        t = np.arange(int(duration * c.sample_rate)) / c.sample_rate
        scale = rng.uniform(0.96, 1.04)
        if label == "double_chime":
            envelope = np.exp(-18 * t) + (t >= duration / 2) * np.exp(-22 * np.maximum(0, t - duration / 2))
            event = (np.sin(2 * np.pi * 950 * scale * t)
                     + 0.45 * np.sin(2 * np.pi * 1430 * scale * t)) * envelope
        else:
            start, end = (650, 2400) if label == "rising_call" else (2400, 650)
            phase = 2 * np.pi * scale * (start * t + (end - start) * t ** 2 / (2 * duration))
            event = (np.sin(phase) + 0.2 * np.sin(2 * phase)) * np.sin(np.pi * t / duration) ** 2
        # Soft onset/offset to avoid synthesiser clicks.
        ramp = min(int(0.008 * c.sample_rate), len(event) // 2)
        event[:ramp] *= np.linspace(0, 1, ramp)
        event[-ramp:] *= np.linspace(1, 0, ramp)
        return event

    def next_frame(self) -> SceneFrame:
        c = self.settings
        if self._step >= c.rounds:
            raise StopIteration
        n = int(c.duration_s * c.sample_rate)
        background = self._background(n)
        signal = np.zeros(n)
        label = None
        nuisance = False
        active_snr = None
        event_start = event_stop = None
        previous_tail_s = max(0., self._source_tail_until_s-self._step*c.duration_s)
        if self._step >= c.warmup:
            if self.schedule_rng.random() < c.event_probability:
                label = str(self.schedule_rng.choice(self.event_labels))
            else:
                nuisance = self.schedule_rng.random() < c.nuisance_probability
            if label is not None or nuisance:
                if label is not None:
                    event = self._event(label)
                else:
                    if c.structured_distractors and self.source_rng.random() < 0.5:
                        # One-off tonal sequences: salient but not recurring
                        # identities. Unlike impacts these pass the onset gate,
                        # allowing the confirmation stage to be tested separately.
                        length = int(c.sample_rate * self.source_rng.uniform(0.2, 0.4))
                        frequencies = np.repeat(self.source_rng.uniform(400, 4500, 3),
                                                (length + 2) // 3)[:length]
                        phase = 2 * np.pi * np.cumsum(frequencies) / c.sample_rate
                        event = np.sin(phase) * np.hanning(length)
                    else:
                        # Random broadband impacts, not a reusable foreground class.
                        event = self.source_rng.normal(size=int(c.sample_rate * 0.035))
                        event *= np.exp(-np.linspace(0, 5, len(event)))
                available_s = c.duration_s - len(event) / c.sample_rate
                margin_s = min(0.12, available_s / 3)
                start = int(self.source_rng.uniform(margin_s, available_s - margin_s) * c.sample_rate)
                section = slice(start, start + len(event))
                target_rms = _rms(background[section]) * 10 ** (c.snr_db / 20)
                signal[section] = event * target_rms / max(_rms(event), 1e-12)
                active_snr = float(20 * np.log10(_rms(signal[section]) / _rms(background[section])))
                if label is not None:
                    event_start, event_stop = start/c.sample_rate, (start+len(event))/c.sample_rate
                self._source_tail_until_s = (self._step*c.duration_s
                    + (start+len(event)+len(self._ir)-1)/c.sample_rate)
        # Continuous overlap-add room response and persistent microphone filter.
        convolved = fftconvolve(background + signal, self._ir)
        if len(self._tail):
            convolved[:len(self._tail)] += self._tail
        self._tail = convolved[n:].copy()
        mixture, self._mic_state = lfilter(self._mic_b, self._mic_a, convolved[:n], zi=self._mic_state)
        changed = c.background_change and self._step == c.rounds // 2
        self._step += 1
        return SceneFrame(mixture.astype(np.float32), label, nuisance, changed, active_snr,
                          event_start, event_stop, previous_tail_s)

    def __iter__(self):
        while self._step < self.settings.rounds:
            yield self.next_frame()
