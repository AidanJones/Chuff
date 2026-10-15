"""Online acoustic foreground discovery; no event labels or clean stems.

This experimental learner detects spectral excess over estimated ambience,
keeps temporal structure, and requires separated recurrences before committing
a prototype. It is deliberately separate from the legacy MFCC/voice contract.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import stft


@dataclass(frozen=True)
class AttentionConfig:
    sample_rate: int = 16000
    n_fft: int = 512
    hop: int = 160
    bands: int = 32
    warmup_clips: int = 3
    background_rate: float = 0.15
    subtraction: float = 2.0
    min_excess_ratio: float = 0.8
    min_duration_s: float = 0.06
    max_duration_s: float = 0.75
    max_flatness: float = 0.35
    match_distance: float = 0.18
    confirmations: int = 2
    candidate_ttl: int = 40
    capacity: int = 32

    def __post_init__(self) -> None:
        if self.sample_rate < 4000 or self.n_fft < 32 or not 0 < self.hop <= self.n_fft:
            raise ValueError("invalid sample rate or FFT/hop sizes")
        if self.bands < 4 or self.warmup_clips < 1 or self.confirmations < 1:
            raise ValueError("bands, warmup and confirmations must be positive")
        if self.capacity < 1 or self.candidate_ttl < 1:
            raise ValueError("memory capacity and candidate lifetime must be positive")
        if not 0 < self.background_rate <= 1 or not 0 < self.match_distance < 2:
            raise ValueError("invalid background rate or match distance")
        if not 0 < self.min_duration_s < self.max_duration_s or self.subtraction < 1:
            raise ValueError("invalid event duration or subtraction factor")
        if not 0 < self.max_flatness <= 1 or self.min_excess_ratio <= 0:
            raise ValueError("invalid event detection thresholds")


@dataclass(frozen=True)
class Observation:
    status: str
    salience: float = 0.0
    novelty: float | None = None
    prototype_id: int | None = None
    embedding: np.ndarray | None = None


@dataclass(eq=False)
class _Pattern:
    embedding: np.ndarray
    hits: int
    last_seen: int
    identifier: int


def _distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.clip(1 - np.dot(a, b), 0, 2))


class ForegroundLearner:
    """Bounded prototype memory with an adaptive, mixture-only input gate.

    `candidate` means attention without long-term admission; `learned` is a
    new persistent pattern; `recognised` is habituation to a stored pattern.
    Continuous sounds and broadband transients are intentionally conservative
    cases here, not solved semantic signal/noise separation.
    """

    def __init__(self, settings: AttentionConfig | None = None) -> None:
        self.settings = settings or AttentionConfig()
        c = self.settings
        frequencies = np.fft.rfftfreq(c.n_fft, 1 / c.sample_rate)
        # Triangular mel bands, without a runtime dependency on librosa defaults.
        mel = lambda f: 2595 * np.log10(1 + f / 700)
        edges = 700 * (10 ** (np.linspace(mel(100), mel(c.sample_rate / 2), c.bands + 2) / 2595) - 1)
        self._filters = np.array([
            np.maximum(0, np.minimum((frequencies - a) / (b - a), (d - frequencies) / (d - b)))
            for a, b, d in zip(edges[:-2], edges[1:-1], edges[2:])
        ])
        self._background: np.ndarray | None = None
        self._tick = 0
        self._next_id = 0
        self._patterns: list[_Pattern] = []
        self._candidates: list[_Pattern] = []

    @property
    def memory_size(self) -> int:
        return len(self._patterns)

    @property
    def candidate_count(self) -> int:
        return len(self._candidates)

    def observe(self, clip: np.ndarray) -> Observation:
        """Consume one mono window; only this waveform reaches the learner.

        Windows must have the configured sample rate and a consistent duration.
        Initial windows calibrate ambience; events during calibration may be
        missed. No simulator warmup labels are inspected.
        """
        c = self.settings
        clip = np.asarray(clip, dtype=np.float64)
        if clip.ndim != 1 or len(clip) < c.n_fft or not np.isfinite(clip).all():
            raise ValueError("expected a finite mono clip at least one FFT window long")
        self._tick += 1
        self._candidates = [p for p in self._candidates if self._tick - p.last_seen <= c.candidate_ttl]
        _, _, spectrum = stft(clip, fs=c.sample_rate, nperseg=c.n_fft,
                              noverlap=c.n_fft - c.hop, boundary=None, padded=False)
        power = np.abs(spectrum) ** 2
        # A low within-window quantile is robust to short foreground bursts.
        # Taking the maximum with history rejects a newly sustained background
        # instead of recording every changed fan as a new foreground identity.
        # Gaussian-noise periodogram power is approximately exponential:
        # quantile(0.3) = -log(0.7) * mean. An uncorrected low quantile would
        # underestimate noise and mark ordinary background frames as active.
        floor = np.maximum(np.quantile(power, 0.3, axis=1) / -np.log(0.7), 1e-12)
        if self._background is None:
            self._background = floor.copy()
        background = np.maximum(self._background, floor)
        self._background += c.background_rate * (floor - self._background)
        if self._tick <= c.warmup_clips:
            return Observation("calibrating")
        excess = np.maximum(power - c.subtraction * background[:, None], 0)
        band_excess = self._filters @ excess
        band_background = self._filters @ background
        ratio = band_excess.sum(axis=0) / max(float(band_background.sum()), 1e-12)
        active = ratio >= c.min_excess_ratio
        salience = float(np.max(ratio))
        indices = np.flatnonzero(active)
        if not len(indices):
            return Observation("background", salience)
        duration = len(indices) * c.hop / c.sample_rate
        span = (indices[-1] - indices[0] + 1) * c.hop / c.sample_rate
        if len(indices) < 3 or duration < c.min_duration_s or span > c.max_duration_s:
            return Observation("background", salience)
        profile = band_excess[:, active].mean(axis=1) + 1e-12
        flatness = float(np.exp(np.log(profile).mean()) / profile.mean())
        if flatness > c.max_flatness:
            return Observation("unstructured", salience)
        # Three ordered spectral summaries distinguish a rise from a fall.
        region = band_excess[:, indices[0]:indices[-1] + 1]
        features = []
        for segment in np.array_split(region, 3, axis=1):
            vector = segment.mean(axis=1)
            vector = np.sqrt(vector / max(float(vector.sum()), 1e-12))
            features.append(vector)
        embedding = np.concatenate(features)
        embedding /= max(float(np.linalg.norm(embedding)), 1e-12)
        return self._remember(embedding.astype(np.float32), salience)

    def _remember(self, embedding: np.ndarray, salience: float) -> Observation:
        c = self.settings
        distances = [_distance(p.embedding, embedding) for p in self._patterns]
        novelty = min(distances, default=1.0)
        if distances and novelty < c.match_distance:
            pattern = self._patterns[int(np.argmin(distances))]
            pattern.hits += 1
            pattern.last_seen = self._tick
            # Keep the admitted prototype fixed: no gradual drift into noise.
            return Observation("recognised", salience, novelty, pattern.identifier, embedding)
        distances = [_distance(p.embedding, embedding) for p in self._candidates]
        if distances and min(distances) < c.match_distance:
            pattern = self._candidates[int(np.argmin(distances))]
            # Adjacent windows of one ongoing sound are not two recurrences.
            if self._tick - pattern.last_seen >= 2:
                pattern.hits += 1
            pattern.last_seen = self._tick
        else:
            pattern = _Pattern(embedding.copy(), 1, self._tick, -1)
            if len(self._candidates) >= c.capacity:
                self._candidates.remove(min(self._candidates, key=lambda p: p.last_seen))
            self._candidates.append(pattern)
        if pattern.hits < c.confirmations:
            return Observation("candidate", salience, novelty, embedding=embedding)
        self._candidates.remove(pattern)
        if len(self._patterns) >= c.capacity:
            self._patterns.remove(min(self._patterns, key=lambda p: p.last_seen))
        pattern.identifier = self._next_id
        self._next_id += 1
        self._patterns.append(pattern)
        return Observation("learned", salience, novelty, pattern.identifier, embedding)
