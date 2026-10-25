"""Online acoustic foreground discovery; no event labels or clean stems.

This experimental learner detects spectral excess over estimated ambience,
keeps temporal structure, and requires separated recurrences before committing
a prototype. It is deliberately separate from the legacy MFCC/voice contract.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
import os
from pathlib import Path
import tempfile

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
    region_start_s: float | None = None
    region_stop_s: float | None = None


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

    FEATURE_SCHEMA = "chuff.foreground.mel-temporal-excess.v1"
    SETTINGS_CLASS = AttentionConfig

    def __init__(self, settings: AttentionConfig | None = None) -> None:
        self.settings = settings or self.SETTINGS_CLASS()
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

    def save(self, path: str | Path) -> None:
        """Atomically persist admitted patterns, not pending candidates or room noise.

        The schema is specific to this frontend. On restart the room estimator,
        pending confirmations, window counter and controller timers recalibrate.
        Prototype IDs, hit counts and eviction ordering survive the restart.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        patterns = sorted(self._patterns, key=lambda pattern: pattern.last_seen)
        metadata = json.dumps({"schema": self.FEATURE_SCHEMA,
                               "settings": asdict(self.settings), "next_id": self._next_id}, allow_nan=False)
        features = np.stack([p.embedding for p in patterns]) if patterns else np.empty((0, 3 * self.settings.bands))
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
                temporary = Path(handle.name)
                np.savez_compressed(handle, metadata=np.array(metadata), features=features,
                                    identifiers=np.array([p.identifier for p in patterns], dtype=np.int64),
                                    hits=np.array([p.hits for p in patterns], dtype=np.int64))
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    @classmethod
    def load(cls, path: str | Path) -> ForegroundLearner:
        """Read a validated, non-pickle foreground snapshot; reject legacy MFCC files."""
        try:
            with np.load(path, allow_pickle=False) as data:
                if set(data.files) != {"metadata", "features", "identifiers", "hits"}:
                    raise ValueError("not a foreground memory snapshot")
                metadata = json.loads(str(data["metadata"].item()))
                if metadata["schema"] != cls.FEATURE_SCHEMA:
                    raise ValueError("unsupported foreground feature schema")
                if set(metadata["settings"]) != set(cls.SETTINGS_CLASS.__dataclass_fields__):
                    raise ValueError("incompatible foreground settings")
                integer_settings = {"sample_rate", "n_fft", "hop", "bands", "warmup_clips",
                                    "confirmations", "candidate_ttl", "capacity"}
                for name, value in metadata["settings"].items():
                    if type(value) not in (int, float) or not np.isfinite(value):
                        raise ValueError("foreground settings must be finite numbers")
                    if name in integer_settings and type(value) is not int:
                        raise ValueError("foreground sizes and counts must be integers")
                learner = cls(cls.SETTINGS_CLASS(**metadata["settings"]))
                features = np.asarray(data["features"], dtype=np.float32)
                identifiers, hits = data["identifiers"], data["hits"]
                count = len(features)
                if features.shape != (count, 3 * learner.settings.bands) or count > learner.settings.capacity:
                    raise ValueError("incompatible foreground feature dimensions or capacity")
                if not np.isfinite(features).all() or (features < 0).any():
                    raise ValueError("foreground features must be finite and nonnegative")
                if count and not np.allclose(np.linalg.norm(features, axis=1), 1, atol=1e-4):
                    raise ValueError("foreground features must be unit vectors")
                for values in (identifiers, hits):
                    if values.shape != (count,) or values.dtype.kind not in "iu":
                        raise ValueError("invalid foreground identity or hit counts")
                next_id = metadata["next_id"]
                if (type(next_id) is not int or next_id < 0 or (identifiers < 0).any()
                        or len(set(identifiers.tolist())) != count
                        or (count and next_id <= int(identifiers.max()))
                        or (hits < learner.settings.confirmations).any()):
                    raise ValueError("invalid foreground identity or confirmation history")
                learner._patterns = [_Pattern(row.copy(), int(hit), 0, int(identifier))
                                     for row, hit, identifier in zip(features, hits, identifiers)]
                learner._next_id = next_id
            return learner
        except (KeyError, TypeError, OverflowError, AttributeError) as error:
            raise ValueError("invalid foreground memory snapshot") from error

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
        observation = self._remember(embedding.astype(np.float32), salience)
        return replace(observation, region_start_s=float(indices[0] * c.hop / c.sample_rate),
                       region_stop_s=float((indices[-1] * c.hop + c.n_fft) / c.sample_rate))

    def _admission_allowed(self, candidate_embedding: np.ndarray) -> bool:
        """Research hook for new prototypes; established matches bypass this gate."""
        return True

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
        if pattern.hits < c.confirmations or not self._admission_allowed(pattern.embedding):
            return Observation("candidate", salience, novelty, embedding=embedding)
        self._candidates.remove(pattern)
        if len(self._patterns) >= c.capacity:
            self._patterns.remove(min(self._patterns, key=lambda p: p.last_seen))
        pattern.identifier = self._next_id
        self._next_id += 1
        self._patterns.append(pattern)
        return Observation("learned", salience, novelty, pattern.identifier, embedding)
