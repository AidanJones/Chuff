"""Experimental band-relative evidence for events masked in whole-window energy.

The original learner is retained unchanged as an algorithmic comparison.
This frontend reuses its bounded, recurrence-confirmed prototype memory but
has a separate feature schema. It is not a pretrained or label-trained model.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.ndimage import binary_closing, uniform_filter1d
from scipy.signal import stft

from chuff_mpu.attention import AttentionConfig, ForegroundLearner, Observation


@dataclass(frozen=True)
class BandAttentionConfig(AttentionConfig):
    min_duration_s: float = 0.08
    max_duration_s: float = 0.95
    local_threshold: float = 3.5
    broad_threshold: float = 1.5
    smoothing_frames: int = 5
    feature_mode: int = 0  # 0: raw excess; 1: background-relative excess ablation
    noise_mode: int = 1  # 0: binwise quantile ablation; 1: shared quiet-frame estimate

    def __post_init__(self) -> None:
        super().__post_init__()
        if (not np.isfinite(self.local_threshold) or self.local_threshold <= 1
                or not np.isfinite(self.broad_threshold) or self.broad_threshold <= 1):
            raise ValueError("band evidence thresholds must be finite and >1")
        if type(self.smoothing_frames) is not int or not 1 <= self.smoothing_frames <= 21:
            raise ValueError("smoothing_frames must be an integer in [1, 21]")
        if type(self.feature_mode) is not int or self.feature_mode not in (0, 1):
            raise ValueError("feature_mode must be 0 (raw) or 1 (relative)")
        if type(self.noise_mode) is not int or self.noise_mode not in (0, 1):
            raise ValueError("noise_mode must be 0 (binwise) or 1 (quiet frames)")


class BandForegroundLearner(ForegroundLearner):
    FEATURE_SCHEMA = "chuff.foreground.band-contrast.v1"
    SETTINGS_CLASS = BandAttentionConfig

    def _prepare_power(self, power):
        """Optional waveform-only spectral transform; the baseline is identity."""
        return power

    def observe(self, clip: np.ndarray) -> Observation:
        c = self.settings
        clip = np.asarray(clip, dtype=np.float64)
        if clip.ndim != 1 or len(clip) < c.n_fft or not np.isfinite(clip).all():
            raise ValueError("expected a finite mono clip at least one FFT window long")
        self._tick += 1
        self._candidates = [p for p in self._candidates if self._tick - p.last_seen <= c.candidate_ttl]
        _, _, spectrum = stft(clip, fs=c.sample_rate, nperseg=c.n_fft,
                              noverlap=c.n_fft - c.hop, boundary=None, padded=False)
        power = np.abs(spectrum) ** 2
        power = self._prepare_power(power)
        if c.noise_mode == 0:
            floor = np.maximum(np.quantile(power, 0.3, axis=1) / -np.log(0.7), 1e-12)
        else:
            # A long event contaminates per-bin percentiles. Use the same quiet
            # time frames for all bins, selected from waveform energy only.
            # This remains a heuristic when a window contains no quiet frames.
            energy = uniform_filter1d(power.sum(axis=0), c.smoothing_frames, mode="nearest")
            quiet = energy <= np.quantile(energy, 0.2)
            floor = np.maximum(power[:, quiet].mean(axis=1), 1e-12)
        if self._background is None:
            self._background = floor.copy()
        background = np.maximum(self._background, floor)
        self._background += c.background_rate * (floor - self._background)
        if self._tick <= c.warmup_clips:
            return Observation("calibrating")
        bands = self._filters @ power
        noise = np.maximum(self._filters @ background, 1e-12)
        relative = bands / noise[:, None]
        smoothed = uniform_filter1d(relative, c.smoothing_frames, axis=1, mode="nearest")
        evidence = np.maximum(smoothed.max(axis=0) / c.local_threshold,
                              smoothed.mean(axis=0) / c.broad_threshold)
        instantaneous = np.maximum(relative.max(axis=0) / c.local_threshold,
                                   relative.mean(axis=0) / c.broad_threshold)
        active = binary_closing(evidence > 1, structure=np.ones(3))
        edges = np.diff(np.r_[False, active, False].astype(int))
        starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
        regions = [(int(a), int(b)) for a, b in zip(starts, stops)
                   if c.min_duration_s <= (b - a) * c.hop / c.sample_rate <= c.max_duration_s
                   and np.count_nonzero(instantaneous[a:b] > 1) * c.hop / c.sample_rate >= c.min_duration_s
                   and b - a >= 3]
        salience = float(evidence.max())
        if not regions:
            return Observation("background", salience)
        start, stop = max(regions, key=lambda region: float(evidence[region[0]:region[1]].sum()))
        if c.feature_mode == 0:
            features = self._filters @ np.maximum(power - c.subtraction * background[:, None], 0)
        else:
            features = np.maximum(smoothed - 1.5, 0)
        vectors = []
        for segment in np.array_split(features[:, start:stop], 3, axis=1):
            vector = segment.mean(axis=1)
            vector = np.sqrt(vector / max(float(vector.sum()), 1e-12))
            vectors.append(vector)
        embedding = np.concatenate(vectors)
        length = float(np.linalg.norm(embedding))
        if length < 1e-12:
            return Observation("background", salience)
        embedding /= length
        observation = self._remember(embedding.astype(np.float32), salience)
        # Boundary=None STFT frame k spans [k*hop, k*hop+n_fft] in samples.
        return replace(observation, region_start_s=float(start * c.hop / c.sample_rate),
                       region_stop_s=float(((stop-1) * c.hop + c.n_fft) / c.sample_rate))
