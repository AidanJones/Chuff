"""Experimental full-band causal gain compensation before foreground learning.

This is not PCEN or source separation. Removing common level variation can
also suppress useful broadband amplitude cues; paired foreground controls
are required. Memories use a distinct schema because features change.
"""

from dataclasses import dataclass

import numpy as np
from scipy.signal import lfilter

from chuff_mpu.band_attention import BandAttentionConfig, BandForegroundLearner


@dataclass(frozen=True)
class GainAttentionConfig(BandAttentionConfig):
    gain_time_s: float = .2
    gain_floor_power: float = 1e-8

    def __post_init__(self):
        super().__post_init__()
        if not np.isfinite(self.gain_time_s) or not 0 < self.gain_time_s <= 10:
            raise ValueError("gain_time_s must be finite and in (0, 10]")
        if not np.isfinite(self.gain_floor_power) or self.gain_floor_power <= 0:
            raise ValueError("gain_floor_power must be finite and positive")


class CausalRMSNormalizer:
    def __init__(self, settings):
        self.settings = settings
        self._beta = float(-np.expm1(-1/(settings.sample_rate*settings.gain_time_s)))
        self._state = np.zeros(1)

    def transform(self, clip):
        clip = np.asarray(clip, dtype=np.float64)
        if clip.ndim != 1 or not len(clip) or not np.isfinite(clip).all():
            raise ValueError("gain normalization requires finite nonempty mono samples")
        try:
            with np.errstate(over="raise", invalid="raise"):
                power = clip*clip
        except FloatingPointError as error:
            raise ValueError("input energy exceeds numeric range") from error
        energy, state = lfilter([self._beta], [1, -(1-self._beta)], power, zi=self._state)
        normalized = clip/np.sqrt(np.maximum(energy, self.settings.gain_floor_power))
        if not np.isfinite(normalized).all():
            raise ValueError("nonfinite gain-normalized audio")
        self._state = state
        return normalized


class GainForegroundLearner(BandForegroundLearner):
    FEATURE_SCHEMA = "chuff.foreground.gain-band-contrast.v1"
    SETTINGS_CLASS = GainAttentionConfig

    def __init__(self, settings=None):
        super().__init__(settings)
        self._normalizer = CausalRMSNormalizer(self.settings)

    def observe(self, clip):
        clip = np.asarray(clip, dtype=np.float64)
        if clip.ndim != 1 or len(clip) < self.settings.n_fft or not np.isfinite(clip).all():
            raise ValueError("expected a finite mono clip at least one FFT window long")
        return super().observe(self._normalizer.transform(clip))


@dataclass(frozen=True)
class SpectralGainConfig(BandAttentionConfig):
    gain_time_s: float = .02

    def __post_init__(self):
        super().__post_init__()
        if not np.isfinite(self.gain_time_s) or not 0 < self.gain_time_s <= 10:
            raise ValueError("gain_time_s must be finite and in (0, 10]")


class SpectralGainForegroundLearner(BandForegroundLearner):
    """Remove common band-level gain against a warmup-only spectral template.

    The band-ratio median reduces influence of spectrally localized signals;
    it is not a clean-noise oracle. A changed background spectrum or a source
    occupying most bands can break that assumption. The template resets and
    recalibrates on load, like the original background estimator.
    """
    FEATURE_SCHEMA = "chuff.foreground.spectral-gain-band.v1"
    SETTINGS_CLASS = SpectralGainConfig

    def __init__(self, settings=None):
        super().__init__(settings)
        self._gain_profile = None
        self._gain_state = np.zeros(1)
        self._gain_cv = None

    def _prepare_power(self, power):
        c = self.settings
        bands = self._filters @ power
        average = np.maximum(bands.mean(axis=1), 1e-12)
        if self._gain_profile is None:
            self._gain_profile = average.copy()
        elif self._tick <= c.warmup_clips:
            self._gain_profile += (average-self._gain_profile)/self._tick
        common = np.median(bands/np.maximum(self._gain_profile[:, None], 1e-12), axis=0)
        beta = float(-np.expm1(-c.hop/(c.sample_rate*c.gain_time_s)))
        level, self._gain_state = lfilter([beta], [1, -(1-beta)], common, zi=self._gain_state)
        skip = min(int(np.ceil(3*c.gain_time_s*c.sample_rate/c.hop)), len(level)-1)
        settled = level[skip:]
        self._gain_cv = float(np.std(settled)/max(float(np.mean(settled)), 1e-12))
        return power/np.maximum(level[None, :], 1e-6)


@dataclass(frozen=True)
class CalibratedGainConfig(SpectralGainConfig):
    gain_cv_threshold: float = .12

    def __post_init__(self):
        super().__post_init__()
        if not np.isfinite(self.gain_cv_threshold) or not 0 < self.gain_cv_threshold <= 10:
            raise ValueError("gain_cv_threshold must be finite and in (0, 10]")


class CalibratedGainForegroundLearner(SpectralGainForegroundLearner):
    """Choose raw versus compensated power from unlabelled calibration audio.

    Only the first warmup windows decide the mode; later events cannot turn
    it on simply by raising energy. Environment changes need separate testing.
    """
    FEATURE_SCHEMA = "chuff.foreground.calibrated-gain-band.v1"
    SETTINGS_CLASS = CalibratedGainConfig

    def __init__(self, settings=None):
        super().__init__(settings)
        self._calibration_cv = []
        self._compensate = None

    def _prepare_power(self, power):
        normalized = super()._prepare_power(power)
        if self._tick <= self.settings.warmup_clips:
            self._calibration_cv.append(self._gain_cv)
            if self._tick == self.settings.warmup_clips:
                self._compensate = bool(np.mean(self._calibration_cv) >= self.settings.gain_cv_threshold)
            # Both modes use the original raw-power background warmup.
            return power
        return normalized if self._compensate else power
