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
