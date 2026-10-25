"""Experimental corroboration of NEW memories using a compensated audio view."""

from collections import deque
from dataclasses import dataclass

import numpy as np

from chuff_mpu.adaptive_gain import AdaptiveGainConfig, AdaptiveGainForegroundLearner
from chuff_mpu.attention import Observation
from chuff_mpu.gain_attention import SpectralGainConfig, SpectralGainForegroundLearner


class _SpectralEvidence(SpectralGainForegroundLearner):
    """A bounded background/profile estimator with no prototype admission store."""
    def _remember(self, embedding, salience):
        return Observation("candidate", salience, embedding=embedding)


@dataclass(frozen=True)
class GuardedGainConfig(AdaptiveGainConfig):
    uncertainty_windows: int = 3
    uncertainty_votes: int = 2
    recovery_windows: int = 3

    def __post_init__(self):
        super().__post_init__()
        if type(self.uncertainty_windows) is not int or not 3 <= self.uncertainty_windows <= 31:
            raise ValueError("uncertainty_windows must be an integer in [3, 31]")
        if type(self.uncertainty_votes) is not int or not self.uncertainty_windows//2 < self.uncertainty_votes <= self.uncertainty_windows:
            raise ValueError("uncertainty_votes must be a strict majority of uncertainty_windows")
        if type(self.recovery_windows) is not int or not 1 <= self.recovery_windows <= 31:
            raise ValueError("recovery_windows must be an integer in [1, 31]")


class GuardedGainForegroundLearner(AdaptiveGainForegroundLearner):
    """Hold a confirmed proposal until its stored vector has current support.

    While raw mode has evidence of uncertain common gain (2/3 high-CV
    windows), the independent always-compensated frontend must detect an event whose
    embedding is within the ordinary cosine matching radius of the proposed
    prototype. Both frontends receive only the same mixture waveform. This
    can reject useful sources that compensation misses and is not separation.
    Candidates remain bounded/expiring; switches discard them as before.
    Existing prototype recognition bypasses this gate. All shadow state resets
    on load; only this actor's admitted prototypes are persisted.
    """
    FEATURE_SCHEMA = "chuff.foreground.guarded-gain-band.v2"
    SETTINGS_CLASS = GuardedGainConfig

    def __init__(self, settings=None):
        super().__init__(settings)
        self._shadow = _SpectralEvidence(SpectralGainConfig(**{
            name: getattr(self.settings, name) for name in SpectralGainConfig.__dataclass_fields__}))
        self._shadow_observation = Observation("calibrating")
        self.held_admissions = 0
        self.admission_support = False
        self.admission_check_required = False
        self._uncertainty_history = deque(maxlen=self.settings.uncertainty_windows)
        self.gain_uncertain = False
        self._recovery_count = 0

    def observe(self, clip):
        self._shadow_observation = self._shadow.observe(clip)
        self.admission_support = False
        self.admission_check_required = False
        return super().observe(clip)

    def _prepare_power(self, power):
        prepared = super()._prepare_power(power)
        c = self.settings
        if self._compensate or self._gain_profile_windows < c.warmup_clips:
            self._uncertainty_history.clear()
            self.gain_uncertain = False
            self._recovery_count = 0
        elif self._tick > c.warmup_clips:
            high = self._gain_cv >= c.gain_cv_threshold
            self._uncertainty_history.append(high)
            if sum(self._uncertainty_history) >= c.uncertainty_votes:
                self.gain_uncertain = True
            self._recovery_count = self._recovery_count+1 if self._gain_cv <= c.gain_cv_off else 0
            if self._recovery_count >= c.recovery_windows:
                self.gain_uncertain = False
                self._uncertainty_history.clear()
        return prepared

    def _admission_allowed(self, candidate_embedding):
        self.admission_check_required = self.gain_uncertain and not self._compensate
        if not self.admission_check_required:
            self.admission_support = True
            return True
        other = self._shadow_observation.embedding
        self.admission_support = bool(other is not None and
            float(np.clip(1-np.dot(candidate_embedding, other), 0, 2)) < self.settings.match_distance)
        self.held_admissions += not self.admission_support
        return self.admission_support
