"""Experimental corroboration of NEW memories using a compensated audio view."""

import numpy as np

from chuff_mpu.adaptive_gain import AdaptiveGainForegroundLearner
from chuff_mpu.attention import Observation
from chuff_mpu.gain_attention import SpectralGainConfig, SpectralGainForegroundLearner


class _SpectralEvidence(SpectralGainForegroundLearner):
    """A bounded background/profile estimator with no prototype admission store."""
    def _remember(self, embedding, salience):
        return Observation("candidate", salience, embedding=embedding)


class GuardedGainForegroundLearner(AdaptiveGainForegroundLearner):
    """Hold a confirmed proposal until its stored vector has current support.

    The independent always-compensated frontend must detect an event whose
    embedding is within the ordinary cosine matching radius of the proposed
    prototype. Both frontends receive only the same mixture waveform. This
    can reject useful sources that compensation misses and is not separation.
    Candidates remain bounded/expiring; switches discard them as before.
    Existing prototype recognition bypasses this gate. All shadow state resets
    on load; only this actor's admitted prototypes are persisted.
    """
    FEATURE_SCHEMA = "chuff.foreground.guarded-gain-band.v1"

    def __init__(self, settings=None):
        super().__init__(settings)
        self._shadow = _SpectralEvidence(SpectralGainConfig(**{
            name: getattr(self.settings, name) for name in SpectralGainConfig.__dataclass_fields__}))
        self._shadow_observation = Observation("calibrating")
        self.held_admissions = 0
        self.admission_support = False

    def observe(self, clip):
        self._shadow_observation = self._shadow.observe(clip)
        self.admission_support = False
        return super().observe(clip)

    def _admission_allowed(self, candidate_embedding):
        other = self._shadow_observation.embedding
        self.admission_support = bool(other is not None and
            float(np.clip(1-np.dot(candidate_embedding, other), 0, 2)) < self.settings.match_distance)
        self.held_admissions += not self.admission_support
        return self.admission_support
