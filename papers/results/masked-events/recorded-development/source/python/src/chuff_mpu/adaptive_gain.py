"""Experimental persistent-evidence gain selection; not a validated live default."""

from collections import deque
from dataclasses import dataclass

import numpy as np

from chuff_mpu.gain_attention import CalibratedGainConfig, CalibratedGainForegroundLearner


@dataclass(frozen=True)
class AdaptiveGainConfig(CalibratedGainConfig):
    history_windows: int = 9
    required_votes: int = 7
    gain_cv_off: float = .09

    def __post_init__(self):
        super().__post_init__()
        if type(self.history_windows) is not int or not 3 <= self.history_windows <= 61:
            raise ValueError("history_windows must be an integer in [3, 61]")
        if type(self.required_votes) is not int or not self.history_windows//2 < self.required_votes <= self.history_windows:
            raise ValueError("required_votes must be a strict majority within history_windows")
        if not np.isfinite(self.gain_cv_off) or not 0 < self.gain_cv_off < self.gain_cv_threshold:
            raise ValueError("gain_cv_off must be positive and below the on threshold")


class AdaptiveGainForegroundLearner(CalibratedGainForegroundLearner):
    """Require 7/9 usable windows above/below hysteretic CV thresholds.

    Switches apply on the NEXT window and suppress three settling windows.
    Background/pending confirmations reset; admitted vectors/IDs survive.
    Their cross-mode matching quality must be measured, not presumed.
    Saving retains admitted memories only; load restarts all adaptation state.
    """
    FEATURE_SCHEMA = "chuff.foreground.adaptive-gain-band.v1"
    SETTINGS_CLASS = AdaptiveGainConfig

    def __init__(self, settings=None):
        super().__init__(settings)
        self._mode_history = deque(maxlen=self.settings.history_windows)
        self._pending_mode = None
        self._settle_until = 0
        self.mode_switches = 0

    def _apply_pending_mode(self):
        if self._pending_mode is None:
            return
        self._compensate = self._pending_mode
        self._pending_mode = None
        self._background = None
        self._candidates.clear()
        self._mode_history.clear()
        self._settle_until = self._tick+self.settings.warmup_clips-1
        self.mode_switches += 1

    def _record_gain_evidence(self, cv):
        c = self.settings
        self._mode_history.append(cv)
        if len(self._mode_history) < c.history_windows:
            return
        high = sum(value >= c.gain_cv_threshold for value in self._mode_history)
        low = sum(value <= c.gain_cv_off for value in self._mode_history)
        if not self._compensate and high >= c.required_votes:
            self._pending_mode = True
        elif self._compensate and low >= c.required_votes:
            self._pending_mode = False

    def _is_calibrating(self):
        return self._tick <= self._settle_until or super()._is_calibrating()

    def _prepare_power(self, power):
        self._apply_pending_mode()
        prepared = super()._prepare_power(power)
        if (self._tick > self.settings.warmup_clips and self._gain_profile_windows >= self.settings.warmup_clips
                and self._tick > self._settle_until):
            self._record_gain_evidence(self._gain_cv)
        return prepared
