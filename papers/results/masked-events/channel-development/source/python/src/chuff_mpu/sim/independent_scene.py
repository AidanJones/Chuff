"""Independent pitch/time interventions with evaluator-only channel components."""

from dataclasses import replace

import librosa
import numpy as np

from chuff_mpu.sim.challenge_scene import ChallengeScene
from chuff_mpu.sim.varied_scene import VariedScene


TRANSFORMS = ("reference", "pitch", "duration", "pitch_duration")


def transform_source(event, sample_rate, pitch_ratio=1., duration_factor=1.):
    """Separate library phase-vocoder operations; artifacts are not natural variation."""
    if not .5 <= pitch_ratio <= 2 or not .5 <= duration_factor <= 2:
        raise ValueError("source factors must be in [0.5, 2]")
    result = np.asarray(event, dtype=np.float64)
    if pitch_ratio != 1:
        result = librosa.effects.pitch_shift(result, sr=sample_rate, n_steps=12*np.log2(pitch_ratio),
                                            n_fft=1024, hop_length=256)
    if duration_factor != 1:
        result = librosa.effects.time_stretch(result, rate=1/duration_factor, n_fft=1024, hop_length=256)
    return result


class IndependentSourceScene(VariedScene):
    def __init__(self, seed, settings, *, environment_seed, transform="reference"):
        if transform not in TRANSFORMS:
            raise ValueError("unknown independent source transform")
        # Always use the same varied physical environment, not coupled source resampling.
        super().__init__(seed, settings, environment_seed=environment_seed, condition="joint")
        self.transform = transform
        self._pitch_rng, self._duration_rng = [np.random.default_rng(s) for s in
            np.random.SeedSequence([seed, environment_seed, 91]).spawn(2)]
        self._source_parameters = None

    def _event(self, label):
        event = ChallengeScene._event(self, label)
        pitch = float(np.exp(self._pitch_rng.uniform(np.log(.85), np.log(1.15))))
        requested_duration = float(self._duration_rng.uniform(.8, 1.2))
        maximum = int(.94*self.settings.sample_rate)
        duration = min(requested_duration, maximum/len(event))
        if self.transform not in ("pitch", "pitch_duration"):
            pitch = 1.
        if self.transform not in ("duration", "pitch_duration"):
            duration = 1.
        result = transform_source(event, self.settings.sample_rate, pitch, duration)
        if len(result) > maximum:
            raise ValueError("transformed event exceeds window margin")
        if self.transform != "reference":
            ramp = min(int(.008*self.settings.sample_rate), len(result)//2)
            result[:ramp] *= np.linspace(0, 1, ramp)
            result[-ramp:] *= np.linspace(1, 0, ramp)
        self._source_parameters = dict(pitch_ratio=pitch, requested_duration_factor=requested_duration,
            applied_duration_factor=duration, realized_duration_factor=len(result)/len(event))
        return result

    def next_frame(self):
        self._source_parameters = None
        return replace(super().next_frame(), source_parameters=self._source_parameters)
