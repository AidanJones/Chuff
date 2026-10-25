"""Factorized procedural stressors; these are not calibrated room acoustics."""

from dataclasses import replace

import numpy as np
from scipy.signal import butter, lfilter, resample

from chuff_mpu.sim.challenge_scene import ChallengeScene
from chuff_mpu.sim.scene import SceneConfig


CONDITIONS = ("reference", "source", "room", "microphone", "background", "joint")


class VariedScene(ChallengeScene):
    """Environment RNGs never draw from the event schedule or source RNG.

    Source variation changes pitch and duration together, independently each
    occurrence. Room and microphone parameters stay fixed within each run.
    Background variance includes a persistent level/spectrum change halfway.
    The inherited SNR is measured before the common room/microphone channel.
    """

    def __init__(self, seed=0, settings=None, *, environment_seed=0, condition="joint"):
        if condition not in CONDITIONS:
            raise ValueError("unknown variation condition")
        settings = settings or SceneConfig()
        if settings.duration_s < 1:
            raise ValueError("extended variable events require >=1 second windows")
        streams = [np.random.default_rng(s) for s in
                   np.random.SeedSequence(environment_seed).spawn(4)]
        room_rng, mic_rng, background_rng, self.variation_rng = streams
        self.condition = condition
        self.parameters = {"environment_seed": environment_seed, "condition": condition}
        if condition in ("room", "joint"):
            settings = replace(settings, room_decay_s=float(room_rng.uniform(0.03, 0.5)))
        super().__init__(seed, settings)
        self.parameters["room_response_length_s"] = settings.room_decay_s
        if condition in ("room", "joint"):
            self._ir[:] = 0
            self._ir[0] = 1
            for _ in range(24):
                delay = int(room_rng.uniform(0.004, settings.room_decay_s) * settings.sample_rate)
                delay = min(delay, len(self._ir) - 1)
                self._ir[delay] += room_rng.uniform(-0.5, 0.5) * np.exp(-4 * delay / len(self._ir))
        low, high, self.gain = 100., .4 * settings.sample_rate, 1.
        if condition in ("microphone", "joint"):
            low = float(mic_rng.uniform(60, 450))
            high = float(mic_rng.uniform(2800, .45 * settings.sample_rate))
            self.gain = float(10 ** (mic_rng.uniform(-12, 12) / 20))
            self._mic_b, self._mic_a = butter(2, [low, high], btype="bandpass", fs=settings.sample_rate)
            self._mic_state[:] = 0
        self.parameters.update(microphone_low_hz=low, microphone_high_hz=high, gain=self.gain)
        self._colour = float(background_rng.uniform(0.35, 0.97))
        self._hum = float(background_rng.uniform(45, 180))
        self._fan = float(background_rng.uniform(220, 950))
        self._jump = float(10 ** (background_rng.uniform(-6, 9) / 20))
        self.parameters.update(background_pole=self._colour, hum_hz=self._hum,
                               fan_hz=self._fan, halfway_level_multiplier=self._jump,
                               source_resampling_range=[0.75, 1.3])

    def _background(self, n):
        if self.condition not in ("background", "joint"):
            return super()._background(n)
        c = self.settings
        white = self.background_rng.normal(size=n)
        coloured, self._noise_state = lfilter([np.sqrt(1-self._colour**2)],
                                              [1, -self._colour], white, zi=self._noise_state)
        t = (self._step * n + np.arange(n)) / c.sample_rate
        noise = .8 * coloured + .2 * white + .4 * np.sin(2*np.pi*self._hum*t)
        noise *= 1 + .45 * np.sin(2*np.pi*.17*t + self._hum_phase)
        if c.background_change and self._step >= c.rounds // 2:
            noise = (noise + .8*np.sin(2*np.pi*self._fan*t)) * self._jump
        return noise * c.background_rms

    def _event(self, label):
        event = super()._event(label)
        if self.condition in ("source", "joint"):
            factor = self.variation_rng.uniform(.75, 1.3)
            # Explicitly cap duration: long alarms must fit, with soft boundaries.
            size = min(int(len(event) * factor), int(.94*self.settings.sample_rate))
            event = resample(event, size)
            ramp = min(int(.012*self.settings.sample_rate), size//2)
            event[:ramp] *= np.linspace(0, 1, ramp)
            event[-ramp:] *= np.linspace(1, 0, ramp)
        return event

    def next_frame(self):
        frame = super().next_frame()
        return replace(frame, audio=(frame.audio*self.gain).astype(np.float32))
