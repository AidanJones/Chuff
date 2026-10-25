"""An extended procedural inventory; no claim to reproduce speech or real recordings."""

import numpy as np
from scipy.signal import butter, sosfilt

from chuff_mpu.sim.scene import AcousticScene, EVENTS


EXTENDED_EVENTS = EVENTS + ("low_rustle", "high_rustle", "long_alarm")


class ChallengeScene(AcousticScene):
    def __init__(self, seed=0, settings=None):
        super().__init__(seed, settings)
        self.event_labels = EXTENDED_EVENTS

    def _event(self, label):
        if label in EVENTS:
            return super()._event(label)
        rng = self.source_rng
        rate = self.settings.sample_rate
        if label == "long_alarm":
            duration = rng.uniform(0.78, 0.84)
            t = np.arange(int(duration * rate)) / rate
            frequency = rng.uniform(1050, 1150)
            event = (np.sin(2 * np.pi * frequency * t) + 0.3 * np.sin(4 * np.pi * frequency * t))
            event *= 0.55 + 0.45 * np.sin(2 * np.pi * 7 * t) ** 2
        else:
            duration = rng.uniform(0.3, 0.48)
            t = np.arange(int(duration * rate)) / rate
            edges = (180, 1100) if label == "low_rustle" else (1800, min(5000, rate * 0.45))
            noise = rng.normal(size=len(t))
            event = sosfilt(butter(2, edges, btype="bandpass", fs=rate, output="sos"), noise)
            event *= np.sin(np.pi * t / duration) ** 2
        ramp = int(0.012 * rate)
        event[:ramp] *= np.linspace(0, 1, ramp)
        event[-ramp:] *= np.linspace(1, 0, ramp)
        return event
