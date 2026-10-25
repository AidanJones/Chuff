"""Controlled background modulation with independent spectrum/channel factors."""

import numpy as np
from scipy.signal import lfilter

from chuff_mpu.sim.challenge_scene import ChallengeScene
from chuff_mpu.sim.varied_scene import VariedScene


MODULATIONS = {"stationary": (0., .17), "slow": (.45, .17), "fast": (.45, 1.3)}


class ModulationScene(VariedScene):
    def __init__(self, seed, settings, *, environment_seed, spectrum, channel, modulation):
        if spectrum not in ("reference", "varied") or channel not in ("reference", "varied"):
            raise ValueError("spectrum/channel must be reference or varied")
        if modulation not in MODULATIONS:
            raise ValueError("unknown modulation")
        if settings.background_change:
            raise ValueError("controlled modulation experiment excludes halfway changes")
        super().__init__(seed, settings, environment_seed=environment_seed,
                         condition="joint" if channel == "varied" else "reference")
        self.spectrum = spectrum
        self.depth, self.rate = MODULATIONS[modulation]
        self.parameters.update(spectrum=spectrum, channel=channel, modulation=modulation,
                               modulation_depth=self.depth, modulation_rate_hz=self.rate,
                               source_variation=False, halfway_change=False)

    def _event(self, label):
        # Exclude source pitch/duration variation to isolate the gate/noise floor.
        return ChallengeScene._event(self, label)

    def _background(self, n):
        c = self.settings
        t = (self._step*n+np.arange(n))/c.sample_rate
        white = self.background_rng.normal(size=n)
        if self.spectrum == "reference":
            coloured, self._noise_state = lfilter([.12], [1, -.88], white, zi=self._noise_state)
            noise = .65*coloured/.253 + .35*white
            noise += .32*np.sin(2*np.pi*100*t+self._hum_phase)
            noise += .12*np.sin(2*np.pi*200*t+self._hum_phase)
        else:
            coloured, self._noise_state = lfilter([np.sqrt(1-self._colour**2)],
                                                  [1, -self._colour], white, zi=self._noise_state)
            noise = .8*coloured + .2*white + .4*np.sin(2*np.pi*self._hum*t)
        modulation = 1+self.depth*np.sin(2*np.pi*self.rate*t+self._hum_phase)
        return noise*modulation*c.background_rms
