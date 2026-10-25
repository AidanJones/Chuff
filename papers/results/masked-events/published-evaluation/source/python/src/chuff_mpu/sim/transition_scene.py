"""Matched changing-background scenes; all condition information is evaluator-only."""

from dataclasses import replace

from chuff_mpu.sim.modulation_scene import ModulationScene


CASES = ("stationary", "slow", "fast", "stationary_fast", "fast_stationary",
         "stationary_slow", "slow_stationary", "silent_fast", "shallow", "spectral_fast")


class TransitionScene(ModulationScene):
    def __init__(self, seed, settings, *, environment_seed, case):
        if case not in CASES or settings.rounds < 30 or settings.warmup < 3:
            raise ValueError("known case, >=30 windows and >=3 warmup windows required")
        super().__init__(seed, settings, environment_seed=environment_seed,
                         spectrum="varied", channel="varied", modulation="stationary")
        self.case = case
        self.change_step = settings.warmup if case == "silent_fast" else settings.rounds//2
        self.parameters.update(case=case, change_step=self.change_step,
            source_resampling_range=None, source_variation=False,
            modulation="time_varying", modulation_depth=None, modulation_rate_hz=None,
            protocol=".45 depth slow=.17Hz fast=1.3Hz; shallow=.15 depth .07Hz; spectral_fast varied->reference spectrum",
            halfway_change=case not in ("stationary", "slow", "fast", "shallow"))

    def phase(self, step):
        return "before" if step < self.change_step else "after"

    def _background(self, n):
        after = self._step >= self.change_step
        name = self.case
        if name in ("stationary_fast", "stationary_slow", "fast_stationary", "slow_stationary"):
            name = name.split("_")[int(after)]
        elif name in ("silent_fast", "spectral_fast"):
            name = "fast"
        self.depth, self.rate = {"stationary": (0., .17), "slow": (.45, .17),
                                 "fast": (.45, 1.3), "shallow": (.15, .07)}[name]
        self.spectrum = "reference" if self.case == "spectral_fast" and after else "varied"
        # Draw/filter even during silence, preserving paired background RNG/state.
        background = super()._background(n)
        if self.case == "silent_fast" and not after:
            background[:] = 0
        return background

    def next_frame(self):
        step = self._step
        frame = super().next_frame()
        changing = self.case not in ("stationary", "slow", "fast", "shallow")
        return replace(frame, background_changed=changing and step == self.change_step)
