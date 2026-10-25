"""Validate controlled factors and waveform pairing, independent of learner success."""

from dataclasses import replace
import unittest

import numpy as np

from chuff_mpu.sim.modulation_scene import ModulationScene
from chuff_mpu.sim.scene import SceneConfig


class ModulationSceneTests(unittest.TestCase):
    def make(self, **changes):
        values = dict(spectrum="varied", channel="reference", modulation="stationary")
        values.update(changes)
        return ModulationScene(71, SceneConfig(rounds=15, background_change=False),
                               environment_seed=100071, **values)

    def test_modulation_is_exact_gain_on_the_same_background(self):
        stationary, slow = self.make(), self.make(modulation="slow")
        n = 16000
        expected = 1+.45*np.sin(2*np.pi*.17*np.arange(n)/16000+slow._hum_phase)
        np.testing.assert_allclose(slow._background(n), stationary._background(n)*expected)

    def test_channel_and_background_do_not_change_sources_or_schedule(self):
        a, b = self.make(), self.make(channel="varied", spectrum="reference", modulation="fast")
        for label in a.event_labels:
            np.testing.assert_array_equal(a._event(label), b._event(label))
        first, second = list(self.make()), list(self.make(channel="varied", modulation="fast"))
        self.assertEqual([(f.label, f.nuisance) for f in first], [(f.label, f.nuisance) for f in second])
        self.assertTrue(all(np.isfinite(f.audio).all() for f in second))

    def test_halfway_changes_rejected(self):
        with self.assertRaises(ValueError):
            ModulationScene(0, SceneConfig(), environment_seed=1,
                            spectrum="reference", channel="reference", modulation="stationary")
