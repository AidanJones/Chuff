import unittest

import numpy as np

from chuff_mpu.sim.one_off_experiment import OneOffScene, measure
from chuff_mpu.sim.scene import SceneConfig


class OneOffTests(unittest.TestCase):
    def test_one_target_schedule_and_component_state(self):
        settings = SceneConfig(rounds=80, nuisance_probability=0, background_change=False,
            collect_stems=True, snr_reference="post_residual", snr_db=-6)
        scene = OneOffScene(2200, settings, environment_seed=102200, case="stationary_slow", label="low_rustle")
        configured = scene.settings
        frames = list(scene)
        self.assertIs(scene.settings, configured)
        self.assertEqual([step for step, frame in enumerate(frames) if frame.label], [66])
        for frame in frames:
            self.assertFalse(frame.nuisance)
            np.testing.assert_allclose(frame.audio, frame.background_audio+frame.source_audio, atol=1e-7)
        self.assertAlmostEqual(frames[66].post_channel_residual_snr_db, -6, places=8)

    def test_no_target_cannot_be_scored_as_one_off_success(self):
        with self.assertRaises(ValueError):
            measure([])
