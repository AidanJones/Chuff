"""Simulation pairing and validity, independently of detector performance."""

from dataclasses import replace
import unittest

import numpy as np

from chuff_mpu.sim.challenge_scene import ChallengeScene
from chuff_mpu.sim.scene import SceneConfig
from chuff_mpu.sim.varied_scene import CONDITIONS, VariedScene


class VariedSceneTests(unittest.TestCase):
    def test_reference_is_original_extended_scene(self):
        settings = SceneConfig(rounds=12)
        for old, new in zip(ChallengeScene(51, settings),
                            VariedScene(51, settings, condition="reference")):
            np.testing.assert_array_equal(old.audio, new.audio)
            self.assertEqual(old.label, new.label)

    def test_environment_changes_audio_without_changing_schedule(self):
        settings = SceneConfig(rounds=30, event_probability=.7)
        first = list(VariedScene(52, settings, environment_seed=1))
        second = list(VariedScene(52, settings, environment_seed=2))
        self.assertEqual([(f.label, f.nuisance) for f in first],
                         [(f.label, f.nuisance) for f in second])
        self.assertFalse(np.array_equal(first[0].audio, second[0].audio))

    def test_all_conditions_deterministic_finite_and_snr_matched(self):
        settings = SceneConfig(rounds=15, event_probability=1, snr_db=-6)
        for condition in CONDITIONS:
            kwargs = dict(environment_seed=7, condition=condition)
            first = list(VariedScene(53, settings, **kwargs))
            second = list(VariedScene(53, settings, **kwargs))
            for a, b in zip(first, second):
                np.testing.assert_array_equal(a.audio, b.audio)
                self.assertEqual(a.audio.shape, (16000,))
                self.assertTrue(np.isfinite(a.audio).all())
                if a.label is not None:
                    self.assertAlmostEqual(a.active_snr_db, -6)
            scene = VariedScene(53, settings, **kwargs)
            for label in scene.event_labels:
                self.assertLessEqual(len(scene._event(label)), 16000)

    def test_rejects_unsupported_windows_and_conditions(self):
        with self.assertRaises(ValueError):
            VariedScene(settings=replace(SceneConfig(), duration_s=.8))
        with self.assertRaises(ValueError):
            VariedScene(condition="unknown")
