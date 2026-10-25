import unittest
from pathlib import Path
import tempfile

import numpy as np

from chuff_mpu.adaptive_gain import AdaptiveGainConfig, AdaptiveGainForegroundLearner
from chuff_mpu.gain_attention import CalibratedGainForegroundLearner
from chuff_mpu.sim.scene import SceneConfig
from chuff_mpu.sim.transition_scene import TransitionScene


class AdaptiveGainTests(unittest.TestCase):
    def test_isolated_bursts_cannot_select_new_mode(self):
        model = AdaptiveGainForegroundLearner()
        model._compensate = False
        for value in [.04]*9+[.8]+[.04]*12:
            model._record_gain_evidence(value)
            self.assertIsNone(model._pending_mode)
        for value in [.6]*9:
            model._record_gain_evidence(value)
        self.assertTrue(model._pending_mode)

    def test_switch_retains_vectors_and_ids_but_clears_pending_and_floor(self):
        model = AdaptiveGainForegroundLearner()
        from chuff_mpu.sim.challenge_scene import ChallengeScene
        for frame in ChallengeScene(1900, SceneConfig(rounds=80, snr_db=6, nuisance_probability=0)):
            model.observe(frame.audio)
        before = [(p.identifier, p.embedding.copy()) for p in model._patterns]
        self.assertGreater(len(before), 0)
        model._pending_mode = not model.gain_compensation_enabled
        switches = model.mode_switches
        model._apply_pending_mode()
        self.assertEqual(model.mode_switches, switches+1)
        self.assertIsNone(model._background)
        self.assertEqual(model._candidates, [])
        self.assertTrue(model._is_calibrating())
        for pattern, (identifier, vector) in zip(model._patterns, before):
            self.assertEqual(pattern.identifier, identifier)
            np.testing.assert_array_equal(pattern.embedding, vector)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"adaptive.npz"
            model.save(path)
            restored = AdaptiveGainForegroundLearner.load(path)
            self.assertEqual(restored.memory_size, len(before))
            self.assertEqual({p.identifier for p in restored._patterns}, {p[0] for p in before})
            vectors = dict(before)
            for pattern in restored._patterns:
                np.testing.assert_array_equal(pattern.embedding, vectors[pattern.identifier])
            self.assertIsNone(restored.gain_compensation_enabled)
            self.assertIsNone(restored._pending_mode)
            self.assertEqual(len(restored._mode_history), 0)
            self.assertEqual(restored.mode_switches, 0)
            with self.assertRaises(ValueError):
                CalibratedGainForegroundLearner.load(path)

    def test_actual_waveform_transition_enables_and_disables(self):
        settings = SceneConfig(rounds=50, background_change=False, event_probability=0, nuisance_probability=0)
        for case, initial, final in (("stationary_fast", False, True), ("fast_stationary", True, False)):
            model = AdaptiveGainForegroundLearner()
            for step, frame in enumerate(TransitionScene(1901, settings, environment_seed=101901, case=case)):
                observation = model.observe(frame.audio)
                if observation.salience is not None:
                    self.assertTrue(np.isfinite(observation.salience))
                if step == 20:
                    self.assertEqual(model.gain_compensation_enabled, initial)
            self.assertEqual(model.gain_compensation_enabled, final)
            self.assertEqual(model.mode_switches, 1)

    def test_invalid_settings(self):
        for values in (dict(history_windows=2), dict(required_votes=4), dict(gain_cv_off=.12)):
            with self.assertRaises(ValueError):
                AdaptiveGainConfig(**values)
