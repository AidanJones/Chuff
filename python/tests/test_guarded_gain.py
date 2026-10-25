from pathlib import Path
import tempfile
import unittest

import numpy as np

from chuff_mpu.adaptive_gain import AdaptiveGainForegroundLearner
from chuff_mpu.attention import Observation
from chuff_mpu.guarded_gain import GuardedGainConfig, GuardedGainForegroundLearner
from chuff_mpu.sim.scene import SceneConfig
from chuff_mpu.sim.transition_scene import TransitionScene
from chuff_mpu.sim.attention_experiment import evaluate
from chuff_mpu.attention import AttentionConfig


class GuardedGainTests(unittest.TestCase):
    def test_invalid_uncertainty_configuration(self):
        for values in (dict(uncertainty_windows=2), dict(uncertainty_votes=1), dict(recovery_windows=0)):
            with self.assertRaises(ValueError):
                GuardedGainConfig(**values)

    def test_hold_preserves_novel_attention_and_does_not_allocate_or_evict(self):
        model = GuardedGainForegroundLearner()
        model.gain_uncertain = True
        a = np.zeros(96, np.float32); a[0] = 1
        b = np.zeros(96, np.float32); b[-1] = 1
        model._shadow_observation = Observation("candidate", embedding=a)
        model._tick = 4
        self.assertEqual(model._remember(a, 3).status, "candidate")
        model._tick = 6
        self.assertEqual(model._remember(a, 3).status, "learned")
        identifier = model._patterns[0].identifier
        model._shadow_observation = Observation("background")
        for tick in (8, 10, 12):
            model._tick = tick
            result = model._remember(b, 5)
            self.assertEqual(result.status, "candidate")
            self.assertEqual(result.novelty, 1)
            self.assertEqual(result.salience, 5)
            self.assertIsNone(result.prototype_id)
        self.assertEqual(model.memory_size, 1)
        self.assertEqual(model._next_id, identifier+1)
        # Familiar recognition works even when corroboration sees no event.
        self.assertEqual(model._remember(a, 3).prototype_id, identifier)
        self.assertEqual(model._remember(a, 3).status, "recognised")
        model._shadow_observation = Observation("candidate", embedding=b)
        model._tick = 14
        self.assertEqual(model._remember(b, 5).status, "learned")
        self.assertEqual(model.memory_size, 2)

    def test_unrelated_shadow_event_cannot_confirm_candidate(self):
        model = GuardedGainForegroundLearner()
        model.gain_uncertain = True
        a, b = np.eye(2, 96, dtype=np.float32)
        model._shadow_observation = Observation("candidate", embedding=b)
        for tick in (4, 6, 8):
            model._tick = tick
            self.assertEqual(model._remember(a, 2).status, "candidate")
        self.assertEqual(model.memory_size, 0)

    def test_actual_transition_blocks_control_admissions_and_load_resets_shadow(self):
        model = GuardedGainForegroundLearner()
        settings = SceneConfig(rounds=120, event_probability=0, nuisance_probability=0, background_change=False)
        for frame in TransitionScene(2200, settings, environment_seed=102200, case="stationary_slow"):
            model.observe(frame.audio)
        self.assertGreater(model.held_admissions, 0)
        self.assertEqual(model.memory_size, 0)
        self.assertEqual(model._shadow.memory_size, 0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"guarded.npz"
            model.save(path)
            loaded = GuardedGainForegroundLearner.load(path)
            self.assertEqual(loaded.held_admissions, 0)
            self.assertIsNone(loaded._shadow._gain_profile)
            self.assertIsNone(loaded.gain_compensation_enabled)
            with self.assertRaises(ValueError):
                AdaptiveGainForegroundLearner.load(path)

    def test_sensory_attention_is_unchanged_while_admission_can_differ(self):
        original, guarded = AdaptiveGainForegroundLearner(), GuardedGainForegroundLearner()
        scene = TransitionScene(2201, SceneConfig(rounds=100, nuisance_probability=0, background_change=False),
                                environment_seed=102201, case="stationary_slow")
        for frame in scene:
            a, b = original.observe(frame.audio), guarded.observe(frame.audio)
            self.assertEqual((a.salience, a.region_start_s, a.region_stop_s),
                             (b.salience, b.region_start_s, b.region_stop_s))
            self.assertEqual(a.embedding is None, b.embedding is None)
            if a.embedding is not None:
                np.testing.assert_array_equal(a.embedding, b.embedding)

    def test_source_deletion_probes_do_not_change_either_real_frontend(self):
        settings = SceneConfig(rounds=40, nuisance_probability=0, background_change=False, collect_stems=True)
        frames = list(TransitionScene(2201, settings, environment_seed=102201, case="stationary_slow"))
        a, b = GuardedGainForegroundLearner(), GuardedGainForegroundLearner()
        summary_a, rows_a, _ = evaluate(2201, settings, AttentionConfig(), learners={"g": a},
            frames=frames, include_baseline=False)
        summary_b, rows_b, _ = evaluate(2201, settings, AttentionConfig(), learners={"g": b},
            frames=frames, include_baseline=False, counterfactual_probes=True)
        self.assertEqual(summary_a, summary_b)
        for left, right in zip(rows_a, rows_b):
            right.pop("without_current_probe", None)
            self.assertEqual(left, right)
        np.testing.assert_array_equal(a._shadow._background, b._shadow._background)
        np.testing.assert_array_equal(a._shadow._gain_profile, b._shadow._gain_profile)
        self.assertEqual(a.held_admissions, b.held_admissions)
