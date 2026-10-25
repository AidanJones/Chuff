"""Masked-signal regression cases, schema isolation, and negative controls."""

from dataclasses import replace
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.io import wavfile

from chuff_mpu.attention import ForegroundLearner
from chuff_mpu.band_attention import BandAttentionConfig, BandForegroundLearner
from chuff_mpu.sim.challenge_scene import ChallengeScene
from chuff_mpu.sim.scene import AcousticScene, SceneConfig
from chuff_mpu.selective_demo import main as selective_main
from chuff_mpu.sim.masked_experiment import aggregate


class BandAttentionTests(unittest.TestCase):
    def test_uncertainty_preserves_pairing_between_identical_seed_differences(self):
        runs = []
        for seed, reference in enumerate([0.1, 0.3, 0.6]):
            runs.append({"suite": "tonal", "snr_db": -6, "seed": seed, "metrics": {
                mode: {"event_detection_rate": value, "repeat_recognition_rate": value,
                       "background_attention_rate": 0, "false_admissions": 0}
                for mode, value in [("original", reference), ("quiet_band", reference + 0.2)]}})
        difference = aggregate(runs)[0]["modes"]["quiet_band"]["repeat_recognition_rate"]["paired_difference"]
        self.assertAlmostEqual(difference["mean"], 0.2)
        np.testing.assert_allclose(difference["bootstrap_95"], [0.2, 0.2])

    def test_masked_tonal_repeats_are_recovered_without_oracle_labels(self):
        settings = SceneConfig(rounds=120, snr_db=-6, structured_distractors=True)
        learners = [ForegroundLearner(), BandForegroundLearner()]
        correct = [0, 0]
        identities = [{}, {}]
        false_admissions = [0, 0]
        for frame in AcousticScene(42, settings):
            for index, learner in enumerate(learners):
                result = learner.observe(frame.audio)
                if result.status == "learned":
                    identities[index][result.prototype_id] = frame.label
                    false_admissions[index] += frame.label is None
                if result.status == "recognised" and frame.label is not None:
                    correct[index] += identities[index].get(result.prototype_id) == frame.label
        self.assertGreater(correct[1], correct[0] + 15)
        self.assertLessEqual(false_admissions[1], 1)

    def test_broadband_and_long_patterns_can_be_discovered(self):
        identities = {}
        recognised = set()
        learner = BandForegroundLearner()
        for frame in ChallengeScene(43, SceneConfig(rounds=180, event_probability=0.55, snr_db=0)):
            result = learner.observe(frame.audio)
            if result.status == "learned":
                identities[result.prototype_id] = frame.label
            elif result.status == "recognised" and identities.get(result.prototype_id) == frame.label:
                recognised.add(frame.label)
        self.assertTrue({"low_rustle", "high_rustle", "long_alarm"} <= recognised)

    def test_room_change_and_random_impacts_are_not_permanent_patterns(self):
        learner = BandForegroundLearner()
        for frame in AcousticScene(44, SceneConfig(rounds=120, event_probability=0, nuisance_probability=0.4)):
            learner.observe(frame.audio)
        self.assertEqual(learner.memory_size, 0)

    def test_silent_and_invalid_inputs_are_handled(self):
        learner = BandForegroundLearner()
        for _ in range(10):
            self.assertIn(learner.observe(np.zeros(16000)).status, ("calibrating", "background"))
        for clip in [np.full(16000, np.inf), np.zeros((2, 16000)), np.zeros(2)]:
            with self.assertRaises(ValueError):
                learner.observe(clip)
        for settings in [{"smoothing_frames": 1.5}, {"noise_mode": 9}, {"local_threshold": float("nan")}]:
            with self.assertRaises(ValueError):
                BandAttentionConfig(**settings)

    def test_saved_schemas_cannot_be_cross_loaded(self):
        with tempfile.TemporaryDirectory() as directory:
            original = Path(directory) / "original.npz"
            band = Path(directory) / "band.npz"
            ForegroundLearner().save(original)
            learner = BandForegroundLearner(replace(BandAttentionConfig(), capacity=4))
            learner.save(band)
            restored = BandForegroundLearner.load(band)
            self.assertEqual(restored.settings.capacity, 4)
            self.assertEqual(restored.settings.noise_mode, 1)
            with self.assertRaises(ValueError):
                ForegroundLearner.load(band)
            with self.assertRaises(ValueError):
                BandForegroundLearner.load(original)

    def test_band_frontend_replay_uses_its_own_persistent_schema(self):
        quiet = np.random.default_rng(45).normal(0, 0.01, 16000).astype(np.float32)
        event = quiet.copy()
        t = np.arange(4000) / 16000
        event[4000:8000] += 0.1 * np.sin(2 * np.pi * 1600 * t) * np.hanning(4000)
        with tempfile.TemporaryDirectory() as directory:
            wav, state = Path(directory) / "input.wav", Path(directory) / "band.npz"
            wavfile.write(wav, 16000, np.concatenate([quiet] * 3 + [event, quiet, event, quiet, event]))
            with (patch("sys.argv", ["chuff-selective", "--frontend", "band", "--input-wav", str(wav),
                                     "--state", str(state)]), redirect_stdout(io.StringIO())):
                selective_main()
            self.assertEqual(BandForegroundLearner.load(state).memory_size, 1)
            with self.assertRaises(ValueError):
                ForegroundLearner.load(state)


if __name__ == "__main__":
    unittest.main()
