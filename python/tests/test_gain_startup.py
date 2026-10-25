"""Unusable calibration must not freeze a floor-energy gain profile."""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from chuff_mpu.band_attention import BandForegroundLearner
from chuff_mpu.gain_attention import SpectralGainForegroundLearner, CalibratedGainForegroundLearner
from chuff_mpu.sim.challenge_scene import ChallengeScene
from chuff_mpu.sim.scene import SceneConfig


class GainStartupTests(unittest.TestCase):
    def compare(self, a, b, frames):
        for frame in frames:
            first, second = a.observe(frame.audio), b.observe(frame.audio)
            self.assertEqual((first.status, first.prototype_id, first.novelty),
                             (second.status, second.prototype_id, second.novelty))
            if first.embedding is not None:
                np.testing.assert_array_equal(first.embedding, second.embedding)

    def test_silent_prefix_recovers_clean_spectral_trajectory(self):
        clean, silent = SpectralGainForegroundLearner(), SpectralGainForegroundLearner()
        for _ in range(3): silent.observe(np.zeros(16000))
        self.assertIsNone(silent._gain_profile)
        self.assertEqual(silent._gain_profile_windows, 0)
        frames = list(ChallengeScene(1700, SceneConfig(rounds=120, snr_db=-6, nuisance_probability=0)))
        self.compare(clean, silent, frames)
        self.assertEqual(silent._gain_profile_windows, 3)
        self.assertGreater(silent.memory_size, 0)

    def test_calibrated_raw_choice_stays_baseline_equivalent_after_silence(self):
        chosen, baseline = CalibratedGainForegroundLearner(), BandForegroundLearner()
        for model in (chosen, baseline):
            for _ in range(3): model.observe(np.zeros(16000))
        self.assertFalse(chosen.gain_compensation_enabled)
        self.compare(chosen, baseline, ChallengeScene(1701, SceneConfig(rounds=40)))

    def test_late_calibration_keeps_admitted_memory_and_suppresses_new_admissions(self):
        source = SpectralGainForegroundLearner()
        for frame in ChallengeScene(1702, SceneConfig(rounds=120)):
            source.observe(frame.audio)
        self.assertGreater(source.memory_size, 0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"spectral.npz"
            source.save(path)
            restored = SpectralGainForegroundLearner.load(path)
            before = [(p.identifier, p.embedding.copy()) for p in restored._patterns]
            for _ in range(3): restored.observe(np.zeros(16000))
            rng = np.random.default_rng(1703)
            for _ in range(3):
                result = restored.observe(rng.normal(size=16000)*.02)
                self.assertEqual(result.status, "calibrating")
            self.assertEqual([p.identifier for p in restored._patterns], [p[0] for p in before])
            for pattern, (_, vector) in zip(restored._patterns, before):
                np.testing.assert_array_equal(pattern.embedding, vector)
