"""Behavioural tests for discovery, habituation, and room reproducibility."""

from dataclasses import replace
import unittest

import numpy as np

from chuff_mpu.attention import AttentionConfig, ForegroundLearner
from chuff_mpu.sim.scene import AcousticScene, SceneConfig


class SceneTests(unittest.TestCase):
    def test_audio_and_truth_are_reproducible_and_snr_is_calibrated(self):
        settings = SceneConfig(rounds=20, event_probability=1, snr_db=3)
        first, second = list(AcousticScene(9, settings)), list(AcousticScene(9, settings))
        for a, b in zip(first, second):
            np.testing.assert_array_equal(a.audio, b.audio)
            self.assertEqual(a.label, b.label)
            self.assertEqual(a.audio.shape, (16000,))
            self.assertTrue(np.isfinite(a.audio).all())
            if a.active_snr_db is not None:
                self.assertAlmostEqual(a.active_snr_db, 3, places=6)
        self.assertTrue(any(frame.background_changed for frame in first))

    def test_source_schedule_is_independent_of_background_and_snr(self):
        settings = SceneConfig(rounds=30)
        dry = list(AcousticScene(7, replace(settings, room_decay_s=0, snr_db=0, background_change=False)))
        wet = list(AcousticScene(7, replace(settings, room_decay_s=0.3, snr_db=12, background_change=True)))
        self.assertEqual([f.label for f in dry], [f.label for f in wet])
        self.assertEqual([f.nuisance for f in dry], [f.nuisance for f in wet])
        self.assertFalse(np.array_equal(dry[-1].audio, wet[-1].audio))


class AttentionTests(unittest.TestCase):
    def test_background_change_and_impacts_do_not_fill_long_term_memory(self):
        settings = SceneConfig(rounds=60, event_probability=0, nuisance_probability=0.5)
        learner = ForegroundLearner()
        statuses = [learner.observe(frame.audio).status for frame in AcousticScene(17, settings)]
        self.assertNotIn("learned", statuses)
        self.assertEqual(learner.memory_size, 0)

    def test_recurring_temporal_events_learn_then_habituate_without_labels(self):
        settings = SceneConfig(rounds=120, snr_db=12, event_probability=0.45)
        learner = ForegroundLearner()
        learned_at = {}
        recognised = set()
        initial, familiar = [], []
        for frame in AcousticScene(29, settings):
            # Truth is inspected only after the waveform-only decision.
            result = learner.observe(frame.audio)
            if result.status == "candidate" and result.novelty is not None:
                initial.append(result.novelty)
            if result.status == "learned":
                learned_at[result.prototype_id] = frame.label
            if result.status == "recognised":
                self.assertEqual(learned_at[result.prototype_id], frame.label)
                recognised.add(frame.label)
                familiar.append(result.novelty)
        self.assertEqual(recognised, {"double_chime", "rising_call", "falling_call"})
        self.assertLess(np.mean(familiar), np.mean(initial))
        self.assertEqual(learner.memory_size, 3)

    def test_candidate_expiry_prevents_distant_accidental_confirmation(self):
        learner = ForegroundLearner(AttentionConfig(candidate_ttl=2))
        noise = np.random.default_rng(5).normal(0, 0.01, 16000)
        for _ in range(3):
            learner.observe(noise)
        # A short tonal event with silence-like background is a candidate.
        t = np.arange(4000) / 16000
        clip = noise.copy()
        clip[4000:8000] += 0.1 * np.sin(2 * np.pi * 1600 * t) * np.hanning(4000)
        self.assertEqual(learner.observe(clip).status, "candidate")
        for _ in range(3):
            learner.observe(noise)
        self.assertEqual(learner.candidate_count, 0)
        self.assertEqual(learner.observe(clip).status, "candidate")

    def test_consecutive_windows_cannot_confirm_one_ongoing_event(self):
        learner = ForegroundLearner()
        rng = np.random.default_rng(6)
        quiet = rng.normal(0, 0.01, 16000)
        for _ in range(3):
            learner.observe(quiet)
        clip = quiet.copy()
        t = np.arange(4000) / 16000
        clip[4000:8000] += 0.1 * np.sin(2 * np.pi * 1600 * t) * np.hanning(4000)
        self.assertEqual(learner.observe(clip).status, "candidate")
        for _ in range(5):
            self.assertEqual(learner.observe(clip).status, "candidate")
        self.assertEqual(learner.memory_size, 0)

    def test_both_memory_stages_stay_bounded_under_new_events(self):
        learner = ForegroundLearner(AttentionConfig(capacity=2, confirmations=1))
        identifiers = set()
        for frame in AcousticScene(11, SceneConfig(rounds=60, event_probability=0.7, snr_db=12)):
            observation = learner.observe(frame.audio)
            if observation.status == "learned":
                self.assertNotIn(observation.prototype_id, identifiers)
                identifiers.add(observation.prototype_id)
            self.assertLessEqual(learner.memory_size, 2)
            self.assertLessEqual(learner.candidate_count, 2)
        self.assertGreater(len(identifiers), 2)

    def test_invalid_audio_and_settings_are_rejected(self):
        learner = ForegroundLearner()
        for clip in [np.zeros((2, 16000)), np.zeros(2), np.full(16000, np.nan)]:
            with self.assertRaises(ValueError):
                learner.observe(clip)
        with self.assertRaises(ValueError):
            AttentionConfig(capacity=0)
        with self.assertRaises(ValueError):
            SceneConfig(event_probability=1.1)


if __name__ == "__main__":
    unittest.main()
