"""Causal streaming normalization and persistent feature-schema isolation."""

from dataclasses import replace
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.io import wavfile
from unittest.mock import patch

from chuff_mpu.band_attention import BandForegroundLearner
from chuff_mpu.gain_attention import (CausalRMSNormalizer, GainAttentionConfig, GainForegroundLearner,
                                      SpectralGainForegroundLearner, CalibratedGainForegroundLearner)
from chuff_mpu.selective_demo import main as selective_main


class GainAttentionTests(unittest.TestCase):
    def test_calibration_choice_freezes_but_restarts_after_loading(self):
        model = CalibratedGainForegroundLearner()
        rng = np.random.default_rng(95)
        for _ in range(3):
            model.observe(rng.normal(size=16000)*.02)
        self.assertFalse(model.gain_compensation_enabled)
        for _ in range(3):
            modulation = 1+.45*np.sin(2*np.pi*1.3*np.arange(16000)/16000)
            model.observe(rng.normal(size=16000)*.02*modulation)
        self.assertFalse(model.gain_compensation_enabled)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"calibrated.npz"
            model.save(path)
            loaded = CalibratedGainForegroundLearner.load(path)
            self.assertIsNone(loaded.gain_compensation_enabled)
            self.assertEqual(loaded._calibration_cv, [])
            with self.assertRaises(ValueError):
                SpectralGainForegroundLearner.load(path)

    def test_raw_choice_preserves_baseline_decisions_after_stationary_warmup(self):
        model = CalibratedGainForegroundLearner()
        baseline = BandForegroundLearner()
        rng = np.random.default_rng(94)
        for i in range(12):
            clip = rng.normal(size=16000)*.02
            if i > 5:
                clip[4000:9000] += .06*np.sin(2*np.pi*1200*np.arange(5000)/16000)
            first, second = model.observe(clip), baseline.observe(clip)
            self.assertEqual((first.status, first.prototype_id, first.novelty),
                             (second.status, second.prototype_id, second.novelty))
            if first.embedding is not None:
                np.testing.assert_array_equal(first.embedding, second.embedding)
        self.assertFalse(model._compensate)

    def test_spectral_replay_is_silent_and_uses_its_own_persistent_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            wav = Path(directory)/"quiet.wav"
            wavfile.write(wav, 16000, np.random.default_rng(93).normal(0, .01, 4*16000).astype(np.float32))
            for name, cls in (("spectral-gain", SpectralGainForegroundLearner),
                              ("calibrated-gain", CalibratedGainForegroundLearner)):
                state = Path(directory)/(name+".npz")
                with patch("sys.argv", ["chuff-selective", "--frontend", name, "--input-wav", str(wav), "--state", str(state)]), \
                     patch("chuff_mpu.audio_safety.safe_play") as play, redirect_stdout(io.StringIO()) as output:
                    selective_main()
                play.assert_not_called()
                self.assertIn(cls.FEATURE_SCHEMA, output.getvalue())
                restored = cls.load(state)
                self.assertIsNone(restored._gain_profile)
                self.assertEqual(restored._tick, 0)
                with self.assertRaises(ValueError):
                    BandForegroundLearner.load(state)

    def test_spectral_profile_freezes_after_unlabelled_warmup(self):
        model = SpectralGainForegroundLearner()
        rng = np.random.default_rng(92)
        for _ in range(model.settings.warmup_clips):
            model.observe(rng.normal(size=16000)*.02)
        profile = model._gain_profile.copy()
        model.observe(rng.normal(size=16000)*.04)
        np.testing.assert_array_equal(profile, model._gain_profile)
        self.assertTrue(np.isfinite(model._gain_state).all())

    def test_streaming_and_prefix_causality(self):
        config = GainAttentionConfig()
        clip = np.random.default_rng(91).normal(size=3200)*.02
        whole = CausalRMSNormalizer(config).transform(clip)
        stream = CausalRMSNormalizer(config)
        split = np.r_[stream.transform(clip[:1000]), stream.transform(clip[1000:])]
        np.testing.assert_array_equal(whole, split)
        future_changed = clip.copy()
        future_changed[1000:] *= 10
        np.testing.assert_array_equal(whole[:1000], CausalRMSNormalizer(config).transform(future_changed)[:1000])

    def test_silence_and_invalid_input_do_not_poison_state(self):
        listener = GainForegroundLearner()
        for _ in range(5):
            self.assertIn(listener.observe(np.zeros(16000)).status, ("calibrating", "background"))
        normalizer = CausalRMSNormalizer(GainAttentionConfig())
        for clip in (np.array([np.nan]), np.array([1e300]), np.zeros((2, 2))):
            with self.assertRaises(ValueError):
                normalizer.transform(clip)
            np.testing.assert_array_equal(normalizer._state, [0])

    def test_configuration_and_memory_schema(self):
        for value in (0, np.nan, 11):
            with self.assertRaises(ValueError):
                GainAttentionConfig(gain_time_s=value)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"gain.npz"
            settings = replace(GainAttentionConfig(), gain_time_s=.05)
            GainForegroundLearner(settings).save(path)
            restored = GainForegroundLearner.load(path)
            self.assertEqual(restored.settings, settings)
            np.testing.assert_array_equal(restored._normalizer._state, [0])
            with self.assertRaises(ValueError):
                BandForegroundLearner.load(path)
