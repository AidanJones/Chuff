from copy import deepcopy
from pathlib import Path
import runpy
import tempfile
import unittest

import librosa
import numpy as np
from scipy.signal import stft

from chuff_mpu.band_attention import BandAttentionConfig, BandForegroundLearner
from chuff_mpu.published_attention import (PCENConfig, PCENState, PCENForegroundLearner,
    MCRAConfig, MCRAState, MCRAForegroundLearner)
from chuff_mpu.sim.challenge_scene import ChallengeScene
from chuff_mpu.sim.scene import SceneConfig


class PublishedAttentionTests(unittest.TestCase):
    def test_pcen_streaming_matches_library_one_shot(self):
        settings = PCENConfig()
        rng = np.random.default_rng(2800)
        magnitude = np.exp(rng.normal(size=(32, 213)))*1e6
        expected, state = librosa.pcen(magnitude, sr=16000, hop_length=160, return_zf=True)
        model = PCENState(settings)
        actual = np.concatenate([model.process(magnitude[:, :17]), model.process(magnitude[:, 17:100]),
                                 model.process(magnitude[:, 100:])], axis=1)
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(model._zi, state, atol=1e-12, rtol=1e-12)

    def test_pcen_magnitude_matches_unnormalized_librosa_mel_reference(self):
        rng = np.random.default_rng(2801)
        clip = rng.normal(size=16000)*.02
        _, _, spectrum = stft(clip, fs=16000, nperseg=512, noverlap=352, boundary=None, padded=False)
        model = PCENForegroundLearner()
        actual = model._pcen_magnitude(np.abs(spectrum)**2)
        expected = librosa.feature.melspectrogram(y=clip, sr=16000, n_fft=512, hop_length=160,
            n_mels=32, fmin=100, fmax=8000, htk=True, norm="slaney", dtype=np.float64,
            center=False, power=1)*2**31
        np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-12)

    def test_pcen_warmup_processes_state_and_background(self):
        model = PCENForegroundLearner()
        result = model.observe(np.random.default_rng(2802).normal(size=16000)*.02)
        self.assertEqual(result.status, "calibrating")
        self.assertIsNotNone(model._pcen._zi)
        self.assertIsNotNone(model._pcen_background)
        self.assertTrue(np.isfinite(model._pcen_relative).all())

    def test_mcra_equation_noise_prediction_and_presence_probability(self):
        model = MCRAState(MCRAConfig())
        noise = model.process(np.array([[2., 4., 6.]]))
        np.testing.assert_allclose(noise, [[2., 2., 2.1]], atol=1e-14)
        model = MCRAState(MCRAConfig())
        model.process(np.array([[2., 100.]]))
        np.testing.assert_allclose(model._presence, [.8], atol=1e-14)
        # alpha=.95+.05*.8=.99; next noise=.99*2+.01*100.
        np.testing.assert_allclose(model._noise, [2.98], atol=1e-14)

    def test_mcra_chunk_parity_and_minimum_reset_tracks_persistent_noise_step(self):
        c = MCRAConfig(mcra_min_frames=10)
        power = np.ones((3, 80)); power[:, 10:] = 100
        a, b = MCRAState(c), MCRAState(c)
        expected = a.process(power)
        actual = np.concatenate([b.process(power[:, :7]), b.process(power[:, 7:23]),
                                 b.process(power[:, 23:])], axis=1)
        np.testing.assert_array_equal(actual, expected)
        self.assertGreater(actual[0, -1], 90)
        self.assertLessEqual(actual.max(), 100)
        self.assertEqual(b.frames, 80)

    def test_invalid_spectral_input_preserves_stream_state(self):
        for model in (MCRAState(MCRAConfig()), PCENState(PCENConfig())):
            model.process(np.ones((4, 10)))
            before = deepcopy(model)
            for bad in (np.full((4, 3), np.nan), -np.ones((4, 3)), np.ones((5, 3)), np.empty((4, 0))):
                with self.assertRaises(ValueError):
                    model.process(bad)
            np.testing.assert_array_equal(model.process(np.ones((4, 3))), before.process(np.ones((4, 3))))

    def test_refactored_band_is_exactly_equal_to_frozen_recorded_pilot(self):
        root = Path(__file__).resolve().parents[2]
        source = root/"papers/results/masked-events/recorded-evaluation/source/python/src/chuff_mpu/band_attention.py"
        old = runpy.run_path(str(source))
        frames = list(ChallengeScene(2803, SceneConfig(rounds=60, snr_db=-6, nuisance_probability=0)))
        for feature_mode in (0, 1):
            a = BandForegroundLearner(BandAttentionConfig(feature_mode=feature_mode))
            b = old["BandForegroundLearner"](old["BandAttentionConfig"](feature_mode=feature_mode))
            for frame in frames:
                first, second = a.observe(frame.audio), b.observe(frame.audio)
                self.assertEqual((first.status, first.prototype_id, first.novelty, first.salience,
                                  first.region_start_s, first.region_stop_s),
                                 (second.status, second.prototype_id, second.novelty, second.salience,
                                  second.region_start_s, second.region_stop_s))
                if first.embedding is not None:
                    np.testing.assert_array_equal(first.embedding, second.embedding)

    def test_new_schemas_preserve_memories_but_reset_frontend_state(self):
        for cls in (PCENForegroundLearner, MCRAForegroundLearner):
            model = cls()
            model.observe(np.ones(16000)*.02)
            vector = (np.ones(96)/np.sqrt(96)).astype(np.float32)
            model._tick = 4; model._remember(vector, 2)
            model._tick = 6; observation = model._remember(vector, 2)
            self.assertEqual(observation.status, "learned")
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory)/"state.npz"
                model.save(path)
                loaded = cls.load(path)
                self.assertEqual(loaded.memory_size, 1)
                np.testing.assert_array_equal(loaded._patterns[0].embedding, model._patterns[0].embedding)
                if cls is PCENForegroundLearner:
                    self.assertIsNone(loaded._pcen._zi)
                else:
                    self.assertIsNone(loaded._mcra._noise)
                    self.assertEqual(loaded._mcra.frames, 0)
                with self.assertRaises(ValueError):
                    BandForegroundLearner.load(path)
