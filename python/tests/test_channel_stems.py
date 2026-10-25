"""Linear component/tail accounting, SNR calibration and independent transforms."""

from dataclasses import replace
import unittest

import numpy as np

from chuff_mpu.attention import AttentionConfig, Observation
from chuff_mpu.band_attention import BandForegroundLearner
from chuff_mpu.sim.attention_experiment import evaluate
from chuff_mpu.sim.channel_metrics import condition_contrasts
from chuff_mpu.sim.independent_scene import IndependentSourceScene, transform_source
from chuff_mpu.sim.scene import AcousticScene, SceneConfig, SceneFrame
from chuff_mpu.sim.varied_scene import VariedScene


class ChannelStemsTests(unittest.TestCase):
    def test_cross_condition_uncertainty_preserves_seed_pairs(self):
        summaries = []
        for seed, value in enumerate((.2, .4, .7)):
            for transform in ("reference", "pitch", "duration", "pitch_duration"):
                for reference in ("input", "post_residual"):
                    score = value-(.02 if reference == "post_residual" else 0)-(.05 if transform != "reference" else 0)
                    summaries.append(dict(seed=seed, suite=transform+"/"+reference, metrics={"test": {}},
                        localization={"test": {"localized_repeat_recognition_rate": {"0.25": score}}}))
        first = condition_contrasts(summaries)[0]
        np.testing.assert_allclose(first["bootstrap_95"], [-.02, -.02])

    def test_probes_preserve_the_complete_real_decision_trajectory(self):
        settings = SceneConfig(rounds=20, event_probability=1, nuisance_probability=0, collect_stems=True)
        frames = list(AcousticScene(105, settings))
        summary, rows, _ = evaluate(0, settings, AttentionConfig(), learners={"band": BandForegroundLearner()}, frames=frames)
        probed_summary, probed_rows, _ = evaluate(0, settings, AttentionConfig(),
            learners={"band": BandForegroundLearner()}, frames=frames, counterfactual_probes=True)
        self.assertEqual(summary, probed_summary)
        self.assertEqual(rows, [{k: v for k, v in r.items() if k != "without_current_probe"} for r in probed_rows])

    def test_stems_preserve_mixture_and_recombine_with_persistent_channel_state(self):
        settings = SceneConfig(rounds=20, event_probability=.8, nuisance_probability=0)
        plain = AcousticScene(101, settings)
        stems = AcousticScene(101, replace(settings, collect_stems=True))
        for a, b in zip(plain, stems):
            np.testing.assert_array_equal(a.audio, b.audio)
            np.testing.assert_allclose(b.audio, b.background_audio+b.source_audio, atol=2e-8, rtol=2e-6)
            np.testing.assert_allclose(b.audio, b.without_current_source_audio+b.current_source_audio,
                                       atol=2e-8, rtol=2e-6)
            if b.label is None:
                np.testing.assert_array_equal(b.audio, b.without_current_source_audio)

    def test_identity_channel_has_analytic_snr_and_post_calibration_is_exact(self):
        settings = SceneConfig(rounds=10, warmup=1, event_probability=1, nuisance_probability=0,
                               room_decay_s=0, collect_stems=True, snr_db=-6)
        scene = AcousticScene(102, settings)
        scene._mic_b, scene._mic_a, scene._mic_state = np.ones(1), np.ones(1), np.zeros(0)
        scene._background = lambda n: np.full(n, .02)
        for frame in scene:
            if frame.label is not None:
                self.assertAlmostEqual(frame.post_channel_background_snr_db, -6, places=10)
                self.assertAlmostEqual(frame.post_channel_residual_snr_db, -6, places=10)
        for frame in VariedScene(102, replace(settings, snr_reference="post_residual"),
                                 condition="joint", environment_seed=100102):
            if frame.label is not None:
                self.assertAlmostEqual(frame.post_channel_residual_snr_db, -6, places=9)
                np.testing.assert_allclose(frame.audio, frame.background_audio+frame.source_audio,
                                           atol=1e-7, rtol=3e-6)

    def test_current_removal_retains_previous_echoes(self):
        settings = SceneConfig(rounds=4, warmup=0, event_probability=0, nuisance_probability=0,
                               room_decay_s=.5, collect_stems=True)
        scene = AcousticScene(103, settings)
        scene._mic_b, scene._mic_a, scene._mic_state = np.ones(1), np.ones(1), np.zeros(0)
        scene._ir[:] = 0
        scene._ir[0], scene._ir[4000] = 1., .5
        # Inject known old source tail; mixture and decomposed source histories
        # must describe the same already-existing physical state.
        n = int(settings.sample_rate*settings.duration_s)
        scene._tail[:100] = .01
        scene._stem_history = dict(background=(np.zeros(len(scene._ir)-1), np.zeros(0)),
                                   source=(scene._tail.copy(), np.zeros(0)))
        frame = scene.next_frame()
        np.testing.assert_allclose(frame.source_audio[:100], .01)
        np.testing.assert_array_equal(frame.current_source_audio, np.zeros(n))
        np.testing.assert_array_equal(frame.audio, frame.without_current_source_audio)

    def test_pitch_and_duration_controls_are_independent_and_fit(self):
        rate = 16000
        tone = np.sin(2*np.pi*1000*np.arange(8000)/rate)
        pitched = transform_source(tone, rate, 1.1, 1)
        stretched = transform_source(tone, rate, 1, 1.2)
        peak = lambda x: np.fft.rfftfreq(len(x), 1/rate)[np.abs(np.fft.rfft(x)).argmax()]
        self.assertEqual(len(pitched), len(tone))
        self.assertEqual(len(stretched), 9600)
        self.assertAlmostEqual(peak(pitched), 1100, delta=5)
        self.assertAlmostEqual(peak(stretched), 1000, delta=5)
        settings = SceneConfig(rounds=14, event_probability=1, nuisance_probability=0)
        scenes = {kind: list(IndependentSourceScene(104, settings, environment_seed=100104, transform=kind))
                  for kind in ("pitch", "duration", "pitch_duration")}
        for a, b, c in zip(*scenes.values()):
            self.assertEqual(a.label, b.label)
            self.assertEqual(a.label, c.label)
            if a.label is not None:
                self.assertEqual(a.source_parameters["pitch_ratio"], c.source_parameters["pitch_ratio"])
                self.assertEqual(b.source_parameters["applied_duration_factor"], c.source_parameters["applied_duration_factor"])
                self.assertEqual(a.source_parameters["applied_duration_factor"], 1)
                self.assertEqual(b.source_parameters["pitch_ratio"], 1)
                self.assertLessEqual(c.event_stop_s-c.event_start_s, .94+1/rate)

    def test_counterfactual_branch_cannot_update_the_real_learner(self):
        class ProbeLearner:
            def __init__(self): self.calls = 0
            @property
            def memory_size(self): return 0
            def observe(self, clip):
                self.calls += 1
                return Observation("candidate" if np.any(clip) else "background")
        learner = ProbeLearner()
        frames = [SceneFrame(np.ones(16000), "a", False, False, 0,
                             without_current_source_audio=np.zeros(16000)) for _ in range(4)]
        _, rows, _ = evaluate(0, SceneConfig(rounds=10), AttentionConfig(), learners={"test": learner},
                              frames=frames, counterfactual_probes=True)
        self.assertEqual(learner.calls, 4)
        self.assertTrue(all(r["status"] == "candidate" for r in rows))
        self.assertTrue(all(r["without_current_probe"]["status"] == "background" for r in rows))
