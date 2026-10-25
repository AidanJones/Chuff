from copy import deepcopy
import unittest

import numpy as np

from chuff_mpu.band_attention import BandAttentionConfig, BandForegroundLearner
from chuff_mpu.sim.recorded_experiment import evaluate
from chuff_mpu.sim.recorded_metrics import episode_metrics
from chuff_mpu.sim.recorded_scene import RecordedScene
from chuff_mpu.sim.scene import SceneConfig


class RecordedSceneTests(unittest.TestCase):
    def scene(self, control=False):
        rate = 8000
        rng = np.random.default_rng(2600)
        t = np.arange(rate*5)/rate
        audio = dict(target=np.sin(2*np.pi*900*t)*(.6+.4*np.sin(2*np.pi*3*t)**2),
                     background=rng.normal(size=len(t)))
        settings = SceneConfig(sample_rate=rate, duration_s=5, rounds=8, warmup=2,
            nuisance_probability=0, background_change=False, collect_stems=True,
            snr_reference="post_residual", snr_db=-6)
        return RecordedScene(2600, settings, environment_seed=202600, audio=audio,
            foregrounds=("target",), backgrounds=("background",), control=control, repetitions=3)

    def test_full_episodes_post_snr_components_and_matched_background(self):
        positive, control = self.scene(), self.scene(True)
        labels = []
        for frame, removed in zip(positive, control):
            self.assertEqual(len(frame.audio), 5*8000)
            np.testing.assert_array_equal(frame.background_audio, removed.background_audio)
            np.testing.assert_allclose(frame.audio, frame.background_audio+frame.source_audio, atol=1e-7)
            np.testing.assert_allclose(frame.audio, frame.without_current_source_audio+frame.current_source_audio, atol=1e-7)
            if frame.label:
                labels.append(frame.label)
                self.assertAlmostEqual(frame.post_channel_residual_snr_db, -6, places=8)
                actual = 20*np.log10(np.linalg.norm(frame.current_source_audio.astype(float))/
                                     np.linalg.norm(frame.without_current_source_audio.astype(float)))
                self.assertAlmostEqual(actual, -6, places=4)
        self.assertEqual(labels, ["target"]*3)

    def test_source_removal_branch_does_not_change_real_trajectory(self):
        a = BandForegroundLearner(BandAttentionConfig(sample_rate=8000))
        b = deepcopy(a)
        without, physical_a = evaluate(self.scene(), {"band": a}, probes=False)
        with_probes, physical_b = evaluate(self.scene(), {"band": b}, probes=True)
        self.assertEqual(physical_a, physical_b)
        for first, second in zip(without, with_probes):
            second.pop("without_episode_probe", None)
            self.assertEqual(first, second)

    def test_same_episode_learning_does_not_count_as_prior_repeat_identity(self):
        rows = []
        for occurrence, episode in ((1, 2), (2, 4), (3, 6)):
            for offset in range(5):
                # A new prototype is only learned INSIDE episode three.
                status = "candidate" if occurrence < 3 else ("learned" if offset == 0 else "recognised")
                rows.append(dict(mode="m", episode_id=episode, recording_id="target", occurrence=occurrence,
                    status=status, prototype_id=0 if occurrence == 3 else None,
                    without_episode_probe=dict(status="background", prototype_id=None)))
        result = episode_metrics(rows)["m"]
        self.assertEqual(result["counts"]["eligible_repeat_episodes"], 1)
        self.assertEqual(result["rates"]["repeat_identity"], 0)
        for offset in range(5):
            rows.append(dict(mode="m", episode_id=8, recording_id="target", occurrence=4,
                status="recognised", prototype_id=0,
                without_episode_probe=dict(status="background", prototype_id=None)))
        result = episode_metrics(rows)["m"]
        self.assertEqual(result["rates"]["repeat_identity"], .5)
        self.assertEqual(result["rates"]["source_dependent_repeat"], .5)

    def test_stricter_exclusion_rejects_a_different_deleted_window_match(self):
        rows = [dict(mode="m", episode_id=2, recording_id="target", occurrence=1,
                     status="learned", prototype_id=0,
                     without_episode_probe=dict(status="background", prototype_id=None))]
        for offset in range(5):
            rows.append(dict(mode="m", episode_id=6, recording_id="target", occurrence=3,
                status="recognised", prototype_id=0,
                without_episode_probe=dict(status="recognised" if offset == 1 else "background",
                                           prototype_id=0 if offset == 1 else None)))
        result = episode_metrics(rows, details=True, exclusion=True)["m"]
        self.assertEqual(result["rates"]["source_dependent_repeat"], 1)
        self.assertEqual(result["rates"]["strict_episode_exclusion"], 0)
        self.assertEqual(result["by_recording"]["target"]["eligible_repeat_episodes"], 1)
