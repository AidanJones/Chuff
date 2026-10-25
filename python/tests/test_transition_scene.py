import unittest

import numpy as np

from chuff_mpu.sim.scene import SceneConfig
from chuff_mpu.sim.transition_metrics import phase_metrics
from chuff_mpu.sim.transition_scene import TransitionScene
from chuff_mpu.sim.transition_adaptation import adaptation_metrics


class TransitionTests(unittest.TestCase):
    def test_paired_schedule_and_components_through_change(self):
        settings = SceneConfig(rounds=32, background_change=False, nuisance_probability=0,
            collect_stems=True, snr_reference="post_residual", snr_db=-6)
        a = list(TransitionScene(1900, settings, environment_seed=101900, case="stationary"))
        b = list(TransitionScene(1900, settings, environment_seed=101900, case="stationary_fast"))
        for step, (left, right) in enumerate(zip(a, b)):
            self.assertEqual((left.label, left.event_start_s, left.event_stop_s),
                             (right.label, right.event_start_s, right.event_stop_s))
            if step < 16:
                np.testing.assert_array_equal(left.audio, right.audio)
            self.assertEqual(right.background_changed, step == 16)
            np.testing.assert_allclose(right.audio, right.background_audio+right.source_audio, atol=1e-7)
            if right.label is not None:
                self.assertAlmostEqual(right.post_channel_residual_snr_db, -6, places=8)
        self.assertFalse(np.array_equal(a[-1].audio, b[-1].audio))

    def test_silent_start_and_spectral_change(self):
        settings = SceneConfig(rounds=32, background_change=False, nuisance_probability=0)
        silent = list(TransitionScene(1901, settings, environment_seed=101901, case="silent_fast"))
        for frame in silent[:6]:
            np.testing.assert_array_equal(frame.audio, 0)
        self.assertGreater(np.linalg.norm(silent[6].audio), 0)
        self.assertTrue(silent[6].background_changed)
        a = list(TransitionScene(1901, settings, environment_seed=101901, case="fast"))
        b = list(TransitionScene(1901, settings, environment_seed=101901, case="spectral_fast"))
        for left, right in zip(a[:16], b[:16]):
            np.testing.assert_array_equal(left.audio, right.audio)
        self.assertFalse(np.array_equal(a[-1].audio, b[-1].audio))

    def test_phase_scoring_retains_prior_identity_and_repeat_history(self):
        rows = []
        for step, status in enumerate(("candidate", "learned", "recognised")):
            rows.append(dict(mode="listener", phase="before" if step < 2 else "after",
                truth="call", status=status, prototype_id=7, event_start_s=.2, event_stop_s=.5,
                region_start_s=.2, region_stop_s=.5,
                without_current_probe=dict(status="background", current_class_recognised=False)))
        result = phase_metrics(rows)["listener"]
        self.assertEqual(result["after"]["rates"]["localized_repeat"], 1)
        self.assertEqual(result["after"]["counts"]["repeats"], 1)
        self.assertEqual(result["before"]["rates"]["localized_repeat"], None)

    def test_latency_requires_sustained_mode_and_counts_from_change(self):
        values = [False]*6+[True, False]+[True]*5
        rows = [dict(mode="adaptive_gain", step=step, gain_compensation_enabled=value,
                     mode_switches=int(step >= 6)+int(step >= 7)+int(step >= 8))
                for step, value in enumerate(values)]
        result = adaptation_metrics(rows, "stationary_fast", 6)["adaptive_gain"]
        self.assertEqual(result["first_five_window_correct_mode_delay"], 2)
        self.assertEqual(result["wrong_mode_windows_after"], 1)
        self.assertEqual(result["switches_after"], 3)


if __name__ == "__main__":
    unittest.main()
