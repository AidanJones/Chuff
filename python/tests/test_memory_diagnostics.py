"""Check error accounting and source-free controls, not expected research gains."""

import unittest

import numpy as np

from chuff_mpu.sim.admission_experiment import make_scene
from chuff_mpu.sim.audit_admission import policy_contrasts
from chuff_mpu.sim.memory_diagnostics import diagnose


class MemoryDiagnosticsTests(unittest.TestCase):
    def test_partition_and_admission_time_identity(self):
        def row(label, status, identifier=None, nuisance=False):
            return dict(mode="test", truth=label, status=status,
                        prototype_id=identifier, nuisance=nuisance)
        rows = [row(None, "learned", 9, True), row(None, "learned", 10),
                row("a", "candidate"), row("a", "candidate"),
                row("a", "background"), row("a", "candidate"),
                row("a", "learned", 1), row("a", "learned", 2),
                row("a", "recognised", 9), row("a", "recognised", 1)]
        result = diagnose(rows)["test"]
        self.assertEqual(result["repeat_events"], 6)
        self.assertEqual(set(result["repeat_outcomes"].values()), {1})
        self.assertEqual(result["first_admission_occurrence"], {"a": 5})
        self.assertEqual(result["admissions_per_class"], {"a": 2})
        self.assertEqual(result["false_admissions_by_current_source"],
                         {"current_nuisance": 1, "no_current_source": 1})

    def test_pure_ambience_has_no_sources_and_snr_is_inapplicable(self):
        _, first = make_scene(61, 15, -6, "joint_ambience")
        _, second = make_scene(61, 15, 0, "joint_ambience")
        for a, b in zip(first, second):
            self.assertIsNone(a.label)
            self.assertFalse(a.nuisance)
            self.assertIsNone(a.active_snr_db)
            np.testing.assert_array_equal(a.audio, b.audio)

    def test_contrasts_resample_seed_pairs(self):
        runs = [dict(suite="test", snr_db=-6, seed=seed,
                     metrics={"quiet_band": dict(repeat_recognition_rate=value, event_detection_rate=value,
                                                  false_admissions=2),
                              "confirm4": dict(repeat_recognition_rate=value-.1, event_detection_rate=value,
                                               false_admissions=1)})
                for seed, value in enumerate([.2, .5, .8])]
        result = policy_contrasts(runs)
        np.testing.assert_allclose(result[0]["bootstrap_95"], [-.1, -.1])
        self.assertEqual(result[2]["mean"], -1)

    def test_foreground_conditions_share_schedule(self):
        schedules = []
        for suite in ("reference", "source", "joint"):
            _, scene = make_scene(62, 20, -6, suite)
            schedules.append([(f.label, f.nuisance) for f in scene])
        self.assertEqual(schedules[0], schedules[1])
        self.assertEqual(schedules[1], schedules[2])
