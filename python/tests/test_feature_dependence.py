"""Denominator, lifecycle and context controls for the feature diagnostic."""

from copy import deepcopy
import unittest

import numpy as np

from chuff_mpu.sim.feature_dependence import admission_pairs, summarize_records
from chuff_mpu.sim.feature_dependence_report import derive, ratio_estimate, tally, window_context


def record(candidate=0, birth=True, match=True, actual_match=True, zero=False):
    feature = f"feature-{candidate}-{'birth' if birth else 'admission'}"
    actual = dict(feature_sha256=feature, feature_norm=0. if zero else 1.,
        matches_stored=actual_match, exact_actual_feature=True, region_start_s=0., region_stop_s=.5, region_iou=1.)
    bed = dict(actual, matches_stored=match, exact_actual_feature=False,
        feature_sha256="bed" if match else None, feature_norm=1. if match else None)
    return dict(mode="m", suite="recorded/0/positive", seed=1, step=3*candidate+(0 if birth else 2),
        candidate_id=candidate, prototype_id=None if birth else candidate, vector_index=0 if birth else 1,
        vector_file="vectors/test.npz", is_birth=birth, is_confirmation=not birth, is_admission=not birth,
        stored_feature_sha256=f"feature-{candidate}-birth", stored_feature_norm=0. if zero else 1.,
        observations=dict(actual=actual, bed=bed, without_current=dict(bed)),
        waveform_sha256=dict(actual="mixture", bed="background", without_current="background"),
        window_context=dict(recording_id="dog", previous_source_tail_s=0.))


def admission(first, last):
    return dict(mode="m", candidate_id=first["candidate_id"], prototype_id=last["prototype_id"],
        birth_step=first["step"], admission_step=last["step"],
        birth_feature_sha256=first["observations"]["actual"]["feature_sha256"],
        stored_feature_sha256=last["stored_feature_sha256"],
        admission_observed_feature_sha256=last["observations"]["actual"]["feature_sha256"])


class FeatureDependenceTests(unittest.TestCase):
    def test_all_windows_include_unadmitted_zero_birth(self):
        rows = [record(), record(birth=False, match=False), record(1, zero=True, actual_match=False)]
        counts = summarize_records(rows, ["m"])["m"]
        self.assertEqual([counts[s]["windows"] for s in ("all", "birth", "confirmation", "admission")], [3, 2, 1, 1])
        self.assertEqual(counts["all"]["actual_zero"], 1)
        for name in ("actual_outside_radius", "probe_no_feature", "probe_matches_stored"):
            self.assertEqual(counts["all"]["bed/"+name], 1)

    def test_joint_pairs_and_vector_pointers(self):
        rows, admissions = [], []
        for candidate, (first_match, last_match, actual_match) in enumerate(
                [(True, True, True), (True, False, True), (False, True, True), (False, False, True), (True, True, False)]):
            first = record(candidate, match=first_match, actual_match=actual_match)
            last = record(candidate, birth=False, match=last_match)
            rows.extend([first, last])
            admissions.append(admission(first, last))
        pairs = admission_pairs(rows, admissions)
        self.assertEqual([r["joint"] for r in pairs], ["both_reproducible", "birth_only_reproducible",
            "admission_only_reproducible", "neither_reproducible", "unmatchable_actual"])
        self.assertEqual(pairs[0]["birth_vector_file"], "vectors/test.npz")
        self.assertEqual(pairs[0]["admission_vector_index"], 1)

    def test_history_mismatch_rejected(self):
        first, last = record(), record(birth=False)
        old = admission(first, last)
        for field, value in (("birth_step", 7), ("stored_feature_sha256", "wrong"), ("prototype_id", 17)):
            with self.subTest(field=field), self.assertRaises(ValueError):
                admission_pairs([first, last], [dict(old, **{field: value})])
        for rows in ([first], [last], [first, first, last]):
            with self.assertRaises(ValueError):
                admission_pairs(rows, [old])

    def test_region_change_is_counted_separately(self):
        row = record()
        row["observations"]["bed"].update(region_start_s=.6, region_stop_s=.9, region_iou=0.)
        counts = tally([row])
        self.assertEqual(counts["bed/probe_matches_stored"], 1)
        self.assertEqual(counts["bed_match_changed_region"], 1)
        self.assertEqual(counts["bed_match_disjoint_region"], 1)

    def test_contexts_distinguish_recording_tail_and_controls(self):
        row = record()
        self.assertEqual(window_context(row), "recording")
        row["window_context"].update(recording_id=None, previous_source_tail_s=.2)
        self.assertEqual(window_context(row), "gap_with_tail")
        row["window_context"]["previous_source_tail_s"] = 0.
        self.assertEqual(window_context(row), "gap_without_tail")
        row["suite"] = "recorded/0/control"
        self.assertEqual(window_context(row), "control")

    def test_equal_seed_weights_and_undefined_denominators(self):
        draws = np.array([[0, 0], [1, 1], [0, 1]])
        counts = [tally([record()]), tally([record(match=False) for _ in range(9)])]
        result = ratio_estimate(counts, "bed", draws)
        self.assertEqual(result["pooled_fraction"], .1)
        self.assertEqual(result["mean"], .5)
        result = ratio_estimate([counts[0], tally([])], "bed", draws)
        self.assertEqual(result["per_seed"], [1., None])
        self.assertEqual(result["valid_seeds"], 1)
        self.assertIsNone(result["mean"])
        self.assertIsNone(result["bootstrap_95"])

    def test_suite_aggregation_precedes_seed_bootstrap(self):
        rows = [record(), dict(record(1, match=False), suite="modulated/0/positive"),
                dict(record(2), seed=2)]
        plan = dict(evaluation_seeds=[1, 2], positive_suites=["recorded/0/positive", "modulated/0/positive"], control_suites=[])
        report = derive(rows, [], plan, ["m"])
        birth = report["modes"]["m"]["contexts"]["recording"]["birth"]["reproducible"]["bed"]
        self.assertEqual(birth["per_seed"], [.5, 1.])
        self.assertEqual(birth["mean"], .75)
        changed = report["modes"]["m"]["contexts"]["recording_changed_audio"]["birth"]["counts"]
        self.assertEqual(changed["windows"], 3)
        noop = dict(record(3), waveform_sha256=dict(actual="same", bed="same", without_current="same"))
        subset = derive(rows+[noop], [], plan, ["m"])["modes"]["m"]["contexts"]
        self.assertEqual(subset["recording"]["birth"]["counts"]["windows"], 4)
        self.assertEqual(subset["recording_changed_audio"]["birth"]["counts"]["windows"], 3)
        with self.assertRaises(ValueError):
            derive(rows+[deepcopy(rows[0])], [], plan, ["m"])
        with self.assertRaises(ValueError):
            derive([dict(record(), seed=9)], [], plan, ["m"])


if __name__ == "__main__":
    unittest.main()
