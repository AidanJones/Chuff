import unittest

from chuff_mpu.sim.guarded_analysis import guarded_analysis


class GuardedAnalysisTests(unittest.TestCase):
    def test_missing_class_is_reported_instead_of_becoming_fast_learning(self):
        rates = dict(localized_repeat=.5, localized_detection=.75, false_attention=0)
        result = guarded_analysis([dict(suite="example", seed=1,
            phases={mode: dict(after=dict(rates=rates, counts={}))
                    for mode in ("adaptive_gain", "guarded_gain")},
            diagnostics={"adaptive_gain": dict(first_admission_occurrence={"a": 2, "b": 2}),
                         "guarded_gain": dict(first_admission_occurrence={"a": 4})})])[0]
        self.assertEqual(result["cohorts"][0]["lost_classes"], ["b"])
        self.assertEqual(result["paired_differences"]["admitted_classes"]["mean"], -1)
        self.assertEqual(result["paired_differences"]["matched_class_first_admission_occurrences"]["mean"], 2)
