"""Broad-region and paired background controls prevent treating raw IoU as attribution."""

import unittest

from chuff_mpu.sim.overlap_controls import alternative_position_overlap, analyze_rows


class OverlapControlsTests(unittest.TestCase):
    def test_whole_window_has_no_alignment_information_for_fixed_length(self):
        self.assertEqual(alternative_position_overlap((0, 1), .3, 1, .25), 1)
        self.assertEqual(alternative_position_overlap((0, 1), .3, 1, .5), 0)
        self.assertEqual(alternative_position_overlap(None, .3, 1, .25), 0)

    def test_nulls_preserve_generator_margins_and_lengths(self):
        self.assertEqual(alternative_position_overlap((0, .1), .3, 1, .1), 0)
        self.assertEqual(alternative_position_overlap((0, 1), 1, 1, .5), 1)
        with self.assertRaises(ValueError):
            alternative_position_overlap((0, 1), 1.1, 1, .25)

    def test_matched_background_cancels_coincident_whole_window_detection(self):
        foreground = dict(suite="test/foreground", seed=1, step=3, mode="test", truth="a",
            nuisance=False, status="candidate", event_start_s=.2, event_stop_s=.5,
            region_start_s=0., region_stop_s=1.)
        background = dict(foreground, suite="test/ambience", truth=None)
        result = analyze_rows([foreground, background])["estimates"]
        primary = next(row for row in result if row["iou_threshold"] == .25)
        self.assertEqual(primary["metrics"]["observed"]["mean"], 1)
        self.assertEqual(primary["metrics"]["excess_over_background"]["mean"], 0)
        self.assertEqual(primary["metrics"]["excess_over_alternatives"]["mean"], 0)
