"""Temporal overlap distinguishes a coincident background trigger from an event."""

from dataclasses import replace
import unittest

import numpy as np

from chuff_mpu.band_attention import BandForegroundLearner
from chuff_mpu.sim.localization import overlap_iou, score_regions
from chuff_mpu.sim.scene import AcousticScene, SceneConfig


class LocalizationTests(unittest.TestCase):
    def row(self, **changes):
        row = dict(mode="test", truth="a", nuisance=False, status="candidate", prototype_id=None,
                   event_start_s=.6, event_stop_s=.9, region_start_s=.1, region_stop_s=.3,
                   previous_source_tail_s=0)
        row.update(changes)
        return row

    def test_wrong_region_is_not_a_localized_detection(self):
        result = score_regions([self.row()])["test"]
        self.assertEqual(result["counts"]["window_detections"], 1)
        self.assertEqual(result["localized_detection_rate"]["0.25"], 0)
        result = score_regions([self.row(region_start_s=.6, region_stop_s=.9)])["test"]
        self.assertEqual(result["localized_detection_rate"]["0.5"], 1)

    def test_recognition_needs_both_correct_identity_and_overlap(self):
        rows = [self.row(status="learned", prototype_id=1), self.row(),
                self.row(status="recognised", prototype_id=1),
                self.row(status="recognised", prototype_id=1, region_start_s=.6, region_stop_s=.9)]
        result = score_regions(rows)["test"]
        self.assertEqual(result["counts"]["repeat_windows"], 2)
        self.assertEqual(result["localized_repeat_recognition_rate"]["0.25"], .5)

    def test_missing_timing_is_unknown_and_room_tails_are_separate(self):
        result = score_regions([self.row(event_start_s=None, event_stop_s=None)])["test"]
        self.assertIsNone(result["localized_detection_rate"]["0.25"])
        result = score_regions([self.row(truth=None, previous_source_tail_s=.1),
                                self.row(truth=None)])["test"]["counts"]
        self.assertEqual(result["tail_attention_windows"], 1)
        self.assertEqual(result["no_room_tail_attention_windows"], 1)

    def test_interval_math_and_invalid_spans(self):
        self.assertAlmostEqual(overlap_iou((0, .5), (.25, .75)), 1/3)
        with self.assertRaises(ValueError):
            overlap_iou((.5, .1), (0, 1))

    def test_scene_annotations_fit_and_do_not_depend_on_snr(self):
        settings = SceneConfig(rounds=25, event_probability=1)
        first = list(AcousticScene(81, settings))
        second = list(AcousticScene(81, replace(settings, snr_db=-6)))
        for a, b in zip(first, second):
            self.assertEqual((a.event_start_s, a.event_stop_s), (b.event_start_s, b.event_stop_s))
            if a.label is not None:
                self.assertTrue(0 <= a.event_start_s < a.event_stop_s <= settings.duration_s)
        learner = BandForegroundLearner()
        detections = []
        for frame in first:
            result = learner.observe(frame.audio)
            if result.status in ("candidate", "learned", "recognised"):
                detections.append(result)
                self.assertTrue(0 <= result.region_start_s < result.region_stop_s <= 1)
        self.assertTrue(detections)
