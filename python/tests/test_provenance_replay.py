from copy import deepcopy
import unittest

import numpy as np

from chuff_mpu.attention import ForegroundLearner
from chuff_mpu.sim.memory_provenance import instrument
from chuff_mpu.sim.provenance_replay import join_trace, score
from chuff_mpu.sim.provenance_report import derive


class ProvenanceReplayTests(unittest.TestCase):
    def fixture(self, birth=None, admission="target", target="target", tail=0.):
        model = instrument(ForegroundLearner())
        vector = np.zeros(96, dtype=np.float32); vector[0] = 1
        later = .95*vector
        later[1] = np.sqrt(1-.95**2)
        rows = []
        for tick, label, occurrence, feature in ((1, birth, 1, vector), (3, admission, 2, later), (11, target, 3, vector)):
            model._tick = tick
            observation = model._remember(feature, 2.)
            rows.append(dict(mode="test", step=tick-1, episode_id=tick-1 if label is not None else None,
                recording_id=label, occurrence=occurrence if label is not None else None,
                status=observation.status, prototype_id=observation.prototype_id,
                previous_source_tail_s=tail if tick == 1 else 0., current_episode_window_rms=.1 if label else 0.,
                without_episode_probe=dict(status="background", prototype_id=None)))
        events = [dict(event, mode="test") for event in model.provenance_events]
        return rows, events

    def test_background_birth_foreground_admission_changes_identity_without_changing_truth(self):
        rows, events = self.fixture()
        original = deepcopy(rows)
        joined, admissions, counts = join_trace(rows, events)
        self.assertEqual(rows, original)
        self.assertEqual(counts["test"]["background_to_foreground"], 1)
        self.assertEqual(counts["test"]["bed_only_birth"], 1)
        self.assertEqual(counts["test"]["stored_differs_from_admission_observation"], 1)
        self.assertEqual(admissions[0]["birth_context"]["recording_id"], None)
        self.assertEqual(joined[0]["window_context"]["recording_id"], None)
        scores = score(rows, admissions)
        self.assertEqual(scores["admission"]["test"]["rates"]["strict_episode_exclusion"], 1)
        self.assertEqual(scores["birth"]["test"]["rates"]["strict_episode_exclusion"], 0)
        self.assertEqual(scores["birth"]["test"]["rates"]["wrong_repeat_identity"], 1)

    def test_foreground_birth_background_admission_can_reverse_the_score_difference(self):
        rows, events = self.fixture(birth="target", admission=None)
        _, admissions, counts = join_trace(rows, events)
        self.assertEqual(counts["test"]["foreground_to_background"], 1)
        scores = score(rows, admissions)
        self.assertEqual(scores["admission"]["test"]["rates"]["strict_episode_exclusion"], 0)
        self.assertEqual(scores["birth"]["test"]["rates"]["strict_episode_exclusion"], 1)

    def test_cross_recording_and_same_recording_labels_are_distinguished(self):
        for birth, expected in (("other", "different_recording"), ("target", "same_recording")):
            rows, events = self.fixture(birth=birth)
            _, admissions, counts = join_trace(rows, events)
            self.assertEqual(counts["test"][expected], 1)
            scores = score(rows, admissions)
            self.assertEqual(scores["birth"]["test"]["rates"]["strict_episode_exclusion"], int(birth == "target"))

    def test_tail_contaminated_background_birth_is_not_called_bed_only(self):
        rows, events = self.fixture(tail=.15)
        _, _, counts = join_trace(rows, events)
        self.assertEqual(counts["test"]["background_birth_with_room_tail"], 1)
        self.assertEqual(counts["test"]["bed_only_birth"], 0)

    def test_missing_or_corrupt_lifecycle_events_are_rejected(self):
        rows, events = self.fixture()
        for kind in ("candidate_created", "prototype_admitted", "candidate_observed", "prototype_recognised"):
            incomplete = deepcopy(events)
            index = next(i for i, event in enumerate(incomplete) if event["kind"] == kind)
            incomplete.pop(index)
            with self.assertRaises(ValueError):
                join_trace(rows, incomplete)
        for field, value in (("stored_feature_sha256", "0"*64), ("birth_tick", 2), ("hits", 99)):
            corrupt = deepcopy(events)
            item = next(event for event in corrupt if event["kind"] == "prototype_admitted")
            item[field] = value
            with self.assertRaises(ValueError):
                join_trace(rows, corrupt)
        with self.assertRaises(ValueError):
            join_trace(rows+[rows[0]], events)

    def test_source_removed_recognition_is_rescored_under_the_same_label_mapping(self):
        rows, events = self.fixture(birth="target", admission=None)
        rows[-1]["without_episode_probe"] = dict(status="recognised", prototype_id=0)
        _, admissions, _ = join_trace(rows, events)
        scores = score(rows, admissions)
        self.assertEqual(scores["birth"]["test"]["rates"]["repeat_identity"], 1)
        self.assertEqual(scores["birth"]["test"]["rates"]["deletion_repeat_identity"], 1)
        self.assertEqual(scores["birth"]["test"]["rates"]["strict_episode_exclusion"], 0)

    def test_paired_seed_report_keeps_the_label_difference_paired_across_suites(self):
        rows, events = self.fixture()
        _, admitted, transitions = join_trace(rows, events)
        scores = score(rows, admitted)
        plan = dict(evaluation_seeds=[1, 2, 3],
            positive_suites=["recorded/0/positive", "modulated/0/positive"],
            control_suites=["recorded/0/control"])
        summaries = []
        for suite in plan["positive_suites"]+plan["control_suites"]:
            for seed, base in zip(plan["evaluation_seeds"], (.2, .5, .8)):
                current = deepcopy(scores)
                offset = .1 if suite.startswith("modulated") else -.1
                current["admission"]["test"]["rates"]["strict_episode_exclusion"] = base+offset
                current["birth"]["test"]["rates"]["strict_episode_exclusion"] = base+offset-.1
                summaries.append(dict(suite=suite, seed=seed, scores=current, transitions=transitions))
        report = derive(summaries, plan, ["test"])
        difference = report["modes"]["test"]["metrics"]["strict_episode_exclusion"]["birth_minus_admission"]
        self.assertAlmostEqual(difference["mean"], -.1)
        np.testing.assert_allclose(difference["bootstrap_95"], [-.1, -.1])
        self.assertEqual(report["modes"]["test"]["transitions"]["positive"]["admissions"], 6)
        for invalid in (summaries[:-1], summaries+[summaries[0]]):
            with self.assertRaises(ValueError):
                derive(invalid, plan, ["test"])

    def test_explicit_stored_vector_changes_are_not_silently_treated_as_birth_bytes(self):
        model = instrument(ForegroundLearner())
        first = np.zeros(96, dtype=np.float32); first[0] = 1
        second = np.zeros(96, dtype=np.float32); second[1] = 1
        observations = []
        model._tick = 1
        observations.append(model._remember(first, 2.))
        model._candidates[0].embedding = second.copy()
        model._tick = 3
        observations.append(model._remember(second, 2.))
        rows = [dict(mode="test", step=tick-1, episode_id=None, recording_id=None, occurrence=None,
            status=observation.status, prototype_id=observation.prototype_id,
            previous_source_tail_s=0., current_episode_window_rms=0.)
            for tick, observation in zip((1, 3), observations)]
        events = [dict(event, mode="test") for event in model.provenance_events]
        _, admissions, counts = join_trace(rows, events)
        self.assertEqual(counts["test"]["stored_hash_changed_since_birth"], 1)
        self.assertNotEqual(admissions[0]["birth_feature_sha256"], admissions[0]["stored_feature_sha256"])
