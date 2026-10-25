from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import numpy as np

from chuff_mpu.attention import AttentionConfig, ForegroundLearner, Observation
from chuff_mpu.band_attention import BandForegroundLearner
from chuff_mpu.gain_attention import SpectralGainForegroundLearner, CalibratedGainForegroundLearner
from chuff_mpu.adaptive_gain import AdaptiveGainForegroundLearner
from chuff_mpu.guarded_gain import GuardedGainForegroundLearner
from chuff_mpu.published_attention import PCENForegroundLearner, MCRAForegroundLearner
from chuff_mpu.sim.feature_probes import (FeatureProbeEvaluator, InterventionScene, feature_packet,
                                        match_dependence, validate_records, waveform_hash)
from chuff_mpu.sim.memory_provenance import feature_hash, instrument
from chuff_mpu.sim.recorded_experiment import evaluate
from chuff_mpu.sim.recorded_scene import RecordedScene
from chuff_mpu.sim.scene import SceneConfig


class SyntheticMemory(ForegroundLearner):
    """Deterministic test-only features; no recording labels are accepted."""
    def observe(self, audio):
        self._tick += 1
        if audio[0] == 3:
            return Observation("background")
        vector = np.zeros(96, dtype=np.float32)
        if audio[0] != 2:
            vector[0 if audio[0] else 1] = 1
        result = self._remember(vector, float(audio.mean()))
        return Observation(result.status, result.salience, result.novelty, result.prototype_id,
                           result.embedding, 0. if audio[1] else .6, .4 if audio[1] else 1.)


class FeatureProbeTests(unittest.TestCase):
    def synthetic(self, steps):
        return FeatureProbeEvaluator(instrument(SyntheticMemory()), steps)

    def clip(self, value):
        return np.full(16000, value, dtype=np.float32)

    def test_source_removal_changes_selected_feature_without_changing_actual_memory(self):
        ev = self.synthetic((0, 2))
        actual = np.concatenate([self.clip(1), self.clip(3), self.clip(1)])
        bed = np.concatenate([self.clip(0), self.clip(3), self.clip(0)])
        ev.set_interventions(bed, bed)
        observed = [ev.observe(clip) for clip in np.split(actual, 3)]
        self.assertEqual([item.status for item in observed], ["candidate", "background", "learned"])
        self.assertEqual(len(ev.records), 2)
        self.assertTrue(ev.records[0]["is_birth"])
        self.assertTrue(ev.records[1]["is_confirmation"])
        self.assertTrue(ev.records[1]["is_admission"])
        for row in ev.records:
            self.assertTrue(row["observations"]["actual"]["matches_stored"])
            self.assertFalse(row["observations"]["bed"]["matches_stored"])
            self.assertEqual(match_dependence(row), "probe_feature_outside_radius")
            self.assertEqual(row["removed_foreground_rms"], 1.)
        stored = ev.model._patterns[0].embedding
        self.assertEqual(stored[0], 1)
        self.assertEqual(stored[1], 0)
        self.assertEqual(ev.model._tick, 3)
        self.assertEqual(len([e for e in ev.model.provenance_events if e["kind"] == "candidate_observed"]), 2)
        ev.finish()

    def test_removing_older_sources_differs_from_removing_no_current_episode(self):
        ev = self.synthetic((0,))
        actual = self.clip(1)
        ev.set_interventions(actual, self.clip(0))
        ev.observe(actual)
        row = ev.records[0]
        self.assertEqual(row["waveform_sha256"]["actual"], row["waveform_sha256"]["without_current"])
        self.assertEqual(row["observations"]["actual"], row["observations"]["without_current"])
        self.assertFalse(row["observations"]["bed"]["matches_stored"])
        self.assertEqual(row["removed_current_rms"], 0)
        self.assertEqual(match_dependence(row, "without_current"), "probe_matches_stored")

    def test_identical_audio_reproduces_zero_features_without_inventing_a_cosine_match(self):
        ev = self.synthetic((0,))
        audio = self.clip(2)
        ev.set_interventions(audio, audio)
        ev.observe(audio)
        row = ev.records[0]
        for packet in row["observations"].values():
            self.assertEqual(packet["feature_norm"], 0.)
            self.assertEqual(packet["distance_to_stored"], 1.)
            self.assertTrue(packet["exact_actual_feature"])
            self.assertFalse(packet["matches_stored"])
        self.assertEqual(row["observations"]["actual"], row["observations"]["bed"])
        self.assertEqual(match_dependence(row), "actual_outside_radius")
        np.testing.assert_array_equal(ev.finish()["actual"], np.zeros((1, 96)))

    def test_matching_feature_can_move_to_a_nonoverlapping_region(self):
        ev = self.synthetic((0,))
        audio = self.clip(1)
        bed = audio.copy(); bed[1] = 0
        ev.set_interventions(bed, bed)
        ev.observe(audio)
        packet = ev.records[0]["observations"]["bed"]
        self.assertTrue(packet["matches_stored"])
        self.assertTrue(packet["exact_actual_feature"])
        self.assertEqual(packet["region_iou"], 0.)
        self.assertNotEqual(waveform_hash(audio), waveform_hash(bed))

    def test_absent_and_zero_features_are_distinct(self):
        vector = np.zeros(96, dtype=np.float32)
        packet = feature_packet(Observation("background"), vector, vector, .18)
        self.assertIsNone(packet["feature_sha256"])
        self.assertIsNone(packet["distance_to_stored"])
        self.assertFalse(packet["exact_actual_feature"])
        with self.assertRaises(ValueError):
            feature_packet(Observation("candidate", embedding=np.ones(3)), vector, vector, .18)

    def test_legacy_evaluator_copy_is_a_bare_branch_without_stems_or_collector(self):
        ev = self.synthetic((0,))
        audio = self.clip(1)
        ev.set_interventions(audio, audio)
        ev.observe(audio)
        before = deepcopy(ev.records)
        branch = deepcopy(ev)
        self.assertFalse(hasattr(branch, "_bed"))
        branch.observe(self.clip(0))
        self.assertEqual(ev.records, before)
        self.assertEqual(ev.step, 1)
        self.assertEqual(ev.model._tick, 1)
        self.assertEqual(branch._tick, 2)

    def test_incomplete_invalid_and_changed_schedules_are_rejected(self):
        for steps in ((True,), (-1,), (0, 0)):
            with self.assertRaises(ValueError):
                self.synthetic(steps)
        ev = self.synthetic((0,))
        with self.assertRaises(ValueError):
            ev.observe(self.clip(1))
        for a, b in ((np.zeros(1), np.zeros(1)), (self.clip(0), np.zeros(1)), (self.clip(np.nan), self.clip(0))):
            with self.assertRaises(ValueError):
                ev.set_interventions(a, b)
        ev.set_interventions(self.clip(0), self.clip(0))
        with self.assertRaises(ValueError):
            ev.finish()
        with self.assertRaises(ValueError):
            ev.set_interventions(self.clip(0), self.clip(0))
        changed = self.synthetic(())
        changed.set_interventions(self.clip(1), self.clip(1))
        with self.assertRaises(ValueError):
            changed.observe(self.clip(1))

    def scene(self, control):
        rate = 8000
        rng = np.random.default_rng(3100)
        time = np.arange(5*rate)/rate
        audio = dict(target=np.sin(2*np.pi*900*time)*(.6+.4*np.sin(2*np.pi*3*time)**2),
                     background=rng.normal(size=len(time)))
        config = SceneConfig(sample_rate=rate, duration_s=5, rounds=8, warmup=2,
            nuisance_probability=0, background_change=False, collect_stems=True,
            snr_reference="post_residual", snr_db=-6)
        return RecordedScene(3100, config, environment_seed=203100, audio=audio,
            foregrounds=("target",), backgrounds=("background",), control=control, repetitions=3)

    def test_saved_vector_validation_rejects_changed_measurements_and_bytes(self):
        ev = self.synthetic((0,))
        ev.set_interventions(self.clip(0), self.clip(0))
        ev.observe(self.clip(1))
        arrays = ev.finish()
        self.assertTrue(validate_records(ev.records, arrays))
        changed = deepcopy(ev.records)
        changed[0]["observations"]["bed"]["distance_to_stored"] = .01
        with self.assertRaises(ValueError):
            validate_records(changed, arrays)
        corrupted = {name: array.copy() for name, array in arrays.items()}
        corrupted["actual"][0, 0] = .5
        with self.assertRaises(ValueError):
            validate_records(ev.records, corrupted)
        changed = deepcopy(ev.records)
        changed[0]["vector_index"] = 2
        with self.assertRaises(ValueError):
            validate_records(changed, arrays)

    def test_exact_legacy_rows_and_features_across_eight_families_and_controls(self):
        classes = (ForegroundLearner, BandForegroundLearner, SpectralGainForegroundLearner,
            CalibratedGainForegroundLearner, AdaptiveGainForegroundLearner,
            GuardedGainForegroundLearner, PCENForegroundLearner, MCRAForegroundLearner)
        probes = 0
        for cls in classes:
            for control in (False, True):
                settings = cls.SETTINGS_CLASS(sample_rate=8000)
                baseline = instrument(cls(settings))
                expected_rows, expected_physical = evaluate(self.scene(control), {"m": baseline})
                steps = [event["tick"]-1 for event in baseline.provenance_events if event["kind"] == "candidate_observed"]
                ev = FeatureProbeEvaluator(instrument(cls(settings)), iter(steps))
                rows, physical = evaluate(InterventionScene(self.scene(control), {"m": ev}), {"m": ev})
                self.assertEqual(rows, expected_rows, cls.__name__)
                self.assertEqual(physical, expected_physical, cls.__name__)
                self.assertEqual(ev.model.provenance_events, baseline.provenance_events)
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory)/"features.npz"
                    ev.finish(path)
                    with self.assertRaises(FileExistsError):
                        ev.finish(path)
                    with np.load(path, allow_pickle=False) as arrays:
                        self.assertTrue(validate_records(ev.records, dict(arrays)))
                        for row in ev.records:
                            index = row["vector_index"]
                            self.assertEqual(feature_hash(arrays["stored"][index]), row["stored_feature_sha256"])
                            for name in ("actual", "without_current", "bed"):
                                packet = row["observations"][name]
                                if packet["feature_sha256"] is not None:
                                    self.assertEqual(feature_hash(arrays[name][index]), packet["feature_sha256"])
                                else:
                                    np.testing.assert_array_equal(arrays[name][index], np.zeros(96))
                                if control:
                                    self.assertEqual(packet, row["observations"]["actual"])
                probes += len(ev.records)
        self.assertGreater(probes, 0)
