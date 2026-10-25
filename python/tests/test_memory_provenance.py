from copy import deepcopy
import unittest

import numpy as np

from chuff_mpu.attention import AttentionConfig, ForegroundLearner
from chuff_mpu.band_attention import BandForegroundLearner
from chuff_mpu.gain_attention import SpectralGainForegroundLearner, CalibratedGainForegroundLearner
from chuff_mpu.adaptive_gain import AdaptiveGainForegroundLearner
from chuff_mpu.guarded_gain import GuardedGainForegroundLearner
from chuff_mpu.published_attention import PCENForegroundLearner, MCRAForegroundLearner
from chuff_mpu.sim.challenge_scene import ChallengeScene
from chuff_mpu.sim.memory_provenance import feature_hash, instrument
from chuff_mpu.sim.scene import SceneConfig


class MemoryProvenanceTests(unittest.TestCase):
    def vector(self, coordinate=0):
        result = np.zeros(96, dtype=np.float32)
        result[coordinate] = 1
        return result

    def remember(self, model, tick, vector):
        model._tick = tick
        return model._remember(vector, 2.)

    def events(self, model, kind):
        return [row for row in model.provenance_events if row["kind"] == kind]

    def assert_observation_equal(self, first, second):
        for name in ("status", "salience", "novelty", "prototype_id", "region_start_s", "region_stop_s"):
            self.assertEqual(getattr(first, name), getattr(second, name), name)
        if first.embedding is None:
            self.assertIsNone(second.embedding)
        else:
            np.testing.assert_array_equal(first.embedding, second.embedding)

    def test_birth_stored_vector_and_confirmation_gaps_are_distinguished(self):
        model = instrument(ForegroundLearner())
        first = self.vector()
        later = .95*first+np.sqrt(1-.95**2)*self.vector(1)
        self.assertEqual(self.remember(model, 1, first).status, "candidate")
        self.assertEqual(self.remember(model, 2, later).status, "candidate")
        observed = self.remember(model, 4, later)
        self.assertEqual(observed.status, "learned")
        np.testing.assert_array_equal(model._patterns[0].embedding, first)
        np.testing.assert_array_equal(observed.embedding, later)
        birth = self.events(model, "candidate_created")
        self.assertEqual(len(birth), 1)
        self.assertEqual(birth[0]["birth_tick"], 1)
        self.assertEqual(birth[0]["candidate_id"], 0)
        seen = self.events(model, "candidate_observed")
        self.assertEqual([row["hits_after"] for row in seen], [1, 1, 2])
        self.assertEqual([row["previous_observation_tick"] for row in seen], [None, 1, 2])
        admission = self.events(model, "prototype_admitted")[0]
        self.assertEqual(admission["tick"], 4)
        self.assertEqual(admission["birth_tick"], 1)
        self.assertEqual(admission["prototype_id"], 0)
        self.assertEqual(admission["stored_feature_sha256"], feature_hash(first))
        self.assertEqual(admission["observed_feature_sha256"], feature_hash(later))
        self.assertNotEqual(admission["observed_feature_sha256"], admission["stored_feature_sha256"])

    def test_immediate_admission_owns_its_vector_and_ids_are_not_reused(self):
        model = instrument(ForegroundLearner(AttentionConfig(confirmations=1, capacity=1)))
        first = self.vector()
        self.remember(model, 1, first)
        first[:] = 0
        self.assertEqual(feature_hash(model._patterns[0].embedding), feature_hash(self.vector()))
        self.remember(model, 2, self.vector(1))
        births = self.events(model, "candidate_created")
        self.assertEqual([row["candidate_id"] for row in births], [0, 1])
        self.assertEqual([row["prototype_id"] for row in self.events(model, "prototype_admitted")], [0, 1])
        retired = self.events(model, "prototype_retired")
        self.assertEqual(len(retired), 1)
        self.assertEqual(retired[0]["prototype_id"], 0)
        self.assertEqual(retired[0]["stored_feature_sha256"], feature_hash(self.vector()))

    def test_candidate_eviction_expiration_and_external_reset_are_observed(self):
        model = instrument(ForegroundLearner(AttentionConfig(capacity=1, candidate_ttl=2)))
        self.remember(model, 1, self.vector())
        self.remember(model, 3, self.vector(1))
        self.assertEqual(self.events(model, "candidate_retired")[0]["candidate_id"], 0)
        for _ in range(3):
            model.observe(np.zeros(16000))
        self.assertEqual([row["candidate_id"] for row in self.events(model, "candidate_retired")], [0, 1])
        self.remember(model, 7, self.vector(2))
        model._candidates.clear()
        model.observe(np.zeros(16000))
        self.assertEqual(self.events(model, "candidate_retired")[-1]["candidate_id"], 2)

    def test_copied_branches_keep_origins_without_changing_actual_history_or_memory(self):
        model = instrument(ForegroundLearner(AttentionConfig(confirmations=1)))
        self.remember(model, 1, self.vector())
        before = model.provenance_events
        branch = deepcopy(model)
        self.assertIs(branch._provenance_history, model._provenance_history)
        self.assertEqual(self.remember(branch, 2, self.vector()).status, "recognised")
        self.remember(branch, 3, self.vector(1))
        self.assertEqual(model.provenance_events, before)
        self.assertEqual(model.memory_size, 1)
        self.assertEqual(branch.memory_size, 2)
        matched = self.events(branch, "prototype_recognised")[0]
        self.assertEqual(matched["candidate_id"], 0)
        self.assertEqual(matched["birth_tick"], 1)
        before[0]["candidate_id"] = 999
        self.assertEqual(model.provenance_events[0]["candidate_id"], 0)

    def test_changed_stored_vector_is_explicit_and_preserves_original_birth(self):
        model = instrument(ForegroundLearner())
        self.remember(model, 1, self.vector())
        model._candidates[0].embedding = self.vector(1)
        self.remember(model, 3, self.vector(1))
        changed = self.events(model, "stored_vector_changed")[0]
        self.assertEqual(changed["previous_feature_sha256"], feature_hash(self.vector()))
        self.assertEqual(changed["stored_feature_sha256"], feature_hash(self.vector(1)))
        self.assertEqual(changed["birth_tick"], 1)
        self.assertEqual(self.events(model, "prototype_admitted")[0]["stored_feature_sha256"], feature_hash(self.vector(1)))

    def test_existing_or_already_instrumented_state_is_rejected(self):
        original = ForegroundLearner()
        original.observe(np.zeros(16000))
        with self.assertRaises(ValueError):
            instrument(original)
        with self.assertRaises(ValueError):
            instrument(instrument(ForegroundLearner()))
        with self.assertRaisesRegex(ValueError, "birth provenance"):
            type(instrument(ForegroundLearner())).load("unused-path.npz")

    def test_invalid_clip_does_not_add_events(self):
        model = instrument(ForegroundLearner())
        for clip in (np.zeros((2, 16000)), np.full(16000, np.nan), np.zeros(10)):
            with self.assertRaises(ValueError):
                model.observe(clip)
        self.assertEqual(model.provenance_events, [])
        self.assertEqual(model._tick, 0)

    def test_exact_actor_and_source_removal_branch_parity_across_eight_families(self):
        frames = list(ChallengeScene(3000, SceneConfig(rounds=65, snr_db=-6,
            nuisance_probability=0, collect_stems=True)))
        classes = (ForegroundLearner, BandForegroundLearner, SpectralGainForegroundLearner,
            CalibratedGainForegroundLearner, AdaptiveGainForegroundLearner,
            GuardedGainForegroundLearner, PCENForegroundLearner, MCRAForegroundLearner)
        for cls in classes:
            a, b = cls(), instrument(cls())
            for index, frame in enumerate(frames):
                modulation = 1 if index < 30 else .8+.7*np.sin(2*np.pi*1.3*np.arange(16000)/16000)
                audio = frame.audio*modulation
                if index % 10 == 0:
                    before = b.provenance_events
                    self.assert_observation_equal(deepcopy(a).observe(frame.without_current_source_audio),
                                                  deepcopy(b).observe(frame.without_current_source_audio))
                    self.assertEqual(b.provenance_events, before)
                self.assert_observation_equal(a.observe(audio), b.observe(audio))
                for first, second in ((a._candidates, b._candidates), (a._patterns, b._patterns)):
                    self.assertEqual(len(first), len(second), cls.__name__)
                    for p, q in zip(first, second):
                        self.assertEqual((p.hits, p.last_seen, p.identifier), (q.hits, q.last_seen, q.identifier))
                        np.testing.assert_array_equal(p.embedding, q.embedding)
            self.assertFalse(self.events(b, "memory_observation_unresolved"), cls.__name__)
            self.assertTrue(all("recording_id" not in row and "episode_id" not in row for row in b.provenance_events))
