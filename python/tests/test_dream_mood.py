"""Deterministic dream replay and explicit mood-event semantics."""

from contextlib import redirect_stdout
import io
import unittest

import numpy as np

from chuff_mpu import config, dream
from chuff_mpu.memory import EmbeddingStore
from chuff_mpu.mood import Mood, MoodState, update_mood
from chuff_mpu.sim.mood_demo import run


class DreamMoodTests(unittest.TestCase):
    def store(self):
        store = EmbeddingStore(8)
        for vector in np.eye(5, dtype=np.float32):
            store.add(vector)
        return store

    def test_pair_stream_is_reproducible_and_not_a_fixed_pair(self):
        store = self.store()
        first, second = np.random.default_rng(1), np.random.default_rng(1)
        pairs = []
        for _ in range(20):
            a, b = store.sample_pair(first)
            c, d = store.sample_pair(second)
            np.testing.assert_array_equal(a, c)
            np.testing.assert_array_equal(b, d)
            self.assertFalse(np.array_equal(a, b))
            pairs.append((int(a.argmax()), int(b.argmax())))
        self.assertGreater(len(set(pairs)), 1)

    def test_prediction_replays_and_does_not_mutate_memory(self):
        store = self.store()
        a, b = np.random.default_rng(4), np.random.default_rng(4)
        for _ in range(10):
            np.testing.assert_array_equal(dream.predict_next(store, .25, rng=a),
                                          dream.predict_next(store, .25, rng=b))
        self.assertEqual(len(store), 5)
        for weight in (-.1, 1.1, np.nan):
            with self.assertRaises(ValueError):
                dream.predict_next(store, weight)
        with self.assertRaises(ValueError):
            EmbeddingStore(1).sample_pair(a)

    def test_all_nine_states_have_explicit_supported_inputs(self):
        current = MoodState(Mood.CONTENTMENT, .3)
        cases = [
            (Mood.ACKNOWLEDGEMENT, dict(novelty=None, surprise=None, idle_s=0, ack_received=True)),
            (Mood.SURPRISE, dict(novelty=None, surprise=1., idle_s=0)),
            (Mood.CONTENTMENT, dict(novelty=None, surprise=0., idle_s=0)),
            (Mood.BORED, dict(novelty=None, surprise=None, idle_s=config.BORED_TIMEOUT_S)),
            (Mood.DREAMING, dict(novelty=None, surprise=None, idle_s=config.IDLE_TIMEOUT_S)),
            (Mood.SONG_RECOGNISED, dict(novelty=0., surprise=None, idle_s=0, learned=False)),
            (Mood.MILD_NOVELTY, dict(novelty=config.NOVELTY_THRESHOLD_LOW, surprise=None, idle_s=0)),
            (Mood.STRONG_NOVELTY, dict(novelty=1., surprise=None, idle_s=0, learned=False)),
            (Mood.NEW_SOUND_LEARNED, dict(novelty=None, surprise=None, idle_s=0, learned=True)),
        ]
        self.assertEqual({expected for expected, _ in cases}, set(Mood))
        for expected, inputs in cases:
            result = update_mood(current, **inputs)
            self.assertEqual(result.mood, expected)
            self.assertTrue(0 <= result.energy <= 1)
        self.assertEqual(update_mood(current, 1., None, 0).mood, Mood.NEW_SOUND_LEARNED)
        self.assertEqual(update_mood(current, 1., 1., config.BORED_TIMEOUT_S,
                                     ack_received=True, learned=True).mood, Mood.ACKNOWLEDGEMENT)
        self.assertEqual(update_mood(current, 1., 1., 0, learned=True).mood, Mood.SURPRISE)

    def test_same_seed_replays_complete_mood_trajectory_with_dreams(self):
        with redirect_stdout(io.StringIO()):
            first = run(45, 13, 180., .2, ack_ticks=(5, 9))
            second = run(45, 13, 180., .2, ack_ticks=(5, 9))
        self.assertEqual(first, second)
        self.assertGreater(first["dream_events"], 0)
        self.assertEqual(first["observations"][4]["mood"], "acknowledgement")
        self.assertTrue(any(row["prediction"] is not None for row in first["observations"]))

    def test_invalid_demo_arguments_fail_before_running(self):
        for args in ((0, 0, 1., .2), (5, 0, 0., .2), (5, 0, 1., 1.1)):
            with self.assertRaises(ValueError):
                run(*args)
