"""Caller/replay scratch buffers cannot silently rewrite learnt memory."""

import unittest

import numpy as np

from chuff_mpu.memory import EmbeddingStore


class MemoryOwnershipTests(unittest.TestCase):
    def test_input_buffer_reuse_preserves_novelty_and_recency_eviction(self):
        store = EmbeddingStore(2, "recency")
        vector = np.array([1., 0., 0.], dtype=np.float32)
        store.add(vector)
        store.add(np.array([0., 1., 0.], dtype=np.float32))
        vector[:] = [0., 0., 1.]
        self.assertAlmostEqual(store.novelty_score(np.array([1., 0., 0.])), 0.)
        store.add(np.array([0., 0., 1.], dtype=np.float32))
        self.assertAlmostEqual(store.novelty_score(np.array([1., 0., 0.])), 0.)
        self.assertAlmostEqual(store.novelty_score(np.array([0., 1., 0.])), 1.)

    def test_sample_mutation_cannot_change_memory_or_sampling_stream(self):
        store = EmbeddingStore(3)
        for vector in np.eye(3, dtype=np.float32): store.add(vector)
        before, tick, history = np.stack(store._items), store._tick, store._last_matched.copy()
        first, second = np.random.default_rng(17), np.random.default_rng(17)
        a, b = store.sample_pair(first)
        c, d = store.sample_pair(second)
        np.testing.assert_array_equal(a, c)
        np.testing.assert_array_equal(b, d)
        self.assertFalse(any(np.shares_memory(a, v) or np.shares_memory(b, v) for v in store._items))
        a[:], b[:] = 10., -10.
        np.testing.assert_array_equal(np.stack(store._items), before)
        self.assertEqual((store._tick, store._last_matched), (tick, history))
