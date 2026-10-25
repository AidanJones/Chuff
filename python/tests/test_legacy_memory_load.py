"""Restart behavior and transactional validation for the legacy memory store."""

from pathlib import Path
import tempfile
import unittest

import numpy as np

from chuff_mpu.memory import EmbeddingStore


class LegacyMemoryLoadTests(unittest.TestCase):
    def test_round_trip_query_and_eviction_follow_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory)/"memory.npz")
            for policy in ("fifo", "recency"):
                a = EmbeddingStore(2, policy)
                a.add(np.array([1., 0., 0.]))
                a.add(np.array([0., 1., 0.]))
                a.save(path)
                b = EmbeddingStore(2, policy)
                b.load(path)
                self.assertAlmostEqual(b.novelty_score(np.array([1., 0., 0.])), 0.)
                b.add(np.array([0., 0., 1.]))
                # Recency protects the first loaded vector after a real query.
                self.assertAlmostEqual(b.novelty_score(np.array([1., 0., 0.])),
                                       0. if policy == "recency" else 1.)
                self.assertEqual(len(b), 2)

    def test_invalid_loads_preserve_vectors_and_eviction_state(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory)/"bad.npz")
            a = EmbeddingStore(2, "recency")
            a.add(np.array([1., 0.]))
            prior = [v.copy() for v in a._items], a._last_matched.copy(), a._tick
            invalid = [dict(arr_0=np.array([np.nan, 0.])),
                       dict(arr_0=np.array([1e300, 0.])),
                       dict(arr_0=np.ones((2, 2))),
                       dict(arr_0=np.ones(2), arr_1=np.ones(3)),
                       dict(arr_0=np.ones(2), arr_1=np.ones(2), arr_2=np.ones(2)),
                       dict(arr_1=np.ones(2)),
                       dict(metadata=np.array("chuff.foreground.band-contrast.v1")),
                       dict(arr_0=np.array([1+2j])), dict(arr_0=np.array([], dtype=float)),
                       dict(arr_0=np.array(["bad"], dtype=object))]
            for content in invalid:
                np.savez(path, **content)
                with self.assertRaises(ValueError):
                    a.load(path)
                np.testing.assert_array_equal(a._items[0], prior[0][0])
                self.assertEqual(a._last_matched, prior[1])
                self.assertEqual(a._tick, prior[2])

    def test_empty_archive_and_numeric_index_order(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory)/"memory.npz")
            np.savez(path)
            a = EmbeddingStore(12, "recency")
            a.add(np.ones(2))
            a.load(path)
            self.assertEqual(len(a), 0)
            self.assertEqual(a._last_matched, [])
            np.savez(path, **{f"arr_{i}": np.array([i, 1]) for i in reversed(range(12))})
            a.load(path)
            self.assertEqual([v[0] for v in a._items], list(range(12)))
            self.assertEqual(a._last_matched, list(range(1, 13)))
