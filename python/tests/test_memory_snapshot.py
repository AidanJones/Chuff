"""Versioned atomic persistence, exact history and legacy compatibility."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from chuff_mpu.attention import ForegroundLearner
from chuff_mpu.memory import EmbeddingStore


class MemorySnapshotTests(unittest.TestCase):
    def populated(self, policy="recency"):
        store = EmbeddingStore(3, policy)
        for row in np.eye(3):
            store.add(row)
        store.novelty_score(np.array([1., 0., 0.]))
        return store

    def test_history_round_trip_preserves_next_eviction_and_clock(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory)/"snapshot.npz")
            for policy in ("fifo", "recency"):
                source = self.populated(policy)
                source.save(path)
                loaded = EmbeddingStore(3, policy)
                loaded.load(path)
                self.assertEqual(loaded._tick, source._tick)
                self.assertEqual(loaded._last_matched, source._last_matched)
                for store in (source, loaded):
                    store.add(np.array([1., 1., 0.]))
                np.testing.assert_array_equal(np.stack(source._items), np.stack(loaded._items))
                self.assertEqual(loaded._last_matched, source._last_matched)

    def test_write_failures_preserve_previous_file_and_clean_temporary(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"snapshot.npz"
            store = self.populated()
            store.save(path)
            previous = path.read_bytes()
            store.add(np.array([1., 1., 0.]))
            for operation in ("np.savez_compressed", "os.fsync", "os.replace"):
                with patch("chuff_mpu.memory."+operation, side_effect=OSError("injected write failure")):
                    with self.assertRaises(OSError):
                        store.save(path)
                self.assertEqual(path.read_bytes(), previous)
                self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_malformed_versioned_files_do_not_mutate_live_state(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"snapshot.npz"
            store = self.populated()
            store.save(path)
            with np.load(path, allow_pickle=False) as data:
                metadata = json.loads(str(data["metadata"].item()))
                features, history = data["features"].copy(), data["last_matched"].copy()
            before = np.stack(store._items), store._last_matched.copy(), store._tick
            invalid = [dict(schema="unknown"), dict(version=2), dict(version=True),
                       dict(eviction="fifo"), dict(tick=-1), dict(capacity=2)]
            for change in invalid:
                np.savez(path, metadata=np.array(json.dumps({**metadata, **change})),
                         features=features, last_matched=history)
                with self.assertRaises(ValueError): store.load(path)
                np.testing.assert_array_equal(np.stack(store._items), before[0])
                self.assertEqual((store._last_matched, store._tick), (before[1], before[2]))
            for bad in (np.array([1, 2]), np.array([0., 1., 2.]), np.array([1, 2, 999]), np.array([-1, 2, 3])):
                np.savez(path, metadata=np.array(json.dumps(metadata)), features=features, last_matched=bad)
                with self.assertRaises(ValueError): store.load(path)
                self.assertEqual((store._last_matched, store._tick), (before[1], before[2]))

    def test_legacy_import_resets_history_and_versioned_save_appends_suffix(self):
        with tempfile.TemporaryDirectory() as directory:
            legacy = Path(directory)/"legacy.npz"
            np.savez(legacy, np.array([1., 0.]), np.array([0., 1.]))
            store = EmbeddingStore(3, "recency")
            store.load(legacy)
            self.assertEqual((store._last_matched, store._tick), ([1, 2], 2))
            destination = Path(directory)/"versioned"
            store.save(destination)
            self.assertTrue(Path(str(destination)+".npz").exists())
            with np.load(str(destination)+".npz", allow_pickle=False) as data:
                self.assertEqual(set(data.files), {"metadata", "features", "last_matched"})

    def test_schema_isolation_capacity_and_empty_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"snapshot.npz"
            source = self.populated()
            source.save(path)
            with self.assertRaises(ValueError): EmbeddingStore(2, "recency").load(path)
            with self.assertRaises(ValueError): ForegroundLearner.load(path)
            ForegroundLearner().save(path)
            with self.assertRaises(ValueError): source.load(path)
            empty = EmbeddingStore(3)
            empty.save(path)
            empty.load(path)
            self.assertEqual((len(empty), empty._last_matched, empty._tick), (0, [], 0))

    def test_external_rng_state_is_not_consumed_by_persistence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"snapshot.npz"
            source = self.populated()
            rng = np.random.default_rng(123)
            source.sample_pair(rng)
            state = deepcopy(rng.bit_generator.state)
            source.save(path)
            restored = EmbeddingStore(3, "recency")
            restored.load(path)
            self.assertEqual(rng.bit_generator.state, state)
            other = np.random.default_rng()
            other.bit_generator.state = state
            for _ in range(10):
                a, b = source.sample_pair(rng)
                c, d = restored.sample_pair(other)
                np.testing.assert_array_equal(a, c)
                np.testing.assert_array_equal(b, d)
