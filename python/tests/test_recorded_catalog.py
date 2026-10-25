import unittest
from pathlib import Path
import io
import json
import tempfile
from unittest.mock import patch
import wave

from chuff_mpu.sim.recorded_catalog import (attributions, filename, select_rows, reserve,
    audit_reservation, prepare, audit, REVISION, BASE, UPSTREAM, digest)


class RecordedCatalogTests(unittest.TestCase):
    def test_same_recording_different_take_cannot_cross_split(self):
        metadata = "filename,fold,target,category,esc10,src_file,take\n" \
            "1-100-A-0.wav,1,0,dog,True,100,A\n" \
            "5-100-B-0.wav,5,0,dog,True,100,B\n"
        with self.assertRaises(ValueError):
            select_rows(metadata, categories=("dog",))
        metadata += "5-200-A-0.wav,5,0,dog,True,200,A\n"
        rows = select_rows(metadata, categories=("dog",))
        self.assertEqual([r["src_file"] for r in rows], ["100", "200"])

    def test_non_subset_and_unsafe_paths_are_rejected(self):
        metadata = "filename,fold,target,category,esc10,src_file,take\n" \
            "1-100-A-0.wav,1,0,dog,False,100,A\n"
        with self.assertRaises(ValueError):
            select_rows(metadata, categories=("dog",))
        with self.assertRaises(ValueError):
            filename("../outside.wav")

    def test_clip_attribution_is_preserved_as_supplied(self):
        text = "- [1-100-A.ogg]: clip derived from example.wav (https://example.org/sounds/100/) by author [CC0]"
        entry = attributions(text)["1-100-A"]
        self.assertEqual(entry["attribution"], text)
        self.assertEqual(entry["creator"], "author")
        self.assertEqual(entry["original_license_tag"], "CC0")

    def test_explicit_folds_exclude_original_sources_not_only_filenames(self):
        metadata = "filename,fold,target,category,esc10,src_file,take\n" \
            "2-100-A-0.wav,2,0,dog,True,100,A\n" \
            "2-200-A-0.wav,2,0,dog,True,200,A\n" \
            "4-100-B-0.wav,4,0,dog,True,100,B\n" \
            "4-200-B-0.wav,4,0,dog,True,200,B\n" \
            "4-300-A-0.wav,4,0,dog,True,300,A\n"
        rows = select_rows(metadata, categories=("dog",), development_fold=2,
                           evaluation_fold=4, excluded_sources=("100",))
        self.assertEqual([r["src_file"] for r in rows], ["200", "300"])
        for development, evaluation in ((2, 2), (0, 4), (2, 6), (True, 4), (2, "4")):
            with self.assertRaises(ValueError):
                select_rows(metadata, development_fold=development, evaluation_fold=evaluation)
        for sources in ((100,), ("../escape",), (None,)):
            with self.assertRaises(ValueError):
                select_rows(metadata, excluded_sources=sources)
        with self.assertRaises(ValueError):
            select_rows(metadata, categories=("dog", "dog"))

    def _pilot(self):
        return Path(__file__).resolve().parents[2]/"papers/data/esc10-pilot"

    def test_reservation_never_fetches_audio_and_is_deterministic(self):
        pilot = self._pilot()
        with tempfile.TemporaryDirectory() as directory, patch(
                "chuff_mpu.sim.recorded_catalog.download", side_effect=AssertionError("network used")):
            folder = Path(directory)/"reservation"
            reserve(folder, pilot, [pilot])
            manifest = audit_reservation(folder)
            old = json.loads((pilot/"catalog.json").read_text())
            old_ids = {r["source_id"] for r in old["clips"]}
            new_ids = {r["src_file"] for r in manifest["selected_rows"]}
            self.assertEqual(len(new_ids), 20)
            self.assertFalse(old_ids & new_ids)
            self.assertEqual({r["fold"] for r in manifest["selected_rows"]}, {"2", "4"})
            self.assertFalse(list(folder.rglob("*.wav")))
            other = Path(directory)/"other"
            reserve(other, pilot, [pilot])
            self.assertEqual((folder/"reservation.json").read_bytes(), (other/"reservation.json").read_bytes())
            with self.assertRaises(FileExistsError):
                reserve(folder, pilot, [pilot])

    def test_reservation_rejects_changed_selection_exclusions_and_unsafe_manifest_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)/"reservation"
            reserve(folder, self._pilot(), [self._pilot()])
            path = folder/"reservation.json"
            original = path.read_bytes()
            for mutation in ("selection", "exclusion", "path"):
                manifest = json.loads(original)
                if mutation == "selection":
                    manifest["selected_rows"][0]["filename"] = "2-1-A-0.wav"
                elif mutation == "exclusion":
                    manifest["selection_plan"]["excluded_sources"] = []
                else:
                    manifest["file_sha256"]["../outside"] = "0"*64
                path.write_text(json.dumps(manifest))
                with self.assertRaises(ValueError):
                    audit_reservation(folder)
            path.write_bytes(original)
            (folder/"excluded/catalog-00.json").write_text("{}")
            with self.assertRaises(ValueError):
                audit_reservation(folder)

    def test_reserved_catalog_materializes_only_reserved_files_and_preserves_old_audit(self):
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as handle:
            handle.setparams((1, 2, 44100, 220500, "NONE", "not compressed"))
            handle.writeframes(b"\0\0"*220500)
        content = buffer.getvalue()
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            reservation = folder/"reservation"
            reserve(reservation, self._pilot(), [self._pilot()])
            with patch("chuff_mpu.sim.recorded_catalog.download", return_value=content) as fetch:
                prepare(folder/"catalog", folder/"cache", reservation=reservation,
                        partitions=("development",))
            manifest = json.loads((folder/"catalog/catalog.json").read_text())
            expected = [r["filename"] for r in audit_reservation(reservation)["selected_rows"]]
            self.assertEqual([r["filename"] for r in manifest["clips"]], expected[:10])
            self.assertEqual([call.args[0] for call in fetch.call_args_list],
                             [r["url"] for r in manifest["clips"]])
            self.assertEqual(len(list((folder/"cache"/REVISION).glob("*.wav"))), 10)
            self.assertFalse(any((folder/"cache"/REVISION/name).exists() for name in expected[10:]))
            self.assertTrue(all(r["sha256"] == digest(content) for r in manifest["clips"]))
            audit(folder/"catalog", folder/"cache")
            # Existing pinned v1 inputs must remain readable by the new auditor.
            responses = {BASE+remote: (self._pilot()/name).read_bytes() for name, remote in UPSTREAM.items()}
            with patch("chuff_mpu.sim.recorded_catalog.download",
                       side_effect=lambda url: responses[url] if url in responses else content):
                prepare(folder/"legacy", folder/"legacy-cache")
            self.assertEqual(json.loads((folder/"legacy/catalog.json").read_text())["schema"],
                             "chuff.recorded.esc10-catalog.v1")
            audit(folder/"legacy", folder/"legacy-cache")
