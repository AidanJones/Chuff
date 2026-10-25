import unittest

from chuff_mpu.sim.recorded_catalog import attributions, filename, select_rows


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
