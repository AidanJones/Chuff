from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from chuff_mpu.sim.published_protocol import FAMILIES, actor, audit_protocol, candidates, protocol, select
from chuff_mpu.sim.published_experiment import runtime
from chuff_mpu.sim.recorded_catalog import digest


class PublishedProtocolTests(unittest.TestCase):
    def _summaries(self, plan):
        return [dict(suite=suite, seed=seed, episodes={name: dict(
            rates=dict(episode_detection=.9, strict_episode_exclusion=.4,
                       first_episode_novel_attention=.8, wrong_repeat_identity=.1),
            counts=dict(background_admissions=5)) for name in [*plan["candidates"], "adaptive_gain"]})
            for suite in plan["positive_suites"]+plan["control_suites"]
            for seed in plan["development_seeds"]]

    def test_matched_grids_and_hop_controls(self):
        grid = candidates()
        self.assertEqual(len(grid), 48)
        for family, (_, _, hop) in FAMILIES.items():
            entries = [spec for spec in grid.values() if spec["family"] == family]
            self.assertEqual(len(entries), 12)
            self.assertEqual({spec["settings"]["feature_mode"] for spec in entries}, {0, 1})
            self.assertTrue(all(spec["settings"]["hop"] == hop for spec in entries))
            self.assertTrue(all(actor(spec).settings.capacity == 32 for spec in entries))

    def test_silent_candidate_cannot_win_by_zero_background_admissions(self):
        plan = protocol("0"*64)
        rows = self._summaries(plan)
        silent = next(iter(plan["candidates"]))
        for row in rows:
            row["episodes"][silent]["rates"].update(episode_detection=0,
                strict_episode_exclusion=0, first_episode_novel_attention=0)
            row["episodes"][silent]["counts"]["background_admissions"] = 0
        result = select(rows, plan)
        self.assertFalse(result["scores"][silent]["feasible"])
        self.assertNotEqual(result["selections"]["quiet160"]["selected"], silent)
        self.assertEqual(len(result["scores"]), 48)

    def test_constraints_apply_per_suite_and_infeasibility_is_explicit(self):
        plan = protocol("0"*64)
        rows = self._summaries(plan)
        for row in rows:
            if row["suite"] == "recorded/-6/positive":
                for metric in row["episodes"].values():
                    metric["rates"]["episode_detection"] = .7
        result = select(rows, plan)
        self.assertTrue(all(item["selected"] is None for item in result["selections"].values()))
        self.assertTrue(all(item["diagnostic"] in plan["candidates"] for item in result["selections"].values()))
        self.assertTrue(all(f"{family}-f0-l3.5-b1.5" in result["evaluation_ids"] for family in FAMILIES))

    def test_complete_development_matrix_is_required(self):
        plan = protocol("0"*64)
        rows = self._summaries(plan)
        for invalid in (rows[:-1], rows+[rows[0]]):
            with self.assertRaises(ValueError):
                select(invalid, plan)
        invalid = deepcopy(rows)
        del invalid[0]["episodes"][next(iter(plan["candidates"]))]
        with self.assertRaises(ValueError):
            select(invalid, plan)

    def test_feasible_ranking_minimizes_nuisance_without_changing_constraints(self):
        plan = protocol("0"*64)
        rows = self._summaries(plan)
        winner = "pcen160-f1-l2.5-b1.15"
        for row in rows:
            row["episodes"][winner]["counts"]["background_admissions"] = 1
        result = select(rows, plan)
        self.assertEqual(result["selections"]["pcen160"]["selected"], winner)

    def test_protocol_audit_rejects_changed_source_declaration_or_grid(self):
        reservation = Path(__file__).resolve().parents[2]/"papers/data/esc10-published-reservation"
        original = protocol(digest((reservation/"reservation.json").read_bytes()))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"protocol.json"
            path.write_text(json.dumps(original))
            self.assertEqual(audit_protocol(path, reservation), original)
            for key in ("method_source_sha256", "candidates", "development_seeds"):
                changed = deepcopy(original)
                changed[key] = {} if isinstance(changed[key], dict) else [1, 2]
                path.write_text(json.dumps(changed))
                with self.assertRaises(ValueError):
                    audit_protocol(path, reservation)

    def test_runtime_summary_excludes_initial_calibration_windows(self):
        rows = [dict(mode="a", step=index, observe_ns=value*1_000_000)
                for index, value in enumerate((1000, 1000, 1000, 2, 4))]
        score = runtime(rows, {"a": {"warmup_clips": 3}})["a"]
        self.assertEqual(score["windows"], 2)
        self.assertEqual(score["mean_ms"], 3)
        self.assertEqual(score["median_ms"], 3)
        self.assertEqual(score["max_ms"], 4)
