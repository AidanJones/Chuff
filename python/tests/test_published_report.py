from copy import deepcopy
import unittest

import numpy as np

from chuff_mpu.sim.published_protocol import protocol
from chuff_mpu.sim.published_report import POSITIVE, estimate, summarize


class PublishedReportTests(unittest.TestCase):
    def fixture(self):
        plan = protocol("0"*64)
        plan["evaluation_seeds"] = [1, 2, 3]
        groups = {}
        names = []
        for family in ("quiet160", "quiet128", "pcen160", "mcra128"):
            for feature in (0, 1):
                name = f"{family}-f{feature}-l3.5-b1.5"
                names.append(name)
                groups[f"{family}/feature{feature}"] = dict(selected=name, diagnostic=name)
            groups[family] = groups[family+"/feature0"]
        selection = dict(evaluation_ids=names, selections=groups)
        modes = names+["adaptive_gain"]
        rows = []
        for suite in plan["positive_suites"]+plan["control_suites"]:
            for seed, base in zip(plan["evaluation_seeds"], (.1, .4, .7)):
                offset = .1 if suite.startswith("modulated") else -.1
                episodes = {}
                for mode in modes:
                    value = base+offset+(.1 if mode.startswith("pcen") else 0)
                    rates = {key: value for key in POSITIVE}
                    rates["background_attention"] = .2
                    episodes[mode] = dict(rates=rates, counts=dict(background_admissions=seed,
                        distinct_recordings=1, admitted_recordings=1, eligible_repeat_episodes=10),
                        by_recording={"one.wav": dict(episodes=4, eligible_repeat_episodes=2)})
                rows.append(dict(suite=suite, seed=seed, episodes=episodes,
                                 runtime={mode: dict(mean_ms=2.) for mode in modes}))
        return plan, selection, rows

    def test_joint_suites_and_paired_seeds_preserve_constant_method_difference(self):
        plan, selection, rows = self.fixture()
        result = summarize(rows, plan, selection)
        quiet = result["modes"]["quiet160-f0-l3.5-b1.5"]
        np.testing.assert_allclose(quiet["seed_values"]["strict_episode_exclusion"], [.1, .4, .7])
        score = result["contrasts"]["pcen_minus_quiet160"]["metrics"]["strict_episode_exclusion"]
        self.assertAlmostEqual(score["mean"], .1)
        np.testing.assert_allclose(score["bootstrap_95"], [.1, .1])
        self.assertEqual(quiet["positive_counts"]["distinct_input_recordings"], 1)
        self.assertEqual(quiet["positive_counts"]["recording_run_pairs"], 12)
        self.assertEqual(quiet["by_recording"]["one.wav"]["eligible_repeat_episodes"], 24)

    def test_report_rejects_incomplete_duplicate_and_changed_actor_matrices(self):
        plan, selection, rows = self.fixture()
        for invalid in (rows[:-1], rows+[rows[0]]):
            with self.assertRaises(ValueError):
                summarize(invalid, plan, selection)
        changed = deepcopy(rows)
        del changed[0]["episodes"][selection["evaluation_ids"][0]]
        with self.assertRaises(ValueError):
            summarize(changed, plan, selection)
        selection["evaluation_ids"].append(selection["evaluation_ids"][0])
        with self.assertRaises(ValueError):
            summarize(rows, plan, selection)

    def test_nonfinite_values_are_rejected(self):
        with self.assertRaises(ValueError):
            estimate([.1, float("nan")], np.array([[0, 1]]))
