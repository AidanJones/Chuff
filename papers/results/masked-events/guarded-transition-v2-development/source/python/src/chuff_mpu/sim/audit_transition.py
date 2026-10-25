"""Audit complete transition matrix, paired schedules, hashes and phase estimates."""

import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path

from chuff_mpu.sim.transition_metrics import estimates, phase_metrics
from chuff_mpu.sim.transition_scene import CASES
from chuff_mpu.sim.transition_adaptation import adaptation_metrics
from chuff_mpu.sim.memory_diagnostics import diagnose


def audit(folder):
    manifest = json.loads((folder/"manifest.json").read_text())
    for name, digest in manifest["source_sha256"].items():
        assert hashlib.sha256((folder/"source"/name).read_bytes()).hexdigest() == digest, name
    for name, digest in manifest["output_sha256"].items():
        assert hashlib.sha256((folder/name).read_bytes()).hexdigest() == digest, name
    with gzip.open(folder/"observations.jsonl.gz", "rt", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    modes = tuple(manifest["settings"])
    assert {"quiet_band", "calibrated_gain", "spectral_gain"} <= set(modes)
    seeds = range(manifest["seed_start"], manifest["seed_start"]+manifest["seeds"])
    suites = {case+"/"+kind for case in CASES for kind in ("positive", "control")}
    assert set(manifest["suites"]) == suites
    indexed = {(r["suite"], r["seed"], r["step"], r["mode"]): r for r in rows}
    expected = {(suite, seed, step, mode) for suite in suites for seed in seeds
                for step in range(manifest["rounds"]) for mode in modes}
    assert len(rows) == len(indexed) and set(indexed) == expected
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["suite"], row["seed"]].append(row)
        change = 6 if row["suite"].startswith("silent_fast/") else manifest["rounds"]//2
        assert row["phase"] == ("before" if row["step"] < change else "after")
        if row["truth"] is not None:
            assert row["suite"].endswith("/positive") and row["step"] >= 6
            assert abs(row["post_channel_residual_snr_db"]+6) < 1e-8
            assert "without_current_probe" in row
        if row["mode"] == "calibrated_gain":
            if row["step"] < 2:
                assert row["gain_compensation_enabled"] is None
            else:
                assert row["gain_compensation_enabled"] == indexed[row["suite"], row["seed"], 2, row["mode"]]["gain_compensation_enabled"]
    # Same targets and placements across cases and modes, despite different backgrounds.
    for seed in seeds:
        for step in range(manifest["rounds"]):
            reference = indexed["stationary/positive", seed, step, "quiet_band"]
            for case in CASES:
                for mode in modes:
                    row = indexed[case+"/positive", seed, step, mode]
                    for field in ("truth", "event_start_s", "event_stop_s"):
                        assert row[field] == reference[field]
            for case in CASES:
                for mode in modes:
                    assert indexed[case+"/control", seed, step, mode]["truth"] is None
    summaries = json.loads((folder/"summary.json").read_text())
    assert len(summaries) == len(suites)*len(seeds)
    assert {(r["suite"], r["seed"]) for r in summaries} == set(grouped)
    for summary in summaries:
        selected = grouped[summary["suite"], summary["seed"]]
        assert summary["phases"] == phase_metrics(selected)
        assert summary["max_component_error"] < 1e-5
        if "adaptation" in summary:
            assert summary["adaptation"] == adaptation_metrics(selected, summary["suite"].split("/")[0],
                summary["environment"]["change_step"])
        if "diagnostics" in summary:
            assert summary["diagnostics"] == diagnose(selected)
        for adaptive_mode in ("adaptive_gain", "guarded_gain"):
            adaptive = [r for r in selected if r["mode"] == adaptive_mode]
            for previous, current in zip(adaptive, adaptive[1:]):
                delta = current["mode_switches"]-previous["mode_switches"]
                assert delta in (0, 1)
                if delta:
                    assert current["gain_compensation_enabled"] != previous["gain_compensation_enabled"]
                    assert current["memory_size"] == previous["memory_size"]
                    assert current["status"] == "calibrating"
        guarded = [r for r in selected if r["mode"] == "guarded_gain"]
        for row in guarded:
            if row["status"] == "learned":
                assert row["admission_support"]
                if row.get("admission_check_required", True):
                    assert row["shadow_status"] == "candidate"
        for previous, current in zip(guarded, guarded[1:]):
            if current["held_admissions"] > previous["held_admissions"]:
                assert current["status"] == "candidate"
                assert current["memory_size"] == previous["memory_size"]
    assert json.loads((folder/"analysis.json").read_text()) == estimates(summaries)
    print(f"{len(rows)} decisions: hashes, complete matrix, paired schedules, SNR, frozen modes, phase scores and seed estimates verified")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    audit(args.folder)


if __name__ == "__main__":
    main()
