"""Audit exhaustive candidate-feature probes, vectors and admission pairing."""

import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path

import numpy as np

from chuff_mpu.sim.feature_dependence import ROLES, admission_pairs, summarize_records
from chuff_mpu.sim.feature_probes import validate_records
from chuff_mpu.sim.provenance_replay import context, join_trace
from chuff_mpu.sim.recorded_catalog import digest


def audit(folder):
    load = lambda name: json.loads((folder/name).read_text())
    manifest = load("manifest.json")
    assert manifest["schema"] == "chuff.recorded.feature-dependence.v1"
    for name, expected in manifest["source_sha256"].items():
        assert digest((folder/"source"/name).read_bytes()) == expected, name
    for group in ("input_sha256", "output_sha256", "vectors"):
        for name, expected in manifest[group].items():
            assert digest((folder/name).read_bytes()) == expected, name
    original, provenance = load("input/original-manifest.json"), load("input/provenance-manifest.json")
    assert digest((folder/"input/original-manifest.json").read_bytes()) == provenance["input_sha256"]["input/original-manifest.json"]
    for local, source in (("observations.jsonl.gz", "observations.jsonl.gz"), ("original-summary.json", "summary.json")):
        assert digest((folder/"input"/local).read_bytes()) == original["output_sha256"][source]
    for name in ("catalog.json", "protocol.json", "LICENSE"):
        assert digest((folder/"input"/name).read_bytes()) == original["input_sha256"]["input/"+name]
    for name in ("trace.jsonl.gz", "admissions.json"):
        assert digest((folder/"input"/name).read_bytes()) == provenance["output_sha256"][name]
    for name, expected in original["source_sha256"].items():
        if name in manifest["source_sha256"]:
            assert manifest["source_sha256"][name] == expected
    tracer = "python/src/chuff_mpu/sim/memory_provenance.py"
    assert manifest["source_sha256"][tracer] == provenance["source_sha256"][tracer]
    assert manifest["settings"] == original["settings"]
    plan = load("input/protocol.json")
    assert manifest["dependencies"] == plan["dependencies"]
    def compressed(name):
        with gzip.open(folder/name, "rt", encoding="utf-8") as stream:
            return [json.loads(line) for line in stream]
    gold, traces, probes = compressed("input/observations.jsonl.gz"), compressed("input/trace.jsonl.gz"), compressed("probes.jsonl.gz")
    indexed = {(r["suite"], r["seed"], r["mode"], r["step"]): r for r in gold}
    matrix = {(suite, seed, mode, step) for suite in plan["positive_suites"]+plan["control_suites"]
              for seed in plan["evaluation_seeds"] for mode in manifest["settings"] for step in range(210)}
    assert len(indexed) == len(gold) and set(indexed) == matrix
    observed = {(r["suite"], r["seed"], r["mode"], r["tick"]-1): r for r in traces if r["kind"] == "candidate_observed"}
    actual = {(r["suite"], r["seed"], r["mode"], r["step"]): r for r in probes}
    assert len(actual) == len(probes) and set(actual) == set(observed)
    by_run, gold_by_run, trace_by_run, admissions_by_run = (defaultdict(list) for _ in range(4))
    for row in probes:
        key = row["suite"], row["seed"], row["mode"], row["step"]
        reference, event = indexed[key], observed[key]
        assert row["window_context"] == context(reference)
        for name in ("candidate_id", "birth_tick", "prototype_id", "hits_before", "hits_after", "previous_observation_tick"):
            assert row[name] == event[name]
        assert row["is_birth"] == (event["hits_before"] == 0)
        assert row["is_confirmation"] == (event["hits_before"] > 0 and event["hits_after"] > event["hits_before"])
        assert row["is_admission"] == (reference["status"] == "learned")
        assert row["stored_feature_sha256"] == event["stored_feature_sha256"]
        assert row["observations"]["actual"]["feature_sha256"] == event["observed_feature_sha256"]
        for name in ("status", "prototype_id", "salience", "novelty", "region_start_s", "region_stop_s"):
            assert row["observations"]["actual"][name] == reference[name]
        if "without_episode_probe" in reference:
            assert {name: row["observations"]["without_current"][name] for name in ("status", "prototype_id")} == reference["without_episode_probe"]
        by_run[row["suite"], row["seed"]].append(row)
    for row in gold:
        gold_by_run[row["suite"], row["seed"]].append(row)
    for event in traces:
        trace_by_run[event["suite"], event["seed"]].append(event)
    for row in load("input/admissions.json"):
        admissions_by_run[row["suite"], row["seed"]].append(row)
    summaries = load("summary.json")
    summary_by_run = {(row["suite"], row["seed"]): row for row in summaries}
    old_summary = {(row["suite"], row["seed"]): row for row in load("input/original-summary.json")}
    assert len(summary_by_run) == len(summaries) and set(summary_by_run) == set(gold_by_run) == set(old_summary)
    expected_files, pairs = set(), []
    for key, summary in summary_by_run.items():
        suite, seed = key
        records = by_run[key]
        name = f"vectors/{suite.replace('/', '--')}/{seed}.npz"
        expected_files.add(name)
        assert summary["verified_decisions"] == len(gold_by_run[key])
        assert summary["probe_windows"] == len(records)
        assert summary["counts"] == summarize_records(records, manifest["settings"])
        for field in ("physical_episodes", "max_component_error", "background_output_sha256"):
            assert summary[field] == old_summary[key][field]
        raw_trace = [{name: value for name, value in event.items() if name not in ("suite", "seed", "window_context")}
                     for event in trace_by_run[key]]
        _, admitted, _ = join_trace(gold_by_run[key], raw_trace)
        assert [dict(row, suite=suite, seed=seed) for row in admitted] == admissions_by_run[key]
        with np.load(folder/name, allow_pickle=False) as arrays:
            assert set(arrays.files) == {mode+"__"+role for mode in manifest["settings"] for role in ROLES}
            for mode in manifest["settings"]:
                selected = [row for row in records if row["mode"] == mode]
                assert all(row["match_distance"] == manifest["settings"][mode]["match_distance"] for row in selected)
                assert arrays[mode+"__actual"].shape == (len(selected), 3*manifest["settings"][mode]["bands"])
                assert all(row["vector_file"] == name for row in selected)
                assert validate_records(selected, {role: arrays[mode+"__"+role] for role in ROLES})
                if suite.endswith("/control"):
                    assert all(row["waveform_sha256"]["bed"] == row["waveform_sha256"]["actual"]
                               and row["observations"]["bed"] == row["observations"]["actual"] for row in selected)
        pairs.extend(admission_pairs(records, admissions_by_run[key]))
    assert set(manifest["vectors"]) == expected_files
    assert pairs == load("admission-pairs.json")
    assert manifest["verified_actual_decisions"] == len(gold)
    assert manifest["source_removal_probes"] == sum("without_episode_probe" in row for row in gold)
    assert manifest["feature_probe_windows"] == len(probes)
    print(f"{len(gold)} original decisions linked; {len(probes)} exhaustive feature windows, {len(pairs)} admission pairs and all vectors/no-op controls verified")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    audit(args.folder)


if __name__ == "__main__":
    main()
