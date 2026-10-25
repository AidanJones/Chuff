"""Offline lifecycle, context and score audit of the observed provenance replay."""

import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path

from chuff_mpu.sim.provenance_metrics import REFERENCE_SCORER_SHA256
from chuff_mpu.sim.provenance_replay import analysis, join_trace, score
from chuff_mpu.sim.recorded_catalog import digest


def audit(folder):
    manifest = json.loads((folder/"manifest.json").read_text())
    assert manifest["schema"] == "chuff.recorded.provenance-replay.v1"
    for name, expected in manifest["source_sha256"].items():
        assert digest((folder/"source"/name).read_bytes()) == expected, name
    for group in ("input_sha256", "output_sha256"):
        for name, expected in manifest[group].items():
            assert digest((folder/name).read_bytes()) == expected, name
    original = json.loads((folder/"input/original-manifest.json").read_text())
    for name in ("summary.json", "observations.jsonl.gz"):
        assert digest((folder/"input"/name).read_bytes()) == original["output_sha256"][name]
    for name in ("protocol.json", "catalog.json", "LICENSE"):
        assert digest((folder/"input"/name).read_bytes()) == original["input_sha256"]["input/"+name]
    plan = json.loads((folder/"input/protocol.json").read_text())
    for name, expected in plan["method_source_sha256"].items():
        assert manifest["source_sha256"]["python/src/chuff_mpu/"+name] == expected
    assert manifest["source_sha256"]["python/src/chuff_mpu/sim/recorded_metrics.py"] == REFERENCE_SCORER_SHA256
    assert manifest["settings"] == original["settings"]
    with gzip.open(folder/"input/observations.jsonl.gz", "rt", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    keys = {(row["suite"], row["seed"], row["mode"], row["step"]) for row in rows}
    expected = {(suite, seed, mode, step) for suite in plan["positive_suites"]+plan["control_suites"]
                for seed in plan["evaluation_seeds"] for mode in original["settings"] for step in range(210)}
    assert keys == expected and len(keys) == len(rows)
    assert manifest["verified_actual_decisions"] == len(rows)
    by_run = defaultdict(list)
    for row in rows:
        by_run[row["suite"], row["seed"]].append(row)
    with gzip.open(folder/"trace.jsonl.gz", "rt", encoding="utf-8") as stream:
        events = [json.loads(line) for line in stream]
    trace_by_run = defaultdict(list)
    for event in events:
        assert (event["suite"], event["seed"]) in by_run
        trace_by_run[event["suite"], event["seed"]].append(event)
    summaries = json.loads((folder/"summary.json").read_text())
    indexed = {(row["suite"], row["seed"]): row for row in summaries}
    gold = {(row["suite"], row["seed"]): row for row in json.loads((folder/"input/summary.json").read_text())}
    assert len(indexed) == len(summaries) and set(indexed) == set(by_run) == set(gold)
    admissions = []
    for key, summary in indexed.items():
        suite, seed = key
        raw = [{k: v for k, v in event.items() if k not in ("suite", "seed", "window_context")}
               for event in trace_by_run[key]]
        joined, admitted, transitions = join_trace(by_run[key], raw)
        assert [dict(event, suite=suite, seed=seed) for event in joined] == trace_by_run[key]
        assert summary["trace_events"] == len(joined)
        assert summary["verified_decisions"] == len(by_run[key])
        assert summary["transitions"] == transitions
        assert summary["scores"] == score(by_run[key], admitted)
        assert summary["scores"]["admission"] == gold[key]["episodes"]
        for name in ("physical_episodes", "max_component_error", "background_output_sha256"):
            assert summary[name] == gold[key][name]
        if suite.endswith("/control"):
            assert all(row["transition"] == "background_to_background" for row in admitted)
            assert all(not row["birth_context"]["previous_source_tail_s"] for row in admitted)
        admissions.extend(dict(row, suite=suite, seed=seed) for row in admitted)
    assert admissions == json.loads((folder/"admissions.json").read_text())
    assert analysis(summaries) == json.loads((folder/"analysis.json").read_text())
    probes = sum("without_episode_probe" in row for row in rows)
    print(f"{len(rows)} original decisions/{probes} probes linked; {len(events)} lifecycle events and {len(admissions)} admissions verified; both label scores recomputed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    audit(args.folder)


if __name__ == "__main__":
    main()
