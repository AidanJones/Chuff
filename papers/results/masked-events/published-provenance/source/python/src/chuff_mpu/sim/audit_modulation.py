"""Audit factorial controls and compare fixed noise estimators by paired seeds."""

import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path

from chuff_mpu.sim.audit_admission import policy_contrasts
from chuff_mpu.sim.audit_masked import audit
from chuff_mpu.sim.memory_diagnostics import diagnose


def audit_modulation(folder):
    summaries = audit(folder)
    manifest = json.loads((folder/"manifest.json").read_text())
    expected = {(suite, seed) for suite in manifest["suites"]
                for seed in range(manifest["seed_start"], manifest["seed_start"]+manifest["seeds"])}
    assert len(summaries) == len(expected)
    assert {(s["suite"], s["seed"]) for s in summaries} == expected
    groups, schedules = defaultdict(list), {}
    with gzip.open(folder/"observations.jsonl.gz", "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            sources = row["suite"].split("/")[-1]
            groups[row["suite"], row["seed"]].append(row)
            key = sources, row["seed"], row["step"]
            truth = row["truth"], row["nuisance"]
            assert schedules.setdefault(key, truth) == truth
            assert not row["nuisance"]
            assert not row["background_changed"]
            if sources == "ambience":
                assert row["truth"] is None and row["active_snr_db_before_channel"] is None
    for summary in summaries:
        assert summary["diagnostics"] == diagnose(groups.pop((summary["suite"], summary["seed"])))
    assert not groups
    controls = []
    for suite in manifest["suites"]:
        if not suite.endswith("/ambience"):
            continue
        runs = [s for s in summaries if s["suite"] == suite]
        controls.append(dict(suite=suite, n_seeds=len(runs), modes={
            mode: dict(false_admissions=sum(s["metrics"][mode]["false_admissions"] for s in runs),
                       affected_streams=sum(s["metrics"][mode]["false_admissions"] > 0 for s in runs))
            for mode in runs[0]["metrics"]}))
    print("Complete factorial matrix, paired schedules, source-free controls and diagnostics verified")
    return dict(controls=controls, paired_noise_estimator_contrasts=policy_contrasts(summaries))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(audit_modulation(args.folder), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
