"""Recompute diagnostic partitions and paired policy contrasts from saved decisions."""

import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path

import numpy as np

from chuff_mpu.sim.audit_masked import audit
from chuff_mpu.sim.memory_diagnostics import diagnose


def policy_contrasts(summaries):
    result = []
    for suite, snr in sorted({(s["suite"], s["snr_db"]) for s in summaries}):
        runs = sorted((s for s in summaries if (s["suite"], s["snr_db"]) == (suite, snr)),
                      key=lambda s: s["seed"])
        for mode in runs[0]["metrics"]:
            if mode in ("original", "quiet_band"):
                continue
            for key in ("repeat_recognition_rate", "event_detection_rate", "false_admissions"):
                values = [s["metrics"][mode][key] for s in runs]
                if any(v is None for v in values):
                    continue
                differences = np.array(values)-np.array([s["metrics"]["quiet_band"][key] for s in runs])
                indices = np.random.default_rng(731).integers(0, len(runs), (10000, len(runs)))
                result.append(dict(suite=suite, snr_db=snr, mode=mode, metric=key,
                    mean=float(differences.mean()),
                    bootstrap_95=np.quantile(differences[indices].mean(axis=1), [.025, .975]).tolist()))
    return result


def audit_admission(folder):
    summaries = audit(folder)
    manifest = json.loads((folder/"manifest.json").read_text())
    expected = {(suite, snr, seed) for suite in manifest["suites"] for snr in manifest["snr_db"]
                for seed in range(manifest["seed_start"], manifest["seed_start"]+manifest["seeds"])}
    assert len(summaries) == len(expected)
    assert {(s["suite"], s["snr_db"], s["seed"]) for s in summaries} == expected
    groups, schedules = defaultdict(list), {}
    modes = {"original", *manifest["variants"]}
    with gzip.open(folder/"observations.jsonl.gz", "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            groups[row["suite"], row["snr_db"], row["seed"]].append(row)
            # Foreground-bearing suites share the full schedule. Controls have
            # different RNG branch consumption; compare each control separately.
            schedule_group = row["suite"] if row["suite"].startswith("joint_") else "events"
            key = (schedule_group, row["seed"], row["step"])
            truth = (row["truth"], row["nuisance"])
            assert schedules.setdefault(key, truth) == truth
            if row["suite"] == "joint_ambience":
                assert row["truth"] is None and not row["nuisance"]
                assert row["active_snr_db_before_channel"] is None
            if row["suite"] == "joint_nuisance":
                assert row["truth"] is None
    for summary in summaries:
        rows = groups.pop((summary["suite"], summary["snr_db"], summary["seed"]))
        diagnostics = diagnose(rows)
        assert diagnostics == summary["diagnostics"]
        assert set(diagnostics) == modes
        for mode, diagnostic in diagnostics.items():
            metrics = summary["metrics"][mode]
            assert diagnostic["repeat_events"] == metrics["counts"]["repeat_windows"]
            assert diagnostic["repeat_outcomes"]["correct_identity"] == metrics["counts"]["correct_repeat_recognition"]
            assert sum(diagnostic["false_admissions_by_current_source"].values()) == metrics["false_admissions"]
    assert not groups
    print("Diagnostic partitions, source-free controls, complete matrix and paired schedules verified")
    return policy_contrasts(summaries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(audit_admission(args.folder), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
