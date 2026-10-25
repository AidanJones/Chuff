"""Audit variance evidence and bootstrap paired differences from reference acoustics."""

import argparse
import gzip
import json
from pathlib import Path

import numpy as np

from chuff_mpu.sim.audit_masked import audit, by_class
from chuff_mpu.sim.varied_scene import CONDITIONS


def audit_variance(folder):
    summaries = audit(folder)
    manifest = json.loads((folder/"manifest.json").read_text())
    seeds = range(manifest["seed_start"], manifest["seed_start"]+manifest["seeds"])
    expected = {(suite, snr, seed) for suite in manifest["suites"]
                for snr in manifest["snr_db"] for seed in seeds}
    actual = {(s["suite"], s["snr_db"], s["seed"]) for s in summaries}
    assert len(summaries) == len(expected) and actual == expected
    schedules = {}
    with gzip.open(folder/"observations.jsonl.gz", "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["suite"] == "joint_background":
                assert row["truth"] is None
                continue
            key = (row["seed"], row["step"])
            truth = (row["truth"], row["nuisance"])
            assert schedules.setdefault(key, truth) == truth
    contrasts = []
    indexed = {(s["suite"], s["snr_db"], s["seed"]): s for s in summaries}
    for snr in manifest["snr_db"]:
        for condition in CONDITIONS[1:]:
            for mode in ("original", "quiet_band"):
                for metric in ("repeat_recognition_rate", "event_detection_rate", "false_admissions"):
                    differences = np.array([
                        indexed[condition, snr, seed]["metrics"][mode][metric]
                        - indexed["reference", snr, seed]["metrics"][mode][metric] for seed in seeds])
                    draws = np.random.default_rng(731).integers(0, len(differences), (10000, len(differences)))
                    contrasts.append(dict(condition=condition, snr_db=snr, mode=mode, metric=metric,
                        mean=float(differences.mean()),
                        bootstrap_95=np.quantile(differences[draws].mean(axis=1), [.025, .975]).tolist()))
    print("Complete condition/seed matrix and cross-condition schedules verified")
    return {"contrasts_to_reference": contrasts, "by_class": by_class(summaries)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(audit_variance(args.folder), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
