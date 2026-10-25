"""Recompute timing overlap from raw rows and bootstrap paired seed localization rates."""

import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path

import numpy as np

from chuff_mpu.sim.audit_modulation import audit_modulation
from chuff_mpu.sim.localization import score_regions


def audit_localization(folder):
    base = audit_modulation(folder)
    summaries = json.loads((folder/"summary.json").read_text())
    groups = defaultdict(list)
    calibrations = {}
    with gzip.open(folder/"observations.jsonl.gz", "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            groups[row["suite"], row["seed"]].append(row)
    for summary in summaries:
        score = score_regions(groups.pop((summary["suite"], summary["seed"])))
        assert score == summary["localization"]
        for mode, metrics in score.items():
            assert not metrics["counts"].get("missing_event_timing", 0)
            assert not metrics["counts"].get("missing_detected_region", 0)
            assert metrics["counts"].get("event_windows", 0) == summary["metrics"][mode]["counts"]["event_windows"]
        if "gain_calibration" in summary:
            calibration = summary["gain_calibration"]
            settings = json.loads((folder/"manifest.json").read_text())["variants"]["calibrated_gain"]
            assert len(calibration["cv"]) == settings["warmup_clips"]
            assert np.isfinite(calibration["cv"]).all()
            assert calibration["compensate"] == (np.mean(calibration["cv"]) >= settings["gain_cv_threshold"])
            key = summary["suite"].rsplit("/", 1)[0], summary["seed"]
            assert calibrations.setdefault(key, calibration) == calibration
    assert not groups
    estimates = []
    for suite in sorted({s["suite"] for s in summaries if s["suite"].endswith("/foreground")}):
        runs = sorted((s for s in summaries if s["suite"] == suite), key=lambda s: s["seed"])
        for mode in runs[0]["metrics"]:
            for key in ("localized_detection_rate", "localized_repeat_recognition_rate"):
                for threshold in ("0.1", "0.25", "0.5"):
                    values = np.array([s["localization"][mode][key][threshold] for s in runs], dtype=float)
                    assert np.isfinite(values).all()
                    draws = np.random.default_rng(731).integers(0, len(runs), (10000, len(runs)))
                    estimate = dict(suite=suite, mode=mode, metric=key, iou_threshold=float(threshold),
                        mean=float(values.mean()),
                        bootstrap_95=np.quantile(values[draws].mean(axis=1), [.025, .975]).tolist())
                    if mode != "quiet_band":
                        reference = np.array([s["localization"]["quiet_band"][key][threshold] for s in runs])
                        differences = values-reference
                        estimate["paired_difference_to_quiet_band"] = dict(mean=float(differences.mean()),
                            bootstrap_95=np.quantile(differences[draws].mean(axis=1), [.025, .975]).tolist())
                    estimates.append(estimate)
    print("Source/region timing coverage and all overlap metrics verified")
    return dict(**base, localization_estimates=estimates)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(audit_localization(args.folder), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
