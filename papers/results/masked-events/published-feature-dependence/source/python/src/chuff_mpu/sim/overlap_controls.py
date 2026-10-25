"""Post-hoc temporal overlap diagnostics, not source-attribution significance tests."""

import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from chuff_mpu.sim.audit_localization import audit_localization
from chuff_mpu.sim.localization import DETECTED, THRESHOLDS, overlap_iou


def alternative_position_overlap(region, length, duration, threshold, positions=200):
    """Midpoint quadrature over the generator's legal placement range.

    Keep length fixed, do not wrap intervals across boundaries, and apply the
    same margin rule as AcousticScene. The selected detector region remains
    fixed; therefore this is alignment sensitivity, not a causal null test.
    """
    if not 0 < length <= duration or positions < 1:
        raise ValueError("invalid duration or quadrature size")
    if region is None:
        return 0.
    available = duration-length
    margin = min(.12, available/3)
    starts = margin+(np.arange(positions)+.5)/positions*(available-2*margin)
    intersection = np.maximum(0., np.minimum(region[1], starts+length)-np.maximum(region[0], starts))
    iou = intersection/(region[1]-region[0]+length-intersection)
    return float(np.mean(iou >= threshold))


def analyze_rows(rows, duration=1.):
    indexed = {(r["suite"], r["seed"], r["step"], r["mode"]): r for r in rows}
    if len(indexed) != len(rows):
        raise ValueError("duplicate observations")
    samples = defaultdict(list)
    classes = defaultdict(list)
    for row in rows:
        if row["truth"] is None or not row["suite"].endswith("/foreground"):
            continue
        event = row["event_start_s"], row["event_stop_s"]
        assert None not in event
        length = event[1]-event[0]
        control_suite = row["suite"].removesuffix("foreground")+"ambience"
        control = indexed[control_suite, row["seed"], row["step"], row["mode"]]
        assert control["truth"] is None and not control["nuisance"]
        def region(item):
            if item["status"] not in DETECTED:
                return None
            result = item["region_start_s"], item["region_stop_s"]
            assert None not in result
            return result
        selected, counterfactual = region(row), region(control)
        for threshold in THRESHOLDS:
            observed = float(selected is not None and overlap_iou(selected, event) >= threshold)
            matched_background = float(counterfactual is not None and overlap_iou(counterfactual, event) >= threshold)
            alternative = alternative_position_overlap(selected, length, duration, threshold)
            values = observed, matched_background, alternative, length
            key = row["suite"], row["mode"], threshold
            samples[*key, row["seed"]].append(values)
            classes[*key, row["truth"]].append(values)
    estimates = []
    for suite, mode, threshold in sorted({key[:3] for key in samples}):
        keys = sorted(key for key in samples if key[:3] == (suite, mode, threshold))
        values = np.array([np.mean(samples[key], axis=0) for key in keys])
        # Equal seed weighting; retain paired within-seed differences.
        statistics = dict(observed=values[:, 0], matched_background=values[:, 1],
                          alternative_positions=values[:, 2],
                          excess_over_background=values[:, 0]-values[:, 1],
                          excess_over_alternatives=values[:, 0]-values[:, 2])
        draws = np.random.default_rng(731).integers(0, len(keys), (10000, len(keys)))
        estimates.append(dict(suite=suite, mode=mode, iou_threshold=threshold, seeds=len(keys),
            metrics={name: dict(mean=float(v.mean()), bootstrap_95=np.quantile(v[draws].mean(axis=1), [.025, .975]).tolist())
                     for name, v in statistics.items()}))
    class_estimates = []
    for (suite, mode, threshold, label), values in sorted(classes.items()):
        mean = np.mean(values, axis=0)
        class_estimates.append(dict(suite=suite, mode=mode, iou_threshold=threshold, label=label,
            windows=len(values), mean_duration_s=float(mean[3]), observed=float(mean[0]),
            matched_background=float(mean[1]), alternative_positions=float(mean[2])))
    return dict(estimates=estimates, class_pooled_descriptive=class_estimates)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output directory must be new")
    audit_localization(args.folder)
    with gzip.open(args.folder/"observations.jsonl.gz", "rt", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    result = analyze_rows(rows)
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[4]
    source_hashes = {}
    for name in ("overlap_controls.py", "localization.py", "audit_localization.py", "audit_modulation.py",
                 "audit_admission.py", "audit_masked.py", "masked_experiment.py", "memory_diagnostics.py"):
        relative = "python/src/chuff_mpu/sim/"+name
        content = (root/relative).read_bytes()
        target = args.output/"source"/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        source_hashes[relative] = hashlib.sha256(content).hexdigest()
    manifest = dict(source_sha256=source_hashes,
        input_manifest_sha256=hashlib.sha256((args.folder/"manifest.json").read_bytes()).hexdigest(),
        input_observations_sha256=hashlib.sha256((args.folder/"observations.jsonl.gz").read_bytes()).hexdigest(),
        quadrature_positions=200, scope="Post-hoc diagnostics on observed seeds 1000-1009; not a causal or permutation significance test",
        counterfactual="Entire matched background-only trajectory; learner history differs from the source-bearing trajectory",
        interval_null="Uniform generator-legal positions, fixed length, no wrap, fixed selected region")
    (args.output/"manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (args.output/"analysis.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
