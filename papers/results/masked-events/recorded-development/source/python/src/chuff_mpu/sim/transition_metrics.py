"""Phase metrics retain the complete admission/occurrence history across changes."""

from collections import Counter

import numpy as np

from chuff_mpu.sim.localization import DETECTED, overlap_iou


def phase_metrics(rows):
    result = {}
    for mode in sorted({r["mode"] for r in rows}):
        assigned, occurrences = {}, Counter()
        counts = {phase: Counter() for phase in ("before", "after")}
        for row in (r for r in rows if r["mode"] == mode):
            count = counts[row["phase"]]
            label, status = row["truth"], row["status"]
            count["windows"] += 1
            count["calibrating"] += status == "calibrating"
            if label is not None:
                repeat = occurrences[label] >= 2
                occurrences[label] += 1
                count["events"] += 1
                count["repeats"] += repeat
                detected = status in DETECTED
                hit = detected and overlap_iou((row["region_start_s"], row["region_stop_s"]),
                    (row["event_start_s"], row["event_stop_s"])) >= .25
                count["localized_events"] += hit
                count["localized_repeat_identity"] += (hit and repeat and status == "recognised"
                    and assigned.get(row["prototype_id"]) == label)
                probe = row["without_current_probe"]
                count["deletion_attention"] += probe["status"] in DETECTED
                count["deletion_repeat_identity"] += repeat and probe["current_class_recognised"]
            else:
                count["background_windows"] += 1
                count["false_attention"] += status in DETECTED
                count["false_admissions"] += status == "learned"
            if status == "learned":
                assigned[row["prototype_id"]] = label
        result[mode] = {}
        for phase, count in counts.items():
            rates = {name: count[numerator]/count[denominator] if count[denominator] else None
                for name, numerator, denominator in (
                    ("localized_detection", "localized_events", "events"),
                    ("localized_repeat", "localized_repeat_identity", "repeats"),
                    ("false_attention", "false_attention", "background_windows"),
                    ("deletion_attention", "deletion_attention", "events"),
                    ("deletion_repeat_identity", "deletion_repeat_identity", "repeats"))}
            result[mode][phase] = dict(counts=dict(count), rates=rates)
    return result


def estimates(summaries):
    """Equal-seed means and paired mode contrasts; no window-level bootstrap."""
    result = []
    for suite in sorted({r["suite"] for r in summaries}):
        runs = sorted((r for r in summaries if r["suite"] == suite), key=lambda r: r["seed"])
        modes = sorted(runs[0]["phases"])
        draws = np.random.default_rng(731).integers(0, len(runs), (10000, len(runs)))
        for mode in modes:
            for phase in ("before", "after"):
                metrics = {}
                for name in runs[0]["phases"][mode][phase]["rates"]:
                    values = [r["phases"][mode][phase]["rates"][name] for r in runs]
                    if any(v is None for v in values):
                        metrics[name] = None
                        continue
                    array = np.array(values)
                    metrics[name] = dict(mean=float(array.mean()),
                        bootstrap_95=np.quantile(array[draws].mean(axis=1), [.025, .975]).tolist())
                for reference in ("quiet_band", "calibrated_gain"):
                    if reference == mode:
                        continue
                    values = [r["phases"][mode][phase]["rates"]["localized_repeat"] for r in runs]
                    baseline = [r["phases"][reference][phase]["rates"]["localized_repeat"] for r in runs]
                    if any(v is None for v in values+baseline):
                        continue
                    array = np.array(values)-np.array(baseline)
                    metrics["localized_repeat_minus_"+reference] = dict(mean=float(array.mean()),
                        bootstrap_95=np.quantile(array[draws].mean(axis=1), [.025, .975]).tolist())
                result.append(dict(suite=suite, mode=mode, phase=phase, n_seeds=len(runs), metrics=metrics))
    return result
