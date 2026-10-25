"""Evaluator-only source-deletion probes from the same prior learner state."""

from collections import Counter

import numpy as np

from chuff_mpu.sim.localization import DETECTED, overlap_iou


def intervention_metrics(rows):
    result = {}
    for mode in sorted({r["mode"] for r in rows}):
        count, occurrences = Counter(), Counter()
        for row in (r for r in rows if r["mode"] == mode and r["truth"] is not None):
            probe = row.get("without_current_probe")
            if probe is None:
                raise ValueError("missing identical-state source-deletion probe")
            count["events"] += 1
            repeat = occurrences[row["truth"]] >= 2
            occurrences[row["truth"]] += 1
            count["repeats"] += repeat
            count["attention_without_current"] += probe["status"] in DETECTED
            count["repeat_identity_without_current"] += repeat and probe["current_class_recognised"]
            count["status_or_id_changed"] += ((row["status"], row["prototype_id"])
                                               != (probe["status"], probe["prototype_id"]))
            if probe["status"] in DETECTED:
                region = probe["region_start_s"], probe["region_stop_s"]
                if None in region:
                    raise ValueError("missing probe-region timing")
                count["overlap_without_current"] += overlap_iou(region,
                    (row["event_start_s"], row["event_stop_s"])) >= .25
        rate = lambda key, denominator: count[key]/count[denominator] if count[denominator] else None
        result[mode] = dict(counts=dict(count),
            attention_without_current_rate=rate("attention_without_current", "events"),
            overlap_without_current_rate=rate("overlap_without_current", "events"),
            repeat_identity_without_current_rate=rate("repeat_identity_without_current", "repeats"),
            status_or_id_change_rate=rate("status_or_id_changed", "events"))
    return result


def paired_estimates(summaries):
    result = []
    for suite in sorted({r["suite"] for r in summaries}):
        runs = sorted((r for r in summaries if r["suite"] == suite), key=lambda r: r["seed"])
        for mode in runs[0]["metrics"]:
            values = {
                "localized_detection": [r["localization"][mode]["localized_detection_rate"]["0.25"] for r in runs],
                "localized_repeat": [r["localization"][mode]["localized_repeat_recognition_rate"]["0.25"] for r in runs],
                "attention_without_current": [r["interventions"][mode]["attention_without_current_rate"] for r in runs],
                "overlap_without_current": [r["interventions"][mode]["overlap_without_current_rate"] for r in runs],
                "repeat_identity_without_current": [r["interventions"][mode]["repeat_identity_without_current_rate"] for r in runs],
            }
            draws = np.random.default_rng(731).integers(0, len(runs), (10000, len(runs)))
            metrics = {}
            for name, numbers in values.items():
                if any(v is None for v in numbers):
                    metrics[name] = None
                    continue
                array = np.array(numbers)
                metrics[name] = dict(mean=float(array.mean()),
                    bootstrap_95=np.quantile(array[draws].mean(axis=1), [.025, .975]).tolist())
            if all(v is not None for v in values["localized_repeat"]+values["repeat_identity_without_current"]):
                differences = np.array(values["localized_repeat"])-np.array(values["repeat_identity_without_current"])
                metrics["localized_repeat_minus_identity_without_current"] = dict(mean=float(differences.mean()),
                    bootstrap_95=np.quantile(differences[draws].mean(axis=1), [.025, .975]).tolist())
            result.append(dict(suite=suite, mode=mode, n_seeds=len(runs), metrics=metrics))
    return result


def condition_contrasts(summaries):
    """Pre/post target and source-factor contrasts, pairing the same seed."""
    indexed = {(r["suite"], r["seed"]): r for r in summaries}
    seeds = sorted({r["seed"] for r in summaries})
    comparisons = [(f"{name}/post_residual", f"{name}/input")
                   for name in ("reference", "pitch", "duration", "pitch_duration")]
    comparisons += [(f"{name}/post_residual", "reference/post_residual")
                    for name in ("pitch", "duration", "pitch_duration")]
    result = []
    for current, reference in comparisons:
        for mode in summaries[0]["metrics"]:
            values = [indexed[current, seed]["localization"][mode]["localized_repeat_recognition_rate"]["0.25"]
                      for seed in seeds]
            baseline = [indexed[reference, seed]["localization"][mode]["localized_repeat_recognition_rate"]["0.25"]
                        for seed in seeds]
            if any(v is None for v in values+baseline):
                continue
            differences = np.array(values)-np.array(baseline)
            draws = np.random.default_rng(731).integers(0, len(seeds), (10000, len(seeds)))
            result.append(dict(current=current, reference=reference, mode=mode, n_seeds=len(seeds),
                mean=float(differences.mean()),
                bootstrap_95=np.quantile(differences[draws].mean(axis=1), [.025, .975]).tolist()))
    return result
