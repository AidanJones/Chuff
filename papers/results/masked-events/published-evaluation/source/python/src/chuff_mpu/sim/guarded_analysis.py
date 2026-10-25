"""Paired admission trade-offs; report missing classes alongside conditional delays."""

import numpy as np


def paired(values):
    values = np.array([v for v in values if v is not None], dtype=float)
    if not len(values):
        return None
    draws = np.random.default_rng(731).integers(0, len(values), (10000, len(values)))
    return dict(mean=float(values.mean()), paired_seeds=len(values),
        bootstrap_95=np.quantile(values[draws].mean(axis=1), [.025, .975]).tolist())


def guarded_analysis(summaries):
    result = []
    for suite in sorted({r["suite"] for r in summaries}):
        runs = sorted((r for r in summaries if r["suite"] == suite), key=lambda r: r["seed"])
        differences = {name: [] for name in ("localized_repeat", "localized_detection", "false_attention",
                                            "false_admissions", "admitted_classes", "matched_class_first_admission_occurrences")}
        cohorts = []
        for run in runs:
            old, new = run["phases"]["adaptive_gain"]["after"], run["phases"]["guarded_gain"]["after"]
            for name in ("localized_repeat", "localized_detection", "false_attention"):
                a, b = old["rates"][name], new["rates"][name]
                differences[name].append(None if a is None or b is None else b-a)
            differences["false_admissions"].append(new["counts"].get("false_admissions", 0)-old["counts"].get("false_admissions", 0))
            old = run["diagnostics"]["adaptive_gain"]["first_admission_occurrence"]
            new = run["diagnostics"]["guarded_gain"]["first_admission_occurrence"]
            matched = sorted(set(old)&set(new))
            differences["admitted_classes"].append(len(new)-len(old))
            differences["matched_class_first_admission_occurrences"].append(
                float(np.mean([new[label]-old[label] for label in matched])) if matched else None)
            cohorts.append(dict(seed=run["seed"], shared_classes=matched,
                lost_classes=sorted(set(old)-set(new)), gained_classes=sorted(set(new)-set(old))))
        result.append(dict(suite=suite, direction="guarded minus adaptive; after-phase primary, full-run class admission",
            paired_differences={name: paired(values) for name, values in differences.items()}, cohorts=cohorts))
    return result
