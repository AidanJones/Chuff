"""Descriptive paired-seed summaries of the frozen held-out frontend comparison."""

import argparse
from collections import Counter
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np

from chuff_mpu.sim.audit_published import audit as audit_experiment
from chuff_mpu.sim.published_protocol import select
from chuff_mpu.sim.recorded_catalog import digest


POSITIVE = ("episode_detection", "first_episode_novel_attention", "repeat_identity",
            "source_dependent_repeat", "strict_episode_exclusion", "wrong_repeat_identity")


def estimate(values, draws):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("nonfinite report metric")
    return dict(mean=float(values.mean()),
                bootstrap_95=np.quantile(values[draws].mean(axis=1), [.025, .975]).tolist())


def summarize(summaries, plan, selection):
    """Resample seeds jointly across suites, never treat replays as new sources."""
    seeds = plan["evaluation_seeds"]
    positive, control = plan["positive_suites"], plan["control_suites"]
    indexed = {(row["suite"], row["seed"]): row for row in summaries}
    expected = {(suite, seed) for suite in positive+control for seed in seeds}
    if len(indexed) != len(summaries) or set(indexed) != expected:
        raise ValueError("incomplete or duplicate evaluation matrix")
    modes = sorted(selection["evaluation_ids"]+["adaptive_gain"])
    if len(modes) != len(set(modes)):
        raise ValueError("duplicate evaluation actor IDs")
    if any(set(row["episodes"]) != set(modes) for row in summaries):
        raise ValueError("evaluation actor matrix differs from frozen selection")
    draws = np.random.default_rng(731).integers(0, len(seeds), (10000, len(seeds)))
    by_seed, result = {}, {}
    for mode in modes:
        metrics = {key: [] for key in (*POSITIVE, "control_admissions", "control_attention", "observe_ms")}
        for seed in seeds:
            for key in POSITIVE:
                metrics[key].append(float(np.mean([indexed[suite, seed]["episodes"][mode]["rates"][key]
                                                  for suite in positive])))
            metrics["control_admissions"].append(float(np.mean([
                indexed[suite, seed]["episodes"][mode]["counts"]["background_admissions"] for suite in control])))
            metrics["control_attention"].append(float(np.mean([
                indexed[suite, seed]["episodes"][mode]["rates"]["background_attention"] for suite in control])))
            metrics["observe_ms"].append(float(np.mean([
                indexed[suite, seed]["runtime"][mode]["mean_ms"] for suite in positive+control])))
        by_seed[mode] = metrics
        counts = Counter()
        per_recording = {}
        for row in summaries:
            item = row["episodes"][mode]
            if row["suite"] in positive:
                counts.update({key: value for key, value in item["counts"].items()
                               if key not in ("distinct_recordings", "admitted_recordings")})
                counts["recording_run_pairs"] += item["counts"]["distinct_recordings"]
                counts["admitted_recording_run_pairs"] += item["counts"]["admitted_recordings"]
                for name, recording in item["by_recording"].items():
                    per_recording.setdefault(name, Counter()).update(recording)
        counts["distinct_input_recordings"] = len(per_recording)
        result[mode] = dict(metrics={key: estimate(values, draws) for key, values in metrics.items()},
            positive_counts=dict(counts), by_recording={key: dict(value) for key, value in per_recording.items()},
            seed_values=metrics)
    choose = lambda group: selection["selections"][group]["selected"] or selection["selections"][group]["diagnostic"]
    quiet160, quiet128, pcen, mcra = [choose(family) for family in ("quiet160", "quiet128", "pcen160", "mcra128")]
    pairs = {
        "pcen_minus_quiet160": (pcen, quiet160),
        "mcra_diagnostic_minus_quiet128": (mcra, quiet128),
        "quiet128_minus_quiet160": (quiet128, quiet160),
        "adaptive_minus_quiet160": ("adaptive_gain", quiet160),
    }
    for family in ("quiet160", "quiet128", "pcen160", "mcra128"):
        pairs[family+"_relative_pipeline_minus_raw_pipeline"] = (choose(family+"/feature1"), choose(family+"/feature0"))
    contrasts = {}
    for name, (current, reference) in pairs.items():
        contrasts[name] = dict(current=current, reference=reference, metrics={
            key: estimate(np.array(by_seed[current][key])-by_seed[reference][key], draws)
            for key in by_seed[current]})
    return dict(schema="chuff.recorded.published-report.v1", modes=result, contrasts=contrasts,
        uncertainty="10000 paired seed-bootstrap draws jointly across suites; conditional on five fixed foreground recordings, not corpus-level confidence",
        scope="Post-run descriptive aggregation; no retuning or new selection. Contrast intervals are exploratory and unadjusted for multiple comparisons.",
        features="Selected raw/relative pipelines may have different thresholds, especially PCEN; their difference is not an isolated stored-feature effect.",
        runtime="Mean warmed actual-observe milliseconds across six equally weighted suites per seed; desktop shared, not CPU-isolated",
        seeds=seeds, positive_suites=positive, control_suites=control)


def markdown(report):
    lines = ["# Held-out frontend comparison", "", report["uncertainty"], "", report["scope"], "",
        "| Configuration | Strict identity % [95% interval] | Background admissions / 210s [95% interval] | Wrong identity % | Observe ms |",
        "|---|---:|---:|---:|---:|"]
    for mode, item in report["modes"].items():
        metrics = item["metrics"]
        interval = lambda key, scale: f"{scale*metrics[key]['mean']:.2f} [{scale*metrics[key]['bootstrap_95'][0]:.2f}, {scale*metrics[key]['bootstrap_95'][1]:.2f}]"
        lines.append(f"| {mode} | {interval('strict_episode_exclusion', 100)} | {interval('control_admissions', 1)} | {100*metrics['wrong_repeat_identity']['mean']:.2f} | {metrics['observe_ms']['mean']:.2f} |")
    lines.extend(["", report["features"], "", report["runtime"], ""])
    return "\n".join(lines)


def build(evaluation, output):
    audit_experiment(evaluation)
    plan = json.loads((evaluation/"input/protocol.json").read_text())
    selection = json.loads((evaluation/"input/development-selection.json").read_text())
    development = json.loads((evaluation/"input/development-summary.json").read_text())
    if selection != select(development, plan):
        raise ValueError("selection provenance mismatch")
    result = summarize(json.loads((evaluation/"summary.json").read_text()), plan, selection)
    output.mkdir(parents=True, exist_ok=False)
    sources, inputs = {}, {}
    package = Path(__file__).resolve().parents[1]
    root = package.parents[2]
    for path in (*package.rglob("*.py"), root/"python/uv.lock"):
        name = path.relative_to(root).as_posix()
        content = path.read_bytes()
        destination = output/"source"/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
        sources[name] = digest(content)
    for name, source in {"evaluation-manifest.json": evaluation/"manifest.json",
        "summary.json": evaluation/"summary.json", "protocol.json": evaluation/"input/protocol.json",
        "selection.json": evaluation/"input/development-selection.json",
        "development-summary.json": evaluation/"input/development-summary.json",
        "catalog.json": evaluation/"input/catalog.json"}.items():
        destination = output/"input"/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        content = source.read_bytes()
        destination.write_bytes(content)
        inputs["input/"+name] = digest(content)
    (output/"report.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    (output/"report.md").write_text(markdown(result), encoding="utf-8")
    manifest = dict(schema="chuff.recorded.published-report-archive.v1", source_sha256=sources,
        input_sha256=inputs, output_sha256={name: digest((output/name).read_bytes()) for name in ("report.json", "report.md")},
        dependencies={name: version(name) for name in ("numpy", "scipy", "librosa")})
    (output/"manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def audit(folder):
    manifest = json.loads((folder/"manifest.json").read_text())
    for name, expected in manifest["source_sha256"].items():
        assert digest((folder/"source"/name).read_bytes()) == expected
    for group in ("input_sha256", "output_sha256"):
        for name, expected in manifest[group].items():
            assert digest((folder/name).read_bytes()) == expected
    load = lambda name: json.loads((folder/"input"/name).read_text())
    source_manifest = load("evaluation-manifest.json")
    assert digest((folder/"input/summary.json").read_bytes()) == source_manifest["output_sha256"]["summary.json"]
    plan, selection = load("protocol.json"), load("selection.json")
    for source, original in (("protocol.json", "protocol.json"), ("selection.json", "development-selection.json"),
                             ("development-summary.json", "development-summary.json"), ("catalog.json", "catalog.json")):
        assert digest((folder/"input"/source).read_bytes()) == source_manifest["input_sha256"]["input/"+original]
    assert select(load("development-summary.json"), plan) == selection
    expected = summarize(load("summary.json"), plan, selection)
    assert json.loads((folder/"report.json").read_text()) == expected
    assert (folder/"report.md").read_text() == markdown(expected)
    print("Comparison source/input/output hashes, original evaluation links, paired-seed estimates and contrasts verified")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("build", "audit"))
    parser.add_argument("folder", type=Path)
    parser.add_argument("--evaluation", type=Path)
    args = parser.parse_args()
    if args.action == "audit":
        audit(args.folder)
    else:
        if args.evaluation is None:
            parser.error("build requires --evaluation")
        build(args.evaluation, args.folder)


if __name__ == "__main__":
    main()
