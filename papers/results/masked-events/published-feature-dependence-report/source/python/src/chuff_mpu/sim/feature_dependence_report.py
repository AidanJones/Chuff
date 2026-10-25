"""Describe selected-feature deletion probes, conditional on actual prior state."""

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path

import numpy as np

from chuff_mpu.sim.audit_feature_dependence import audit as audit_replay
from chuff_mpu.sim.feature_dependence import KINDS
from chuff_mpu.sim.feature_probes import match_dependence
from chuff_mpu.sim.recorded_catalog import digest


STAGES = ("all", "birth", "confirmation", "adjacent", "admission")
CONTEXTS = ("recording", "recording_changed_audio", "gap_with_tail", "gap_without_tail", "control")


def window_context(row):
    if row["suite"].endswith("/control"):
        return "control"
    context = row["window_context"]
    if context["recording_id"] is not None:
        return "recording"
    return "gap_with_tail" if context["previous_source_tail_s"] > 0 else "gap_without_tail"


def in_stage(row, stage):
    return dict(all=True, birth=row["is_birth"], confirmation=row["is_confirmation"],
        adjacent=not row["is_birth"] and not row["is_confirmation"], admission=row["is_admission"])[stage]


def tally(rows):
    result = Counter(windows=0, actual_zero=0, stored_zero=0, bed_noop=0,
                     bed_match_changed_region=0, bed_match_disjoint_region=0)
    result.update({role+"/"+kind: 0 for role in ("bed", "without_current") for kind in KINDS})
    for row in rows:
        result["windows"] += 1
        result["actual_zero"] += row["observations"]["actual"]["feature_norm"] == 0
        result["stored_zero"] += row["stored_feature_norm"] == 0
        result["bed_noop"] += row["waveform_sha256"]["actual"] == row["waveform_sha256"]["bed"]
        for role in ("bed", "without_current"):
            result[role+"/"+match_dependence(row, role)] += 1
        if match_dependence(row) == "probe_matches_stored":
            probe, actual = row["observations"]["bed"], row["observations"]["actual"]
            result["bed_match_disjoint_region"] += probe["region_iou"] == 0
            result["bed_match_changed_region"] += any(probe[k] != actual[k] for k in ("region_start_s", "region_stop_s"))
    return dict(result)


def ratio_estimate(counts, role, draws):
    """Equal seed weights; never silently drop an undefined seed denominator."""
    numerator = [c[role+"/probe_matches_stored"] for c in counts]
    denominator = [c["windows"]-c[role+"/actual_outside_radius"] for c in counts]
    values = [n/d if d else None for n, d in zip(numerator, denominator)]
    mean, interval = None, None
    if all(value is not None for value in values):
        values_array = np.asarray(values)
        mean = float(values_array.mean())
        interval = np.quantile(values_array[draws].mean(axis=1), [.025, .975]).tolist()
    return dict(numerator=sum(numerator), denominator=sum(denominator),
        pooled_fraction=sum(numerator)/sum(denominator) if sum(denominator) else None,
        per_seed=values, valid_seeds=sum(v is not None for v in values), mean=mean, bootstrap_95=interval)


def derive(records, pairs, plan, modes):
    seeds = plan["evaluation_seeds"]
    suites = plan["positive_suites"]+plan["control_suites"]
    if len(set(seeds)) != len(seeds) or not seeds:
        raise ValueError("invalid seed plan")
    if any(r["mode"] not in modes or r["seed"] not in seeds or r["suite"] not in suites for r in records+pairs):
        raise ValueError("unplanned feature record")
    keys = [(r["mode"], r["suite"], r["seed"], r["step"]) for r in records]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate feature window")
    draws = np.random.default_rng(1741).integers(0, len(seeds), (10000, len(seeds)))
    result = {}
    for mode in modes:
        selected = [r for r in records if r["mode"] == mode]
        contexts = {}
        for context in CONTEXTS:
            rows = [r for r in selected if (window_context(r) == context or
                (context == "recording_changed_audio" and window_context(r) == "recording"
                 and r["waveform_sha256"]["bed"] != r["waveform_sha256"]["actual"]))]
            stages = {}
            for stage in STAGES:
                chosen = [r for r in rows if in_stage(r, stage)]
                counts = [tally([r for r in chosen if r["seed"] == seed]) for seed in seeds]
                stages[stage] = dict(counts=tally(chosen), by_seed=counts,
                    reproducible={role: ratio_estimate(counts, role, draws) for role in ("bed", "without_current")})
            contexts[context] = stages
        joint = {}
        for scope in ("positive", "control"):
            admitted = [r for r in pairs if r["mode"] == mode and r["suite"] in plan[scope+"_suites"]]
            transitions = {}
            for transition in sorted({r["transition"] for r in admitted}):
                transitions[transition] = dict(Counter(r["joint"] for r in admitted if r["transition"] == transition))
            joint[scope] = dict(admissions=len(admitted), categories=dict(Counter(r["joint"] for r in admitted)),
                transitions=transitions)
        by_recording = {}
        for recording in sorted({r["window_context"]["recording_id"] for r in selected
                                 if window_context(r) == "recording"}):
            rows = [r for r in selected if r["window_context"]["recording_id"] == recording]
            by_recording[recording] = {stage: tally([r for r in rows if in_stage(r, stage)]) for stage in ("birth", "admission")}
        result[mode] = dict(contexts=contexts, admission_pairs=joint, by_recording=by_recording)
    return dict(schema="chuff.recorded.feature-dependence-report.v1", modes=result, seeds=seeds,
        uncertainty="10000 seed-bootstrap draws; aggregate counts across suites within seed, then equally weight seeds; conditional on five fixed foreground recordings",
        scope="Observed selected-feature intervention with actual prior exposure; regions can change; no semantic causality or new held-out policy",
        denominator="Actual feature must match its stored vector. Zero/unmatchable actuals are excluded explicitly. Undefined seed denominators make the mean and interval null.",
        contexts="Recording, gap with tail, gap without tail and control partition observations. recording_changed_audio is an overlapping subset excluding identical-audio removals, not an active-energy SNR measure.",
        stages="Birth, confirmation and adjacent partition all observations; admission overlaps confirmation. Never-admitted candidates are included.")


def markdown(report):
    lines = ["# Selected-feature foreground-removal diagnostic", "", report["scope"], "", report["uncertainty"], "",
        report["denominator"], "", report["stages"], "",
        "| Configuration | Target-window births: bed match / eligible | Equal-seed % [95% interval] | Target-window admissions: bed match / eligible | Pure-control pairs reproduced at both ends / all |",
        "|---|---:|---:|---:|---:|"]
    for mode, item in report["modes"].items():
        birth = item["contexts"]["recording"]["birth"]["reproducible"]["bed"]
        admission = item["contexts"]["recording"]["admission"]["reproducible"]["bed"]
        control = item["admission_pairs"]["control"]
        estimate = "undefined"
        if birth["mean"] is not None:
            lo, hi = birth["bootstrap_95"]
            estimate = f"{100*birth['mean']:.2f} [{100*lo:.2f}, {100*hi:.2f}]"
        lines.append(f"| {mode} | {birth['numerator']}/{birth['denominator']} | {estimate} | {admission['numerator']}/{admission['denominator']} | {control['categories'].get('both_reproducible', 0)}/{control['admissions']} |")
    lines.extend(["", "A bed match can select a different region. A failed match can mean no detection or an outside-radius feature; inspect the separate counts in report.json. Pure controls are identical-audio negative controls, not a source-deletion test. Source stems are evaluator-only.", ""])
    return "\n".join(lines)


def build(replay, output):
    audit_replay(replay)
    load = lambda name: json.loads((replay/name).read_text())
    original = load("manifest.json")
    with gzip.open(replay/"probes.jsonl.gz", "rt", encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream]
    result = derive(records, load("admission-pairs.json"), load("input/protocol.json"), sorted(original["settings"]))
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(schema="chuff.recorded.feature-dependence-report-archive.v1", source_sha256={}, input_sha256={}, output_sha256={})
    package = Path(__file__).resolve().parents[1]
    root = package.parents[2]
    for source in (*package.rglob("*.py"), root/"python/uv.lock", root/"python/pyproject.toml"):
        name = source.relative_to(root).as_posix()
        content = source.read_bytes()
        destination = output/"source"/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
        manifest["source_sha256"][name] = digest(content)
    for name, source in {"replay-manifest.json": replay/"manifest.json", "probes.jsonl.gz": replay/"probes.jsonl.gz",
        "admission-pairs.json": replay/"admission-pairs.json", "protocol.json": replay/"input/protocol.json"}.items():
        content = source.read_bytes()
        destination = output/"input"/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
        manifest["input_sha256"]["input/"+name] = digest(content)
    (output/"report.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    (output/"report.md").write_text(markdown(result), encoding="utf-8")
    manifest["output_sha256"] = {name: digest((output/name).read_bytes()) for name in ("report.json", "report.md")}
    (output/"manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def audit(folder):
    load = lambda name: json.loads((folder/name).read_text())
    manifest = load("manifest.json")
    assert manifest["schema"] == "chuff.recorded.feature-dependence-report-archive.v1"
    for name, expected in manifest["source_sha256"].items():
        assert digest((folder/"source"/name).read_bytes()) == expected, name
    for group in ("input_sha256", "output_sha256"):
        for name, expected in manifest[group].items():
            assert digest((folder/name).read_bytes()) == expected, name
    original = load("input/replay-manifest.json")
    for name in ("probes.jsonl.gz", "admission-pairs.json"):
        assert manifest["input_sha256"]["input/"+name] == original["output_sha256"][name]
    assert manifest["input_sha256"]["input/protocol.json"] == original["input_sha256"]["input/protocol.json"]
    with gzip.open(folder/"input/probes.jsonl.gz", "rt", encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream]
    result = derive(records, load("input/admission-pairs.json"), load("input/protocol.json"), sorted(original["settings"]))
    assert load("report.json") == result
    assert (folder/"report.md").read_text() == markdown(result)
    print("Feature-dependence report hashes, replay links, context counts and seed intervals verified")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("build", "audit"))
    parser.add_argument("folder", type=Path)
    parser.add_argument("--replay", type=Path)
    args = parser.parse_args()
    if args.action == "audit":
        audit(args.folder)
    else:
        if args.replay is None:
            parser.error("build requires --replay")
        build(args.replay, args.folder)


if __name__ == "__main__":
    main()
