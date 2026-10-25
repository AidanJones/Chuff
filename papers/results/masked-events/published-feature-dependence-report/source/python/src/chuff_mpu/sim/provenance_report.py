"""Paired-seed diagnostic impact of birth-window versus admission-window labels."""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from chuff_mpu.sim.audit_provenance import audit as audit_replay
from chuff_mpu.sim.recorded_catalog import digest


ENDPOINTS = ("repeat_identity", "source_dependent_repeat", "strict_episode_exclusion",
             "wrong_repeat_identity", "deletion_repeat_identity")


def derive(summaries, plan, modes):
    seeds, positives, controls = plan["evaluation_seeds"], plan["positive_suites"], plan["control_suites"]
    indexed = {(row["suite"], row["seed"]): row for row in summaries}
    if len(indexed) != len(summaries) or set(indexed) != {(suite, seed) for suite in positives+controls for seed in seeds}:
        raise ValueError("incomplete or duplicate provenance matrix")
    if any(set(row["scores"][label]) != set(modes) for row in summaries for label in ("admission", "birth")):
        raise ValueError("provenance actor matrix mismatch")
    draws = np.random.default_rng(731).integers(0, len(seeds), (10000, len(seeds)))
    def estimate(values):
        values = np.asarray(values, dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("nonfinite provenance metric")
        return dict(mean=float(values.mean()), bootstrap_95=np.quantile(values[draws].mean(axis=1), [.025, .975]).tolist())
    result = {}
    for mode in modes:
        transitions = {}
        for name, suites in (("positive", positives), ("control", controls)):
            tally = Counter()
            for row in summaries:
                if row["suite"] in suites:
                    tally.update(row["transitions"][mode])
            transitions[name] = dict(tally)
        metrics = {}
        for key in ENDPOINTS:
            values = {label: [float(np.mean([indexed[suite, seed]["scores"][label][mode]["rates"][key]
                       for suite in positives])) for seed in seeds] for label in ("admission", "birth")}
            metrics[key] = {label: estimate(sequence) for label, sequence in values.items()}
            metrics[key]["birth_minus_admission"] = estimate(np.array(values["birth"])-values["admission"])
        by_recording = {}
        for row in summaries:
            if row["suite"] not in positives:
                continue
            for label in ("admission", "birth"):
                item = row["scores"][label][mode]
                if item["rates"]["episode_detection"] != row["scores"]["admission"][mode]["rates"]["episode_detection"]:
                    raise ValueError("label diagnostic changed physical detection")
                for recording, counts in item["by_recording"].items():
                    by_recording.setdefault(recording, {key: Counter() for key in ("admission", "birth")})[label].update(counts)
        result[mode] = dict(transitions=transitions, metrics=metrics,
            by_recording={recording: {label: dict(counts) for label, counts in values.items()}
                          for recording, values in by_recording.items()})
    return dict(schema="chuff.recorded.provenance-report.v1", modes=result,
        scope="Observed-data label sensitivity, not a replacement accuracy estimate or a newly held-out method",
        uncertainty="10000 paired seed-bootstrap draws, jointly across positive suites; conditional on five fixed foreground recordings",
        causality="Birth and admission window labels do not identify the acoustic cause of a stored vector. Background births with prior source tails are explicit.",
        seeds=seeds)


def markdown(report):
    lines = ["# Candidate birth/admission label sensitivity", "", report["scope"], "", report["uncertainty"], "",
        "| Configuration | Mixed-stream admissions | FG to BG | BG to FG | Different recording | Admission strict % | Birth strict % | Difference pp [95% interval] |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for mode, item in report["modes"].items():
        tally = item["transitions"]["positive"]
        score = item["metrics"]["strict_episode_exclusion"]
        difference = score["birth_minus_admission"]
        lines.append(f"| {mode} | {tally['admissions']} | {tally['foreground_to_background']} | {tally['background_to_foreground']} | {tally['different_recording']} | {100*score['admission']['mean']:.2f} | {100*score['birth']['mean']:.2f} | {100*difference['mean']:+.2f} [{100*difference['bootstrap_95'][0]:+.2f}, {100*difference['bootstrap_95'][1]:+.2f}] |")
    lines.extend(["", report["causality"], ""])
    return "\n".join(lines)


def build(replay, output):
    audit_replay(replay)
    plan = json.loads((replay/"input/protocol.json").read_text())
    provenance_manifest = json.loads((replay/"manifest.json").read_text())
    summaries = json.loads((replay/"summary.json").read_text())
    result = derive(summaries, plan, sorted(provenance_manifest["settings"]))
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(schema="chuff.recorded.provenance-report-archive.v1",
                    source_sha256={}, input_sha256={}, output_sha256={})
    package = Path(__file__).resolve().parents[1]
    root = package.parents[2]
    for source in (*package.rglob("*.py"), root/"python/uv.lock"):
        name = source.relative_to(root).as_posix()
        destination = output/"source"/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
        manifest["source_sha256"][name] = digest(source.read_bytes())
    for name, source in {"provenance-manifest.json": replay/"manifest.json", "summary.json": replay/"summary.json",
        "protocol.json": replay/"input/protocol.json"}.items():
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
    manifest = json.loads((folder/"manifest.json").read_text())
    for name, expected in manifest["source_sha256"].items():
        assert digest((folder/"source"/name).read_bytes()) == expected
    for group in ("input_sha256", "output_sha256"):
        for name, expected in manifest[group].items():
            assert digest((folder/name).read_bytes()) == expected
    original = json.loads((folder/"input/provenance-manifest.json").read_text())
    assert digest((folder/"input/summary.json").read_bytes()) == original["output_sha256"]["summary.json"]
    assert digest((folder/"input/protocol.json").read_bytes()) == original["input_sha256"]["input/protocol.json"]
    expected = derive(json.loads((folder/"input/summary.json").read_text()),
        json.loads((folder/"input/protocol.json").read_text()), sorted(original["settings"]))
    assert json.loads((folder/"report.json").read_text()) == expected
    assert (folder/"report.md").read_text() == markdown(expected)
    print("Provenance report hashes, original replay links and paired-seed label contrasts verified")


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
