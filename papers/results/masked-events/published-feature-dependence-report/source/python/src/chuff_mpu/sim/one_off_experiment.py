"""One foreground episode per stream: measure novel attention, not recurrence learning."""

import argparse
from dataclasses import asdict, replace
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from chuff_mpu.adaptive_gain import AdaptiveGainForegroundLearner
from chuff_mpu.attention import AttentionConfig
from chuff_mpu.guarded_gain import GuardedGainForegroundLearner
from chuff_mpu.sim.attention_experiment import evaluate
from chuff_mpu.sim.challenge_scene import EXTENDED_EVENTS
from chuff_mpu.sim.localization import DETECTED, overlap_iou
from chuff_mpu.sim.masked_experiment import save_manifest
from chuff_mpu.sim.scene import SceneConfig
from chuff_mpu.sim.transition_scene import TransitionScene


CASES = ("stationary", "stationary_slow")
MODES = ("adaptive_gain", "guarded_gain")


class OneOffScene(TransitionScene):
    def __init__(self, *args, label, **kwargs):
        super().__init__(*args, **kwargs)
        if self.settings.rounds <= 66:
            raise ValueError("one-off schedule needs window 66")
        if label not in EXTENDED_EVENTS:
            raise ValueError("unknown source factory")
        self.event_labels = (label,)

    def next_frame(self):
        original = self.settings
        # Override only event scheduling; reuse the complete physical renderer.
        self.settings = replace(original, event_probability=float(self._step == 66))
        try:
            return super().next_frame()
        finally:
            self.settings = original


def measure(rows):
    result = {}
    for mode in MODES:
        selected = [r for r in rows if r["mode"] == mode]
        events = [r for r in selected if r["truth"] is not None]
        if len(events) != 1 or events[0]["step"] != 66:
            raise ValueError("exactly one target at window 66 required")
        row = events[0]
        detected = row["status"] in DETECTED
        hit = detected and overlap_iou((row["region_start_s"], row["region_stop_s"]),
            (row["event_start_s"], row["event_stop_s"])) >= .25
        result[mode] = dict(localized_attention=int(hit),
            localized_novel_attention=int(hit and row["status"] in ("candidate", "learned")),
            target_status=row["status"], target_novelty=row["novelty"],
            background_admissions=sum(r["truth"] is None and r["status"] == "learned" for r in selected))
    return result


def aggregate(summaries):
    result = []
    for case in CASES:
        seeds = sorted({r["seed"] for r in summaries})
        for mode in MODES:
            metrics = {}
            for metric in ("localized_attention", "localized_novel_attention", "background_admissions"):
                values = np.array([np.mean([r["one_off"][mode][metric] for r in summaries
                    if r["case"] == case and r["seed"] == seed]) for seed in seeds])
                other = np.array([np.mean([r["one_off"]["adaptive_gain"][metric] for r in summaries
                    if r["case"] == case and r["seed"] == seed]) for seed in seeds])
                draws = np.random.default_rng(731).integers(0, len(seeds), (10000, len(seeds)))
                metrics[metric] = dict(mean=float(values.mean()),
                    bootstrap_95=np.quantile(values[draws].mean(axis=1), [.025, .975]).tolist(),
                    paired_difference_vs_adaptive=float((values-other).mean()),
                    paired_difference_95=np.quantile((values-other)[draws].mean(axis=1), [.025, .975]).tolist())
            result.append(dict(case=case, mode=mode, seeds=len(seeds),
                class_episodes_per_seed=len(EXTENDED_EVENTS), metrics=metrics))
    return result


def audit(folder):
    manifest = json.loads((folder/"manifest.json").read_text())
    for name, digest in manifest["source_sha256"].items():
        assert hashlib.sha256((folder/"source"/name).read_bytes()).hexdigest() == digest
    for name, digest in manifest["output_sha256"].items():
        assert hashlib.sha256((folder/name).read_bytes()).hexdigest() == digest
    with gzip.open(folder/"observations.jsonl.gz", "rt", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    seeds = range(manifest["seed_start"], manifest["seed_start"]+manifest["seeds"])
    indexed = {(r["suite"], r["seed"], r["step"], r["mode"]): r for r in rows}
    suites = {case+"/"+label for case in CASES for label in EXTENDED_EVENTS}
    assert len(indexed) == len(rows)
    assert set(indexed) == {(suite, seed, step, mode) for suite in suites for seed in seeds
                           for step in range(120) for mode in MODES}
    summaries = json.loads((folder/"summary.json").read_text())
    assert len(summaries) == len(suites)*len(seeds)
    assert {(r["suite"], r["seed"]) for r in summaries} == {(suite, seed) for suite in suites for seed in seeds}
    for summary in summaries:
        selected = [indexed[summary["suite"], summary["seed"], step, mode]
                    for step in range(120) for mode in MODES]
        assert summary["one_off"] == measure(selected)
        for row in selected:
            assert row["truth"] == (summary["label"] if row["step"] == 66 else None)
            if row["truth"] is not None:
                assert abs(row["post_channel_residual_snr_db"]+6) < 1e-8
        for field in ("event_start_s", "event_stop_s", "truth"):
            assert selected[66*2][field] == selected[66*2+1][field]
    assert json.loads((folder/"analysis.json").read_text()) == aggregate(summaries)
    print(f"{len(rows)} decisions: complete one-off matrix, hashes, single-target schedule, SNR and seed estimates verified")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--phase", choices=("development", "evaluation"))
    parser.add_argument("--seed-start", type=int)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.audit:
        audit(args.audit)
        return
    if args.phase is None or args.output is None or args.seed_start is None or args.seed_start < 0 or args.seeds < 1:
        parser.error("phase, output, nonnegative seed-start and positive seeds required")
    args.rounds, args.snr_db = 120, [-6.]
    args.suites = [case+"/"+label for case in CASES for label in EXTENDED_EVENTS]
    args.output.mkdir(parents=True, exist_ok=False)
    save_manifest(args.output, args, AttentionConfig())
    path = args.output/"manifest.json"
    manifest = json.loads(path.read_text())
    for unused in ("original", "selected_band", "variants", "scene"):
        manifest.pop(unused, None)
    package = Path(__file__).resolve().parents[1]
    root = package.parents[2]
    for source in package.rglob("*.py"):
        name = source.relative_to(root).as_posix()
        destination = args.output/"source"/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        content = source.read_bytes()
        destination.write_bytes(content)
        manifest["source_sha256"][name] = hashlib.sha256(content).hexdigest()
    manifest.update(target_step=66, scope="Exactly one procedural target per 120-window stream; -6dB post-residual",
        uncertainty="Average six factory outcomes within seed, then bootstrap seeds; descriptive paired differences",
        settings={"adaptive_gain": asdict(AdaptiveGainForegroundLearner().settings),
                  "guarded_gain": asdict(GuardedGainForegroundLearner().settings)})
    summaries = []
    with gzip.open(args.output/"observations.jsonl.gz", "wt", encoding="utf-8") as stream:
        for case in CASES:
            for label in EXTENDED_EVENTS:
                for seed in range(args.seed_start, args.seed_start+args.seeds):
                    settings = SceneConfig(rounds=120, nuisance_probability=0, background_change=False,
                        collect_stems=True, snr_reference="post_residual", snr_db=-6, event_probability=0)
                    scene = OneOffScene(seed, settings, environment_seed=100000+seed, case=case, label=label)
                    learners = dict(adaptive_gain=AdaptiveGainForegroundLearner(), guarded_gain=GuardedGainForegroundLearner())
                    summary, rows, _ = evaluate(seed, settings, AttentionConfig(), learners=learners,
                        frames=scene, include_baseline=False, counterfactual_probes=True)
                    summary.update(suite=case+"/"+label, case=case, label=label, one_off=measure(rows))
                    summaries.append(summary)
                    for row in rows:
                        row.update(suite=case+"/"+label)
                        stream.write(json.dumps(row, allow_nan=False)+"\n")
                    print(f"{case}/{label} seed={seed}", flush=True)
    (args.output/"summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    (args.output/"analysis.json").write_text(json.dumps(aggregate(summaries), indent=2), encoding="utf-8")
    manifest["output_sha256"] = {name: hashlib.sha256((args.output/name).read_bytes()).hexdigest()
        for name in ("observations.jsonl.gz", "summary.json", "analysis.json")}
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
