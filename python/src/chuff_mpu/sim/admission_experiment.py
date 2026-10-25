"""Paired frozen admission/matching policies and pure ambience controls."""

import argparse
from dataclasses import asdict, replace
import gzip
import hashlib
import json
from pathlib import Path

from chuff_mpu.attention import AttentionConfig, ForegroundLearner
from chuff_mpu.band_attention import BandAttentionConfig, BandForegroundLearner
from chuff_mpu.sim.attention_experiment import evaluate
from chuff_mpu.sim.masked_experiment import aggregate, class_metrics, save_manifest
from chuff_mpu.sim.memory_diagnostics import diagnose
from chuff_mpu.sim.scene import SceneConfig
from chuff_mpu.sim.varied_scene import VariedScene


SUITES = ("reference", "source", "joint", "joint_nuisance", "joint_ambience")
POLICIES = {"quiet_band": {}, "confirm4": {"confirmations": 4},
            "wide": {"match_distance": .28},
            "wide_confirm4": {"confirmations": 4, "match_distance": .28}}


def make_scene(seed, rounds, snr, suite):
    if suite not in SUITES:
        raise ValueError("unknown suite")
    config = SceneConfig(rounds=rounds, snr_db=snr, structured_distractors=True,
                        event_probability=0 if suite in ("joint_nuisance", "joint_ambience") else .35,
                        nuisance_probability=0 if suite == "joint_ambience" else .12)
    scene = VariedScene(seed, config, environment_seed=100000+seed,
                        condition="joint" if suite.startswith("joint") else suite)
    return config, scene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-start", type=int, default=700)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=120)
    parser.add_argument("--snr-db", type=float, nargs="+", default=[-6, 0])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.seed_start < 0 or args.seeds < 1 or args.rounds <= 6:
        parser.error("nonnegative seed-start, positive seeds and >6 rounds required")
    for snr in args.snr_db:
        SceneConfig(rounds=args.rounds, snr_db=snr)
    args.phase, args.suites = "admission-diagnostic", list(SUITES)
    args.output.mkdir(parents=True, exist_ok=False)
    save_manifest(args.output, args, BandAttentionConfig())
    manifest_path = args.output/"manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = Path(__file__).resolve().parents[4]
    paths = ["python/src/chuff_mpu/sim/"+name for name in
             ("varied_scene.py", "memory_diagnostics.py", "admission_experiment.py", "audit_admission.py")]
    for name in paths:
        content = (root/name).read_bytes()
        target = args.output/"source"/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        manifest["source_sha256"][name] = hashlib.sha256(content).hexdigest()
    manifest.update(variants=POLICIES, repeat_eligibility="after two true occurrences for all policies",
                    environment_seed_rule="100000 + event seed",
                    scope="Fixed policies, procedural diagnostics; no tuning or deployment selection")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    summaries = []
    with gzip.open(args.output/"observations.jsonl.gz", "wt", encoding="utf-8") as stream:
        for suite in SUITES:
            for snr in args.snr_db:
                for seed in range(args.seed_start, args.seed_start+args.seeds):
                    config, scene = make_scene(seed, args.rounds, snr, suite)
                    learners = {"original": ForegroundLearner()}
                    learners.update({name: BandForegroundLearner(replace(BandAttentionConfig(), **changes))
                                     for name, changes in POLICIES.items()})
                    summary, rows, _ = evaluate(seed, config, AttentionConfig(), include_baseline=False,
                                                learners=learners, frames=scene)
                    summary.update(suite=suite, by_class=class_metrics(rows), diagnostics=diagnose(rows),
                                   scene=asdict(config), environment=scene.parameters)
                    summaries.append(summary)
                    for row in rows:
                        row.update(suite=suite, snr_db=snr)
                        stream.write(json.dumps(row, allow_nan=False)+"\n")
                    print(f"{suite} SNR={snr} seed={seed}", flush=True)
    (args.output/"summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    (args.output/"aggregate.json").write_text(json.dumps(aggregate(summaries), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
