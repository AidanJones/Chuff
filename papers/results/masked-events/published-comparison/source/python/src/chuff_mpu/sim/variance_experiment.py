"""Frozen-detector, paired factorized variance experiment."""

import argparse
from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path

from chuff_mpu.attention import AttentionConfig, ForegroundLearner
from chuff_mpu.band_attention import BandAttentionConfig, BandForegroundLearner
from chuff_mpu.sim.attention_experiment import evaluate
from chuff_mpu.sim.masked_experiment import aggregate, class_metrics, save_manifest
from chuff_mpu.sim.scene import SceneConfig
from chuff_mpu.sim.varied_scene import CONDITIONS, VariedScene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-start", type=int, default=600)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=120)
    parser.add_argument("--snr-db", type=float, nargs="+", default=[-6, 0])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.seed_start < 0 or args.seeds < 1 or args.rounds <= 6:
        parser.error("nonnegative seed-start, positive seeds and >6 rounds required")
    # Validate before creating any archive.
    for snr in args.snr_db:
        SceneConfig(rounds=args.rounds, snr_db=snr)
    args.phase = "variance-evaluation"
    args.suites = list(CONDITIONS) + ["joint_background"]
    args.output.mkdir(parents=True, exist_ok=False)
    save_manifest(args.output, args, BandAttentionConfig())
    manifest_path = args.output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = Path(__file__).resolve().parents[4]
    for name in ["python/src/chuff_mpu/sim/varied_scene.py",
                 "python/src/chuff_mpu/sim/variance_experiment.py"]:
        content = (root/name).read_bytes()
        target = args.output/"source"/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        manifest["source_sha256"][name] = hashlib.sha256(content).hexdigest()
    manifest["variants"] = {"quiet_band": {"feature_mode": 0, "noise_mode": 1}}
    manifest["environment_seed_rule"] = "100000 + event seed; reused across conditions/SNR"
    manifest["scope"] = ("Fixed room/microphone per run; per-occurrence coupled pitch/duration variation; "
                         "pre-channel active SNR. No overlaps, recordings, channel switches or detector retuning.")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    summaries = []
    with gzip.open(args.output/"observations.jsonl.gz", "wt", encoding="utf-8") as stream:
        for suite in args.suites:
            for snr in args.snr_db:
                config = SceneConfig(rounds=args.rounds, snr_db=snr, structured_distractors=True,
                                     event_probability=0 if suite == "joint_background" else .35)
                for seed in range(args.seed_start, args.seed_start+args.seeds):
                    scene = VariedScene(seed, config, environment_seed=100000+seed,
                                        condition="joint" if suite == "joint_background" else suite)
                    summary, rows, _ = evaluate(seed, config, AttentionConfig(), include_baseline=False,
                        learners={"original": ForegroundLearner(), "quiet_band": BandForegroundLearner()},
                        frames=scene)
                    summary.update(suite=suite, by_class=class_metrics(rows),
                                   environment=scene.parameters, scene=asdict(scene.settings))
                    summaries.append(summary)
                    for row in rows:
                        row.update(suite=suite, snr_db=snr)
                        stream.write(json.dumps(row, allow_nan=False)+"\n")
                    print(f"{suite} SNR={snr} seed={seed}", flush=True)
    (args.output/"summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    (args.output/"aggregate.json").write_text(json.dumps(aggregate(summaries), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
