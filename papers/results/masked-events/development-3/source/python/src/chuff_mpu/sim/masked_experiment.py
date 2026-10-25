"""Paired masked-event comparisons with raw evidence and independent seed intervals."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import gzip
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import subprocess

import numpy as np

from chuff_mpu.attention import AttentionConfig, ForegroundLearner
from chuff_mpu.band_attention import BandAttentionConfig, BandForegroundLearner
from chuff_mpu.sim.attention_experiment import evaluate
from chuff_mpu.sim.challenge_scene import ChallengeScene
from chuff_mpu.sim.scene import AcousticScene, SceneConfig


def aggregate(summaries):
    """Paired bootstrap over seeds, not over windows or correlated SNRs."""
    groups = []
    for suite, snr in sorted({(s["suite"], s["snr_db"]) for s in summaries}):
        runs = sorted([s for s in summaries if (s["suite"], s["snr_db"]) == (suite, snr)],
                      key=lambda s: s["seed"])
        rng = np.random.default_rng(731)
        indices = rng.integers(0, len(runs), size=(10000, len(runs)))
        group = {"suite": suite, "snr_db": snr, "n_seeds": len(runs), "modes": {}}
        for mode in runs[0]["metrics"]:
            metrics = {}
            for key in ["event_detection_rate", "repeat_recognition_rate", "background_attention_rate", "false_admissions"]:
                values = [s["metrics"][mode][key] for s in runs]
                if any(value is None for value in values):
                    metrics[key] = None
                    continue
                values = np.array(values, dtype=float)
                estimate = {"mean": float(values.mean()), "range": [float(values.min()), float(values.max())],
                            "bootstrap_95": np.quantile(values[indices].mean(axis=1), [0.025, 0.975]).tolist()}
                if mode != "original":
                    reference = np.array([s["metrics"]["original"][key] for s in runs], dtype=float)
                    differences = values - reference
                    estimate["paired_difference"] = {"mean": float(differences.mean()),
                        "bootstrap_95": np.quantile(differences[indices].mean(axis=1), [0.025, 0.975]).tolist()}
                metrics[key] = estimate
            group["modes"][mode] = metrics
        groups.append(group)
    return groups


def save_manifest(output, args, settings):
    root = Path(__file__).resolve().parents[4]
    paths = ["python/src/chuff_mpu/attention.py", "python/src/chuff_mpu/band_attention.py",
             "python/src/chuff_mpu/sim/scene.py", "python/src/chuff_mpu/sim/challenge_scene.py",
             "python/src/chuff_mpu/sim/attention_experiment.py", "python/src/chuff_mpu/sim/masked_experiment.py",
             "python/pyproject.toml", "python/uv.lock"]
    hashes = {}
    for name in paths:
        content = (root / name).read_bytes()
        target = output / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        hashes[name] = hashlib.sha256(content).hexdigest()
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    manifest = {"phase": args.phase, "created_utc": datetime.now(timezone.utc).isoformat(),
                "base_revision": revision, "source_sha256": hashes, "source_snapshot": "source/",
                "python": platform.python_version(),
                "dependencies": {name: version(name) for name in ["numpy", "scipy", "librosa"]},
                "original": asdict(AttentionConfig()), "selected_band": asdict(settings),
                "variants": {"band_gate_raw": {"feature_mode": 0, "noise_mode": 0},
                             "band_relative": {"feature_mode": 1, "noise_mode": 0},
                             "quiet_band": {"feature_mode": 0, "noise_mode": 1}},
                "seed_start": args.seed_start, "seeds": args.seeds, "rounds": args.rounds,
                "snr_db": args.snr_db, "suites": args.suites,
                "scene": asdict(SceneConfig(rounds=args.rounds, structured_distractors=True)),
                "uncertainty": "10000 paired seed bootstrap draws per suite/SNR; descriptive percentile intervals"}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["development", "evaluation", "stress"], required=True)
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=120)
    parser.add_argument("--snr-db", type=float, nargs="+", default=[-6, 0, 6])
    parser.add_argument("--suites", choices=["tonal", "extended"], nargs="+", default=["tonal", "extended"])
    parser.add_argument("--local-threshold", type=float, default=3.5)
    parser.add_argument("--broad-threshold", type=float, default=1.5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.seed_start < 0 or args.seeds < 1 or args.rounds <= 6:
        parser.error("nonnegative seed-start, positive seeds and >6 rounds required")
    settings = BandAttentionConfig(local_threshold=args.local_threshold, broad_threshold=args.broad_threshold)
    args.output.mkdir(parents=True, exist_ok=False)
    save_manifest(args.output, args, settings)
    summaries = []
    with gzip.open(args.output / "observations.jsonl.gz", "wt", encoding="utf-8") as stream:
        for suite in args.suites:
            for snr in args.snr_db:
                scene_config = SceneConfig(rounds=args.rounds, snr_db=snr, structured_distractors=True)
                for seed in range(args.seed_start, args.seed_start + args.seeds):
                    learners = {"original": ForegroundLearner(),
                                "band_gate_raw": BandForegroundLearner(replace(settings, feature_mode=0, noise_mode=0)),
                                "band_relative": BandForegroundLearner(replace(settings, feature_mode=1, noise_mode=0)),
                                "quiet_band": BandForegroundLearner(replace(settings, feature_mode=0, noise_mode=1))}
                    scene = (AcousticScene if suite == "tonal" else ChallengeScene)(seed, scene_config)
                    summary, rows, _ = evaluate(seed, scene_config, AttentionConfig(),
                                                include_baseline=False, learners=learners, frames=scene)
                    summary["suite"] = suite
                    summaries.append(summary)
                    for row in rows:
                        row.update(suite=suite, snr_db=snr)
                        stream.write(json.dumps(row, allow_nan=False) + "\n")
                    print(json.dumps({"suite": suite, "snr": snr, "seed": seed,
                                      "metrics": {mode: {key: value for key, value in metrics.items()
                                                  if key in ["repeat_recognition_rate", "event_detection_rate", "false_admissions"]}
                                                  for mode, metrics in summary["metrics"].items()}}), flush=True)
    (args.output / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    (args.output / "aggregate.json").write_text(json.dumps(aggregate(summaries), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
