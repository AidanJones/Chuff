"""Fixed factorial test of modulation, spectral shape and channel colouring."""

import argparse
from dataclasses import asdict, replace
import gzip
import hashlib
import itertools
import json
from pathlib import Path

from chuff_mpu.attention import AttentionConfig, ForegroundLearner
from chuff_mpu.band_attention import BandAttentionConfig, BandForegroundLearner
from chuff_mpu.gain_attention import (GainAttentionConfig, GainForegroundLearner,
                                      SpectralGainConfig, SpectralGainForegroundLearner,
                                      CalibratedGainConfig, CalibratedGainForegroundLearner)
from chuff_mpu.sim.attention_experiment import evaluate
from chuff_mpu.sim.masked_experiment import aggregate, class_metrics, save_manifest
from chuff_mpu.sim.memory_diagnostics import diagnose
from chuff_mpu.sim.localization import score_regions
from chuff_mpu.sim.modulation_scene import MODULATIONS, ModulationScene
from chuff_mpu.sim.scene import SceneConfig


FACTORS = list(itertools.product(("reference", "varied"), ("reference", "varied"),
                                MODULATIONS, ("ambience", "foreground")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("development", "evaluation"), required=True)
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=120)
    parser.add_argument("--snr-db", type=float, nargs="+", default=[-6])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--localization", action="store_true",
                        help="Add detector/source timing, overlap metrics and a six-cell focused matrix")
    parser.add_argument("--gain-study", action="store_true", help="Include fixed .05/.2/.5 second causal RMS frontends")
    parser.add_argument("--gain-time-s", type=float, nargs="+", default=[.05, .2, .5],
                        help="Gain candidates in seconds; freeze these before evaluation")
    parser.add_argument("--spectral-gain", action="store_true", help="Include the fixed warmup-template median-band gain frontend")
    parser.add_argument("--calibrated-gain", action="store_true", help="Choose compensation from unlabelled warmup variation")
    args = parser.parse_args()
    if args.seed_start < 0 or args.seeds < 1 or args.rounds <= 6 or len(args.snr_db) != 1:
        parser.error("nonnegative seeds, >6 rounds and one SNR required; ambience has no SNR")
    if args.gain_study and not args.localization:
        parser.error("gain study requires --localization and its positive/negative controls")
    if args.spectral_gain and not args.gain_study:
        parser.error("spectral gain requires --gain-study")
    if args.calibrated_gain and not args.spectral_gain:
        parser.error("calibrated gain requires --spectral-gain")
    gain_times = [(f"{time:g}s", time) for time in args.gain_time_s]
    if len(set(args.gain_time_s)) != len(args.gain_time_s):
        parser.error("gain time constants must be unique")
    if args.gain_study:
        for _, time in gain_times:
            GainAttentionConfig(gain_time_s=time)
    SceneConfig(rounds=args.rounds, snr_db=args.snr_db[0])
    factors_to_run = ([f for f in FACTORS if f[0] == f[1] == "varied"] if args.localization else FACTORS)
    args.suites = ["/".join(factors) for factors in factors_to_run]
    args.output.mkdir(parents=True, exist_ok=False)
    save_manifest(args.output, args, BandAttentionConfig())
    manifest_path = args.output/"manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = Path(__file__).resolve().parents[4]
    for name in ("varied_scene.py", "modulation_scene.py", "modulation_experiment.py", "memory_diagnostics.py",
                 "localization.py"):
        relative = "python/src/chuff_mpu/sim/"+name
        data = (root/relative).read_bytes()
        target = args.output/"source"/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        manifest["source_sha256"][relative] = hashlib.sha256(data).hexdigest()
    if args.gain_study:
        relative = "python/src/chuff_mpu/gain_attention.py"
        data = (root/relative).read_bytes()
        target = args.output/"source"/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        manifest["source_sha256"][relative] = hashlib.sha256(data).hexdigest()
    manifest.update(variants={"quiet_band": {"noise_mode": 1}, "binwise": {"noise_mode": 0}},
                    environment_seed_rule="100000 + event seed", modulation=MODULATIONS,
                    scope="No source variation or halfway changes. Foreground has no nuisances. Three frozen methods.")
    if args.gain_study:
        manifest["variants"].update({f"gain_{name}": asdict(GainAttentionConfig(gain_time_s=time))
                                     for name, time in gain_times})
        manifest["scope"] = "Existing baselines and declared causal full-band gain normalizers; fixed memory matching"
    if args.spectral_gain:
        manifest["variants"]["spectral_gain"] = asdict(SpectralGainConfig())
        manifest["scope"] += "; plus fixed median-band spectral gain with warmup-only profile"
    if args.calibrated_gain:
        manifest["variants"]["calibrated_gain"] = asdict(CalibratedGainConfig())
        manifest["scope"] += "; plus warmup-derived compensation choice without condition labels"
    if args.localization:
        manifest["localization"] = dict(iou_thresholds=[.1, .25, .5],
            primary_threshold=.25, region="full FFT-frame sample support",
            event="pre-channel active source interval; no room-tail extension",
            tail_flag="previous source finite room-IR support; IIR mic tail not bounded")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    summaries = []
    with gzip.open(args.output/"observations.jsonl.gz", "wt", encoding="utf-8") as stream:
        for spectrum, channel, modulation, sources in factors_to_run:
            suite = "/".join((spectrum, channel, modulation, sources))
            config = SceneConfig(rounds=args.rounds, snr_db=args.snr_db[0], background_change=False,
                                 event_probability=.35 if sources == "foreground" else 0,
                                 nuisance_probability=0)
            for seed in range(args.seed_start, args.seed_start+args.seeds):
                scene = ModulationScene(seed, config, environment_seed=100000+seed,
                    spectrum=spectrum, channel=channel, modulation=modulation)
                learners = {"original": ForegroundLearner(), "quiet_band": BandForegroundLearner(),
                            "binwise": BandForegroundLearner(replace(BandAttentionConfig(), noise_mode=0))}
                if args.gain_study:
                    learners.update({f"gain_{name}": GainForegroundLearner(GainAttentionConfig(gain_time_s=time))
                                     for name, time in gain_times})
                if args.spectral_gain:
                    learners["spectral_gain"] = SpectralGainForegroundLearner()
                if args.calibrated_gain:
                    learners["calibrated_gain"] = CalibratedGainForegroundLearner()
                summary, rows, _ = evaluate(seed, config, AttentionConfig(), include_baseline=False,
                                            learners=learners, frames=scene)
                summary.update(suite=suite, scene=asdict(scene.settings), environment=scene.parameters,
                               by_class=class_metrics(rows), diagnostics=diagnose(rows))
                if args.calibrated_gain:
                    model = learners["calibrated_gain"]
                    summary["gain_calibration"] = dict(cv=model._calibration_cv, compensate=model._compensate)
                if args.localization:
                    summary["localization"] = score_regions(rows)
                summaries.append(summary)
                for row in rows:
                    row.update(suite=suite, snr_db=config.snr_db)
                    stream.write(json.dumps(row, allow_nan=False)+"\n")
                print(f"{suite} seed={seed}", flush=True)
    (args.output/"summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    (args.output/"aggregate.json").write_text(json.dumps(aggregate(summaries), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
