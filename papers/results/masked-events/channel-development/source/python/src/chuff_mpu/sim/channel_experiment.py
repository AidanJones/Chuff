"""Post-channel SNR, independent source transforms and identical-state probes."""

import argparse
from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from chuff_mpu.attention import AttentionConfig, ForegroundLearner
from chuff_mpu.band_attention import BandAttentionConfig, BandForegroundLearner
from chuff_mpu.gain_attention import CalibratedGainConfig, CalibratedGainForegroundLearner
from chuff_mpu.sim.attention_experiment import evaluate
from chuff_mpu.sim.channel_metrics import intervention_metrics, paired_estimates
from chuff_mpu.sim.independent_scene import IndependentSourceScene, TRANSFORMS
from chuff_mpu.sim.localization import score_regions
from chuff_mpu.sim.masked_experiment import aggregate, class_metrics, save_manifest
from chuff_mpu.sim.memory_diagnostics import diagnose
from chuff_mpu.sim.scene import SceneConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("development", "evaluation"), required=True)
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=120)
    parser.add_argument("--snr-db", type=float, default=-6)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.seed_start < 0 or args.seeds < 1 or args.rounds <= 6:
        parser.error("nonnegative seeds and >6 rounds required")
    target_snr = args.snr_db
    args.snr_db = [target_snr]
    SceneConfig(rounds=args.rounds, snr_db=target_snr)
    conditions = [(transform, reference) for transform in TRANSFORMS
                  for reference in ("input", "post_residual")]+[("background", "input")]
    args.suites = ["/".join(c) for c in conditions]
    args.output.mkdir(parents=True, exist_ok=False)
    save_manifest(args.output, args, BandAttentionConfig())
    manifest_path = args.output/"manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = Path(__file__).resolve().parents[4]
    sources = ["python/src/chuff_mpu/gain_attention.py"] + ["python/src/chuff_mpu/sim/"+name for name in
        ("varied_scene.py", "independent_scene.py", "channel_experiment.py", "channel_metrics.py",
         "localization.py", "memory_diagnostics.py", "audit_channel.py")]
    for name in sources:
        content = (root/name).read_bytes()
        path = args.output/"source"/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        manifest["source_sha256"][name] = hashlib.sha256(content).hexdigest()
    manifest.update(variants={"quiet_band": asdict(BandAttentionConfig()),
                              "calibrated_gain": asdict(CalibratedGainConfig())},
        scope="Source active interval before filtering; post target is current-source/residual after channel, retaining prior source tails",
        transforms="Independent librosa pitch/time operations, no natural-source equivalence claim",
        examples={}, counterfactual="Copied identical prior learner state; remove only current source, discard branch state")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    summaries = []
    with gzip.open(args.output/"observations.jsonl.gz", "wt", encoding="utf-8") as stream:
        for transform, reference in conditions:
            suite = transform+"/"+reference
            settings = SceneConfig(rounds=args.rounds, collect_stems=True, snr_reference=reference,
                snr_db=target_snr, nuisance_probability=0, event_probability=0 if transform == "background" else .35)
            for seed in range(args.seed_start, args.seed_start+args.seeds):
                scene = IndependentSourceScene(seed, settings, environment_seed=100000+seed,
                                               transform="reference" if transform == "background" else transform)
                events, checks = [], []
                def frames():
                    example_saved = False
                    for step, frame in enumerate(scene):
                        # Validate actual component arrays before any learner sees the mixture.
                        recombined = frame.background_audio+frame.source_audio
                        counterfactual_sum = frame.without_current_source_audio+frame.current_source_audio
                        scale = max(float(np.max(np.abs(frame.audio))), 1e-6)
                        error = max(float(np.max(np.abs(frame.audio-recombined))),
                                    float(np.max(np.abs(frame.audio-counterfactual_sum))))
                        if error > 2e-6*scale+1e-8:
                            raise ValueError("linear component recombination failed")
                        checks.append(error)
                        if frame.label is not None:
                            info = dict(step=step, label=frame.label, before=frame.active_snr_db,
                                post_background=frame.post_channel_background_snr_db,
                                post_residual=frame.post_channel_residual_snr_db,
                                event_start_s=frame.event_start_s, event_stop_s=frame.event_stop_s,
                                source_parameters=frame.source_parameters)
                            events.append(info)
                            if seed == args.seed_start and not example_saved:
                                path = args.output/"examples"/(suite.replace("/", "-")+".npz")
                                path.parent.mkdir(parents=True, exist_ok=True)
                                np.savez_compressed(path, mixture=frame.audio, background=frame.background_audio,
                                    all_sources=frame.source_audio, current_source=frame.current_source_audio,
                                    without_current=frame.without_current_source_audio, metadata=np.array(json.dumps(info)))
                                manifest["examples"][str(path.relative_to(args.output))] = hashlib.sha256(path.read_bytes()).hexdigest()
                                example_saved = True
                        yield frame
                learners = dict(original=ForegroundLearner(), quiet_band=BandForegroundLearner(),
                                calibrated_gain=CalibratedGainForegroundLearner())
                summary, rows, _ = evaluate(seed, settings, AttentionConfig(), learners=learners,
                    include_baseline=False, frames=frames(), counterfactual_probes=True)
                summary.update(suite=suite, by_class=class_metrics(rows), diagnostics=diagnose(rows),
                    localization=score_regions(rows), interventions=intervention_metrics(rows),
                    scene=asdict(scene.settings), environment=scene.parameters,
                    snr_events=events, max_component_recombination_error=max(checks))
                summaries.append(summary)
                for row in rows:
                    row.update(suite=suite, snr_db=target_snr)
                    stream.write(json.dumps(row, allow_nan=False)+"\n")
                print(f"{suite} seed={seed}", flush=True)
    (args.output/"summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    (args.output/"aggregate.json").write_text(json.dumps(aggregate(summaries), indent=2), encoding="utf-8")
    (args.output/"analysis.json").write_text(json.dumps(paired_estimates(summaries), indent=2), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
