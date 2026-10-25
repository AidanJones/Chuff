"""Frozen paired gain-mode transition study with source-deletion branches."""

import argparse
from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path

from chuff_mpu.attention import AttentionConfig
from chuff_mpu.adaptive_gain import AdaptiveGainForegroundLearner
from chuff_mpu.guarded_gain import GuardedGainForegroundLearner
from chuff_mpu.band_attention import BandAttentionConfig, BandForegroundLearner
from chuff_mpu.gain_attention import (CalibratedGainForegroundLearner, SpectralGainForegroundLearner)
from chuff_mpu.sim.attention_experiment import evaluate
from chuff_mpu.sim.masked_experiment import save_manifest
from chuff_mpu.sim.scene import SceneConfig
from chuff_mpu.sim.transition_scene import CASES, TransitionScene
from chuff_mpu.sim.transition_metrics import estimates, phase_metrics
from chuff_mpu.sim.transition_adaptation import adaptation_metrics
from chuff_mpu.sim.memory_diagnostics import diagnose


class Telemetry:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.telemetry = []

    def observe(self, clip):
        result = super().observe(clip)
        record = dict(gain_cv=self._gain_cv,
            gain_compensation_enabled=getattr(self, "gain_compensation_enabled", True),
            mode_switches=getattr(self, "mode_switches", 0))
        if hasattr(self, "held_admissions"):
            record.update(held_admissions=self.held_admissions, admission_support=self.admission_support,
                          shadow_status=self._shadow_observation.status,
                          admission_check_required=self.admission_check_required, gain_uncertain=self.gain_uncertain)
        self.telemetry.append(record)
        return result


class Fixed(Telemetry, CalibratedGainForegroundLearner):
    pass


class Spectral(Telemetry, SpectralGainForegroundLearner):
    pass


class Adaptive(Telemetry, AdaptiveGainForegroundLearner):
    pass


class Guarded(Telemetry, GuardedGainForegroundLearner):
    pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("development", "evaluation"), required=True)
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=120)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adaptive", action="store_true", help="Include the experimental persistent-evidence selector")
    parser.add_argument("--guarded", action="store_true", help="Also require compensated-view support for new admissions")
    args = parser.parse_args()
    if args.seed_start < 0 or args.seeds < 1 or args.rounds < 30:
        parser.error("nonnegative seeds, positive seed count and >=30 rounds required")
    if args.guarded and not args.adaptive:
        parser.error("--guarded requires --adaptive for its paired comparator")
    args.snr_db = [-6.]
    args.suites = [case+"/"+kind for case in CASES for kind in ("positive", "control")]
    args.output.mkdir(parents=True, exist_ok=False)
    save_manifest(args.output, args, BandAttentionConfig())
    path = args.output/"manifest.json"
    manifest = json.loads(path.read_text())
    # The shared manifest helper carries old-study defaults; replace unused fields.
    for unused in ("original", "selected_band", "variants", "scene"):
        manifest.pop(unused, None)
    root = Path(__file__).resolve().parents[4]
    # Capture every implementation transitively used by this runner, plus the auditor.
    for name in ("gain_attention.py", "adaptive_gain.py", "guarded_gain.py", "sim/varied_scene.py", "sim/modulation_scene.py",
                 "sim/transition_scene.py", "sim/transition_metrics.py", "sim/transition_experiment.py",
                 "sim/audit_transition.py", "sim/localization.py", "sim/transition_adaptation.py",
                 "sim/memory_diagnostics.py"):
        name = "python/src/chuff_mpu/"+name
        content = (root/name).read_bytes()
        destination = args.output/"source"/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
        manifest["source_sha256"][name] = hashlib.sha256(content).hexdigest()
    manifest.update(primary="after-change localized repeat identity, IoU>=.25; phase scorer retains prior admissions",
        settings={"quiet_band": asdict(BandAttentionConfig()),
                  "calibrated_gain": asdict(Fixed().settings), "spectral_gain": asdict(Spectral().settings)},
        information_boundary="All actors observe only mixture audio; truth/stems/phase are evaluator-only",
        scope="Fixed source factories; varied room/mic; no nuisance or additional simultaneous masker; -6dB post-channel current/residual")
    if args.adaptive:
        manifest["settings"]["adaptive_gain"] = asdict(Adaptive().settings)
    if args.guarded:
        manifest["settings"]["guarded_gain"] = asdict(Guarded().settings)
    manifest["scene_protocol"] = dict(rounds=args.rounds, warmup=6, duration_s=1.,
        snr_db=-6., snr_reference="post_residual", positive_probability=.35,
        control_probability=0., nuisance_probability=0., collect_stems=True,
        background_change="TransitionScene defines the declared condition changes")
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    summaries = []
    with gzip.open(args.output/"observations.jsonl.gz", "wt", encoding="utf-8") as stream:
        for case in CASES:
            for kind in ("positive", "control"):
                settings = SceneConfig(rounds=args.rounds, collect_stems=True, snr_reference="post_residual",
                    snr_db=-6, background_change=False, nuisance_probability=0,
                    event_probability=.35 if kind == "positive" else 0)
                for seed in range(args.seed_start, args.seed_start+args.seeds):
                    scene = TransitionScene(seed, settings, environment_seed=100000+seed, case=case)
                    learners = dict(quiet_band=BandForegroundLearner(), calibrated_gain=Fixed(), spectral_gain=Spectral())
                    if args.adaptive:
                        learners["adaptive_gain"] = Adaptive()
                    if args.guarded:
                        learners["guarded_gain"] = Guarded()
                    checks = []
                    def frames():
                        import numpy as np
                        for frame in scene:
                            error = max(float(np.max(np.abs(frame.audio-frame.background_audio-frame.source_audio))),
                                float(np.max(np.abs(frame.audio-frame.without_current_source_audio-frame.current_source_audio))))
                            if error > 2e-6*max(float(np.max(np.abs(frame.audio))), 1e-6)+1e-8:
                                raise ValueError("component recombination failed")
                            if frame.label is not None and abs(frame.post_channel_residual_snr_db+6) > 1e-8:
                                raise ValueError("post-channel SNR target failed")
                            checks.append(error)
                            yield frame
                    summary, rows, _ = evaluate(seed, settings, AttentionConfig(), frames=frames(),
                        learners=learners, include_baseline=False, counterfactual_probes=True)
                    for row in rows:
                        row.update(suite=case+"/"+kind, phase=scene.phase(row["step"]))
                        if hasattr(learners[row["mode"]], "telemetry"):
                            row.update(learners[row["mode"]].telemetry[row["step"]])
                        stream.write(json.dumps(row, allow_nan=False)+"\n")
                    summary.update(suite=case+"/"+kind, phases=phase_metrics(rows), scene=asdict(settings),
                        environment=scene.parameters, max_component_error=max(checks))
                    if args.adaptive:
                        summary["adaptation"] = adaptation_metrics(rows, case, scene.change_step)
                    if args.guarded:
                        summary["diagnostics"] = diagnose(rows)
                    summaries.append(summary)
                    print(f"{case}/{kind} seed={seed}", flush=True)
    (args.output/"summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    (args.output/"analysis.json").write_text(json.dumps(estimates(summaries), indent=2), encoding="utf-8")
    manifest["output_sha256"] = {name: hashlib.sha256((args.output/name).read_bytes()).hexdigest()
        for name in ("observations.jsonl.gz", "summary.json", "analysis.json")}
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
