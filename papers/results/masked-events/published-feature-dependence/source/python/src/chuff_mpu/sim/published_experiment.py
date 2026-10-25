"""Run the frozen, source-separated recorded PCEN/MCRA frontend comparison."""

import argparse
from dataclasses import asdict
import gzip
import json
import os
from pathlib import Path
import platform
from types import SimpleNamespace

import numpy as np

from chuff_mpu.adaptive_gain import AdaptiveGainForegroundLearner
from chuff_mpu.band_attention import BandAttentionConfig
from chuff_mpu.sim.masked_experiment import save_manifest
from chuff_mpu.sim.published_protocol import actor, audit_protocol, select
from chuff_mpu.sim.recorded_catalog import audit as audit_catalog, digest
from chuff_mpu.sim.recorded_experiment import evaluate
from chuff_mpu.sim.recorded_metrics import aggregate, episode_metrics
from chuff_mpu.sim.recorded_scene import BACKGROUNDS, FOREGROUNDS, RecordedScene, load_audio
from chuff_mpu.sim.scene import SceneConfig


def actors(plan, names):
    return {**{name: actor(plan["candidates"][name]) for name in names},
            "adaptive_gain": AdaptiveGainForegroundLearner()}


def runtime(rows, settings):
    result = {}
    for mode, config in settings.items():
        durations = np.array([row["observe_ns"]/1e6 for row in rows
            if row["mode"] == mode and row["step"] >= config["warmup_clips"]])
        result[mode] = dict(windows=len(durations), mean_ms=float(durations.mean()),
            median_ms=float(np.median(durations)), p95_ms=float(np.quantile(durations, .95)),
            max_ms=float(durations.max()))
    return result


def run(phase, catalog_folder, reservation, protocol_path, cache, output, development=None):
    plan = audit_protocol(protocol_path, reservation)
    audit_catalog(catalog_folder, cache)
    catalog = json.loads((catalog_folder/"catalog.json").read_text())
    if (catalog["schema"] != "chuff.recorded.esc10-catalog.v2"
            or catalog["partitions"] != [phase]
            or catalog["reservation_sha256"] != plan["reservation_sha256"]):
        raise ValueError("a matching reservation and phase-only catalog are required")
    names = sorted(plan["candidates"])
    input_files = {"protocol.json": protocol_path, "catalog.json": catalog_folder/"catalog.json",
                   "LICENSE": catalog_folder/"upstream/LICENSE",
                   "reservation.json": reservation/"reservation.json"}
    if phase == "evaluation":
        if development is None:
            raise ValueError("evaluation requires a complete frozen development archive")
        dev_manifest = json.loads((development/"manifest.json").read_text())
        if dev_manifest["input_sha256"]["input/protocol.json"] != digest(protocol_path.read_bytes()):
            raise ValueError("development used a different protocol")
        for name, expected in dev_manifest["output_sha256"].items():
            if digest((development/name).read_bytes()) != expected:
                raise ValueError("development output checksum mismatch")
        chosen = json.loads((development/"selection.json").read_text())
        if chosen != select(json.loads((development/"summary.json").read_text()), plan):
            raise ValueError("development selection differs from the declared rule")
        names = chosen["evaluation_ids"]
        input_files.update({"development-selection.json": development/"selection.json",
                            "development-summary.json": development/"summary.json"})
    clips = catalog["clips"]
    audio = load_audio(clips, cache)
    by_category = {row["category"]: row["filename"] for row in clips}
    targets = tuple(by_category[name] for name in FOREGROUNDS)
    backgrounds = tuple(by_category[name] for name in BACKGROUNDS)
    seeds = plan[phase+"_seeds"]
    suites = plan["positive_suites"]+plan["control_suites"]
    settings = {name: asdict(model.settings) for name, model in actors(plan, names).items()}
    # Warm library/JIT state on disposable actors, before any measured trajectory.
    noise = np.random.default_rng(2799).normal(0, .01, 16000)
    for model in actors(plan, names).values():
        for _ in range(4):
            model.observe(noise)
    output.mkdir(parents=True, exist_ok=False)
    args = SimpleNamespace(phase=phase, seed_start=seeds[0], seeds=len(seeds), rounds=42,
                           snr_db=[0., -6.], suites=suites)
    save_manifest(output, args, BandAttentionConfig())
    manifest = json.loads((output/"manifest.json").read_text())
    for key in ("original", "selected_band", "variants", "scene"):
        manifest.pop(key, None)
    package = Path(__file__).resolve().parents[1]
    root = package.parents[2]
    for source in package.rglob("*.py"):
        name = source.relative_to(root).as_posix()
        content = source.read_bytes()
        destination = output/"source"/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
        manifest["source_sha256"][name] = digest(content)
    manifest.update(schema="chuff.recorded.published-experiment.v1", partition=phase,
        settings=settings, foregrounds=list(targets), backgrounds=list(backgrounds),
        source_processing="Full 5s, resample_poly to16k,10ms edge fades; no energy/event cropping",
        per_recording_metrics=True, strict_episode_exclusion=True,
        input_sha256={}, output_sha256={}, examples={}, seconds_per_stream=210,
        processed_audio_sha256={name: digest(signal.astype("<f8").tobytes()) for name, signal in audio.items()},
        runtime_hardware=dict(system=platform.platform(), processor=platform.processor(),
            machine=platform.machine(), logical_cpus=os.cpu_count(),
            thread_environment={name: os.environ.get(name) for name in
                                ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}),
        runtime_scope=plan["runtime"], uncertainty="Seed bootstrap conditional on fixed recordings; development is selection data",
        controls="One no-target stream per condition/seed; no duplicate nominal-SNR controls",
        data_license="Upstream attribution and ESC-10 license apply to inputs and derived examples; input/LICENSE")
    for name, source in input_files.items():
        content = source.read_bytes()
        destination = output/"input"/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
        manifest["input_sha256"]["input/"+name] = digest(content)
    summaries = []
    with gzip.open(output/"observations.jsonl.gz", "wt", encoding="utf-8") as stream:
        for suite in suites:
            condition, snr, kind = suite.split("/")
            for seed in seeds:
                scene = RecordedScene(seed, SceneConfig(**plan["scene"], snr_db=float(snr)),
                    environment_seed=200000+seed, audio=audio, foregrounds=targets,
                    backgrounds=backgrounds, modulation=condition == "modulated", control=kind == "control")
                example = None
                if seed == seeds[0] and kind == "positive":
                    example = output/"examples"/(suite.replace("/", "-")+".npz")
                rows, physical = evaluate(scene, actors(plan, names), example=example, timing=True)
                for row in rows:
                    row.update(seed=seed, suite=suite)
                    stream.write(json.dumps(row, allow_nan=False)+"\n")
                summaries.append(dict(seed=seed, suite=suite, scene=asdict(scene.settings),
                    environment=scene.parameters, episodes=episode_metrics(rows, details=True, exclusion=True),
                    runtime=runtime(rows, settings), **physical))
                if example is not None:
                    manifest["examples"][example.relative_to(output).as_posix()] = digest(example.read_bytes())
                print(f"{phase}: {suite} seed={seed} actors={len(settings)}", flush=True)
    outputs = dict(summary=summaries, analysis=aggregate(summaries))
    if phase == "development":
        outputs["selection"] = select(summaries, plan)
    for name, content in outputs.items():
        (output/(name+".json")).write_text(json.dumps(content, indent=2, allow_nan=False), encoding="utf-8")
    manifest["output_sha256"] = {name: digest((output/name).read_bytes())
        for name in ("observations.jsonl.gz", *(key+".json" for key in outputs))}
    (output/"manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("development", "evaluation"), required=True)
    for name in ("catalog", "reservation", "protocol", "cache", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--development", type=Path)
    args = parser.parse_args()
    run(args.phase, args.catalog, args.reservation, args.protocol, args.cache, args.output, args.development)


if __name__ == "__main__":
    main()
