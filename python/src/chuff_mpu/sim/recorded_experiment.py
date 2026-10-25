"""Full-duration recorded replay: episode-aware identity and source-deletion controls."""

import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from chuff_mpu.attention import AttentionConfig, ForegroundLearner
from chuff_mpu.band_attention import BandAttentionConfig, BandForegroundLearner
from chuff_mpu.gain_attention import CalibratedGainForegroundLearner
from chuff_mpu.adaptive_gain import AdaptiveGainForegroundLearner
from chuff_mpu.guarded_gain import GuardedGainForegroundLearner
from chuff_mpu.sim.masked_experiment import save_manifest
from chuff_mpu.sim.recorded_catalog import audit as audit_catalog
from chuff_mpu.sim.recorded_scene import BACKGROUNDS, FOREGROUNDS, RecordedScene, load_audio
from chuff_mpu.sim.recorded_metrics import aggregate, episode_metrics
from chuff_mpu.sim.scene import SceneConfig


def learners():
    return dict(original=ForegroundLearner(), quiet_band=BandForegroundLearner(),
        calibrated_gain=CalibratedGainForegroundLearner(), adaptive_gain=AdaptiveGainForegroundLearner(),
        guarded_gain=GuardedGainForegroundLearner())


def rms(signal):
    return float(np.sqrt(np.mean(np.asarray(signal, dtype=np.float64)**2)))


def evaluate(scene, actors, *, probes=True, example=None):
    rows, episodes, occurrences = [], [], Counter()
    max_error = 0.
    background_digest = hashlib.sha256()
    example_saved = False
    c = scene.settings
    for chunk, frame in enumerate(scene):
        error = max(float(np.max(np.abs(frame.audio-frame.background_audio-frame.source_audio))),
            float(np.max(np.abs(frame.audio-frame.without_current_source_audio-frame.current_source_audio))))
        if error > 2e-6*max(float(np.max(np.abs(frame.audio))), 1e-6)+1e-8:
            raise ValueError("recorded component recombination failed")
        max_error = max(max_error, error)
        background_digest.update(frame.background_audio.astype("<f4").tobytes())
        episode = chunk if frame.label is not None else None
        if episode is not None:
            occurrences[frame.label] += 1
            if abs(frame.post_channel_residual_snr_db-c.snr_db) > 1e-8:
                raise ValueError("whole-episode post-channel target failed")
            episodes.append(dict(episode_id=episode, recording_id=frame.label,
                occurrence=occurrences[frame.label], start_step=5*chunk, stop_step=5*chunk+5,
                post_snr_db=frame.post_channel_residual_snr_db,
                source_rms=rms(frame.current_source_audio), residual_rms=rms(frame.without_current_source_audio)))
            if example is not None and not example_saved:
                example.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(example, mixture=frame.audio, background=frame.background_audio,
                    all_sources=frame.source_audio, current_episode=frame.current_source_audio,
                    without_episode=frame.without_current_source_audio,
                    metadata=np.array(json.dumps(episodes[-1])))
                example_saved = True
        for offset in range(5):
            section = slice(offset*c.sample_rate, (offset+1)*c.sample_rate)
            audio = frame.audio[section]
            removed = frame.without_current_source_audio[section]
            numerator, denominator = rms(frame.current_source_audio[section]), rms(removed)
            window_snr = float(20*np.log10(numerator/denominator)) if min(numerator, denominator) > 1e-15 else None
            for mode, actor in actors.items():
                branch = deepcopy(actor).observe(removed) if episode is not None and probes else None
                observation = actor.observe(audio)
                row = dict(step=chunk*5+offset, chunk=chunk, mode=mode, episode_id=episode,
                    recording_id=frame.label, occurrence=occurrences[frame.label] if episode is not None else None,
                    episode_offset=offset if episode is not None else None,
                    episode_post_snr_db=frame.post_channel_residual_snr_db,
                    window_post_snr_db=window_snr if episode is not None else None,
                    current_episode_window_rms=numerator, residual_window_rms=denominator,
                    previous_source_tail_s=max(0., frame.previous_source_tail_s-offset),
                    status=observation.status, prototype_id=observation.prototype_id,
                    novelty=observation.novelty, salience=observation.salience, memory_size=actor.memory_size,
                    region_start_s=observation.region_start_s, region_stop_s=observation.region_stop_s)
                if branch is not None:
                    row["without_episode_probe"] = dict(status=branch.status, prototype_id=branch.prototype_id)
                rows.append(row)
    return rows, dict(physical_episodes=episodes, max_component_error=max_error,
                     background_output_sha256=background_digest.hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("development", "evaluation"), required=True)
    parser.add_argument("--partition", choices=("development", "evaluation"), required=True)
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.seed_start < 0 or args.seeds < 1 or args.phase != args.partition:
        parser.error("positive seed count, nonnegative seed-start and matching phase/partition required")
    audit_catalog(args.catalog, args.cache)
    catalog = json.loads((args.catalog/"catalog.json").read_text())
    clips = [r for r in catalog["clips"] if r["split"] == args.partition]
    audio = load_audio(clips, args.cache)
    by_category = {r["category"]: r["filename"] for r in clips}
    targets, backgrounds = tuple(by_category[name] for name in FOREGROUNDS), tuple(by_category[name] for name in BACKGROUNDS)
    args.rounds, args.snr_db = 42, [0., -6.]
    args.suites = [f"{condition}/{snr:g}/{kind}" for condition in ("recorded", "modulated")
        for snr in args.snr_db for kind in ("positive", "control")]
    args.output.mkdir(parents=True, exist_ok=False)
    save_manifest(args.output, args, BandAttentionConfig())
    path = args.output/"manifest.json"
    manifest = json.loads(path.read_text())
    for unused in ("original", "selected_band", "variants", "scene"):
        manifest.pop(unused, None)
    package = Path(__file__).resolve().parents[1]
    root = package.parents[2]
    for source in package.rglob("*.py"):
        name = source.relative_to(root).as_posix()
        content = source.read_bytes()
        destination = args.output/"source"/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
        manifest["source_sha256"][name] = hashlib.sha256(content).hexdigest()
    manifest.update(partition=args.partition, window_seconds=1, episode_seconds=5, seconds_per_stream=210,
        foregrounds=list(targets), backgrounds=list(backgrounds),
        settings={mode: asdict(actor.settings) for mode, actor in learners().items()},
        primary="Episode repeat identity requiring a pre-episode admitted prototype, minus same-window source-removed recognition",
        repeat_eligibility="Third/fourth replay of each recording; five adjacent windows remain one episode",
        scope="Recording injection and replay identity, not natural onset localization or semantic category recognition",
        interventions="Remove the entire current episode channel contribution in each window; copy actual prior learner state for each probe; discard branch",
        source_processing="Resample_poly44.1->16kHz full5s;10ms edge fades; no event/energy cropping",
        data_license="Recorded input/derived examples retain upstream attribution; see input/catalog.json and input/LICENSE",
        per_recording_metrics=True,
        input_sha256={}, examples={})
    manifest["processed_audio_sha256"] = {name: hashlib.sha256(signal.astype("<f8").tobytes()).hexdigest()
        for name, signal in audio.items()}
    for source, name in ((args.catalog/"catalog.json", "catalog.json"), (args.catalog/"upstream/LICENSE", "LICENSE")):
        content = source.read_bytes()
        destination = args.output/"input"/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
        manifest["input_sha256"]["input/"+name] = hashlib.sha256(content).hexdigest()
    summaries = []
    with gzip.open(args.output/"observations.jsonl.gz", "wt", encoding="utf-8") as stream:
        for condition in ("recorded", "modulated"):
            for snr in args.snr_db:
                for kind in ("positive", "control"):
                    suite = f"{condition}/{snr:g}/{kind}"
                    for seed in range(args.seed_start, args.seed_start+args.seeds):
                        settings = SceneConfig(duration_s=5, rounds=42, warmup=2, nuisance_probability=0,
                            background_change=False, collect_stems=True, snr_reference="post_residual", snr_db=snr)
                        scene = RecordedScene(seed, settings, environment_seed=200000+seed, audio=audio,
                            foregrounds=targets, backgrounds=backgrounds,
                            modulation=condition == "modulated", control=kind == "control")
                        example = None
                        if seed == args.seed_start and kind == "positive":
                            example = args.output/"examples"/(suite.replace("/", "-")+".npz")
                        rows, physical = evaluate(scene, learners(), example=example)
                        for row in rows:
                            row.update(seed=seed, suite=suite)
                            stream.write(json.dumps(row, allow_nan=False)+"\n")
                        summary = dict(seed=seed, suite=suite, scene=asdict(scene.settings),
                            environment=scene.parameters, episodes=episode_metrics(rows, details=True), **physical)
                        summaries.append(summary)
                        if example is not None:
                            manifest["examples"][example.relative_to(args.output).as_posix()] = hashlib.sha256(example.read_bytes()).hexdigest()
                        print(f"{suite} seed={seed}", flush=True)
    (args.output/"summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    (args.output/"analysis.json").write_text(json.dumps(aggregate(summaries), indent=2), encoding="utf-8")
    manifest["output_sha256"] = {name: hashlib.sha256((args.output/name).read_bytes()).hexdigest()
        for name in ("observations.jsonl.gz", "summary.json", "analysis.json")}
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
