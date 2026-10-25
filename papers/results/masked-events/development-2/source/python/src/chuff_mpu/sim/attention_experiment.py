"""Paired, mixture-only foreground discovery versus legacy whole-clip memory.

Run `chuff-sim-attention --help`. Labels are evaluator-only, and every mode
receives identical audio. Saved manifests and JSONL support independent audits.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import wave

import numpy as np

from chuff_mpu import audio
from chuff_mpu.attention import AttentionConfig, ForegroundLearner, Observation
from chuff_mpu.sim.scene import AcousticScene, SceneConfig


class WholeClipLearner:
    """Legacy MFCC/cosine admission rule at the same committed-memory budget.

    Vectorised lookup is equivalent to the production 1-NN rule, and keeps
    stable IDs available to the evaluator without inspecting private stores.
    """

    def __init__(self, sample_rate: int, capacity: int) -> None:
        self.sample_rate = sample_rate
        self.capacity = capacity
        self._items: list[np.ndarray] = []
        self._ids: list[int] = []
        self._next_id = 0

    @property
    def memory_size(self) -> int:
        return len(self._items)

    def observe(self, clip: np.ndarray) -> Observation:
        from chuff_mpu import config

        embedding = audio.extract_mfcc(clip, self.sample_rate, config.N_MFCC)
        norm = max(float(np.linalg.norm(embedding)), 1e-12)
        embedding = embedding / norm
        distances = [float(np.clip(1 - np.dot(embedding, item), 0, 2)) for item in self._items]
        novelty = min(distances, default=1.0)
        if novelty >= config.NOVELTY_THRESHOLD_HIGH:
            identifier = self._next_id
            self._next_id += 1
            self._items.append(embedding)
            self._ids.append(identifier)
            if len(self._items) > self.capacity:
                self._items.pop(0)
                self._ids.pop(0)
            return Observation("learned", novelty=novelty, prototype_id=identifier)
        if distances and novelty < config.RECOGNITION_THRESHOLD:
            return Observation("recognised", novelty=novelty, prototype_id=self._ids[int(np.argmin(distances))])
        return Observation("uncertain", novelty=novelty)


def evaluate(seed: int, scene_config: SceneConfig, attention_config: AttentionConfig,
             *, include_baseline: bool = True, learners: dict | None = None,
             frames=None) -> tuple[dict, list[dict], np.ndarray]:
    if scene_config.sample_rate != attention_config.sample_rate:
        raise ValueError("scene and learner sample rates must match")
    if learners is None:
        learners = {"foreground": ForegroundLearner(attention_config),
                    "foreground_single_sighting": ForegroundLearner(
                        AttentionConfig(**{**asdict(attention_config), "confirmations": 1}))}
        if include_baseline:
            learners["whole_clip_mfcc"] = WholeClipLearner(scene_config.sample_rate, attention_config.capacity)
    assigned = {mode: {} for mode in learners}
    learned_labels = {mode: set() for mode in learners}
    counters = {mode: Counter() for mode in learners}
    novelty = {mode: {"first": [], "repeat": []} for mode in learners}
    occurrences: Counter = Counter()
    rows = []
    clips = []
    for step, frame in enumerate(AcousticScene(seed, scene_config) if frames is None else frames):
        clips.append(frame.audio)
        repeat = frame.label is not None and occurrences[frame.label] >= attention_config.confirmations
        for mode, learner in learners.items():
            # The complete learner interface: no label, stem, event mask or SNR.
            observation = learner.observe(frame.audio)
            count = counters[mode]
            count["windows"] += 1
            count["event_windows"] += frame.label is not None
            count["background_windows"] += frame.label is None
            count["repeat_windows"] += repeat
            count[observation.status] += 1
            if frame.label is not None:
                count["detected_events"] += observation.status in ("candidate", "learned", "recognised")
                if observation.novelty is not None:
                    novelty[mode]["repeat" if repeat else "first"].append(observation.novelty)
            elif observation.status in ("candidate", "learned", "recognised"):
                count["background_attention"] += 1
            correct = (observation.status == "recognised" and frame.label is not None
                       and assigned[mode].get(observation.prototype_id) == frame.label)
            count["correct_repeat_recognition"] += bool(repeat and correct)
            if observation.status == "recognised" and not correct:
                count["wrong_recognition"] += 1
            if observation.status == "learned":
                assigned[mode][observation.prototype_id] = frame.label
                if frame.label is None:
                    count["false_admissions"] += 1
                else:
                    learned_labels[mode].add(frame.label)
                    count["event_admissions"] += 1
            rows.append({"seed": seed, "step": step, "mode": mode,
                         "truth": frame.label, "nuisance": frame.nuisance,
                         "background_changed": frame.background_changed,
                         "active_snr_db_before_channel": frame.active_snr_db,
                         "status": observation.status, "novelty": observation.novelty,
                         "salience": observation.salience, "prototype_id": observation.prototype_id,
                         "memory_size": learner.memory_size})
        if frame.label is not None:
            occurrences[frame.label] += 1
    metrics = {}
    for mode, count in counters.items():
        rate = lambda numerator, denominator: count[numerator] / count[denominator] if count[denominator] else None
        metrics[mode] = {
            "counts": dict(count), "learned_labels": sorted(learned_labels[mode]),
            "observed_labels": sorted(occurrences),
            "event_detection_rate": rate("detected_events", "event_windows"),
            "repeat_recognition_rate": rate("correct_repeat_recognition", "repeat_windows"),
            "background_attention_rate": rate("background_attention", "background_windows"),
            "false_admissions": count["false_admissions"], "final_memory_size": learners[mode].memory_size,
            "novelty_mean": {group: float(np.mean(values)) if values else None
                             for group, values in novelty[mode].items()},
        }
    return {"seed": seed, "snr_db": scene_config.snr_db,
            "structured_distractors": scene_config.structured_distractors,
            "metrics": metrics}, rows, np.concatenate(clips)


def _source_manifest() -> dict:
    package = Path(__file__).resolve().parents[1]
    root = package.parents[2]
    paths = [package / "attention.py", package / "audio.py", package / "config.py",
             Path(__file__).resolve(), Path(__file__).with_name("scene.py"), root / "python/uv.lock"]
    digests = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, None
    return {"revision": revision, "dirty": dirty, "sha256": digests,
            "dependencies": {name: version(name) for name in ("numpy", "scipy", "librosa")}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--rounds", type=int, default=120)
    parser.add_argument("--snr-db", type=float, nargs="+", default=[6.0])
    parser.add_argument("--event-probability", type=float, default=0.35)
    parser.add_argument("--structured-distractors", action="store_true",
                        help="Include one-off tonal sequences as well as impacts")
    parser.add_argument("--output", type=Path, default=None, help="New directory for manifest, summary and per-window JSONL")
    parser.add_argument("--save-audio", action="store_true", help="Save first mixture as PCM WAV (requires --output)")
    parser.add_argument("--no-baseline", action="store_true", help="Skip the slower legacy MFCC comparator")
    args = parser.parse_args()
    if args.seeds < 1 or args.seed_start < 0 or args.rounds <= 6:
        parser.error("positive seeds, nonnegative seed-start and >6 rounds are required")
    if args.save_audio and args.output is None:
        parser.error("--save-audio requires --output")
    settings = AttentionConfig()
    scenes = [SceneConfig(rounds=args.rounds, snr_db=snr, event_probability=args.event_probability,
                          structured_distractors=args.structured_distractors)
              for snr in args.snr_db]
    if args.output:
        args.output.mkdir(parents=True, exist_ok=False)
        manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "software": _source_manifest(),
                    "attention": asdict(settings), "scenes": [asdict(c) for c in scenes],
                    "seed_start": args.seed_start, "seeds": args.seeds,
                    "include_baseline": not args.no_baseline,
                    "snr_definition": "active source segment before common room/microphone channel",
                    "independence": "paired mixtures within seed; use seeds, not clips, for uncertainty"}
        (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    summaries = []
    for scene in scenes:
        for seed in range(args.seed_start, args.seed_start + args.seeds):
            summary, rows, mixture = evaluate(seed, scene, settings, include_baseline=not args.no_baseline)
            summaries.append(summary)
            print(json.dumps(summary, allow_nan=False), flush=True)
            if args.output:
                name = f"seed-{seed}-snr-{scene.snr_db:g}"
                (args.output / f"{name}.jsonl").write_text(
                    "\n".join(json.dumps(row, allow_nan=False) for row in rows) + "\n", encoding="utf-8")
                if args.save_audio and len(summaries) == 1:
                    # File-only preview, with headroom; does not play on hardware.
                    pcm = np.clip(mixture, -1, 1) * 0.8
                    with wave.open(str(args.output / "mixture.wav"), "wb") as handle:
                        handle.setnchannels(1)
                        handle.setsampwidth(2)
                        handle.setframerate(scene.sample_rate)
                        handle.writeframes((pcm * 32767).astype("<i2").tobytes())
    if args.output:
        (args.output / "summary.json").write_text(json.dumps(summaries, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    main()
