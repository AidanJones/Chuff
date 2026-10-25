"""Archived-versus-current silent-start regression, without model retuning."""

import argparse
from collections import Counter
import importlib.util
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np

from chuff_mpu.gain_attention import SpectralGainForegroundLearner
from chuff_mpu.sim.challenge_scene import ChallengeScene
from chuff_mpu.sim.scene import SceneConfig


def load_reference(folder):
    def module(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        result = importlib.util.module_from_spec(spec)
        sys.modules[name] = result
        spec.loader.exec_module(result)
        return result
    attention = module("startup_reference_attention", folder/"attention.py")
    with patch.dict(sys.modules, {"chuff_mpu.attention": attention}):
        band = module("startup_reference_band", folder/"band_attention.py")
    with patch.dict(sys.modules, {"chuff_mpu.band_attention": band}):
        gain = module("startup_reference_gain", folder/"gain_attention.py")
    return gain.SpectralGainForegroundLearner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-source", type=Path, required=True,
                        help="Archived package folder containing attention/band_attention/gain_attention.py")
    parser.add_argument("--seed-start", type=int, default=1800)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.seed_start < 0 or args.seeds < 1:
        parser.error("nonnegative seed-start and positive seeds required")
    if args.output.exists(): parser.error("output directory must be new")
    old_class = load_reference(args.reference_source)
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[4]
    hashes = {}
    for group, source, names in (
        ("reference", args.reference_source, ("attention.py", "band_attention.py", "gain_attention.py")),
        ("current", root/"python/src/chuff_mpu", ("attention.py", "band_attention.py", "gain_attention.py",
                                                  "sim/scene.py", "sim/challenge_scene.py", "sim/startup_recovery.py"))):
        for name in names:
            content = (source/name).read_bytes()
            target = args.output/"source"/group/name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            hashes[(Path(group)/name).as_posix()] = hashlib.sha256(content).hexdigest()
    results, observations = [], []
    for seed in range(args.seed_start, args.seed_start+args.seeds):
        models = {"old_clean": old_class(), "old_silent": old_class(),
                  "new_clean": SpectralGainForegroundLearner(), "new_silent": SpectralGainForegroundLearner()}
        for name, model in models.items():
            if name.endswith("silent"):
                for _ in range(3): model.observe(np.zeros(16000))
        counts = {name: Counter() for name in models}
        scene = ChallengeScene(seed, SceneConfig(rounds=120, snr_db=-6, nuisance_probability=0))
        for step, frame in enumerate(scene):
            outcomes = {name: model.observe(frame.audio) for name, model in models.items()}
            comparable = lambda o: (o.status, o.prototype_id, o.novelty, o.salience)
            assert comparable(outcomes["new_clean"]) == comparable(outcomes["old_clean"])
            assert comparable(outcomes["new_silent"]) == comparable(outcomes["new_clean"])
            for first, second in (("new_clean", "old_clean"), ("new_silent", "new_clean")):
                if outcomes[first].embedding is not None:
                    np.testing.assert_array_equal(outcomes[first].embedding, outcomes[second].embedding)
            for name, outcome in outcomes.items():
                counts[name]["events"] += frame.label is not None
                counts[name]["detected_events"] += frame.label is not None and outcome.status in ("candidate", "learned", "recognised")
                counts[name]["false_admissions"] += frame.label is None and outcome.status == "learned"
                observations.append(dict(seed=seed, step=step, mode=name, truth=frame.label,
                    status=outcome.status, prototype_id=outcome.prototype_id, novelty=outcome.novelty))
        results.append(dict(seed=seed, modes={name: dict(counts[name], final_memory_size=model.memory_size)
                                           for name, model in models.items()}))
        print(json.dumps(results[-1]), flush=True)
    manifest = dict(source_sha256=hashes, seed_start=args.seed_start, seeds=args.seeds,
        scenario="3 exact-zero prefix windows, followed by identical120-window ChallengeScene at input SNR-6dB",
        invariant="New silent-prefix trajectory equals new clean trajectory; new clean equals archived clean, including features",
        scope="Numerical regression/compatibility fixtures, not startup/field generalization or a new detector benchmark")
    (args.output/"manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (args.output/"summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (args.output/"observations.json").write_text(json.dumps(observations, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
