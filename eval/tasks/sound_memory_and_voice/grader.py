#!/usr/bin/env python3
"""Grader for the sound_memory_and_voice task.

Scores a candidate's `memory.py` (EmbeddingStore) and `comms.py`
(Repertoire) against two rubric dimensions:

  1. sound_memory  — of the sounds Chuff hears, how many does it
     correctly recognise as repeats vs. flag as genuinely new?
  2. communication — does a single unit's call-to-context weighting
     behave the way the spec describes (stable choice per context,
     reinforcement sticks, decay causes it to try alternatives, one
     context's learning doesn't clobber another's)?

Usage:
    uv run python grader.py --candidate-dir path/to/dir [--out report.json]

`--candidate-dir` must contain `memory.py` and/or `comms.py` implementing
the classes declared in python/src/chuff_mpu/{memory,comms}.py. A missing
or broken file scores 0 on that dimension rather than crashing the run —
every check is isolated so one bug doesn't zero out the whole report.

Runs as a fresh subprocess per invocation (see harness/run_task.py), so
there's no import-caching to worry about between repeated grading runs
in a loop.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import traceback
from pathlib import Path
from types import ModuleType

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
TASK_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(TASK_DIR))
import fixtures  # noqa: E402  (needs TASK_DIR on sys.path first)


def _load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load module spec for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_config() -> ModuleType:
    return _load_module(REPO_ROOT / "python" / "src" / "chuff_mpu" / "config.py", "chuff_task_config")


class Check:
    """One pass/fail (or partial-credit) check, isolated from the rest."""

    def __init__(self, name: str, weight: float) -> None:
        self.name = name
        self.weight = weight
        self.score = 0.0  # 0.0-1.0
        self.error: str | None = None

    def run(self, fn) -> None:
        try:
            result = fn()
            self.score = 1.0 if result is True else float(result)
        except Exception:  # noqa: BLE001 - a broken candidate must not crash grading
            self.score = 0.0
            self.error = traceback.format_exc(limit=3)

    def to_dict(self) -> dict:
        d = {"name": self.name, "weight": self.weight, "score": self.score}
        if self.error:
            d["error"] = self.error
        return d


def grade_sound_memory(candidate_dir: Path, config: ModuleType) -> dict:
    checks: list[Check] = []
    mem_path = candidate_dir / "memory.py"

    recognition = Check("recognition_accuracy", weight=0.8)
    sample_pair_ok = Check("sample_pair_works", weight=0.1)
    capacity_ok = Check("respects_max_embeddings", weight=0.1)
    checks += [recognition, sample_pair_ok, capacity_ok]

    if not mem_path.exists():
        return _finish("sound_memory", checks, note="memory.py not found in candidate dir")

    try:
        memory_mod = _load_module(mem_path, "candidate_memory")
    except Exception:  # noqa: BLE001
        err = traceback.format_exc(limit=3)
        for c in checks:
            c.error = err
        return _finish("sound_memory", checks, note="memory.py failed to import")

    def _run_recognition() -> float:
        store = memory_mod.EmbeddingStore(max_embeddings=64)
        seen_classes: set[int] = set()
        correct = 0
        total = 0
        for emb, true_cls in fixtures.memory_test_sequence():
            score = store.novelty_score(emb)
            predicted_recognised = score < config.RECOGNITION_THRESHOLD
            expected_recognised = true_cls is not None and true_cls in seen_classes
            correct += int(predicted_recognised == expected_recognised)
            total += 1
            if true_cls is not None and true_cls not in seen_classes:
                store.add(emb)
                seen_classes.add(true_cls)
        return correct / total if total else 0.0

    def _run_sample_pair() -> bool:
        store = memory_mod.EmbeddingStore(max_embeddings=64)
        for emb, _ in fixtures.memory_test_sequence()[:4]:
            store.add(emb)
        a, b = store.sample_pair()
        return a is not None and b is not None and not np.array_equal(np.asarray(a), np.asarray(b))

    def _run_capacity() -> bool:
        store = memory_mod.EmbeddingStore(max_embeddings=3)
        rng = np.random.default_rng(0)
        for _ in range(5):
            store.add(rng.normal(size=fixtures.N_DIM).astype(np.float32))
        return len(store) <= 3

    recognition.run(_run_recognition)
    sample_pair_ok.run(_run_sample_pair)
    capacity_ok.run(_run_capacity)
    return _finish("sound_memory", checks)


def grade_communication(candidate_dir: Path) -> dict:
    checks: list[Check] = []
    repertoire_sane = Check("repertoire_in_range", weight=0.2)
    deterministic = Check("deterministic_without_change", weight=0.2)
    reinforcement_sticks = Check("reinforcement_sticks", weight=0.2)
    context_independent = Check("context_independent", weight=0.2)
    decay_causes_turnover = Check("decay_causes_turnover", weight=0.2)
    checks += [repertoire_sane, deterministic, reinforcement_sticks, context_independent, decay_causes_turnover]

    comms_path = candidate_dir / "comms.py"
    if not comms_path.exists():
        return _finish("communication", checks, note="comms.py not found in candidate dir")

    try:
        comms_mod = _load_module(comms_path, "candidate_comms")
    except Exception:  # noqa: BLE001
        err = traceback.format_exc(limit=3)
        for c in checks:
            c.error = err
        return _finish("communication", checks, note="comms.py failed to import")

    ctx1, ctx2, _rng = fixtures.comms_test_contexts()

    def _build_repertoire():
        return comms_mod.Repertoire(size=12, freq_min_hz=400, freq_max_hz=3000, seed=42)

    def _run_repertoire_sane() -> bool:
        r = _build_repertoire()
        freqs = np.asarray(r.frequencies)
        return (
            freqs.shape[0] == 12
            and bool(np.all(freqs >= 400))
            and bool(np.all(freqs <= 3000))
            and len(set(np.round(freqs, 1))) >= 10  # mostly distinct, allow tiny collisions
        )

    try:
        r = _build_repertoire()
    except Exception:  # noqa: BLE001 - constructor itself is broken/unimplemented
        err = traceback.format_exc(limit=3)
        for c in checks:
            c.error = err
        return _finish("communication", checks, note="Repertoire() constructor raised")

    repertoire_sane.run(_run_repertoire_sane)

    # The remaining checks share state (idx1/idx2_before computed once, then
    # built on by later checks) so a broken closest_call()/reinforce()/decay()
    # needs to fail all of them, not crash the whole grader.
    remaining = [deterministic, reinforcement_sticks, context_independent, decay_causes_turnover]
    try:
        idx1 = r.closest_call(ctx1)
        idx2_before = r.closest_call(ctx2)

        def _run_deterministic() -> bool:
            return r.closest_call(ctx1) == idx1

        def _run_reinforcement() -> bool:
            for _ in range(3):
                r.reinforce(idx1, ctx1, rate=0.3)
            return r.closest_call(ctx1) == idx1

        def _run_context_independent() -> bool:
            for _ in range(3):
                r.reinforce(idx1, ctx1, rate=0.3)  # more ctx1 reinforcement
            return r.closest_call(ctx2) == idx2_before

        def _run_decay_turnover() -> bool:
            for _ in range(8):
                r.decay(idx2_before, ctx2, rate=0.5)
            choices = {r.closest_call(ctx2) for _ in range(5)}
            return choices != {idx2_before}

        deterministic.run(_run_deterministic)
        reinforcement_sticks.run(_run_reinforcement)
        context_independent.run(_run_context_independent)
        decay_causes_turnover.run(_run_decay_turnover)
    except Exception:  # noqa: BLE001
        err = traceback.format_exc(limit=3)
        for c in remaining:
            if c.score == 0.0 and c.error is None:
                c.error = err

    return _finish("communication", checks)


def _finish(dimension: str, checks: list[Check], note: str | None = None) -> dict:
    total_weight = sum(c.weight for c in checks) or 1.0
    score = sum(c.weight * c.score for c in checks) / total_weight
    result = {"dimension": dimension, "score": round(score, 4), "checks": [c.to_dict() for c in checks]}
    if note:
        result["note"] = note
    return result


def grade(candidate_dir: Path) -> dict:
    config = _load_config()
    memory_report = grade_sound_memory(candidate_dir, config)
    comms_report = grade_communication(candidate_dir)
    overall = round(0.5 * memory_report["score"] + 0.5 * comms_report["score"], 4)
    return {
        "task": "sound_memory_and_voice",
        "overall_score": overall,
        "dimensions": [memory_report, comms_report],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    report = grade(args.candidate_dir)
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        args.out.write_text(text)


if __name__ == "__main__":
    main()
