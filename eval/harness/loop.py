#!/usr/bin/env python3
"""Iterative improve-and-grade loop for a single agent on a single task.

This is the "automl-style" loop: the agent attempts the task, the grader
scores it, the score + failing-check detail is fed back to the agent as
feedback for the next attempt, and the best-scoring attempt is kept. Runs
until `--target-score` is reached or `--max-attempts` is used up.

With `--agent null` or `--agent reference` this just demonstrates the
mechanics (null never improves since it never changes anything;
reference hits the target on attempt 1) without needing a real model.
Point `--agent gemini` at it (after `uv add google-genai` and setting
GEMINI_API_KEY) to actually improve a model's attempts round over round.

Usage:
    uv run python eval/harness/loop.py --task sound_memory_and_voice \\
        --agent gemini --max-attempts 5 --target-score 0.95
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import EVAL_DIR, get_agent, load_task, run_grader  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="sound_memory_and_voice")
    parser.add_argument("--agent", default="null", choices=["null", "reference", "gemini"])
    parser.add_argument("--max-attempts", type=int, default=5)
    parser.add_argument("--target-score", type=float, default=0.95)
    args = parser.parse_args()

    prompt, starting_files, task_dir = load_task(args.task)
    agent = get_agent(args.agent)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = EVAL_DIR / "results" / args.task / f"{args.agent}_loop_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    history_path = out_dir / "history.jsonl"

    current_files = starting_files
    feedback: dict | None = None
    best_score = -1.0
    best_files = current_files
    best_report: dict = {}

    for attempt in range(1, args.max_attempts + 1):
        attempt_files = agent.solve(prompt, current_files, feedback)
        report = run_grader(task_dir, attempt_files)
        score = report.get("overall_score", 0.0)

        print(f"attempt {attempt}: overall_score={score:.4f}")
        with history_path.open("a") as f:
            f.write(json.dumps({"attempt": attempt, "score": score, "report": report}) + "\n")

        if score > best_score:
            best_score, best_files, best_report = score, attempt_files, report

        current_files = attempt_files
        feedback = report

        if score >= args.target_score:
            print(f"Reached target score {args.target_score} on attempt {attempt}.")
            break

    for name, content in best_files.items():
        (out_dir / name).write_text(content)
    (out_dir / "best_report.json").write_text(json.dumps(best_report, indent=2))

    print(f"\nBest overall_score={best_score:.4f}. Saved to {out_dir}")


if __name__ == "__main__":
    main()
