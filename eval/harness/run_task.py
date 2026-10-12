#!/usr/bin/env python3
"""Run one agent against one task, once, and print the grader's report.

Usage:
    uv run python eval/harness/run_task.py --task sound_memory_and_voice --agent null
    uv run python eval/harness/run_task.py --task sound_memory_and_voice --agent reference
    uv run python eval/harness/run_task.py --task sound_memory_and_voice --agent gemini

For iterating an agent across multiple attempts with feedback, see loop.py.
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
    parser.add_argument(
        "--save", action="store_true", help="Save attempt files + report under eval/results/"
    )
    args = parser.parse_args()

    prompt, starting_files, task_dir = load_task(args.task)
    agent = get_agent(args.agent)

    attempt_files = agent.solve(prompt, starting_files)
    report = run_grader(task_dir, attempt_files)

    print(json.dumps(report, indent=2))

    if args.save:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_dir = EVAL_DIR / "results" / args.task / f"{args.agent}_{ts}"
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, content in attempt_files.items():
            (out_dir / name).write_text(content)
        (out_dir / "report.json").write_text(json.dumps(report, indent=2))
        print(f"\nSaved to {out_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
