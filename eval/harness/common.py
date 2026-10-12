"""Shared plumbing for run_task.py and loop.py.

Deliberately subprocess-isolates the grader (see run_grader()) rather
than importing it in-process: a candidate's files get loaded fresh every
time with no import-cache leakage between attempts in a loop, and a
candidate that hangs or segfaults can't take the harness down with it.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

HARNESS_DIR = Path(__file__).resolve().parent
EVAL_DIR = HARNESS_DIR.parent
REPO_ROOT = EVAL_DIR.parent
TASKS_DIR = EVAL_DIR / "tasks"

sys.path.insert(0, str(HARNESS_DIR))
from agents.base import Agent  # noqa: E402
from agents.null_agent import NullAgent  # noqa: E402
from agents.reference_agent import ReferenceAgent  # noqa: E402

CHUFF_MPU_SRC = REPO_ROOT / "python" / "src" / "chuff_mpu"


def get_agent(name: str) -> Agent:
    if name == "null":
        return NullAgent()
    if name == "reference":
        return ReferenceAgent()
    if name == "gemini":
        from agents.gemini_agent import GeminiAgent  # deferred: optional dependency

        return GeminiAgent()
    raise ValueError(f"unknown agent: {name!r} (expected null, reference, or gemini)")


def load_task(task_name: str) -> tuple[str, dict[str, str], Path]:
    """Return (prompt text, {filename: starting contents}, task_dir)."""
    task_dir = TASKS_DIR / task_name
    prompt = (task_dir / "prompt.md").read_text()

    manifest_path = task_dir / "files.json"
    filenames = json.loads(manifest_path.read_text())["files"]

    starting_files = {name: (CHUFF_MPU_SRC / name).read_text() for name in filenames}
    return prompt, starting_files, task_dir


def run_grader(task_dir: Path, files: dict[str, str]) -> dict:
    """Write `files` to a temp dir and run tasks/<task>/grader.py against it."""
    with tempfile.TemporaryDirectory(prefix="chuff_eval_") as tmp:
        candidate_dir = Path(tmp)
        for name, content in files.items():
            (candidate_dir / name).write_text(content)

        proc = subprocess.run(
            [sys.executable, str(task_dir / "grader.py"), "--candidate-dir", str(candidate_dir)],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            return {
                "task": task_dir.name,
                "overall_score": 0.0,
                "dimensions": [],
                "grader_error": proc.stderr.strip()[-4000:],
            }
        return json.loads(proc.stdout)
