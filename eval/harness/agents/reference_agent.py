"""Known-good agent: returns the task's reference solution verbatim.

Used to validate the harness end-to-end (should score ~1.0 through
run_task.py/loop.py, the same way grader.py alone already does when
pointed straight at tasks/<task>/reference_solution/).
"""

from __future__ import annotations

from pathlib import Path

TASK_DIR = Path(__file__).resolve().parents[2] / "tasks" / "sound_memory_and_voice"


class ReferenceAgent:
    def solve(
        self,
        prompt: str,
        starting_files: dict[str, str],
        feedback: dict | None = None,
    ) -> dict[str, str]:
        out: dict[str, str] = {}
        for name in starting_files:
            ref_path = TASK_DIR / "reference_solution" / name
            out[name] = ref_path.read_text() if ref_path.exists() else starting_files[name]
        return out
