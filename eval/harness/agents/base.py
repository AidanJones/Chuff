"""Agent interface the harness drives.

An "agent" here is anything that can read a task prompt plus the current
file contents and produce new file contents to be graded. Swap in a real
model (see gemini_agent.py) without touching run_task.py or loop.py.
"""

from __future__ import annotations

from typing import Protocol


class Agent(Protocol):
    def solve(
        self,
        prompt: str,
        starting_files: dict[str, str],
        feedback: dict | None = None,
    ) -> dict[str, str]:
        """Produce new contents for the files named in `starting_files`.

        Args:
            prompt: The task prompt (see tasks/<task>/prompt.md).
            starting_files: filename -> current contents (e.g. the stub,
                or the previous attempt's files on a retry).
            feedback: The previous attempt's grader report (None on the
                first attempt), for agents that can use it to self-correct.

        Returns:
            filename -> new contents, for exactly the files it changed.
            Filenames not present are left as `starting_files` had them.
        """
        ...
