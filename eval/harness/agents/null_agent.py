"""Baseline agent: never changes anything.

Always scores ~0 (the unmodified stub raises NotImplementedError
everywhere). Used to smoke-test the harness and loop without needing any
model, API key, or network access.
"""

from __future__ import annotations


class NullAgent:
    def solve(
        self,
        prompt: str,
        starting_files: dict[str, str],
        feedback: dict | None = None,
    ) -> dict[str, str]:
        return dict(starting_files)
