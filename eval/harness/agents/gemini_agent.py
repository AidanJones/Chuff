"""Real model agent, wired for Gemini.

Not exercised by default — the null/reference agents are enough to prove
the harness and grader work without any API key or cost. Flip this on
when you want an actual model attempting the task.

Setup:
    uv add google-genai          # (in python/, adds the SDK to the project)
    export GEMINI_API_KEY=...    # or GOOGLE_API_KEY

Usage: pass `--agent gemini` to harness/run_task.py or harness/loop.py.

The prompt asks the model to return each changed file as a fenced code
block whose opening fence is the filename, e.g.:

    ```memory.py
    ...full file contents...
    ```

which this module parses back out. This keeps the contract readable by a
human reviewing a transcript, and keeps the harness model-agnostic — any
other model wired up the same way (one prompt in, filename-fenced files
out) can reuse run_task.py/loop.py unchanged.
"""

from __future__ import annotations

import os
import re

_FENCE_RE = re.compile(r"```(?P<name>[\w./-]+)\n(?P<body>.*?)```", re.DOTALL)

DEFAULT_MODEL = "gemini-3-pro"


class GeminiAgent:
    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        self.model = model

    def solve(
        self,
        prompt: str,
        starting_files: dict[str, str],
        feedback: dict | None = None,
    ) -> dict[str, str]:
        try:
            from google import genai
        except ImportError as exc:
            raise RuntimeError(
                "google-genai is not installed. Run `uv add google-genai` "
                "in python/ to enable the Gemini agent."
            ) from exc

        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Set GEMINI_API_KEY (or GOOGLE_API_KEY) in the environment to use GeminiAgent."
            )

        full_prompt = self._build_prompt(prompt, starting_files, feedback)
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(model=self.model, contents=full_prompt)
        return self._parse_response(response.text, starting_files)

    def _build_prompt(self, prompt: str, starting_files: dict[str, str], feedback: dict | None) -> str:
        parts = [prompt, "\n\n## Current file contents\n"]
        for name, content in starting_files.items():
            parts.append(f"### {name}\n```{name}\n{content}\n```\n")
        if feedback is not None:
            parts.append(
                "\n## Feedback from the previous attempt (grader report)\n"
                f"```json\n{feedback}\n```\n"
                "Fix the failing checks above.\n"
            )
        parts.append(
            "\nRespond with the complete new contents of each file you change, "
            "each as its own fenced code block opened with the filename, e.g.:\n"
            "```memory.py\n...\n```\n"
            "Return every file listed above, even ones you don't change."
        )
        return "\n".join(parts)

    def _parse_response(self, text: str, starting_files: dict[str, str]) -> dict[str, str]:
        found = {m.group("name"): m.group("body") for m in _FENCE_RE.finditer(text)}
        return {name: found.get(name, starting_files[name]) for name in starting_files}
