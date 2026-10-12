# Eval harness

A small harness for scoring a coding agent's attempt at a Chuff task:
a written **prompt** telling the agent what to build, and a **grader**
that turns its attempt into a numerical score with a per-check
breakdown — no human judgement needed to tell a good attempt from a bad
one.

## Philosophy

One deeply-specified, realistic task beats a pile of shallow synthetic
ones. `tasks/sound_memory_and_voice/` is deliberately a single task that
spans two real files or real behaviour (`memory.py`'s novelty detection,
`comms.py`'s call reinforcement) rather than ten trivial ones — closer to
"implement this feature correctly" than "fill in this one-line stub".
Add new tasks in that spirit: pick a piece of Chuff's actual spec that's
non-trivial to get right, write a prompt for it, and write a grader that
checks the *behaviour* the spec cares about (here: correctly telling
repeated sounds from new ones, and a believable reinforcement/decay
response) rather than just "does it run".

## Layout

```
eval/
  tasks/
    sound_memory_and_voice/
      prompt.md             task description, handed to the agent
      files.json             which chuff_mpu files this task covers
      stub/                   frozen starting point (what the agent edits) — see note below
      grader.py              scores an attempt, 0.0-1.0, with a breakdown
      fixtures.py             deterministic synthetic test data
      reference_solution/    known-good implementation (validates the grader itself)
  harness/
    common.py                shared plumbing (load a task, run the grader)
    run_task.py               run one agent on one task, once
    loop.py                   automl-style loop: attempt -> grade -> feed score back -> retry
    agents/
      base.py                 the Agent interface
      null_agent.py            does nothing (baseline, always ~0)
      reference_agent.py       returns the known-good solution (sanity check, ~1.0)
      gemini_agent.py          real model wiring (optional dependency, needs an API key)
  results/                    loop/run output lands here, per task per run (gitignored)
```

## Running it

From the repo root, using the `python/` uv project's environment (it
already has everything `grader.py` needs: just `numpy`):

```
cd python
uv run python ../eval/harness/run_task.py --task sound_memory_and_voice --agent null       # baseline, ~0.0
uv run python ../eval/harness/run_task.py --task sound_memory_and_voice --agent reference   # sanity check, ~1.0
```

Or run the grader directly against any directory containing `memory.py`
/ `comms.py` (e.g. your own in-progress files, no harness involved):

```
uv run python ../eval/tasks/sound_memory_and_voice/grader.py --candidate-dir path/to/dir
```

### The automl-style loop

```
uv run python ../eval/harness/loop.py --task sound_memory_and_voice --agent null --max-attempts 3
```

Each attempt's score and full grader report get appended to
`eval/results/<task>/<run>/history.jsonl`; the best attempt's files and
report are saved alongside it. With `--agent null` or `--agent
reference` this only demonstrates the mechanics (neither one actually
learns across attempts). To get a real improving loop, wire up a model:

```
cd python
uv add google-genai
export GEMINI_API_KEY=...        # or GOOGLE_API_KEY
uv run python ../eval/harness/loop.py --task sound_memory_and_voice --agent gemini --max-attempts 5 --target-score 0.95
```

Each failed attempt's grader report (which checks failed, on which
inputs) is fed back into the next prompt, so the loop is closer to
iterative self-correction than blind resampling.

## Adding a new task

1. `mkdir eval/tasks/<name>`, write `prompt.md` describing the real
   behaviour you want implemented (link to the relevant spec section).
2. `files.json`: `{"files": ["some_module.py"]}` — the chuff_mpu files
   the agent is allowed to touch.
3. `grader.py`: load the candidate's file(s) with `importlib`, run a set
   of isolated checks (one try/except each — a broken file should fail
   checks, not crash the grader), print a JSON report with an
   `overall_score` in [0, 1].
4. `fixtures.py` if you need deterministic synthetic inputs.
5. `reference_solution/`: a correct implementation, used only to prove
   the grader actually gives it a high score (run `run_task.py --agent
   reference` and confirm ~1.0) — this is the step that catches a grader
   that's secretly impossible to satisfy, or secretly satisfiable by a
   trivial wrong answer.

**On `stub/` vs. the live `python/src/chuff_mpu/` package:** `harness/
common.py:load_task()` reads an agent's starting files from the task's
own `stub/` directory, not from the live package. That split matters the
moment anyone actually finishes implementing the real feature (as
happened here — `memory.py`/`comms.py` are no longer stubs in
`python/`, see `TASKS.md` CH-32) — without it, every future attempt at
the task would start from the already-solved answer. Keep `stub/` frozen
at whatever the files looked like when the task was authored, even as
the real package moves on.
