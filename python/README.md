# chuff-mpu

Chuff's Arduino UNO Q MPU-side Python package (Track B). Runs as full
Python 3 on the UNO Q's Debian Linux side (or any dev machine, for the
MVP demo and for running the eval harness's grader).

See the top-level repo `README.md` and `TASKS.md` for how this fits into
the rest of Chuff, and `../shared/` for the design docs this package
implements against.

## Setup

```
uv sync
```

## Running the MVP demo (real hardware, partial pipeline)

Mic loudness -> mood -> serial JSON line to `arduino/chuff_mcu/chuff_mcu.ino`
(any Arduino, no UNO Q required yet):

```
uv run chuff-mvp-demo --port COM5    # or /dev/ttyACM0, etc.
```

## Running the in-silico test (no hardware, full pipeline)

Listen -> remember -> repeat end to end against synthetic audio, using
the real `audio.py`/`memory.py`/`comms.py` — no mic/speaker required:

```
uv run chuff-sim
```

And a two-unit version — two independent simulated Chuffs talking over
one noisy acoustic channel, no shared data structure:

```
uv run chuff-sim-2unit --condition social
uv run chuff-sim-2unit --condition passive   # control
```

See `src/chuff_mpu/sim/README.md` for what both do and don't prove, and
`../shared/research_backlog.md` for the research questions they
answer (social gating, repertoire convergence).

## Package layout (`src/chuff_mpu/`)

- `config.py` — all tunable thresholds (fully implemented), including
  the `SAFE_*` output-safety ceilings in `audio_safety.py`.
- `audio.py` — real MFCC extraction (librosa) + mic capture (sounddevice).
- `memory.py` — real cosine-distance KNN store (`EmbeddingStore`).
- `comms.py` — real call repertoire + reinforcement/decay (`Repertoire`),
  with `play_call()` always routed through `audio_safety`.
- `audio_safety.py` — the output guardrail everything above routes
  through before any sample reaches a real speaker.
- `dream.py`, `mood.py` — still stubs; see `TASKS.md`.
- `ipc.py` — serial JSON bridge to the MCU side (fully implemented).
- `mvp_demo.py` — the MVP demo described above.
- `sim/` — the in-silico test described above.

`memory.py` and `comms.py`'s designs were validated against
`../eval/tasks/sound_memory_and_voice/` before landing here — running
that task's grader against this package's own files scores 1.0.
