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

## Running the MVP demo

Mic loudness -> mood -> serial JSON line to `arduino/chuff_mcu/chuff_mcu.ino`
(any Arduino, no UNO Q required yet):

```
uv run chuff-mvp-demo --port COM5    # or /dev/ttyACM0, etc.
```

## Package layout (`src/chuff_mpu/`)

- `config.py` — all tunable thresholds (fully implemented).
- `audio.py`, `memory.py`, `dream.py`, `mood.py`, `comms.py` — stubs
  matching the Chuff spec's module layout; see `TASKS.md` for what's
  implemented vs. still to do, and `../eval/` for a graded task covering
  `memory.py` and `comms.py`.
- `ipc.py` — serial JSON bridge to the MCU side (fully implemented).
- `mvp_demo.py` — the runnable MVP described above.
