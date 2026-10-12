# Chuff

Chuff is a robotic pet designed to learn from its acoustic environment,
express emotional states through light and motion, and — in a second
phase — develop a primitive shared language with a second unit through
sound-only communication.

The full concept, hardware BOM, software architecture, and emotional
("Culture colour") grammar are written up in
`Chuff — Agent Spec & Implementation Plan.md`. This README covers what's
actually in this repo and how the pieces fit together; see `TASKS.md`
for the current backlog.

## Two hardware tracks

| | Track A — ESP32-S3 | Track B — Arduino UNO Q |
| --- | --- | --- |
| Status | Board in hand | Board + Modulino kit on order |
| Folder | `esp32/` | `arduino/` (MCU) + `python/` (MPU) |
| Language | C (ESP-IDF) | C++ (Arduino) + Python 3 |
| Display | 1.8" AMOLED (LVGL) | Built-in 8x13 LED matrix + Modulino Pixels |

### `esp32/` — deviates from the original spec, deliberately

The Chuff spec's Software Architecture section specifies MicroPython +
emlearn-micropython for Track A. This repo's `esp32/` is instead a direct
copy of a working ESP-IDF (C) project — a touch-reactive animated face
on the same board (LVGL, ES8311 audio, FT3168 touch) — that already had
display, touch, and sound working on this exact hardware. Starting from
working C/LVGL code rather than a from-scratch MicroPython rewrite was a
deliberate call to get a real MVP in hand quickly; the audio-capture ->
MFCC -> KNN pipeline the spec describes for this track is not
implemented yet (see TASKS.md). `pet_config.h` / `pet_face.c` already
add one small real piece of mood behaviour on top of the inherited code:
Chuff slows down and stops blinking after 5 minutes of no touch
("dreaming"), and wakes immediately on touch.

### `arduino/` + `python/` — MVP proven end-to-end without the real board

The UNO Q and Modulino kit haven't arrived yet, so `arduino/chuff_mcu/`
is a minimal, dependency-free sketch that runs on *any* Arduino: it
reads `{"mood": ..., "energy": ...}` JSON lines over serial and blinks
`LED_BUILTIN` at a rate set by `energy`, standing in for the real LED
matrix / Modulino Pixels output until the hardware lands.
`python/src/chuff_mpu/mvp_demo.py` (installed as `chuff-mvp-demo`) is the
other end: it reads mic loudness (RMS) from this computer's default
input device and sends mood/energy lines to that sketch over serial —
so the mood-over-serial protocol (`shared/comms_protocol.md`'s MPU<->MCU
IPC, not the two-unit acoustic protocol) is demonstrable today with just
a laptop mic and a $5 Arduino, ahead of the real audio/ML pipeline.

## Repo layout

```
esp32/              Track A — ESP-IDF C project (copied from esp32_digital_pet)
arduino/
  chuff_mcu/        Track B, MCU side — Arduino sketch (LED matrix / Pixels, IPC)
python/             Track B, MPU side — uv-managed Python package (chuff_mpu)
  src/chuff_mpu/    config, audio, memory, dream, mood, comms, ipc + mvp_demo
shared/             Design docs shared by both tracks (mood grammar, MFCC notes, comms protocol,
                    research_backlog.md — the open ethology/cognition/comms questions behind them)
body/               3D-printed body (placeholder until design is finalised)
eval/               Prompt + grader harness for scoring a coding agent's attempt
                    at implementing a Chuff module (see eval/README.md)
TASKS.md            Backlog (plain markdown for now — see note below)
```

## Working on the Python side

Python tooling uses [uv](https://docs.astral.sh/uv/) exclusively — no
bare `pip`/`venv`.

```
cd python
uv sync                       # create .venv, install locked deps
uv run chuff-mvp-demo --port COM5   # (or /dev/ttyACM0) run the MVP demo
uv run pytest                 # once tests exist
uv add <package>              # add a new dependency
```

## Task tracking

Tracked in [beads](https://github.com/steveyegge/beads) (`bd`) — run
`bd ready` for actionable work, `bd list` for everything. See
`TASKS.md` for a plain-text snapshot (no `bd` required to read it) and
where the live data actually lives.
