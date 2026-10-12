# Chuff Task Backlog

Plain-markdown stand-in for [beads](https://github.com/steveyegge/beads)
(`bd`) issues — `bd.exe` is currently blocked on this machine by Windows
**Smart App Control** (Settings -> Privacy & security -> Windows
Security -> App & browser control -> Smart App Control -> Off; this is a
one-way switch, no per-app exception). Each task below carries an ID,
status, and a one-line body so it can be bulk-imported into real `bd`
issues later with minimal rework.

Status: `todo` / `in-progress` / `done` / `blocked`

## Phase 0 — Scaffolding

- [x] **CH-1** `done` — Folder structure (`esp32/`, `arduino/`, `python/`,
  `shared/`, `body/`, `eval/`) and top-level README/LICENSE.
- [x] **CH-2** `done` — Copy `esp32_digital_pet`'s working ESP-IDF C project
  into `esp32/` as the Track A base (plain copy, no submodule).
- [x] **CH-3** `done` — `python/` initialised with `uv`, `chuff_mpu`
  package scaffolded (config fully populated; audio/memory/dream/mood/
  comms stubbed per the spec's module layout; `ipc.py` fully implemented).
- [x] **CH-4** `done` — `arduino/chuff_mcu/chuff_mcu.ino`: minimal
  dependency-free serial-JSON mood receiver, bench-testable on any
  Arduino ahead of the UNO Q arriving.
- [x] **CH-5** `done` — `shared/mood_grammar.md`, `comms_protocol.md`,
  `mfcc_notes.md` written up from the spec, with a "current status"
  section in each so they don't silently go stale.
- [x] **CH-6** `done` — `eval/` prompt+grader harness for
  `sound_memory_and_voice` (memory.py + comms.py), validated against a
  null baseline (~0.0) and a reference solution (~1.0).

## Phase 1 — ESP32-S3 single unit (Track A)

- [x] **CH-10** `done` — Minimal real mood behaviour: "dreaming" after 5
  minutes idle (slower breathing-pace animation, no blink; wakes on
  touch). `esp32/main/pet_config.h`, `pet_face.c`. **Unverified** — no
  ESP-IDF toolchain available in this environment to build/flash; verify
  with the ESP-IDF VS Code extension before trusting it.
- [ ] **CH-11** `todo` — I2S mic capture (ES8311 ADC side is wired but
  unused — see `esp32/README.md`'s I2S DIN note) and an FFT/MFCC
  pipeline. This is the big gap vs. the original spec's audio pipeline.
- [ ] **CH-12** `todo` — Port `EmbeddingStore`-equivalent KNN memory to C
  (PSRAM-backed), using the same cosine-distance convention validated in
  `eval/tasks/sound_memory_and_voice/`.
- [ ] **CH-13** `todo` — Novelty score -> mood state machine; wire mood
  states to real Culture-colour expression. Today's "dreaming" state has
  behaviour but no colour — needs either the WS2812B ring (hardware: "to
  order") or a redesigned AMOLED background treatment, since the current
  sprite nearly fills the screen (only a few px of margin to tint).
- [ ] **CH-14** `todo` — Dream loop (interpolate two stored embeddings,
  surprise score on next observation).
- [ ] **CH-15** `todo` — microSD persistence of the embedding store
  across power cycles.
- [ ] **CH-16** `todo` — Wire SG90 servo (hardware: "to order"), tail/beak
  wag on high novelty.
- [ ] **CH-17** `todo` — Design + print egg body v1 (see `body/README.md`
  for the brief).
- [ ] **CH-18** `todo` — Decide: reskin the inherited face sprites
  (currently Flowey/Undertale-styled) as the chough, or leave as
  placeholder art until the body/display design is final.

## Phase 2 — Two-unit emergent comms

- [ ] **CH-20** `todo` — Real tone synthesis + playback for `Repertoire`
  on both tracks (today's reference solution in `eval/`'s
  `Repertoire.play_call` is a no-op by design — it's not audio-graded).
- [ ] **CH-21** `todo` — Port the `Repertoire` design validated in
  `eval/tasks/sound_memory_and_voice/reference_solution/comms.py` into
  `python/src/chuff_mpu/comms.py` for real (currently a stub) and an
  ESP32 C equivalent.
- [ ] **CH-22** `todo` — JSONL call-context/acknowledgement logging
  (`comms.py: log_event()`).
- [ ] **CH-23** `blocked` — Bring up a second unit, 2-3m apart, observe
  repertoire evolution over 48h+. Blocked on a second ESP32-S3 board (or
  the UNO Q) being available.

## Phase 3 — Arduino UNO Q track (parallel)

- [ ] **CH-30** `blocked` — UNO Q + Modulino kit on order; everything
  below needs the real hardware to verify.
- [x] **CH-31** `done` — MVP proven without the hardware: mic RMS loudness
  -> mood -> serial JSON -> `chuff_mcu.ino` LED blink, via
  `chuff-mvp-demo`. Real novelty scoring (CH-32/33) replaces the RMS
  stand-in once it exists.
- [ ] **CH-32** `todo` — Implement `audio.py` (real MFCC via librosa) and
  `memory.py` (`EmbeddingStore`) for real — reuse the exact design
  validated in `eval/tasks/sound_memory_and_voice/reference_solution/`.
- [ ] **CH-33** `todo` — Implement `mood.py`'s state machine, `dream.py`.
- [ ] **CH-34** `todo` — Replace `chuff_mcu.ino`'s LED_BUILTIN stand-in
  with the real 8x13 LED matrix + Modulino Pixels once the board/kit
  arrive; confirm the exact onboard-matrix API (not verified against
  real hardware yet).
- [ ] **CH-35** `todo` — Modulino Movement / Distance polling, `{"event":
  ...}` lines back to the MPU.
- [ ] **CH-36** `todo` — Port the comms loop (CH-21) to this track.

## Tooling / process

- [ ] **CH-40** `todo` — Once `bd.exe` is unblocked, import this file into
  real beads issues and delete it (or keep as a human-readable export).
- [ ] **CH-41** `todo` — Wire `eval/harness/agents/gemini_agent.py` to a
  real Gemini run (`uv add google-genai`, set `GEMINI_API_KEY`) and
  confirm the automl-style loop in `eval/harness/loop.py` actually
  improves a real model's attempts round over round, not just the
  null/reference smoke tests.
- [ ] **CH-42** `todo` — Add a second `eval/tasks/` task once the first
  one has been run against a real model — candidates: the dream loop
  (CH-14/33), or the novelty->mood state machine (CH-13/33).
