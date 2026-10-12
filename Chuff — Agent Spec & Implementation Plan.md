# Chuff — Agent Spec & Implementation Plan

Oct 12, 2026 · @Aidan Jones

## Project Overview

Chuff is a robotic pet designed to learn from its acoustic environment, express emotional states through light and motion, and — in a second phase — develop a primitive shared language with a second unit through sound-only communication. The project runs as two parallel hardware tracks: a cheap ESP32-S3 track (already in hand) and an Arduino UNO Q track (more powerful, richer expansion). Both run the same conceptual model; the ESP32-S3 version validates the approach quickly and cheaply, the UNO Q version has headroom for more ambitious ML.

The name comes from the chough (*Pyrrhocorax pyrrhocorax*), a red-billed corvid native to the Pembrokeshire coastline near St Davids. The body is egg-shaped — suggesting a baby creature still learning — 3D printed in red PLA to reference the chough's red bill and legs. There are no moving parts in v1; a single servo for a beak or tail wag is earmarked for v2.

The emotional grammar of the lights is inspired by the Culture ship Minds in Iain M. Banks's novels, where light is language rather than decoration. Chuff does not simulate emotion — it expresses the actual state of its internal learning loop.

**Core behaviours**

- Listens continuously to its acoustic environment
- Builds MFCC-based embeddings of sounds it hears, stored in on-device memory
- Scores incoming audio for novelty (distance to nearest stored embedding)
- Runs a dream loop when idle: interpolates between stored embeddings, generates a predicted next observation, and compares it to reality on wake
- Expresses states — curiosity, surprise, satisfaction, boredom — through LEDs and display
- In two-unit mode: transmits compressed audio signals over the acoustic channel (speaker to microphone, no WiFi) and reinforces calls that the other unit successfully recognises

## Hardware

Two parallel tracks. All components listed are either already purchased or recommended for immediate order.

### Track A — ESP32-S3 (in hand)

| Component | Detail | Status |
| --- | --- | --- |
| Waveshare ESP32-S3-Touch-AMOLED-1.8 | Dual-core LX7 @ 240MHz, 8MB PSRAM, 16MB flash, 1.8" AMOLED 368×448, built-in mic, speaker connector, 6-axis IMU, RTC, microSD, WiFi, BT | Ordered |
| LiPo battery 3.7V | Matched to board's onboard power management | Ordered |
| SG90 micro servo | For v2 beak/tail wag — GPIO PWM via solder pad | To order |
| WS2812B 16-LED ring | Optional Culture colour ring alongside AMOLED | To order |

The 7 exposed GPIO pins on the ESP32-S3 board are on 1mm-pitch solder pads along the board edge. A servo or LED ring requires soldering fine wires to these pads.

### Track B — Arduino UNO Q (to order)

| Component | Detail | Status |
| --- | --- | --- |
| Arduino UNO Q 4GB | Qualcomm QRB2210 quad Cortex-A53 @ 2GHz + STM32U585 MCU, 4GB LPDDR4, 32GB eMMC, built-in 8×13 red LED matrix, Debian Linux, full Python 3 | To order |
| Modulino Pixels | 8 addressable RGB LEDs, Qwiic connector | To order |
| Modulino Movement | 6-axis IMU, Qwiic connector | To order |
| Modulino Distance | Time-of-flight proximity sensor | To order |
| Modulino Vibro | Vibration motor for haptic response | To order |
| Elegoo Super Starter Kit | Includes SG90 servo, stepper motor, L293D motor driver, ultrasonic sensor, breadboard, 65 jumper wires, LEDs | To order |
| USB microphone | Plug-and-play via USB-C port on UNO Q | To order |
| MAX98357 I2S amp + small speaker | Audio output for UNO Q | To order |

The UNO Q has a built-in 8×13 red LED matrix driven by the STM32 side — no shield required for basic emotional display. The Modulino Pixels adds RGB colour capability (Culture palette) via Qwiic with no soldering. Arduino are offering 10% off when a UNO Q and a Modulino are bought together from the Arduino Store.

## Software Architecture

### Track A — ESP32-S3 (MicroPython)

No laptop training. All learning happens on-device at runtime. The key library is **emlearn-micropython** — a C99 ML library for microcontrollers with MicroPython bindings, providing FFT, IIR filters, Random Forest, and K-Nearest Neighbours (KNN) with on-device runtime learning. KNN is the core: new audio embeddings are added to PSRAM at runtime without reflashing.

```
Audio pipeline:
  I2S mic → raw PCM buffer (8KHz, 16-bit)
    → FFT (emlearn) → mel filterbank → 13-coefficient MFCC vector
    → KNN memory (PSRAM) → novelty score (distance to nearest neighbour)
    → mood state update → AMOLED + LED expression

Dream loop (triggered after 5 min idle):
  Sample two stored MFCC vectors from PSRAM
    → interpolate (weighted average) → store as prediction
  On next audio capture:
    → compare new MFCC to prediction
    → large mismatch = high surprise, small mismatch = confirmation
```

Key constraints:

- 8MB PSRAM sets the memory ceiling for stored embeddings (\~hundreds of 13-float vectors)
- No numpy, scipy, or librosa — all DSP is hand-rolled or via emlearn
- MicroPython async (uasyncio) for concurrent audio capture, dream loop, and display update
- microSD for persistent storage of embedding memory across power cycles

### Track B — Arduino UNO Q (Python 3 / Debian Linux)

The Qualcomm MPU side runs full Debian Linux with pip-installable Python. The STM32 MCU side handles real-time hardware (LEDs, servo, sensor polling) over a defined IPC bridge.

```
MPU side (Python 3):
  librosa / numpy / scikit-learn — full MFCC extraction and KNN
  sounddevice — USB mic capture
  Serial IPC → STM32 side (mood state as JSON)

MCU side (Arduino C++):
  Receives mood state from MPU
  Drives 8×13 LED matrix (face expression)
  Drives Modulino Pixels (Culture colour)
  Drives servo via PWM
  Polls Modulino Movement, Distance, Light
  Sends sensor events to MPU
```

The IPC protocol is intentionally simple: the MPU sends a JSON line per mood update (`{"mood": "curious", "energy": 0.8}`), the MCU sends sensor events as JSON lines (`{"event": "picked_up"}`). Both sides are dumb about the other's internals.

### Repo Structure (for coding agent)

```
chuff/
  esp32/              # Track A — MicroPython
    main.py           # entry point, async task runner
    audio.py          # I2S capture, MFCC extraction
    memory.py         # KNN store, PSRAM management
    dream.py          # dream loop, prediction, surprise scoring
    display.py        # AMOLED face, LED ring
    mood.py           # mood state machine
    comms.py          # two-unit sound comms (phase 2)
    config.py         # pin assignments, tunable constants

  uno_q/              # Track B — UNO Q
    mpu/              # Debian Python side
      main.py
      audio.py
      memory.py
      dream.py
      mood.py
      ipc.py          # serial bridge to MCU
      comms.py
    mcu/              # Arduino C++ side
      chuff_mcu.ino
      display.cpp     # LED matrix face
      pixels.cpp      # Modulino Pixels
      servo.cpp
      sensors.cpp
      ipc.cpp

  shared/             # Shared design docs, not code
    mood_grammar.md   # Culture colour spec
    mfcc_notes.md     # MFCC implementation notes
    comms_protocol.md # two-unit comms design

  body/               # 3D print files
    chuff_body_v1.stl
    chuff_body_v1.f3d

  README.md
  SPEC.md             # this document, exported
```

## Dream Loop and Emergent Language

### The Dream Loop

The dream loop is what distinguishes Chuff from a novelty-detection toy. Rather than simply reacting to sounds, Chuff builds expectations about what it will hear next — and its emotional response depends on whether those expectations are met.

**Implementation:**

1. After 5 minutes of inactivity, Chuff enters dream mode (dim violet glow)
2. Two stored MFCC vectors are sampled at random from the KNN store
3. A predicted vector is computed as a weighted interpolation between them
4. The prediction is stored as the current expectation
5. On next audio capture, the new MFCC is compared to the prediction
6. `surprise = cosine_distance(new_mfcc, predicted_mfcc)`
7. High surprise → excited expression (chough heard something it didn't expect)
8. Low surprise → warm satisfied expression (world matched the dream)

Over time, as the KNN store grows, predictions become more accurate — the dream loop naturally biases toward sounds Chuff has heard before, making the surprise response increasingly selective and meaningful.

### Two-Unit Emergent Language

The communication channel between two Chuff units is **acoustic only** — speaker to microphone, across a room. No WiFi messaging, no shared data structure, no pre-programmed vocabulary. Whatever language emerges is shaped entirely by the acoustic environment and the mutual reinforcement loop.

**Starting conditions:**

- Each unit initialises with a small random call repertoire: 8–12 synthesised tones at distinct frequencies and durations, generated from a fixed seed so the two units start with overlapping but not identical repertoires
- Each unit maintains a call-to-context map: which call it played when it heard which sound

**The reinforcement loop:**

```
Unit A hears novel sound S
  → extracts MFCC(S)
  → finds closest call C in repertoire (nearest neighbour by MFCC distance)
  → plays call C through speaker

Unit B hears call C through microphone
  → extracts MFCC(C)
  → checks against its own call-to-context map
  → if recognised: internal novelty state updates; plays short acknowledgement tone
  → if not recognised: no response

Unit A hears acknowledgement (or silence)
  → acknowledgement: reinforces call C for context MFCC(S) — C's weight increases
  → silence: decays call C for context MFCC(S); tries a random variant next time
```

**Why language emerges:**

The lossy acoustic channel creates compression pressure. A long complex tone travels less reliably than a short distinctive one. Calls that survive repeated transmission become shorter and more distinctive. Calls that reliably produce acknowledgement survive; calls that don't are replaced by random variants. Over days of interaction, the surviving repertoire will be shaped by three forces: the acoustics of the room, the sounds Chuff most frequently encounters, and mutual recognisability between the two units.

**Key design constraints:**

- Units should be at least 2–3 metres apart — close enough for signals to travel, far enough to lose detail
- Each unit logs its call-to-context map and acknowledgement rates to microSD
- Logging enables post-hoc analysis of which calls converged and which were dropped
- No shared clock, no synchronisation — each unit acts independently

## Emotional Expression — Culture Colour Grammar

Inspired by the Culture ship Minds in Iain M. Banks's novels. Light is language, not decoration. Each state maps to a colour and a motion pattern; the combination is unambiguous to a child who has spent time with Chuff.

| State | Trigger | Colour | LED behaviour |
| --- | --- | --- | --- |
| Dreaming | Idle >5 min | Deep indigo | Very slow breathing pulse |
| Mild novelty | Novelty score 0.4–0.7 | Soft cyan | Gentle pulse |
| Strong novelty | Novelty score >0.7 | Bright white-blue | Fast ripple around ring |
| Song recognised | KNN match <0.2 distance | Amber → gold | Slow wave |
| New sound learned | KNN store updated | Cyan → white → gold burst | Burst then settle to gold |
| Bored / low input | No new sounds >30 min | Dim violet | Barely moving |
| Surprise (dream mismatch) | Dream prediction error >0.6 | Rapid flicker: two colours | Irregular 3–5 seconds |
| Acknowledgement received | Comms loop confirmation | Warm green pulse | Single slow pulse |
| Baseline contentment | Default / after recognition | Warm pale blue-green | Steady soft glow |

**Implementation notes:**

- On ESP32-S3: AMOLED screen renders an animated face (eyes, brow, glow background) corresponding to the state; WS2812B ring (if wired) mirrors the colour
- On UNO Q: built-in 8×13 red LED matrix renders the face expression; Modulino Pixels ring renders the Culture colour
- Mood states are an enum in `mood.py` / `mood.h`; the display layer subscribes to state changes and renders independently
- Transition animations matter as much as steady states — the burst-then-settle for a new learned sound should feel like a small celebration
- All thresholds (0.4, 0.7, 0.2 etc.) are tunable constants in `config.py` — they will need calibration against the real acoustic environment

## Git Repo Spec for Coding Agent

The coding agent should create a new git repository called `chuff` with the following structure, stub files, and content. No implementation code yet — stubs with docstrings and type signatures only, except for `config.py` which should be fully populated with the pin assignments and tunable constants below.

### Instructions for coding agent

1. Create a new git repo called `chuff` with a `main` branch
2. Create the full directory structure from the Software Architecture section above
3. Populate every `.py` and `.ino` / `.cpp` file as a stub: module docstring explaining its role, all function signatures with type hints and docstrings, no implementation bodies (use `pass` or `raise NotImplementedError`)
4. Fully implement `esp32/config.py` and `uno_q/mpu/config.py` with the constants below
5. Write a thorough `README.md` explaining the project concept, the two hardware tracks, and the build order
6. Write `shared/mood_grammar.md` as a complete reference for the Culture colour table (copy from the Emotional Expression section of this spec)
7. Write `shared/comms_protocol.md` documenting the two-unit acoustic comms design (copy from the Dream Loop section)
8. Write `shared/mfcc_notes.md` explaining the MFCC pipeline and the emlearn-micropython dependency
9. Create a `requirements_mpu.txt` for the UNO Q MPU side: `numpy`, `librosa`, `scikit-learn`, `sounddevice`, `pyserial`
10. Create a placeholder `body/` directory with a `README.md` noting that STL files will be added after the body design is finalised (egg shape, red PLA, AMOLED window in front face)
11. First commit message: `chore: initial repo scaffold for Chuff robotic pet project`

### Config constants for `esp32/config.py`

```python
# Pin assignments (ESP32-S3 AMOLED board — 1mm pitch solder pads)
MIC_SCK_PIN = 4
MIC_WS_PIN = 5
MIC_SD_PIN = 6
SPEAKER_PIN = 7       # I2S out to amp
SERVO_PIN = 8         # SG90 via PWM (v2)
LED_RING_PIN = 10     # WS2812B data (optional)

# Audio
SAMPLE_RATE = 8000    # Hz
FFT_SIZE = 256
N_MFCC = 13
CLIP_DURATION_S = 2   # seconds per audio sample

# KNN memory
MAX_EMBEDDINGS = 512  # PSRAM ceiling
NOVELTY_THRESHOLD_LOW = 0.4
NOVELTY_THRESHOLD_HIGH = 0.7
RECOGNITION_THRESHOLD = 0.2

# Dream loop
IDLE_TIMEOUT_S = 300  # 5 minutes before dream mode
DREAM_SURPRISE_THRESHOLD = 0.6

# Comms
CALL_DURATION_MS = 500
CALL_FREQ_MIN_HZ = 400
CALL_FREQ_MAX_HZ = 3000
REPERTOIRE_SIZE = 12
ACK_FREQ_HZ = 800
ACK_DURATION_MS = 200
REINFORCEMENT_RATE = 0.1
DECAY_RATE = 0.05

# Timing
BORED_TIMEOUT_S = 1800  # 30 minutes no new sounds
```

### Build order (for README)

Phase 1 — ESP32-S3 single unit:

1. Flash MicroPython, verify mic and speaker
2. Implement FFT and MFCC extraction in `audio.py`
3. Implement KNN store in `memory.py`
4. Implement novelty scoring and mood state machine in `mood.py`
5. Implement AMOLED face and Culture colour expressions in `display.py`
6. Implement dream loop in `dream.py`
7. Wire SG90 servo to GPIO8, implement tail wag in `mood.py`
8. Design and print egg body

Phase 2 — Two-unit emergent comms: 9. Implement call synthesis and comms loop in `comms.py` 10. Test single-unit call-and-listen cycle 11. Bring up second unit, place 2–3 metres apart 12. Enable microSD logging of call-context map 13. Observe and log repertoire evolution over days

Phase 3 — UNO Q track (parallel): 14. Set up UNO Q with Debian, install MPU dependencies 15. Port audio and memory logic to `uno_q/mpu/` 16. Implement STM32 MCU firmware in `uno_q/mcu/` 17. Implement IPC bridge between MPU and MCU 18. Wire Modulino Pixels and Movement via Qwiic

## Implementation Plan

Three phases. Phase 1 is the priority — it produces a working Chuff on the ESP32-S3 hardware already in hand. Phases 2 and 3 run in parallel once Phase 1 is stable.

### Phase 1 — ESP32-S3 Single Unit

| Step | Task | Key files | Done when |
| --- | --- | --- | --- |
| 1 | Flash MicroPython v1.23+ to ESP32-S3 board; verify mic records audio; verify speaker plays tone | — | Live FFT output prints to REPL |
| 2 | Implement I2S capture and FFT pipeline | `audio.py` | MFCC vector prints for live audio |
| 3 | Implement KNN store with PSRAM allocation | `memory.py` | 100 embeddings stored and queried |
| 4 | Implement novelty score and mood state machine | `mood.py` | Mood state changes on novel audio |
| 5 | Implement AMOLED face and LED expressions | `display.py` | All 9 Culture states render correctly |
| 6 | Implement dream loop | `dream.py` | Surprise/satisfaction events fire on wake |
| 7 | Implement microSD persistence | `memory.py` | Embedding store survives power cycle |
| 8 | Wire servo, implement tail wag | `mood.py`, `config.py` | Wag triggers on high novelty |
| 9 | Design and print egg body v1 | `body/` | Board, battery, servo fit inside |

### Phase 2 — Two-Unit Emergent Comms

| Step | Task | Key files | Done when |
| --- | --- | --- | --- |
| 10 | Implement call synthesis (tone generation) | `comms.py` | 12-tone repertoire plays correctly |
| 11 | Implement call recognition (MFCC match against repertoire) | `comms.py` | Unit recognises its own calls |
| 12 | Implement reinforcement loop (single unit) | `comms.py` | Call weights update on ack/silence |
| 13 | Bring up second unit, enable logging | `comms.py`, microSD | Both units log call-context maps |
| 14 | Place units 2–3m apart, observe | — | Repertoire evolution visible in logs after 48h |

### Phase 3 — Arduino UNO Q Track (parallel)

| Step | Task | Key files | Done when |
| --- | --- | --- | --- |
| 15 | Set up UNO Q with Debian, install MPU deps | `requirements_mpu.txt` | `import librosa` works |
| 16 | Port audio and memory logic to MPU Python | `uno_q/mpu/` | Novelty detection matches ESP32 behaviour |
| 17 | Implement STM32 MCU firmware | `uno_q/mcu/` | LED matrix face renders all states |
| 18 | Implement IPC bridge | `ipc.py`, `ipc.cpp` | Mood state flows MPU → MCU reliably |
| 19 | Wire Modulino Pixels and Movement via Qwiic | — | Culture colour + IMU events working |
| 20 | Port comms loop to UNO Q | `uno_q/mpu/comms.py` | UNO Q and ESP32 units can talk to each other |

### Notes for coding agent

- Target MicroPython ≥ 1.23 for ESP32-S3; test on the Waveshare ESP32-S3-Touch-AMOLED-1.8 board specifically
- The AMOLED uses SH8601 driver over QSPI — Waveshare's MicroPython example code is the reference; import it into `display.py` as a dependency
- emlearn-micropython must be cross-compiled for ESP32-S3 and included as a `.mpy` file in the repo
- All tunable thresholds live in `config.py` only — never hardcoded elsewhere
- All async logic uses `uasyncio` (ESP32) or `asyncio` (UNO Q MPU); the audio capture, dream loop, display update, and comms loop are separate async tasks
- The logging format for comms experiments should be JSONL (one JSON object per line) to make post-hoc analysis straightforward
