# Culture Colour Grammar

Chuff's emotional expression, inspired by the Culture ship Minds in Iain M.
Banks's novels: light is language, not decoration. Each state maps to a
colour and a motion pattern; the combination should be unambiguous to a
child who has spent time with Chuff.

| State | Trigger | Colour | LED behaviour |
| --- | --- | --- | --- |
| Dreaming | Idle >5 min | Deep indigo | Very slow breathing pulse |
| Mild novelty | Novelty score 0.4-0.7 | Soft cyan | Gentle pulse |
| Strong novelty | Novelty score >0.7 | Bright white-blue | Fast ripple around ring |
| Song recognised | KNN match <0.2 distance | Amber to gold | Slow wave |
| New sound learned | KNN store updated | Cyan to white to gold burst | Burst then settle to gold |
| Bored / low input | No new sounds >30 min | Dim violet | Barely moving |
| Surprise (dream mismatch) | Dream prediction error >0.6 | Rapid flicker: two colours | Irregular 3-5 seconds |
| Acknowledgement received | Comms loop confirmation | Warm green pulse | Single slow pulse |
| Baseline contentment | Default / after recognition | Warm pale blue-green | Steady soft glow |

## Implementation notes

- On ESP32-S3: AMOLED screen renders an animated face (eyes, brow, glow
  background) corresponding to the state; a WS2812B ring (if wired) mirrors
  the colour.
- On UNO Q: the built-in 8x13 LED matrix renders the face expression; the
  Modulino Pixels ring renders the Culture colour.
- Mood states are an enum (`Mood` in `python/src/chuff_mpu/mood.py`; a C
  equivalent on the esp32/MCU side) — the display layer subscribes to state
  changes and renders independently.
- Transition animations matter as much as steady states — the burst-then-
  settle for a new learned sound should feel like a small celebration.
- All thresholds (0.4, 0.7, 0.2, etc.) are tunable constants in each
  track's config module — they will need calibration against the real
  acoustic environment.

## Current implementation status (MVP)

The Python `update_mood` API supports all nine labels with explicit event
inputs, verified by transition/precedence fixtures (25 October 2026). Pass
`learned=True` for an actual admission and `learned=False` to permit strong
novelty without admission; omission retains the legacy high-novelty/immediate
admission convention. Acknowledgement remains an external protocol event.
The seeded mood demo can inject explicit synthetic ACK ticks for replay,
but this does not implement an acoustic ACK detector or physical displays.

Neither hardware track implements the full table yet. What's running today:

- **esp32/**: a "dreaming after idle" behaviour only (slow idle-animation
  cadence after 5 minutes with no touch, wakes on touch) — see
  `esp32/main/pet_config.h` and `pet_face.c`. No colour/LED-ring expression
  yet; no audio input yet.
- **arduino/ + python/**: a `contentment` / `mild_novelty` / `strong_novelty`
  three-band stand-in, driven by raw mic RMS loudness rather than real MFCC
  novelty scoring — see `python/src/chuff_mpu/mvp_demo.py` and
  `arduino/chuff_mcu/chuff_mcu.ino`. Rendered today as an LED blink-rate
  proxy, not yet the real LED matrix/Pixels colour grammar.

The rest of the table is the backlog — see `TASKS.md`.
