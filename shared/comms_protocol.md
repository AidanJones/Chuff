# Two-Unit Emergent Comms Protocol

Phase 2 of the Chuff spec. The communication channel between two Chuff
units is **acoustic only** — speaker to microphone, across a room. No
WiFi messaging, no shared data structure, no pre-programmed vocabulary.
Whatever language emerges is shaped entirely by the acoustic environment
and the mutual reinforcement loop.

## Starting conditions

- Each unit initialises with a small random call repertoire: 8-12
  synthesised tones at distinct frequencies and durations, generated from a
  fixed seed so the two units start with overlapping but not identical
  repertoires.
- Each unit maintains a call-to-context map: which call it played when it
  heard which sound.

## The reinforcement loop

```
Unit A hears novel sound S
  -> extracts MFCC(S)
  -> finds closest call C in repertoire (nearest neighbour by MFCC distance)
  -> plays call C through speaker

Unit B hears call C through microphone
  -> extracts MFCC(C)
  -> checks against its own call-to-context map
  -> if recognised: internal novelty state updates; plays short acknowledgement tone
  -> if not recognised: no response

Unit A hears acknowledgement (or silence)
  -> acknowledgement: reinforces call C for context MFCC(S) - C's weight increases
  -> silence: decays call C for context MFCC(S); tries a random variant next time
```

## Why language emerges

The lossy acoustic channel creates compression pressure. A long complex
tone travels less reliably than a short distinctive one. Calls that
survive repeated transmission become shorter and more distinctive. Calls
that reliably produce acknowledgement survive; calls that don't are
replaced by random variants. Over days of interaction, the surviving
repertoire will be shaped by three forces: the acoustics of the room, the
sounds Chuff most frequently encounters, and mutual recognisability
between the two units.

## Key design constraints

- Units should be at least 2-3 metres apart — close enough for signals to
  travel, far enough to lose detail.
- Each unit logs its call-to-context map and acknowledgement rates (JSONL,
  one object per line — see `comms.py: log_event()` on each track).
- Logging enables post-hoc analysis of which calls converged and which
  were dropped.
- No shared clock, no synchronisation — each unit acts independently.

## Relationship to the MPU<->MCU IPC protocol

This is a *separate* protocol from the simpler MPU<->MCU serial IPC used
within a single UNO Q unit (mood/event JSON lines, see
`python/src/chuff_mpu/ipc.py` and `arduino/chuff_mcu/chuff_mcu.ino`). The
acoustic comms protocol above is between two separate physical Chuff
units, speaker-to-microphone; it has no wired connection at all.

## Status

Not implemented on either track yet — `comms.py` (esp32 and python) are
both stubs. This is Phase 2, after the single-unit novelty/mood pipeline
(audio.py / memory.py / mood.py) lands. See `TASKS.md`.
