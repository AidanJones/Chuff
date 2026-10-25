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

For an opt-in capture/replay controller that turns foreground outcomes into
mood messages and saves compatible memories, see
[selective capture and mood control](src/chuff_mpu/SELECTIVE.md):

```powershell
uv run chuff-selective --input-wav path/to/16khz-recording.wav --state ../eval/results/selective-state.npz
uv run chuff-selective --mic --port COM5 --state ../eval/results/selective-state.npz
```

The default is silent; only an explicit microphone learning-cue option can
play a short notification. Replay needs no microphone or MCU.

For the newer **foreground discovery** experiment, which learns recurring
events instead of memorising the room background, run:

```powershell
uv run chuff-sim-attention --seeds 5 --seed-start 100 --snr-db 0 6 12 --structured-distractors --output ../eval/results/my-attention-run --save-audio
```

Use a new output directory for each run. It saves settings, source hashes,
per-window decisions, per-seed summaries, and an optional mixed-audio WAV.
The learner receives waveforms only; labels are held by the evaluator.
See [the experiment guide](src/chuff_mpu/sim/ATTENTION.md) and
[the first results](../papers/01-foreground-novelty.md) for the noise model,
metrics, and important failure cases. The historical demos below retain
their original whole-clip MFCC behaviour for comparison.

The [masked-event follow-up](src/chuff_mpu/sim/MASKED.md) adds band-relative
detection, broadband/long-event stress cases, source snapshots, and paired
seed intervals. Its frontend can be tried with `chuff-selective --frontend band`
and a separate state file. The original frontend remains the default.
The same guide includes a frozen-detector variance study with independently
varied sources, rooms, microphones and backgrounds, plus reproduction and
audit commands. Broader variation exposes substantial recognition and
background-memory failures documented in section 10 of the follow-up paper.
The [shared-gain study](../papers/04-shared-spectral-gain.md) iteratively reduces
modulation-driven attention and adds a calibration-derived raw/compensated
choice that preserves stationary recall on its controlled evaluation. Try
`chuff-selective --frontend calibrated-gain` experimentally with a separate
state file and representative background calibration. The default remains
original; recordings, changing environments and imperfect calibration still
need independent evaluation.

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
- `attention.py` — experimental mixture-only background estimation, ordered
  foreground features, recurrence confirmation, and bounded novelty memory.
- `memory.py` — real cosine-distance KNN store (`EmbeddingStore`).
- `comms.py` — real call repertoire + reinforcement/decay (`Repertoire`),
  with `play_call()` always routed through `audio_safety`.
- `audio_safety.py` — the output guardrail everything above routes
  through before any sample reaches a real speaker.
- `dream.py` — real dream-loop prediction + cosine-distance surprise scoring.
- `mood.py` — real 9-state mood machine (`Mood`, `MoodState`, `update_mood()`).
- `ipc.py` — serial JSON bridge to the MCU side (fully implemented).
- `mvp_demo.py` — the MVP demo described above.
- `sim/` — the in-silico test described above.

`memory.py` and `comms.py`'s designs were validated against
`../eval/tasks/sound_memory_and_voice/` before landing here — running
that task's grader against this package's own files scores 1.0.

### Legacy embedding-store restarts

`EmbeddingStore.save()` now writes a versioned, non-pickle NPZ with feature
vectors, eviction policy, recency timestamps and clock. It serializes to a
sibling temporary file, flushes/fsyncs it, closes it, then atomically replaces
the destination. Injected serialization, fsync and replacement failures leave
the previous checkpoint intact and remove the temporary file (ch-f5q).
As with the older `np.savez` API, `.npz` is appended if the path lacks it.

`load()` validates schema/version, capacity, vector dimensions/finite values,
history shape and timestamp bounds before replacing any live state. The
receiving store must use the saved eviction policy; its configured capacity
still applies and must fit all vectors. With the same capacity/policy, the
restored clock and history preserve subsequent eviction decisions. Choosing
a different receiving capacity deliberately changes future capacity pressure.
Malformed snapshots leave vectors and history untouched.

Valid older positional `arr_0`, `arr_1`, … files remain readable in numeric
order. Because those files lack history, imports reset recency to archive
insertion order (oldest first). Empty snapshots are supported. Object/pickle
arrays and selective-foreground schema files are rejected; legacy untyped
arrays still rely on caller provenance for their feature meaning.

Sampling generators belong to the caller (`sample_pair(rng=...)`) and are not
serialized or consumed by save/load. Exact sampling continuation requires
retaining that external RNG state separately; a store snapshot is not a full
agent/environment/controller checkpoint. Tests cover history/eviction replay,
legacy imports, schema isolation, invalid-load rollback and interrupted writes.
The store owns copies of added vectors and returns copied sample pairs, so
reusing capture buffers or editing sampled vectors cannot rewrite live memory.
Atomic replacement does not claim concurrent-writer coordination or full
power-loss durability of the filesystem directory entry.
