# sim/ — Chuff, tested in silico

A from-the-ground-up test of the listen -> remember -> repeat loop with
no physical hardware at all: `world.py` synthesises a handful of
distinct sounds and background noise standing in for a room + mic;
`agent.py` wraps the real (not stub) `audio.py` / `memory.py` /
`comms.py` — the same production code a UNO Q would run — and actually
hears, remembers, and "repeats" those sounds in its own voice.

```
uv run chuff-sim                              # 12 rounds, low noise
uv run chuff-sim --rounds 20 --noise 0.4      # noisier room — watch it misrecognise itself sometimes
uv run chuff-sim --play                       # also attempt real playback (best-effort; safe either way)
uv run chuff-sim --log run1.jsonl             # append per-round JSONL
```

## What each round does

1. **World presents a sound** (`doorbell`, `car_horn`, `dog_bark`, ... —
   see `world.SOUND_LIBRARY`), with a little frequency jitter so repeats
   aren't bit-identical.
2. **Chuff hears it** mixed with background noise (`Environment.hear()`),
   extracts a real MFCC embedding (`audio.extract_mfcc`), and scores its
   novelty against everything it's heard before
   (`memory.EmbeddingStore.novelty_score`) — genuinely novel sounds
   (score >= `config.NOVELTY_THRESHOLD_HIGH`) get committed to memory.
3. **Chuff repeats it in its own voice**: picks the closest call in its
   repertoire for this context (`comms.Repertoire.closest_call`),
   synthesises that call, and runs it through the safety limiter
   (`audio_safety.limit_audio`) before anything else touches it.
4. **Chuff hears its own echo** — that same call, mixed with *fresh*
   background noise, the same way a real speaker's output would bleed
   back into a real mic. If it still recognises itself
   (`agent.recognise_own_echo`), it reinforces that call for this
   context; if the noise swamped it, it decays the call instead and will
   lean toward a different one next time.

Step 4 stands in for a second unit's acknowledgement
(`shared/comms_protocol.md`) — there's no second unit here, just Chuff
checking whether its own voice survived the room. At low noise
(`--noise 0.03`, the default) it basically always recognises itself and
one call ends up handling everything. Push the noise up
(`--noise 0.4`) and recognition starts failing sometimes — that's when
you see the repertoire actually diversify, call indices spreading out
round to round as `reinforce`/`decay` pull the choice around. Both
regimes are worth running once to see the difference.

## The guardrail, proven, not just asserted

Before any rounds run, `run_demo.py` feeds `audio_safety.limit_audio()`
a deliberately pathological signal (50x full-scale amplitude, with NaN
and Inf samples injected) and prints the before/after peak amplitude and
non-finite sample count, asserting the output is within
`config.SAFE_PEAK_AMPLITUDE` and fully finite. Every call Chuff "says"
in the rounds that follow goes through the exact same function — see
`audio_safety.py`'s module docstring for the three protections (NaN/Inf
sanitising, duration cap, soft-knee limiter + hard ceiling) and why each
exists.

## What this does and doesn't prove

Proves: the architecture (hear -> embed -> score novelty -> maybe learn
-> pick a call -> safety-limit -> hear the echo -> reinforce/decay)
runs end to end against real audio, with real MFCC extraction and a
real cosine-distance KNN store, not just on the abstract feature vectors
used in `eval/`'s grader.

Doesn't prove: real two-unit emergent language (that needs a second
physical unit, a real acoustic channel, and days of observation — see
`shared/comms_protocol.md` and `TASKS.md` CH-23). Doesn't prove the
thresholds hold on real-world audio (these are synthesised tones/chirps,
not recordings) — see `TASKS.md` CH-32 for porting this to a real mic.
