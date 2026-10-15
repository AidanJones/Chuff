# sim/ — Chuff, tested in silico

For **foreground novelty in a changing room**, start with
[ATTENTION.md](ATTENTION.md). The newer `chuff-sim-attention` experiment
uses coloured ambience, hum, nuisance transients, echoes and microphone
filtering, and learns recurring foreground patterns from the mixture alone.
It compares background-aware admission with the historical whole-clip rule
described below. [Initial results](../../../../papers/01-foreground-novelty.md)
include both the successful noise levels and a strongly masked failure case.

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

There's a second demo, `two_unit_demo.py` (`chuff-sim-2unit`): two
*independent* simulated units talking over one acoustic channel, no
shared data structure — the software stand-in for `shared/
comms_protocol.md`'s real two-unit experiment. See
`../../../../shared/research_backlog.md` (Theme 1.3 / 3.2) for what it
found: contingent acknowledgement converges to a stable shared
call-per-sound mapping within a handful of rounds; passive exposure
alone never converges at all.

```
uv run chuff-sim-2unit --condition social     # contingent acknowledgement
uv run chuff-sim-2unit --condition passive    # control: B never acknowledges
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
`shared/comms_protocol.md` and `TASKS.md` CH-23b). Doesn't prove the
thresholds hold on real-world audio (these are synthesised tones/chirps,
not recordings) — see `TASKS.md` CH-32 for porting this to a real mic.

## Two more experiments, each testing one research_backlog.md item

```
uv run chuff-sim-2unit --cutoff 900           # Theme 1.2: a band-limited channel
uv run chuff-sim-2unit --sweep-seeds 15       # ...averaged over many seeds (one seed is too noisy to trust)

uv run chuff-sim-memory                       # Theme 2.2: FIFO vs. recency-weighted eviction
uv run chuff-sim-memory --capacity 3 --revisit-every 10
```

See `../../../../shared/research_backlog.md` Themes 1.2 and 2.2 for
what each found — including two real bugs the channel-attenuation test
exposed along the way (a too-gentle filter, and an accidental
frequency/index bias in `comms.Repertoire` that was silently hiding the
effect being tested for).

## A compositional call DSL + evolutionary search (`call_dsl.py`)

Inspired by a separate project on this machine
(`clebran-arc-agi` — evolves typed programs over a grid DSL for ARC
tasks; see `../../../../shared/research_backlog.md`'s "Related work"
section for the full comparison). Where `comms.Repertoire` represents
each call as a single fixed frequency, `call_dsl.CallProgram`
represents a call as a small program: one generator op (`tone`/`chirp`)
plus up to 3 modifier ops (`am`, `envelope`), with mutation and
crossover operators mirroring clebran's genome vocabulary at Chuff's
scale.

```
uv run chuff-sim-dsl --target kettle_whistle   # pure-tone target
uv run chuff-sim-dsl --target dog_bark         # wide-chirp target - a structural test
```

Both converge to ~0.97-0.98 fitness, but — honestly reported, not
papered over — neither recovers the "intended" generator kind: the tone
target gets matched by a near-zero-width chirp, the chirp target by a
tone with layered AM. The MFCC-cosine fitness metric doesn't
discriminate "right primitive" from "structural shortcut with a similar
bulk spectral envelope" as sharply as hoped. See the research backlog's
write-up for why that specifically blocks the next step (mining
recurring motifs across evolved programs, Stitch-style) until the
fitness metric is sharpened.

## Two units, symmetric: does grounding actually happen? (`grounding_demo.py`)

`two_unit_demo.py` is one-directional (A speaks, B only listens).
`chuff-sim-grounding` makes B a full second `ChuffAgent` too — both
units independently hear every sound, both speak, both acknowledge what
they recognise, both reinforce their own choice:

```
uv run chuff-sim-grounding --rounds 200
uv run chuff-sim-grounding --rounds 200 --cutoff 900
```

A 10-seed sweep (not just the first couple of runs — those looked like
a fixed pattern and weren't) always gets exactly 2 distinct pairs at
100% purity, and the majority pair is always call `#0`/`#0` (the
untrained tie-break's structural favorite) — but *which* 1-2 labels
break away from it varies by seed, with no consistent acoustic theme
across which ones. Coarse, stable discrimination driven by
presentation-order lock-in, not a fixed acoustic category boundary.

```
uv run chuff-sim-grounding --sweep-thresholds 10   # ch-63: does a tighter relevance gate help?
```

It doesn't — 0.3, 0.5, 0.7, and even 0.9 (near-impossible to clear) all
land at ~2.0 distinct pairs, 100% purity. A clean negative result that
redirects the real suspect to `Repertoire._base_default()`'s
index-based tie-break instead (see
`../../../../shared/research_backlog.md` Theme 3.3 for the full
write-up and `TASKS.md` ch-64 for the correctly-targeted follow-up).

## Does babbling before listening change anything? (`babbling_experiment.py`)

`ChuffAgent.babble()` self-reinforces each repertoire call against its
own clean signature before any real sound ever arrives — a "play each
call once" phase alone would do nothing (`respond()` already caches
self-reference lazily on first real use), so babbling had to be made
self-directed reinforcement to matter at all:

```
uv run chuff-sim-babble --seeds 40
```

A clean positive result (240 trials, holds at both 20 and 40 seeds):
babbling changes the very first real-world response 73% of the time,
landing ~735-770 Hz closer to the target sound's own characteristic
frequency on average than a cold start does. Without babbling, that
first response is decided almost entirely by `Repertoire`'s untrained
index-order tie-break, unrelated to what was actually heard.

## A bug `--log` had been hiding since two_unit_demo.py was first written

All three demos' `--log` flag reimplemented JSONL writing by hand
instead of using `comms.log_event()` (already implemented, never
called). Wiring them up to actually use it surfaced a real latent crash:
`_detect_tone()` returned `numpy.bool_` (from comparisons on numpy FFT
scalars), which `json.dumps` rejects — so `--log` had been silently
broken on every acknowledgement-bearing demo since the day it was
written, just never caught because `--log` had never actually been
exercised in testing. Fixed by casting explicitly to Python `float`/`bool`
in `_detect_tone()`. The lesson, not just the fix: a flag nobody runs is
a flag nobody's tested, no matter how simple the code behind it looks.

## Mood and the dream loop, actually exercised (`mood_demo.py`)

Every other demo presents a sound every tick, so idle time never really
accumulates. `chuff-sim-mood` has real gaps — most ticks nothing
happens — so `config.IDLE_TIMEOUT_S` actually gets crossed, the dream
loop (`dream.predict_next()`) gets a chance to fire, and the next real
sound that arrives gets compared against the prediction
(`dream.surprise_score()`), driving `mood.update_mood()`'s full 9-state
machine from real data rather than hand-picked test inputs:

```
uv run chuff-sim-mood --ticks 300 --seed 5 --sound-probability 0.03
```

Across 5 seeds, every mood state fires from a real triggering
condition — one seed alone hit `contentment`, `new_sound_learned`,
`song_recognised`, `dreaming`, *and* `surprise` in a single run. Whether
waking from a dream lands on `surprise` or `contentment` varies run to
run, as it should (it depends on whether the interpolated prediction
happened to resemble whatever sound came next) — see
`../../../../shared/research_backlog.md` Theme 2.1.
