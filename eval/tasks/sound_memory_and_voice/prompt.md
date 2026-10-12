# Task: give Chuff a working memory and a working voice

Chuff is a small robotic pet that listens to its environment, remembers
distinct sounds it has heard before, and — in a two-unit setup — develops
calls that let it signal another Chuff unit about what it heard. You're
implementing the two pieces of logic that make that possible, in an
existing Python package.

Your starting point is `stub/memory.py` and `stub/comms.py` in this
task's own directory — full docstrings, type hints, and method
signatures, every body `raise NotImplementedError`. Fill in the bodies.
Don't change the public signatures — other code (`mood.py`, `dream.py`,
the MVP demo script, `sim/`) calls them as specified.

(The live `python/src/chuff_mpu/memory.py` / `comms.py` now contain a
real implementation of this exact task — don't just copy it in. The
point is doing the work; the grader is what tells you whether your own
attempt is right.)

Worth reading before you start:

- `python/src/chuff_mpu/config.py` — tunable thresholds. Read it, don't
  change it.
- `shared/mood_grammar.md`, `shared/comms_protocol.md`,
  `shared/mfcc_notes.md` — background on why these exist and how they fit
  the rest of the system.

## 1. `EmbeddingStore` (memory.py)

A fixed-capacity nearest-neighbour store of audio embeddings ("sounds
Chuff has heard"). It needs to:

- `add(embedding)`: remember a new embedding, evicting the oldest once
  `max_embeddings` is exceeded.
- `novelty_score(embedding)`: distance from `embedding` to its nearest
  stored neighbour, or `1.0` if the store is empty. **Use cosine
  distance** — it's what the rest of this codebase uses for the same
  kind of comparison (see the dream-loop surprise formula in the spec),
  and `config.py`'s thresholds are calibrated assuming it.
- `sample_pair()`: return two distinct stored embeddings at random, for
  the dream loop to interpolate between. Raise `ValueError` if fewer than
  two are stored.
- `save`/`load`: persist to/from disk so memory survives a restart. Any
  reasonable format (npz, JSON) is fine — nothing else in this task reads
  the file back.
- `__len__`: how many embeddings are currently stored.

You're graded on whether the store correctly tells repeated/near-
identical sounds apart from genuinely new ones — i.e. whether, across a
sequence of sounds (some repeated with small variation, some brand new),
`novelty_score()` comes out below `config.RECOGNITION_THRESHOLD` for
things Chuff has heard before and above it for things it hasn't.

## 2. `Repertoire` (comms.py)

A single unit's set of calls (tones) plus its learned sense of which call
fits which context — the data structure behind the reinforcement loop
described in `shared/comms_protocol.md`. It needs to:

- `__init__(size, freq_min_hz, freq_max_hz, seed)`: build `size` call
  frequencies within range, seeded so repeated construction is
  reproducible. **Store them on `self.frequencies`** — that's the public
  contract other code relies on.
- `play_call(call_index, duration_ms, sample_rate)`: synthesise and play
  the tone. (No speaker in this grading environment — a no-op or stub
  body is fine; it isn't scored.)
- `closest_call(context_embedding)`: which call this unit currently
  prefers for a given context. Repeated calls with no learning in between
  must return the same answer. Calls that have been `reinforce()`d for a
  context should keep winning for that context; calls that have been
  `decay()`ed for a context should eventually lose to something else.
  Learning tied to one context shouldn't silently hijack a *different*
  context's preference.
- `reinforce(call_index, context_embedding, rate)` /
  `decay(call_index, context_embedding, rate)`: nudge a call's weight for
  a context up or down. How you store the weighting (per-call exemplar
  list, a small table, whatever) is up to you — only the behaviour above
  is checked.

The full two-unit emergent-language experiment (two physical units,
acoustic channel, days of observation) isn't something a grader can run
in CI — this task only checks the single-unit learning behaviour
(repertoire generation + call-to-context reinforcement) that the real
experiment depends on.

## Scoring

Run the grader yourself at any point, against whatever directory holds
your in-progress `memory.py`/`comms.py` (e.g. a copy of `stub/` you're
editing):

```
uv run python eval/tasks/sound_memory_and_voice/grader.py --candidate-dir path/to/your/attempt
```

(from the repo root). It prints a JSON report: an `overall_score` in
[0, 1], and a breakdown by
dimension (`sound_memory`, `communication`) and individual check, each
with the specific input/assertion that failed if it did. Fix what's
failing and re-run — there's no hidden test set beyond what the grader
already shows you.
