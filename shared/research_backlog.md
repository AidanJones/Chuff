# Research Backlog

Nine open-ended investigative tracks behind Chuff's design choices,
organised into three themes. Each item below carries: the original
question, what's actually known about it (grounded with real sources,
not asserted), where it connects to code already in this repo, and a
concrete next step — buildable here, or flagged as genuinely open if
it isn't.

Two of the nine (1.3 and 3.2) now have a first computational answer,
not just a citation — see `python/src/chuff_mpu/sim/two_unit_demo.py`.

## Theme 1 — Ethology & Bio-Acoustics of Mimicry

### 1.1 The Sensorimotor Bottleneck

**Question:** How do juvenile vocal learners (songbirds, parrots)
compress continuous spectrotemporal streams into discrete internal
templates during the critical developmental window?

**What's known:** The textbook picture — a passive "sensory phase"
(forming the auditory template) strictly before an active "sensorimotor
phase" (motor practice against it) — has been complicated by recent
zebra finch work: active babbling appears necessary to *open* the
sensory template-formation window in the first place, not just to act
on a template already formed ([Babbling opens the sensory phase for
imitative vocal learning, PNAS 2024](https://www.pnas.org/doi/10.1073/pnas.2312323121)).
The template itself is explicitly the compressed bridge between a
continuous auditory signal and a discrete motor program ([Tutor
auditory memory for guiding sensorimotor learning in
birdsong](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11246853/)).

**In this repo:** `memory.py`'s `EmbeddingStore` does the raw
compression (continuous MFCC -> a bounded set of stored vectors);
`comms.py`'s `Repertoire._exemplars` is the closer analogue to a
template — a small, incrementally-built set of (context, weight) pairs
per call.

**In this repo — tested (CH-53, done):** a naive "play each call once"
babble phase would do *nothing* observable here — `respond()` already
lazily caches a call's self-reference the moment it's first needed, so
there's no cold-start gap to fill. `ChuffAgent.babble()` instead
self-reinforces each call against its own clean signature (a weak "home"
association) before any real sound arrives — closer to what active
babbling is thought to do (building the template via self-generated
motor-auditory pairing, not just passive exposure).

**Result (`babbling_experiment.py` / `chuff-sim-babble`, 40 seeds x 6
sounds = 240 trials):** babbling changes the very first real-world
response **73% of the time**, and the chosen call's frequency lands
**~735-770 Hz closer** to the target sound's own characteristic
frequency on average (1007 Hz error cold-start vs. 271 Hz babbled,
consistent across a 20-seed and a 40-seed run). Without babbling, the
first encounter with any sound is decided almost entirely by
`Repertoire`'s untrained index-order tie-break, with no relationship to
what was actually heard; with babbling, the self-anchors give the first
real response something acoustically relevant to land near, before any
external reinforcement has happened at all. A clean, repeatable
positive result — unlike CH-60's fitness-metric snag, this one held up
exactly as hypothesised.

### 1.2 Acoustic Niche Adaptation

**Question:** How do environmental constraints (reverberation, ambient
noise, distance attenuation) shape the evolution of short, distinct
call repertoires versus complex continuous song?

**What's known:** The Acoustic Adaptation Hypothesis (AAH) predicts
lower frequency, narrower bandwidth, and longer inter-element intervals
in reverberant/closed habitats, since high frequencies and rapid
repetition degrade faster under reverberation and atmospheric
absorption — but a meta-analysis found the empirical support genuinely
mixed, not settled ([Habitat structure and the evolution of bird song: a
meta-analysis, Functional Ecology
2007](https://besjournals.onlinelibrary.wiley.com/doi/10.1111/j.1365-2435.2006.01207.x)).

**In this repo — tested:** `world.py` now has `lowpass_filter()` (a
4th-order Butterworth) and `Environment.transmit()` (filter + distance
attenuation + fresh noise), and `two_unit_demo.py --cutoff` runs the
whole two-unit loop through it. `--sweep-seeds N` averages the
no-cutoff-vs-900Hz-cutoff comparison over N seeds (a single seed wasn't
trustworthy — see the two false starts below). Result, 15 seeds, 60
rounds each:

| channel | avg frequency of reinforced calls | avg 2nd-half ack rate |
|---|---|---|
| no cutoff | 1903 Hz | 0.99 |
| 900 Hz cutoff | 1483 Hz | 0.86 |

A band-limited channel both pushes the surviving repertoire toward lower
frequencies *and* makes convergence less reliable overall — both
consistent with AAH's prediction, for this toy channel model.

**Two false starts worth recording, because they're the actual lesson:**
1. A single one-pole low-pass (6 dB/octave) turned out far too gentle —
   even at 3000 Hz through a 900 Hz cutoff it only lost ~10 dB, nowhere
   near enough to sink a tone below the ~0.03-amplitude noise floor.
   Switched to a 4th-order Butterworth (~-46 dB at 3000 Hz/900 Hz
   cutoff) after measuring the one-pole's output directly, not assuming
   it was steep enough.
2. The very first filtered run showed *zero* effect, because
   `comms.Repertoire.__init__` used to `np.sort()` its random
   frequencies — meaning call index 0 was always the lowest frequency,
   and `closest_call()`'s untrained tie-break already favours low
   index. So "no learning yet" was *already* maximally biased toward
   low frequencies regardless of the channel, hiding any AAH effect
   behind a confound nobody had noticed. Fixed by not sorting
   (`comms.py`, with a comment explaining why) — a real bug this
   research question exposed, not just a parameter to tune.

### 1.3 Social Motivation vs. Exposure — answered in software, here

**Question:** What threshold conditions trigger vocal learning —
specifically, is passive audio exposure insufficient without social
contingency (the "social gating" hypothesis)?

**What's known:** Human infants acquire phonetic structure from live
social interaction but not from passive/TV exposure at the same
quantity — contingent caregiver responses to babbling (touch, smiling,
vocal feedback) measurably accelerate and mature vocalisation
([Social interaction shapes babbling, PNAS
2003](https://www.pnas.org/doi/10.1073/pnas.1332441100)). The same
contingency effect shows up in zebra finch tutoring: live, responsive
tutors outperform passive playback of the identical song ([Human speech
and birdsong: communication and the social
brain](https://www.pnas.org/doi/10.1073/pnas.1733998100)).

**In this repo — tested, not just read about:**
`python/src/chuff_mpu/sim/two_unit_demo.py` runs this exact A/B test.
Unit A hears a sound, picks a call from its repertoire, and "says" it
to Unit B over a simulated noisy channel. Under `--condition social`, B
acknowledges once it recognises a call; under `--condition passive`, B
never acknowledges, no matter what — a literal passive-exposure control.

Result (seed 5, 60 rounds, both logged verbatim, not cherry-picked):

| | first-half ack rate | second-half ack rate | call choice over time |
|---|---|---|---|
| `social` | 0.90 | 1.00 | converges within ~5 rounds, stays stable |
| `passive` | 0.00 | 0.00 | never converges — same input sound gets a different call nearly every time, even late in the run |

That's the social-gating hypothesis's prediction, reproduced
computationally: contingent acknowledgement is what produces a stable,
shared convention; exposure to the same signals with no acknowledgement
produces none, no matter how long it runs.

**Caveat:** this is a *minimal* simulation (plain tones/chirps, a fixed
ack tone, simple FFT-peak detection) — it shows the mechanism in
`comms.py`'s reinforce/decay design is sufficient to reproduce the
qualitative effect, not that it explains the effect in real birds or
infants.

### 1.4 (bonus, not in the original list) — nothing further queued here;
see 3.2 below, which is the same mechanism viewed as a convergence-speed
question rather than a contingency question.

## Theme 2 — Neuro-Inspired Architectures & Memory Models

### 2.1 Predictive Coding & Free Energy in Audio

**Question:** How does top-down expectation generation interact with
bottom-up prediction error to drive emergent affective states (surprise,
boredom, satisfaction)?

**What's known:** Under the free-energy/predictive-coding framing, the
brain maintains a generative model and treats "surprise" as an
information-theoretic measure of how unlikely current sensory input is
under that model; sustained, unresolved prediction error maps onto
negative affect (stress, anxiety), while accurate prediction corresponds
to calm ([The free-energy principle: a unified brain theory?, Nat Rev
Neurosci 2010](https://www.nature.com/articles/nrn2787)).

**In this repo — done (CH-33):** `dream.py`'s
`surprise_score = cosine_distance(observed, predicted)` *is* this
framework, reduced to one line, now implemented for real (not a stub),
alongside `predict_next()` (interpolates two stored embeddings) and
`mood.py`'s full 9-state `update_mood()` state machine (precedence:
acknowledgement > dream-loop surprise > idle-time states > novelty
score > hold). `sim/mood_demo.py` (`chuff-sim-mood`) wires both into a
"day in the life" with real idle gaps (every other demo presents a
sound every tick, so idle time never actually accumulates) — most
ticks nothing happens; once idle crosses `IDLE_TIMEOUT_S`, the agent
occasionally dreams, and the next real sound that arrives is compared
against the dream's prediction.

**Result, across 5 seeds (300 ticks, p(sound)=0.03 so idle gaps reliably
cross the threshold):** every mood state the grammar describes shows up
from real triggering conditions, not hand-picked test inputs — one seed
alone (`--seed 5`) hit `contentment`, `new_sound_learned`,
`song_recognised`, `dreaming`, *and* `surprise`, all from the same run.
The dream loop fired 2-6 times per 300-tick run depending on seed, and
whether waking from a dream produced `surprise` vs. `contentment`
varied run to run (as it should — it depends on whether the
interpolated prediction happened to resemble whatever sound came next).
That's the actual test of whether the one-line free-energy
operationalisation earns its name: it does produce a state that tracks
whether reality matched an internally-generated expectation, driven
by real audio, not a scripted "and then surprise happens" moment.

**Near-term experiment, still open:** check — by ear, watching
`sim/mood_demo.py`'s logs — whether a human would actually *read* the
resulting state change as "surprised" vs. just "a number changed."
That's a different, harder bar than the one this round cleared
(the mechanism fires correctly); it's about whether the mapping from
score to felt state is actually convincing, not just functioning.

### 2.2 Distributed vs. Local Memory Decay

**Question:** How do non-parametric, bounded-capacity memory stores
handle catastrophic forgetting and reinforcement-driven retention over
extended timelines?

**What's known:** Catastrophic forgetting is the standard failure mode
when new learning overwrites old associations rather than coexisting
with them; current mitigation work frames it as a stability-plasticity
trade-off, with reinforcement-weighted retention (preferentially
protecting high-value or frequently-revisited memories, not just recent
ones) as an active mitigation strategy ([Overcoming Catastrophic
Forgetting in Visual Continual Learning with Reinforcement Fine-Tuning,
2026](https://arxiv.org/html/2605.09640v1); [CORE: Mitigating
Catastrophic Forgetting through Cognitive
Replay](https://arxiv.org/html/2402.01348v2)).

**In this repo — a real asymmetry worth noticing:** `memory.py`'s
`EmbeddingStore` evicts strictly oldest-first (FIFO) once full —
a sound heard once years ago is evicted exactly as readily as one from
yesterday, with zero regard for how often it recurred. `comms.py`'s
`Repertoire`, by contrast, already has reinforcement-weighted retention
per call-context (that's what `reinforce()`/`decay()` *are*). The two
modules this project ships side by side currently embody two different
answers to this exact research question.

**In this repo — tested:** `EmbeddingStore` now takes
`eviction="fifo"` (default, unchanged behaviour) or `"recency"` (evicts
whichever entry has gone longest without being any query's nearest
neighbour — so a sound that keeps recurring stays protected even if
it's never re-added, as long as something keeps recognising it).
`sim/memory_eviction_experiment.py` (`chuff-sim-memory`) runs both
policies against the identical sequence — a fixed rare sound revisited
every few rounds, competing with a stream of well-separated one-off
fillers for a small capacity — averaged over 10 seeds:

| capacity : revisit-gap ratio | fifo still-recognised-at-end | recency still-recognised-at-end |
|---|---|---|
| 3 : 10 (severe churn) | 90% of seeds | 100% of seeds |
| 4 : 6 (moderate) | 100% | 100%, but final novelty 0.005 vs 0.000 and revisit-recognition 57% vs 71% |
| 6 : 8 (mild) | 80% | 100%, revisit-recognition 78% vs 89% |

Recency-weighted eviction wins on every metric, every time, but the
*size* of the win depends heavily on how much the churn-to-capacity
ratio overwhelms it — under severe churn (9 fillers competing for 3
slots between each revisit), even recency-weighting can't fully protect
a rarely-revisited memory, because every single capacity slot gets
contested within a couple of filler additions. That's an honest finding
worth keeping, not a disappointing one: "better, but not immune" is the
actually-correct characterisation of what reinforcement-weighted
retention buys you here.

### 2.3 Functional Mirroring in Software — already true by construction

**Question:** Can the logic of mirror systems — where perception and
generation share one feature space — be translated into a lightweight
loop where an agent processes external sounds through its own
generation engine?

**What's known:** Computational mirror-neuron models show that
mirror-like units emerge "for free" when perception and action-
generation are trained to share one latent representation, rather than
running as two independent pipelines — the same code recognises and
produces ([Embodied Representation Alignment with Mirror Neurons,
2025](https://arxiv.org/pdf/2509.21136)).

**In this repo:** this one's largely already done, not open.
`sim/agent.py`'s `respond()` synthesises a call and immediately
re-runs it through the *exact same* `audio.extract_mfcc()` used for
everything it hears externally (cached as `_call_reference`), and
`comms.py`'s `closest_call()` operates in that one shared feature space
for both "what did I hear" and "what should I say" — there's no separate
perception-space/production-space split anywhere in this design.

**Follow-up, not built:** the two-unit simulation (1.3/3.2) is
positioned to test whether this shared-feature-space choice actually
*matters* for convergence, by comparing against a variant with
deliberately separate perception/production embeddings. Not done here —
flagged for later.

## Theme 3 — Decentralized Emergent Communication

### 3.1 Information Bottleneck & Signal Compression

**Question:** How does a noisy acoustic channel naturally select for
brevity, distinctiveness, and resistance to degradation?

**What's known:** Two findings in tension, both real: human languages
and emergent multi-agent protocols both appear to compress under an
information-bottleneck trade-off between informativeness and complexity
([Zaslavsky-style IB framing surveyed in recent emergent-communication
work](https://dl.acm.org/doi/pdf/10.5555/3545946.3598944)) — but once
channel *noise* specifically is modelled, agents in signaling-game
simulations have been found to add *redundancy* to messages rather than
simply shortening them, when shortening alone would hurt task success
under noise ([Emergent Communication: Generalization and Overfitting in
Lewis Games, 2022](https://arxiv.org/pdf/2209.15342)).

**In this repo:** `shared/comms_protocol.md`'s "calls become shorter
and more distinctive under a lossy channel" claim is only the
compression half of this story — it doesn't yet account for the
competing redundancy-under-noise finding, which is a real gap in the
design rationale, not just an omission in the prose.

**Near-term experiment — currently blocked by a real design gap:**
`comms.py`'s `Repertoire` only varies *which* fixed-duration,
fixed-frequency call gets picked — a call's own duration/bandwidth isn't
a learnable parameter. Testing compression-vs-redundancy properly needs
that to change first (new backlog item, not yet in `TASKS.md` — adding
it below).

### 3.2 Repertoire Convergence Dynamics — answered in software, here

**Question:** How long does it take two uncoordinated nodes, starting
with randomised call repertoires, to converge on stable shared
conventions via mutual reinforcement and decay?

**What's known:** This is the textbook question Lewis signaling games
were built to study — two agents, no shared dictionary, convergence
through repeated cooperative interaction and reinforcement (surveyed
in [Emergent Communication: Generalization and Overfitting in Lewis
Games, 2022](https://arxiv.org/pdf/2209.15342), among much other
signaling-game literature).

**In this repo — tested:** the same `two_unit_demo.py` run reported
under 1.3 answers this directly for Chuff's specific mechanism, not
signaling games in the abstract: under the `social` condition, with
these thresholds (`config.RECOGNITION_THRESHOLD = 0.2`,
`REINFORCEMENT_RATE = 0.1`) and this channel (`CHANNEL_ATTENUATION =
0.5`, white-noise-only room), convergence to a stable call choice per
sound happened within roughly 5 rounds, and held for the rest of a
60-round run. That number is specific to this toy acoustic world, not a
general claim — but it's a real, re-runnable measurement
(`uv run chuff-sim-2unit --log out.jsonl`), not a guess.

**Natural next step:** sweep `--attenuation` and the reinforcement/decay
rates across multiple seeds to see how convergence time actually
depends on channel quality and learning rate, rather than reporting one
seed's result as if it were the whole answer.

### 3.3 Contextual Grounding Without Semantics

**Question:** How do signals acquire functional meaning purely through
state-action-acknowledgement loops, independent of any human-defined
semantic dictionary?

**What's known:** This is the classical signaling-game/convention
formation framing (Lewis, *Convention*, 1969; Skyrms, *Signals*, 2010):
meaning is *defined* as whatever mapping a state-action-acknowledgement
loop stabilises on, with no pre-existing shared reference required —
it's the theoretical foundation `shared/comms_protocol.md`'s design
already rests on, whether or not it cites it.

**In this repo:** `comms.py` has no semantic layer anywhere by design —
`reinforce()`/`decay()` operate purely on (call, context-embedding)
pairs, never on a label or meaning. `two_unit_demo.py` makes this
concrete: Unit B never learns *what* `doorbell` means, only that a
particular acoustic pattern recurs and is worth acknowledging.

**In this repo — tested (CH-54, done):** `grounding_demo.py`
(`chuff-sim-grounding`) makes Unit B a full second `ChuffAgent`, not a
passive `Listener` — both units independently hear every sound (through
their own noisy mic), both speak and both acknowledge, both
reinforce/decay their own choice. After a run, for each of the 6 sound
labels: has each unit settled on one call (purity of its last 30% of
choices), and — the actual grounding question — do *different* labels
map to *different* (A's call, B's call) pairs?

**Result — corrected after a proper sweep, not the first read:** an
initial 3 runs (2 seeds + a channel cutoff) all happened to produce the
same split and were written up here as "it's the same partition every
time." A 10-seed sweep (seeds 0-9, 150 rounds each) shows that was
wrong — it isn't the same partition, but two things *about* it are
reliably true:

- **Always exactly 2 distinct (A-call, B-call) pairs**, and **always
  100% purity** on every label, every seed — no indecision anywhere,
  every run.
- **The majority pair (capturing 4-5 of the 6 labels) is always call
  `#0`/`#0`** — `Repertoire`'s untrained index-0 tie-break has a
  structural head start that nothing in this run ever overcomes for
  *most* labels.
- **Which 1-2 labels break away from that majority varies by seed**:
  `{kettle_whistle, bird_call}` in 3/10 seeds, `{car_horn, dog_bark}` in
  4/10, `{phone_ring}` alone in 3/10. Three different minority groups,
  no consistent acoustic theme across them (the first pairing is
  high-frequency; the second is the two *lowest*-frequency sounds;
  the third is a single mid-frequency tone).

**Reading:** the earlier "acoustically coherent high/low split" claim
doesn't survive more seeds — what's actually stable is a path-dependent
dynamic, not a fixed category boundary. Call `#0`'s structural
head-start means whichever label or two happens to arrive in a way that
falls outside its relevance-gate radius *first* claims a different call
and keeps it (100% purity once settled); everything else piles onto
`#0` regardless of its own acoustic character. This is still
coarse-but-real discrimination (2 distinct, perfectly stable
conventions every time, not indiscriminate acking) — it's just driven
by presentation-order lock-in rather than the sounds' own acoustic
properties, which is a materially different, and more honest, story
than the first write-up told.

**Why, mechanistically — tested, hypothesis refuted (CH-63, done):**
the obvious guess was the relevance gate (`RELEVANCE_SIM_THRESHOLD =
0.5`, now in `config.py` and swept via a constructor override rather
than monkeypatching a module constant — a small refactor this required).
`grounding_demo.py --sweep-thresholds` tried 0.3, 0.5, 0.7, and
**0.9** (near-impossible to clear — almost no exemplar would count as
"relevant" to a different context) across 10 seeds each:

| threshold | avg distinct pairs | avg purity |
|---|---|---|
| 0.30 | 2.0/6 | 100% |
| 0.50 | 2.0/6 | 100% |
| 0.70 | 2.1/6 | 100% |
| 0.90 | 2.0/6 | 100% |

**No effect, at any threshold tried.** This refutes the relevance-gate
hypothesis cleanly rather than confirming it — worth stating plainly,
not quietly dropping the question once the first guess didn't pan out.
The real mechanism has to be `_base_default()`'s index-based tie-break:
even at threshold 0.9, once call `#0` picks up *any* exemplar, a brand
new context that fails to clear that strict a gate gets scored purely
on `_base_default()` for every call — and call `#0` still wins that
contest by construction, regardless of relevance, because nothing else
has an exemplar either. The gate was never the thing keeping `#0`
dominant; the unconditional index-ordering fallback was. Follow-up,
correctly targeted this time, tracked as **CH-64**: test whether
replacing `_base_default()`'s index ramp with a fixed-at-construction
*random* per-call tie-break (still deterministic across repeated calls,
just not correlated with index) produces finer-grained grounding.

---

## Status of the backlog items this raised (see `TASKS.md`)

- **CH-23a** `done` (software only): two-unit convergence, `two_unit_demo.py`.
- **CH-50** `done`: frequency-dependent channel attenuation in
  `world.py`, tested in `two_unit_demo.py --cutoff`/`--sweep-seeds`
  (Theme 1.2). Found and fixed two real bugs along the way (see above).
- **CH-52** `done`: recency-weighted eviction policy for
  `EmbeddingStore`, compared against FIFO in
  `memory_eviction_experiment.py` (Theme 2.2).
- **CH-51** `todo`, not attempted this round: learnable call
  duration/bandwidth in `Repertoire` — needed before Theme 3.1's
  compression-vs-redundancy question is even testable. More invasive
  than the others (extends the graded `Repertoire` contract), so left
  for a dedicated pass.
- **CH-53** `todo`, not attempted this round: babbling warm-up phase.
  Worth flagging *why* it wasn't trivial: in the current architecture,
  `ChuffAgent.respond()` already lazily caches a call's self-reference
  the first time it's used, in the same round it's needed — so naively
  adding a "babble first" phase wouldn't change anything observable
  unless babbling also seeds a *self-directed* exemplar (reinforcing
  each call against its own clean signature before any real sound
  arrives). That's a real design decision, not a one-line add — worth
  doing properly rather than rushing to get a number.
- **CH-54** `todo`, not attempted this round: comparing whether A and B
  converge on the *same* call-to-sound mapping needs Unit B to be a full
  second `ChuffAgent` with its own repertoire, not just a `Listener` —
  a bigger change to `two_unit_demo.py`'s symmetry than the others.

## Related work: clebran-arc-agi

A separate project on this machine (`C:\Users\aidan\repos\clebran-arc-agi`,
not part of this repo) is worth drawing on directly. It evolves typed
programs over a grid DSL ([michaelhodel/arc-dsl](https://github.com/michaelhodel/arc-dsl))
to solve ARC-AGI tasks, with multiple populations ("lineages") that
diverge into different strategies and gossip fragments of what works to
each other over a deliberately narrow channel — structurally close to
Chuff's own comms track, even though the domain (abstract grid
reasoning vs. acoustic signaling) is completely different. Two pieces
transfer close to directly; one finding is a genuine caution.

### 1. The gossip channel's design (directly reusable pattern)

`clebran/channel.py` is a blackboard: lineages `publish()` scored
messages each epoch, a channel variant (`ChNone` / `ChConstrained(K, M)`
/ `ChUnconstrained`) admits a subset, other lineages `receive()` them.
`ChConstrained` caps bandwidth two ways at once — message *complexity*
(statement count for code, demo-pair count for a teaching task — not
raw bytes) and message *count* per sender per epoch — then keeps the
top-scoring messages that fit. `two_unit_demo.py`'s single fixed
acknowledgement tone is the degenerate case of this (unconstrained
content, one bit of information, capacity 1). If Chuff ever moves past
a single ack tone — e.g. richer acknowledgement payloads, or more than
two units — this admission-by-score-under-a-cap pattern is the one to
reuse, not reinvent.

### 2. The B5 central-library result and its Stitch-style miner (adapted: CH-60/61)

clebran's headline finding: at matched compute, **a single learner that
mines its own solved programs into a growing reusable abstraction
library (B5) beats every multi-lineage gossip arm**, 7-10x more
cost-efficient on top of solving more tasks. `dsl/abstraction.py`'s
miner is how: find recurring dependency-closed program slices across
many solved programs, group by structural skeleton, antiunify each
group (a parameter stays an inline constant *only* if it's identical
across every occurrence, otherwise it's lifted to an argument — minimal-
arity generalisation), score by compression utility
(`occurrences × (size − 1) − size − arity_penalty`), keep the best.

**Built here, adapted to Chuff's scale (CH-60, done):**
`python/src/chuff_mpu/sim/call_dsl.py` gives repertoire calls actual
compositional structure — a `CallProgram` is one generator op
(`tone`/`chirp`) plus 0-3 modifier ops (`am`, `envelope`), with
`mutate_const`/`swap`/`grow`/`delete` and `crossover` mirroring
clebran's genome vocabulary at a much smaller scale. `chuff-sim-dsl`
(`dsl_evolution_demo.py`) evolves a population toward a target sound's
MFCC via cosine-similarity fitness — the actual search loop, not just
the representation.

**What running it found, honestly — a real methodological snag, not a
clean win:** both a pure-tone target (`kettle_whistle`, 2600 Hz) and a
wide-chirp target (`dog_bark`, 350→520 Hz) converge to ~0.97-0.98
fitness within 15-20 generations — but neither converges to the
"intended" generator. The tone target gets matched by a *near-zero-width
chirp* (2576→2529 Hz — functionally a tone, structurally a chirp); the
chirp target gets matched by a *tone with three layered AM modifiers*.
The MFCC-cosine fitness metric (same convention as `memory.py`'s
novelty scoring, deliberately) is evidently not very discriminating
between "the right spectral shape via the right primitive" and "a
structural shortcut that happens to produce a similar bulk spectral
envelope." **This directly blocks CH-61 (mining recurring motifs across
evolved programs) until it's addressed** — mining structurally-diverse
"successful" programs whose success is itself a fitness-metric artifact
would mine artifacts, not reusable acoustic building blocks. Needs
either a stricter/multi-target fitness (reward matching across several
contexts at once, not cosine similarity to one target) or an explicit
structural-match term, before CH-61 is worth running.

### 3. A caution for the two-unit track (CH-62, decision)

The B5-beats-gossip result isn't a side note — it's the paper's central,
hard-won finding, replicated across many control conditions:
**"communication is a transport mechanism, not a discovery mechanism"**
in that domain. Translated: don't expect `two_unit_demo.py`'s two-unit
setup to make either Chuff *learn calls faster or better* than a single,
well-tuned Chuff mining its own memory would. That's consistent with
what the two-unit demo already showed under the `social` condition — it
produced a *stable shared convention*, not demonstrably *better*
recognition than a single unit would achieve alone. The two-unit
track's real value is the convention-formation/signaling phenomenon
itself (Theme 3.3's grounding question — does a *shared* mapping emerge,
not does anyone get smarter), and it should stay framed that way rather
than drifting into an implicit "two units learn better than one" claim
the evidence (here and in clebran) doesn't support.

### What doesn't transfer

ARC's substrate is a large, structured, typed program space over 160
hand-designed primitives, searched for exact symbolic correctness
against held-out test grids. Chuff's acoustic domain is low-dimensional,
continuous, and success is "close enough in MFCC space," not "bit-exact."
Lineage/population mechanics (hall-of-fame restarts, per-task RNG
substreams, progress-weighted scheduling) solve problems specific to
that search's scale and are not worth porting at Chuff's current scale.
