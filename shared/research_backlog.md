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

**Near-term experiment:** `sim/`'s `ChuffAgent` is purely reactive —
it only ever responds to something it just heard, with no
babbling-before-listening phase. A direct test of the PNAS finding: add
an optional "babbling" warm-up to `run_demo.py` (play random repertoire
calls with no external trigger, before any real sounds arrive) and
compare how quickly/cleanly the repertoire specialises afterward against
a cold start with no babbling phase.

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

**In this repo:** `dream.py`'s planned
`surprise_score = cosine_distance(observed, predicted)` *is* this
framework, reduced to one line — but `dream.py` is still a stub
(`TASKS.md` CH-33).

**Near-term experiment:** Implement `predict_next()` /
`surprise_score()` for real (same cosine-distance convention already
validated for `memory.py`), wire it into a minimal mood transition, and
check — by ear, watching `sim/`'s logs — whether a human would actually
read the resulting state change as "surprised" vs. just "a number
changed." That's the real test of whether this one-line
operationalisation earns the word "affect" or not.

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

**Near-term experiment:** now that two independent repertoires exist
side by side (`two_unit_demo.py`), a genuinely answerable question
opens up that wasn't answerable with one agent alone: does Unit A's
call-to-context mapping converge toward the *same* association Unit B
independently forms for recognising it, or do the two settle into
self-consistent-but-different conventions — each internally stable, but
not provably "about" the same thing from outside? Logging both units'
full call-index-per-label history (not just A's, which
`two_unit_demo.py` already logs) and comparing them after a long run
would make "grounding" here something measured, not assumed.

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
