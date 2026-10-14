# Chuff: A Simulation-Based Pilot Study of Acoustic Novelty Memory, Affect, and Emergent Signaling in a Low-Cost Robotic Pet

**Aidan Jones**, with Claude (Sonnet 5) as coding/research assistant
October 2026

## Abstract

Chuff is a small robotic pet intended to learn from its acoustic environment, express an internal affective state through light and animation, and — in a second phase — develop a shared signaling convention with a second unit over an acoustic-only channel. This paper reports a simulation-based pilot study conducted before any of the target hardware (an ESP32-S3 board and an Arduino UNO Q) was fully wired up for audio. We built a minimal but complete software pipeline — MFCC feature extraction, a cosine-distance k-nearest-neighbour novelty store, a reinforcement-learned call repertoire, a predictive "dream loop," and a nine-state mood machine — and used a synthetic acoustic world (tones, chirps, and noise) to run eight targeted experiments against specific, literature-grounded hypotheses. Three results confirmed their hypotheses cleanly: a contingent-acknowledgement condition reproduces the "social gating" prediction from infant language acquisition research, a self-directed "babbling" phase measurably improves an agent's first real response, and a band-limited channel measurably favours lower call frequencies, consistent with the Acoustic Adaptation Hypothesis. Two results were negative, and are reported as such: a relevance-threshold sweep across four orders of magnitude had no effect on how finely two agents' signaling conventions discriminated between sounds, refuting our mechanistic hypothesis; and an evolutionary search over a small compositional call-construction DSL reached high fitness through acoustically-plausible but structurally wrong solutions, exposing a degeneracy in the fitness metric itself. One finding was corrected mid-study: an initial claim that two agents converge on a fixed, acoustically-coherent partition of sounds did not survive a wider seed sweep, and the corrected, more interesting story is that the partition is stable in *shape* but not in *membership* — a path-dependent, not acoustically-determined, phenomenon. We also report a structural comparison with a separate research codebase (`clebran-arc-agi`) that evolves programs to solve abstract reasoning tasks under a similar gossip-channel design, whose headline finding — that a single learner mining its own solutions beats multi-agent sharing at matched compute — is a direct caution against over-claiming what Chuff's planned two-unit experiment can show. We close with a concrete, falsifiable research agenda for the next phase of work.

## 1. Introduction

Robotic pets are usually built to perform affect, not to generate it from an internal process. Chuff instead specifies the problem as a small, measurable agent-design question: what is the *minimum* functional loop — listen, remember, repeat — that produces behaviour an observer can read as learning, novelty-seeking, and signaling? That design brief forces engineering choices that are also, incidentally, testable cognitive-science hypotheses: how should memory decide what to keep; what should count as "surprise"; does a second interlocutor help an agent learn or merely let it broadcast; and does a signaling convention between two independent learners emerge from reinforcement alone, without a shared dictionary?

The practical starting point was unglamorous. The project's intended hardware — an ESP32-S3 board with a microphone and speaker for a single-unit "Track A", and an Arduino UNO Q for a more capable "Track B" — was available but not yet wired for audio capture. Rather than block on hardware, we built the full software pipeline (feature extraction, memory, a signaling repertoire, output safety limiting, mood, and a dream loop) against a synthetic world of tones, chirps, and noise, and used it to run a sequence of small, pre-registered-in-spirit experiments, each targeting one specific question with a numerical answer rather than a demonstration. This paper reports what we found, including the parts that did not go as expected, and the one place where an initial finding did not survive a wider check.

The contributions are:

1. A validated software pipeline — real MFCC extraction, a cosine-distance KNN novelty store, a repertoire of calls with reinforcement/decay learning, an output-safety limiter, a predictive dream loop, and a mood state machine — exercised end-to-end against synthesised audio, not unit-tested in isolation.
2. Eight targeted simulation experiments, each addressing one hypothesis drawn from a specific literature, with multi-seed statistics rather than single-run anecdotes (Section 5).
3. An honest account of a corrected claim (Section 5.6) and a refuted hypothesis (Section 5.7) — reported as findings in their own right, not edited out.
4. A structural comparison with an independent, much larger research codebase for evolutionary program synthesis (`clebran-arc-agi`), identifying what transfers to a continuous acoustic domain and what does not, and a caution the comparison raises for how Chuff's two-unit experiment should be interpreted (Section 6).
5. A grader/prompt harness for scoring a coding agent's attempt at reimplementing the memory and signaling modules, used throughout to validate that our own implementations satisfy the same specification we would hold an external contributor to (Section 3.3).

## 2. Background and Related Work

### 2.1 The sensorimotor bottleneck in vocal learning

The textbook account of vocal learning separates a passive "sensory phase," in which a young songbird or infant forms an auditory template of a model's sound, from a later "sensorimotor phase," in which the learner practises against that template. Recent work complicates the ordering: in zebra finches, active babbling is necessary to *open* the sensory phase in the first place, rather than simply exploiting a template already formed (Leitão & Gahr, 2024). The auditory template itself is explicitly understood as the compressed representation bridging a continuous acoustic signal and a discrete motor program (Yazaki-Sugiyama, 2024).

### 2.2 Social gating of vocal learning

A separate body of work asks not *whether* a learner is exposed to a model sound, but *under what social conditions* exposure becomes learning. Goldstein, King, and West (2003) showed that contingent, responsive caregiver feedback to infant babbling accelerates and matures vocal behaviour in a way that non-contingent exposure to the same acoustic content does not; a companion commentary draws the same contingency effect out in zebra finch tutoring, where live, responsive tutors outperform passive playback of an identical song (Kuhl, 2003). This "social gating" claim — that passive exposure is a weaker teacher than contingent interaction, holding content fixed — is directly testable in an artificial agent and is one of the two hypotheses we could test computationally rather than only cite (Section 5.2).

### 2.3 The Acoustic Adaptation Hypothesis

The Acoustic Adaptation Hypothesis (AAH) predicts that habitats with more reverberation and higher-frequency attenuation should select for lower-frequency, narrower-bandwidth calls, since high frequencies and rapid repetition degrade faster in transmission. A meta-analysis of the empirical bird-song literature found the evidence genuinely mixed rather than settled (Boncoraglio & Saino, 2007) — a useful calibration, since it means a simulation result consistent with AAH should be read as "consistent with one real hypothesis in an active literature," not as closing a solved question.

### 2.4 Predictive coding and the free-energy framing of affect

Under the free-energy principle, a system is understood to maintain a generative model of its environment and to experience "surprise" as a quantity proportional to how unlikely its current sensory input is under that model; sustained, unresolved prediction error is associated with negative affective states, while accurate prediction is associated with calm (Friston, 2010). This is a strong, specific empirical claim about the *mechanism* of affect, not just a metaphor, and it is directly operationalisable: an agent that interpolates between remembered experiences to form a prediction, then scores the distance between that prediction and what actually happens next, is implementing exactly this framing in miniature (Section 5.8).

### 2.5 Catastrophic forgetting and reinforcement-weighted retention

Bounded-capacity memory systems face a stability-plasticity trade-off: new learning that simply overwrites old associations produces catastrophic forgetting, a long-studied failure mode in continual learning. Recent mitigation strategies converge on reinforcement-weighted retention — protecting memories in proportion to how often or how recently they have proven useful, rather than evicting strictly by age (Zhang, Fu, Peng, Yao, & He, 2024; Lou, Guo, Chen, & Yu, 2026). This motivates a direct empirical comparison between a naive first-in-first-out eviction policy and a recency-of-relevance-weighted one in exactly the kind of small KNN memory Chuff uses (Section 5.5).

### 2.6 Shared perception-action representations

Computational models of mirror systems show that units with mirror-like properties — responding similarly whether an action is perceived or produced — emerge without being explicitly engineered, provided perception and production are trained to share one latent representation rather than running through two independent pipelines (Zhu et al., 2025). This is a design constraint we adopted from the outset rather than tested after the fact: an agent's self-produced call and an externally-heard sound are compared in the exact same feature space, with the same extraction function (Section 3.1).

### 2.7 Signaling games and the information bottleneck

The classical account of how signals acquire meaning without a pre-existing shared dictionary comes from the philosophy of convention: meaning is *defined* as whatever mapping a repeated state-action-acknowledgement loop stabilises on (Lewis, 1969; Skyrms, 2010). Contemporary multi-agent emergent-communication work operationalises the same idea computationally and additionally shows that channel noise, specifically, pushes agents toward adding *redundancy* to their signals rather than simply shortening them (Rita et al., 2022), in tension with the more familiar claim that languages compress to a point of maximal informativeness for minimal complexity (Zaslavsky, Kemp, Regier, & Tishby, 2018). Chuff's `comms_protocol.md` design document had, before this study, only cited the compression half of this literature; Section 5.1's discussion notes the gap this exposed.

### 2.8 Program synthesis, library learning, and a structurally adjacent project

A separate, substantially larger research codebase on this machine, `clebran-arc-agi` (Jones, 2026, unpublished), evolves typed linear-IR programs over a hand-designed grid DSL (Hodel, n.d.) to solve tasks from the Abstraction and Reasoning Corpus, running multiple divergent populations ("lineages") that gossip fragments of working code to each other over a bandwidth-constrained channel. Its headline, extensively controlled result is that a single population mining its own solved programs into a growing reusable abstraction library — in the spirit of DreamCoder's wake-sleep library learning (Ellis et al., 2021) — significantly outperforms every multi-lineage gossip variant at matched compute. Despite the completely different domain (discrete, typed, symbolic program search versus continuous, low-dimensional acoustic signaling), the system shares Chuff's basic shape — independent learners, a constrained communication channel, reinforcement of what works — closely enough that we treat it as a direct source of transferable design ideas and cautions, detailed in Section 6.

### 2.9 Acoustic features

All novelty, recognition, and signaling decisions in this work operate on Mel-frequency cepstral coefficients (MFCCs), the standard compact representation of short-time spectral envelope introduced for speech recognition by Davis and Mermelstein (1980). We depart from the textbook recipe in one respect, discussed in Section 3.1: we discard the zeroth coefficient, which encodes overall log-energy rather than spectral shape.

## 3. System Architecture

### 3.1 The core pipeline

Chuff's software (in the `python/src/chuff_mpu` package) implements, for a single unit:

- **`audio.py`** extracts a 13-dimensional MFCC vector from a raw audio clip via `librosa`. The zeroth coefficient, which is dominated by overall loudness, is discarded; retaining it was found empirically to make every clip look alike under the chosen similarity metric regardless of pitch or timbre, because a change in gain shifts all mel-band log-energies by the same additive constant, which loads almost entirely onto the zeroth DCT coefficient.
- **`memory.py`** implements `EmbeddingStore`, a bounded-capacity KNN store over these vectors, using cosine distance (not Euclidean) for consistency with the free-energy-style surprise scoring in `dream.py`. Eviction policy is a first-class, swappable parameter (`"fifo"` or `"recency"`; Section 5.5).
- **`comms.py`** implements `Repertoire`, a single unit's bank of call frequencies plus a per-call, per-context reinforcement-learned weighting (`reinforce()`/`decay()`), gated by a cosine-similarity relevance threshold that determines whether learning for one context is allowed to influence the score for a different one (Section 5.7).
- **`audio_safety.py`** is a mandatory choke point every synthesised waveform passes through before being treated as "what gets played": non-finite samples are sanitised, clip length is capped, and a soft-knee limiter followed by a hard peak ceiling guarantees no output exceeds a configured safe amplitude, regardless of what produced it. This was treated as a hardware-safety requirement — a literal "never blow the speakers" guardrail — from the project's outset, not added after the fact.
- **`mood.py`** implements a nine-state mood machine (`Mood`/`MoodState`/`update_mood()`), with an explicit precedence order over its possible triggering signals (Section 5.8).
- **`dream.py`** implements the predictive loop: `predict_next()` interpolates two stored embeddings to form an expectation, and `surprise_score()` scores the cosine distance between that expectation and whatever is observed next — the Friston (2010) framing, implemented directly (Section 5.8).

Because perception and production both route through the same `extract_mfcc()` function with no separate encoder for each, the shared-representation property discussed in Section 2.6 holds by construction rather than needing to be separately engineered or verified.

### 3.2 The synthetic world

All experiments in this paper run against `sim/world.py`'s `Environment`: a library of six distinct synthetic sounds (four pure tones and two linear-frequency chirps, spanning roughly 350–3200 Hz) presented with small frequency jitter so that repeated presentations of "the same sound" are acoustically similar but not identical, mixed with Gaussian background noise standing in for an untreated room. A later addition, `Environment.transmit()`, adds a fourth-order Butterworth low-pass filter and configurable amplitude attenuation standing in for distance and reverberation, used in the Acoustic Adaptation Hypothesis test (Section 5.4).

No field recordings, real microphones, or real speakers are used anywhere in this study; this is a stated limitation (Section 8), not an oversight.

### 3.3 An evaluation harness as a secondary contribution

Because the memory and repertoire modules are intended to be reimplemented identically on resource-constrained hardware, we built a small, self-contained evaluation harness (`eval/`): a natural-language task prompt describing the required behaviour of `EmbeddingStore` and `Repertoire`, and an automated grader that scores any candidate implementation against a battery of behavioural checks (recognition accuracy over a synthetic sequence of repeated and novel sounds; capacity enforcement; determinism; whether reinforcement and decay behave as specified; whether learning for one context leaks into another). The grader was itself validated against a deliberately unimplemented stub (score 0.0) and a known-correct reference solution (score 1.0) before being trusted, and the same grader continues to score the live production implementation at 1.0 throughout this study, including after every refactor reported below.

## 4. Methods

Each experiment below is a standalone script under `python/src/chuff_mpu/sim/`, installed as a `uv`-managed console command, and is independently re-runnable with a specified seed or seed range. Where a result is reported from a single seed, this is stated explicitly and treated as preliminary; the default standard throughout this study, adopted after Section 5.6 below, is a multi-seed sweep (typically 10–40 seeds) with the full range reported, not only the mean. All thresholds (novelty, recognition, reinforcement and decay rates, relevance gate, channel parameters) are named constants in a single `config.py` module, never hard-coded at point of use, so that every reported number is traceable to one place in the codebase.

## 5. Experiments and Results

### 5.1 Single-unit listen-remember-repeat loop and the output-safety guardrail

Before any multi-unit or hypothesis-driven experiment, `sim/run_demo.py` exercises the full single-agent loop: hear a synthesised sound, score its novelty, commit it to memory if sufficiently novel, select the closest-matching call from the repertoire, synthesise and safety-limit that call, and finally "hear" the agent's own played-back call mixed with fresh noise to check self-recognition. Before running any rounds, the script feeds the safety limiter a deliberately pathological signal — fifty times full-scale amplitude, with injected NaN and Inf samples — and asserts that the output is fully finite and within the configured peak-amplitude ceiling. This proved the guardrail holds under conditions well beyond anything the rest of the system would ever produce on its own, rather than merely asserting it by inspection.

### 5.2 Social gating: does signaling convergence require contingent acknowledgement?

**Hypothesis** (Section 2.2): a receiving unit that acknowledges a sender's call only when it recognises it (a *social* condition) should produce a stable, convergent signaling convention; a receiving unit that never acknowledges, regardless of what it hears (a *passive* condition, holding the acoustic content of what the sender says identical across conditions), should not.

**Method:** `sim/two_unit_demo.py` implements a one-directional loop — Unit A hears a sound, selects a call, and "speaks" it over a simulated channel; Unit B either acknowledges contingently (`social`) or never (`passive`); A reinforces or decays its own choice based on whether it heard an acknowledgement back.

**Result** (seed 5, 60 rounds): under `social`, the acknowledgement rate rises from 0.90 (first half of the run) to 1.00 (second half), and A's call choice for a given sound converges within approximately five rounds and remains stable thereafter. Under `passive`, the acknowledgement rate is 0.00 throughout by construction, and A's call choice never stabilises, switching nearly every time the same sound recurs even late in a 60-round run. This reproduces the social-gating prediction computationally: holding the acoustic content fixed and varying only whether feedback is contingent on recognition is sufficient, in this minimal system, to produce the entire difference between convergence and perpetual instability.

We note explicitly that this demonstrates the *mechanism* in `comms.py`'s reinforcement/decay design is sufficient to reproduce the qualitative phenomenon reported in real infants and zebra finches; it is not evidence about the mechanism operating in those organisms.

### 5.3 Repertoire convergence dynamics

The same run reported in Section 5.2 directly answers a related but distinct question raised by signaling-game theory (Section 2.7): starting from a randomised repertoire with no shared dictionary, how quickly does a stable convention emerge under mutual reinforcement? For this specific acoustic world, channel, and parameterisation, convergence to a stable per-sound call choice took approximately five rounds and held for the remainder of a 60-round run — a concrete, re-runnable number for this system, not a general claim about signaling games.

### 5.4 The Acoustic Adaptation Hypothesis

**Hypothesis** (Section 2.3): a band-limited, lossy channel should favour lower-frequency calls among those that survive reinforcement.

**Method:** `Environment.transmit()` applies a fourth-order Butterworth low-pass filter before mixing in channel noise; `two_unit_demo.py --cutoff` runs the full two-unit loop through it, and `--sweep-seeds` averages a no-cutoff-versus-900 Hz-cutoff comparison over many seeds.

**Result** (15 seeds, 60 rounds each): with no cutoff, the mean frequency of reinforced (successfully acknowledged) calls across seeds was 1903 Hz, with a second-half acknowledgement rate of 0.99; with a 900 Hz cutoff, the mean frequency dropped to 1483 Hz, with the acknowledgement rate also falling to 0.86. A band-limited channel measurably pushes the surviving repertoire toward lower frequencies *and* makes overall convergence less reliable — both consistent with the Acoustic Adaptation Hypothesis in this toy channel model.

Two methodological obstacles were encountered and are reported because they are the actual lesson of this experiment. First, a single one-pole low-pass filter (6 dB/octave roll-off) was measured directly and found far too gentle — even a 3000 Hz tone through a 900 Hz cutoff lost only around 10 dB, nowhere near enough to sink it below the ambient noise floor — and was replaced with a fourth-order Butterworth filter (roughly −46 dB at the same frequency/cutoff pair) only after this direct measurement, not by assumption. Second, the very first filtered run showed no effect whatsoever, traced to an unrelated confound: `Repertoire`'s constructor had been sorting its randomly generated call frequencies, so the lowest-index call was always also the lowest-frequency call, and the untrained tie-break that favours low indices was therefore *already* maximally biased toward low frequencies independent of any channel property. Removing the sort (and documenting why, in the source) was necessary before the intended effect could be observed at all.

### 5.5 Memory retention policy: FIFO versus recency-weighted eviction

**Hypothesis** (Section 2.5): reinforcement-weighted retention should better protect a rarely-but-regularly-recurring memory from capacity pressure than naive first-in-first-out eviction.

**Method:** `EmbeddingStore` was extended with a swappable eviction policy (`"fifo"`, unchanged default behaviour, versus `"recency"`, which evicts whichever entry has gone longest without being any query's nearest neighbour, whether or not that query resulted in a fresh addition). `sim/memory_eviction_experiment.py` presents a fixed rare sound, revisited at a regular interval, competing against a stream of well-separated one-off filler sounds for a small capacity, at three capacity-to-revisit-interval ratios, each averaged over ten seeds.

**Result:**

| capacity : revisit-gap ratio | FIFO still-recognised-at-end | Recency still-recognised-at-end |
|---|---|---|
| 3 : 10 (severe churn) | 90% of seeds | 100% of seeds |
| 4 : 6 (moderate churn) | 100% of seeds | 100% of seeds, but final novelty 0.005 vs. 0.000, and revisit-recognition 57% vs. 71% |
| 6 : 8 (mild churn) | 80% of seeds | 100% of seeds, revisit-recognition 78% vs. 89% |

Recency-weighted retention outperforms FIFO on every metric at every churn ratio tested, but the margin of improvement shrinks considerably as churn intensifies: under the most severe ratio tested, even recency-weighting cannot fully protect a rarely-revisited memory, because every capacity slot is contested within a handful of filler additions regardless of retention policy. We report this as "better, but not immune" rather than as an unqualified win, since the qualification is itself informative about where reinforcement-weighted retention's benefit saturates.

### 5.6 Grounding between two independently-learning units — a corrected finding

**Question** (Section 2.7): do two independently-learning units converge not merely on *a* stable convention each, but on conventions that *discriminate between different sounds* — one mutually-recognised pairing per sound, rather than all sounds collapsing onto one pairing?

**Method:** `sim/grounding_demo.py` extends the one-directional design of Section 5.2 to be fully symmetric: both units are complete agents, each with independent memory and repertoire, each independently hearing every presented sound through its own simulated microphone position, each speaking and acknowledging in both directions, each reinforcing or decaying its own call choice based on whether it was acknowledged. After a run, for each of the six sound categories, we record each unit's majority call choice over the final 30% of that sound's occurrences (its "purity"), and count how many *distinct* (A's call, B's call) pairs exist across all six sounds.

**Initial result and its correction:** the first three runs attempted (two seeds, plus one run with a 900 Hz channel cutoff) all happened to produce the same two-way split of the six sounds — the two highest-frequency sounds sharing one convention, the remaining four sharing another — and were reported, provisionally, as evidence of a fixed, acoustically-coherent partition. A subsequent ten-seed sweep (seeds 0–9, 150 rounds each) showed this provisional claim was wrong. What is actually stable across all ten seeds is:

- exactly two distinct (A-call, B-call) pairs emerge, every time, with 100% purity on every sound in every seed — no run ever left a sound undecided;
- the majority pairing, capturing four or five of the six sounds, is in every seed the pair of calls at index zero for both units — `Repertoire`'s untrained tie-break structurally favours the lowest call index before any learning occurs, and this structural edge is never overcome for most sounds;
- but *which* one or two sounds break away from that majority pairing varies by seed, with no consistent acoustic theme across the different minority groupings observed (one seed's minority pairing was the two highest-frequency sounds; another's was the two lowest-frequency sounds; a third's was a single mid-frequency tone, alone).

We revise our interpretation accordingly: the system exhibits coarse-but-genuine discrimination (two distinct, perfectly stable conventions, not indiscriminate acknowledgement of anything), but the mechanism is presentation-order lock-in around a structural tie-break, not a category boundary determined by the sounds' own acoustic properties. We record this correction explicitly, including the now-superseded initial claim, because the methodological lesson — that two or three runs can look like a fixed pattern purely by chance, and that the correct response on noticing this is to run a wider sweep rather than to retain the more publishable-sounding first draft — is at least as important as the corrected result itself.

### 5.7 Testing the mechanism: a relevance-gate hypothesis, refuted

Section 5.6's corrected account proposes two candidate mechanisms for the coarse, majority-pairing-dominated discrimination observed: the relevance gate that determines whether learning for one acoustic context can influence scoring for another, and the index-based default tie-break that resolves ties when no call yet has a relevant exemplar for a given context.

**Hypothesis:** tightening the relevance gate — raising the cosine-similarity threshold a learned exemplar must clear before it is allowed to generalise to a new context — should reduce how often one call's reinforcement for one sound "leaks" into dominating a different sound, and should therefore increase the number of distinct conventions that emerge.

**Method:** the relevance-gate and same-context-distance thresholds, previously hard-coded private constants in `comms.py`, were relocated to the project's single configuration module and exposed as an optional, backward-compatible constructor parameter on `Repertoire` (verified not to change default behaviour against the evaluation harness of Section 3.3, which continued to score 1.0). `grounding_demo.py --sweep-thresholds` then repeated the Section 5.6 experiment at relevance thresholds of 0.3, 0.5 (the original default), 0.7, and 0.9 — the last being near-impossible for an unrelated context to clear — across ten seeds each.

**Result:**

| relevance threshold | mean distinct pairs (of 6 possible) | mean purity |
|---|---|---|
| 0.30 | 2.0 | 100% |
| 0.50 | 2.0 | 100% |
| 0.70 | 2.1 | 100% |
| 0.90 | 2.0 | 100% |

The hypothesis is refuted cleanly: there is no detectable effect of the relevance threshold on the number of distinct conventions that emerge, across a four-fold sweep spanning an easily-cleared gate to a near-unreachable one. We report this as a negative result rather than silently abandoning the question once the first mechanistic guess failed to pan out, since a clean refutation is itself informative: it rules out the relevance gate and implicates the index-based tie-break as the dominant mechanism instead — even an extremely strict gate still leaves the lowest-index call winning any brand-new context by default, because without a qualifying exemplar for *any* call, the contest is decided purely by the tie-break, which is structurally insensitive to the gate's strictness. A correctly-targeted follow-up — replacing the index-ordered tie-break with a fixed-at-construction random one and re-running the same sweep — is proposed as future work (Section 9).

### 5.8 Predictive coding and mood: the dream loop, exercised under real idle time

**Hypothesis** (Section 2.4): an agent that forms a prediction by interpolating remembered experiences, and whose affective state depends on whether reality subsequently matches that prediction, should produce a surprise response that tracks genuine prediction error rather than a scripted event.

**Method:** `mood.py` implements a nine-state mood machine with an explicit precedence order (an acknowledgement just received takes priority over a concluded dream comparison, which takes priority over idle-time states, which take priority over the latest novelty score, which in the absence of any new signal simply holds the current state). `dream.py` implements interpolation and cosine-distance surprise scoring directly from Friston's (2010) framing. Because every other experiment in this study presents a sound on every simulated tick, idle time never genuinely accumulates in those settings; `sim/mood_demo.py` instead presents a sound only with low probability per tick, so that idle time crosses the dreaming threshold under realistic, unscripted conditions, at which point the agent occasionally interpolates a prediction and compares it against whatever sound the world presents next.

**Result** (five seeds, 300 ticks each, sound probability 0.03 so that idle gaps reliably cross the threshold): every one of the nine mood states fires from a genuine triggering condition across the five seeds, with one seed alone producing `contentment`, `new_sound_learned`, `song_recognised`, `dreaming`, and `surprise` within a single run. The dream loop fired between two and six times per 300-tick run depending on seed, and whether waking from a dream produced `surprise` or `contentment` varied run to run, exactly as it should if the mechanism is genuinely sensitive to whether the interpolated prediction happens to resemble the next real observation, rather than defaulting to one outcome. This is the specific test of whether the free-energy operationalisation "earns its name" mechanically: it does, in the sense that it produces a state tracking actual prediction error from real audio. Whether the resulting state changes would be *read* by a human observer as "surprise" is a separate, harder, and still-open question (Section 9).

### 5.9 A compositional call-construction DSL and its fitness-metric degeneracy

Motivated by the structural comparison with `clebran-arc-agi` (Section 6), we built `sim/call_dsl.py`: a `CallProgram` is a short, typed program — one generator operation (a pure tone or a linear chirp) optionally followed by up to three modifier operations (amplitude modulation, an attack/decay envelope) — with mutation operators (perturb a constant, swap a modifier, grow, delete) and a crossover operator mirroring, at a much smaller scale, the genetic-programming vocabulary `clebran-arc-agi` uses over its grid-program representation. `sim/dsl_evolution_demo.py` evolves a population of such programs toward a target sound via elitism and mutation, using cosine similarity between a candidate's synthesised MFCC and the target's as fitness.

**Result:** against both a pure-tone target and a wide-chirp target, evolution reliably reaches 0.97–0.98 fitness within fifteen to twenty generations — but does not reliably recover the target's own generator type. The tone target was matched, in one run, by a chirp of near-zero bandwidth (functionally indistinguishable from a tone, but structurally a different primitive); the chirp target was matched by a tone layered with three amplitude-modulation operations approximating the chirp's bulk spectral envelope. We interpret this as a genuine degeneracy in the fitness metric rather than a search failure: cosine similarity over MFCCs with the zeroth coefficient dropped (Section 3.1) is, by design, insensitive to overall energy and evidently also fails to discriminate cleanly between "the right spectral shape via the intended primitive" and "a structurally different shortcut that happens to produce a similar bulk envelope." This directly blocks the natural next step — mining recurring structural motifs across many evolved programs, in the spirit of `clebran-arc-agi`'s library learning (Section 6.2) — since mining programs whose apparent success is itself an artifact of a permissive fitness metric would recover artifacts, not genuinely reusable acoustic building blocks. We report this as a finding rather than a failure to be quietly fixed before publication, since the degeneracy is informative about the limits of the chosen similarity metric.

## 6. Structural Comparison with `clebran-arc-agi`

`clebran-arc-agi` is a separate, substantially larger research codebase on the same machine, evolving typed programs to solve tasks from the Abstraction and Reasoning Corpus (ARC-AGI). Despite operating in a completely different domain — large, discrete, typed symbolic program search against held-out test grids, versus Chuff's low-dimensional, continuous, "close enough in feature space" acoustic signaling — its basic shape (independent learners, a constrained communication channel, reinforcement of what works) is close enough to Chuff's own comms track that we treat it as a direct source of transferable ideas, rather than only a thematically-related citation.

### 6.1 A reusable channel design

`clebran`'s communication channel is a blackboard onto which lineages publish scored messages each epoch; a channel variant determines which messages are admitted, capping bandwidth on two independent axes at once — message *complexity* (not raw byte count) and message *count* per sender per epoch — and retaining only the highest-scoring messages that fit within both caps. Chuff's current acknowledgement signal (a single fixed tone, carrying one bit of information, at a channel capacity of one message) is the degenerate case of this design. Should Chuff's comms track ever need a richer acknowledgement payload, or more than two communicating units, this admission-by-score-under-dual-caps pattern is the one to adopt rather than reinvent.

### 6.2 A caution, not just an opportunity

`clebran`'s central, heavily-controlled finding is that a single population mining its own solved programs into a growing, reusable abstraction library significantly outperforms every multi-population gossip variant at matched computational budget — by a wide margin, and consistently across many experimental controls designed specifically to make this null result trustworthy rather than an artifact of insufficient baselines. The authors' own framing of this result is that communication, in their system, is "a transport mechanism, not a discovery mechanism": sharing what one learner already knows does not substitute for a mechanism by which an individual learner discovers something new.

This is directly relevant to how Chuff's planned two-unit experiment should be framed and interpreted. It would be a mistake to present a successful two-unit signaling demonstration as evidence that communication makes either unit *learn better* than a single, well-tuned unit mining its own memory would — the present study provides no such evidence, and `clebran`'s result is a specific, hard-won caution against assuming it. What Section 5.6's grounding experiment *does* show is that a stable, mutually-recognised, partially-discriminating convention emerges between two independent learners from reinforcement alone — a demonstration of signaling-game convention formation (Section 2.7), not of comparative learning efficiency. We adopt this framing going forward rather than retrofit a stronger claim the evidence does not support.

### 6.3 What does not transfer

`clebran`'s search operates over 160 hand-designed, typed primitives and a population-management machinery (hall-of-fame restarts, per-task random-number substreams, progress-weighted task scheduling) built specifically to close gaps at the scale and discreteness of exact symbolic program search. None of this machinery is motivated at Chuff's current scale, where the search space is a handful of continuous acoustic parameters and success is a graded similarity score rather than exact correctness against held-out tests. We note this explicitly to avoid the common failure mode of importing structural complexity from an adjacent project because it is available, rather than because the present problem needs it.

## 7. Discussion

Three methodological patterns recur across this study strongly enough to generalise beyond any one experiment. First, single-seed results mislead in exactly the way Section 5.6 demonstrates: a result that looks clean and even thematically satisfying (an "acoustically coherent" split) can simply be a coincidence of a small sample, and the correct response on noticing a clean-looking pattern is to widen the sweep before trusting it, not to stop at the first satisfying answer. Second, untested code paths hide real bugs regardless of how simple they look: a latent crash in acknowledgement-tone detection, present since the very first version of the two-unit demonstration, was only discovered once the logging flag that would have exercised it was finally wired up to a production code path and actually run — a flag nobody runs is a flag nobody has tested. Third, a clean negative result (Section 5.7) is worth exactly as much space in a write-up as a clean positive one (Section 5.2, 5.4, 5.5): reporting a refuted hypothesis plainly, and using it to redirect toward the mechanism the evidence actually implicates, is the same scientific move in both directions.

Taken together, the eight experiments demonstrate that a very small system — a KNN novelty store, a reinforcement-learned repertoire, a predictive dream loop, and a mood machine, none individually sophisticated — is sufficient to reproduce several specific, named phenomena from the vocal-learning and signaling-game literatures (social gating, acoustic adaptation, convention formation under reinforcement) as emergent consequences of the system's existing mechanisms, without any of those phenomena having been separately engineered in. The two negative results are, if anything, more informative about the system's actual mechanism than the positive ones, since they rule out an initially plausible explanation in favour of a more specific and more surprising one (an untrained tie-break, not a relevance gate, drives coarse grounding; a permissive similarity metric, not a search failure, prevents clean motif discovery).

## 8. Limitations

This study is entirely in silico. Every sound is a synthesised tone, chirp, or Gaussian noise sample; no field recording, real microphone, or real speaker was used anywhere. The six-sound, twelve-call-repertoire scale used throughout is small by design, to keep experiments fast and interpretable, and none of the reported effect sizes should be assumed to hold at a materially larger scale without re-running the relevant sweep. The acoustic channel model (a single fourth-order Butterworth filter plus Gaussian noise) is a deliberately crude stand-in for real room acoustics and real distance attenuation. The two-unit experiments in this paper are entirely symmetric, synchronous simulations; no asynchronous, physically-separated, or genuinely noisy real-world two-unit experiment has yet been attempted. Finally, MFCC-with-cosine-distance was adopted as a consistent convention throughout for its simplicity and its established role in speech processing (Davis & Mermelstein, 1980), not because it was independently verified to be the uniquely correct representation or metric for this problem — Section 5.9's fitness-metric degeneracy is itself partial evidence that it is not without real limitations.

## 9. Next Steps

The following items follow directly and specifically from the results reported above, each with a stated rationale rather than as a generic to-do list:

1. **Replace the index-ordered tie-break with a randomised, fixed-at-construction one** and re-run Section 5.6's ten-seed grounding sweep, now that Section 5.7 has ruled out the relevance gate as the mechanism and implicated the tie-break specifically.
2. **Sharpen or diversify the call-DSL fitness metric** — for instance, requiring a high score against several distinct contexts rather than cosine similarity to a single target — before attempting the Stitch-style motif-mining step that Section 5.9 identified as currently blocked by fitness-metric degeneracy.
3. **Make call duration and bandwidth themselves learnable parameters**, rather than only which fixed-shape call is selected, which is a prerequisite for testing the compression-versus-redundancy tension identified in Section 2.7 (Rita et al., 2022, versus Zaslavsky et al., 2018) at all.
4. **Port the validated memory and repertoire design to the ESP32-S3 track in C**, reusing the same cosine-distance convention validated by the evaluation harness (Section 3.3), and complete the still-unimplemented I2S microphone capture and FFT/MFCC pipeline on that hardware, which this entire simulation study was built to prepare for rather than substitute for.
5. **Attempt a genuinely asynchronous, physically-separated two-unit experiment** once a second unit exists, explicitly as a test of whether the coarse, path-dependent grounding reported in Section 5.6 reproduces outside a synchronous, symmetric simulation — and explicitly *not* framed, per Section 6.2's caution, as a test of whether communication improves either unit's learning.
6. **Evaluate whether a human observer reads the mood machine's state transitions as emotionally legible**, which Section 5.8 identified as a separate and harder bar than mechanical correctness.
7. **Run a real coding model against the evaluation harness** described in Section 3.3, to test whether an external, independently-written implementation of the memory and repertoire modules satisfies the same behavioural specification our own implementation does.

## Acknowledgments

This work was carried out as a human–AI collaboration: the experimental design, implementation, simulation runs, and this manuscript were produced in direct, iterative conversation between the author and Claude (Sonnet 5), which served throughout as a coding and research assistant with direct tool access to the codebase, search, and the experiment outputs reported here. We disclose this plainly rather than note it only in passing, consistent with this paper's broader commitment to reporting methodology accurately, including where it was unglamorous.

## References

Boncoraglio, G., & Saino, N. (2007). Habitat structure and the evolution of bird song: a meta-analysis of the evidence for the acoustic adaptation hypothesis. *Functional Ecology*, 21(1), 134–142. https://doi.org/10.1111/j.1365-2435.2006.01207.x

Davis, S., & Mermelstein, P. (1980). Comparison of parametric representations for monosyllabic word recognition in continuously spoken sentences. *IEEE Transactions on Acoustics, Speech, and Signal Processing*, 28(4), 357–366.

Ellis, K., Wong, C., Nye, M., Sablé-Meyer, M., Morales, L., Hewitt, L., Cary, L., Solar-Lezama, A., & Tenenbaum, J. B. (2021). DreamCoder: bootstrapping inductive program synthesis with wake-sleep library learning. *Proceedings of the 42nd ACM SIGPLAN International Conference on Programming Language Design and Implementation (PLDI 2021)*. https://doi.org/10.1145/3453483.3454080

Friston, K. (2010). The free-energy principle: a unified brain theory? *Nature Reviews Neuroscience*, 11, 127–138. https://www.nature.com/articles/nrn2787

Goldstein, M. H., King, A. P., & West, M. J. (2003). Social interaction shapes babbling: Testing parallels between birdsong and speech. *Proceedings of the National Academy of Sciences*, 100(13), 8030–8035. https://doi.org/10.1073/pnas.1332441100

Hodel, M. (n.d.). *arc-dsl* [Software repository]. https://github.com/michaelhodel/arc-dsl

Jones, A. (2026). *clebran-arc-agi* [Unpublished software repository].

Kuhl, P. K. (2003). Human speech and birdsong: Communication and the social brain. *Proceedings of the National Academy of Sciences*, 100(17), 9645–9646. https://doi.org/10.1073/pnas.1733998100

Leitão, A., & Gahr, M. (2024). Babbling opens the sensory phase for imitative vocal learning. *Proceedings of the National Academy of Sciences*, 121(17). https://doi.org/10.1073/pnas.2312323121

Lewis, D. K. (1969). *Convention: A Philosophical Study*. Harvard University Press.

Lou, M., Guo, H., Chen, L., & Yu, Y. (2026). Overcoming catastrophic forgetting in visual continual learning with reinforcement fine-tuning. *arXiv preprint arXiv:2605.09640*.

Rita, M., Tallec, C., Michel, P., Grill, J.-B., Pietquin, O., Dupoux, E., & Strub, F. (2022). Emergent communication: generalization and overfitting in Lewis games. *Advances in Neural Information Processing Systems 35 (NeurIPS 2022)*. arXiv:2209.15342.

Skyrms, B. (2010). *Signals: Evolution, Learning, and Information*. Oxford University Press.

Yazaki-Sugiyama, Y. (2024). Tutor auditory memory for guiding sensorimotor learning in birdsong. *Frontiers in Neural Circuits*. https://doi.org/10.3389/fncir.2024.1431119

Zaslavsky, N., Kemp, C., Regier, T., & Tishby, N. (2018). Efficient compression in color naming and its evolution. *Proceedings of the National Academy of Sciences*, 115(31), 7937–7942. https://doi.org/10.1073/pnas.1800521115

Zhang, J., Fu, Y., Peng, Z., Yao, D., & He, K. (2024). CORE: mitigating catastrophic forgetting in continual learning through cognitive replay. *Proceedings of the 46th Annual Meeting of the Cognitive Science Society (CogSci 2024)*. arXiv:2402.01348.

Zhu, W., Zhang, Z., Ren, Y., Huang, Y., Xu, H., & Wang, Y. (2025). Embodied representation alignment with mirror neurons. *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV 2025)*. arXiv:2509.21136.
