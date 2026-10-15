# Chuff: An Exploratory Simulation Study of Acoustic Memory, Reinforced Call Selection, and Affective Displays in a Robotic Pet

**Aidan Jones**

October 2026

Original implementation and draft developed with Claude as a coding/research assistant; literature and methodological revision assisted by Codex.

## Abstract

Chuff is a low-cost robotic-pet project that combines acoustic novelty memory, adaptive call selection, and expressive state transitions. This exploratory pilot evaluates the software before deployment on the intended ESP32-S3 and Arduino UNO Q hardware. A synthetic world supplies four tones and two chirps to a pipeline comprising time-averaged Mel-frequency cepstral coefficients (MFCCs), bounded nearest-neighbour memory, context-dependent reinforcement of a fixed call repertoire, an idle-time embedding-interpolation routine, and a mood state machine. Selected checks reproduced during manuscript review include a behavioural-grader score of 1.0, a seed-5 acknowledgement rate rising from 0.933 to 1.000, and a babbling comparison across 20 seeds in which self-reinforcement reduced mean first-response frequency error from 1,021 to 251 Hz. The original pilot summaries also report lower frequencies among acknowledged calls under low-pass filtering, retention advantages on some measures for recency-based eviction, and stable but coarse call pairings whose category membership varies across seeds. A relevance-threshold sweep did not produce the predicted increase in discrimination. Program evolution achieved high MFCC similarity with differing generator structures, revealing a need to distinguish acoustic equivalence from failures of temporal fidelity. These findings establish useful implementation behaviour, but do not establish biological social gating, a free-energy mechanism of affect, or receiver understanding of an environmental referent. The principal next experiments are matched-feedback controls, sender-only observation with receiver decoding, causal message interventions, temporal acoustic evaluation, and measured hardware transfer.

## 1. Introduction

Chuff asks how a small robotic pet can acquire acoustic memories, adapt its responses, and display internal state through light and animation. Affective robot control already has substantial precedent: Breazeal (2003), for example, studied emotion-inspired internal regulation and expressive behaviour in Kismet. The present contribution is a compact, inspectable implementation and an exploratory evaluation of its learning rules, rather than a claim to introduce internally regulated robot affect.

The intended hardware comprises an ESP32-S3 track and an Arduino UNO Q track. At the time of the pilot, neither had been fully wired for the intended audio experiments. Synthetic audio therefore provided a way to exercise the software and expose design problems before physical deployment. Environmental labels such as “dog_bark” identify synthesis recipes; they are not recordings or realistic models of those sound classes.

The central questions are engineering questions informed by cognitive science: how should a bounded memory retain recurring inputs; how does acknowledgement affect call selection; what acoustic information survives the representation and channel; and what would count as evidence that a call conveys information to another agent?

This study contributes:

1. An integrated simulation pipeline for acoustic feature extraction, novelty memory, reinforced call selection, waveform limiting, and expressive state control.
2. Exploratory comparisons of acknowledgement conditions, self-reinforcement, channel filtering, retention policies, and call-selection stability, with the sampling and evidential limits stated separately for each.
3. An account of a corrected category-partition claim, an unsupported relevance-gate prediction, and a mismatch between program structure and the acoustic fitness objective.
4. A literature-grounded research agenda that separates implementation verification, communicative function, and human interpretation.

The experiments were developed iteratively; they were not preregistered. Hypotheses in this manuscript organise that exploratory work and should not be read as evidence that outcomes, stopping rules, or analyses were specified in advance.

## 2. Background and Related Work

### 2.1 Vocal self-organisation and sensorimotor experience

A close computational precedent is Oudeyer’s (2005) model of speech-sound self-organisation, in which coupled perceptual and motor systems in interacting agents generate structured vocal repertoires. This work makes clear that shared acoustic structure can arise from the interaction of architecture, experience, and learning constraints. Chuff is substantially simpler: its main experiments select among fixed pure-tone calls rather than learning a vocal tract or a combinatorial sound system.

Biological studies provide motivation, with a different evidential role. Leitão and Gahr (2024) found that babbling was necessary for initiating tutor-song sensory learning in juvenile zebra finches under their experimental conditions. Yazaki-Sugiyama (2024) reviews auditory memories and their role in guiding birdsong sensorimotor learning. Chuff’s “babbling” instead explicitly reinforces each call against its own acoustic signature. Testing that warm start is useful, but it does not test a biological critical period or demonstrate that passive exposure cannot support learning.

### 2.2 Social contingency and the appropriate control

Goldstein, King, and West (2003) manipulated the contingency of maternal responses to infant vocalisation, including a yoked comparison, and found changes in babbling quality. Kuhl, Tsao, and Liu (2003) compared live foreign-language exposure with recorded exposure in infant phonetic learning. These are complementary findings about social conditions; they do not establish that every benefit of feedback in an artificial learner is social gating.

For Chuff, the first question is whether the programmed reinforcement rule behaves differently with and without acknowledgement. A stronger contingency claim requires feedback matched in amount and acoustic content but disconnected from the current action, as well as equivalent listener exposure and learning. The existing comparison does not meet those requirements.

### 2.3 Acoustic adaptation and transmission

Boncoraglio and Saino’s (2007) meta-analysis evaluates the acoustic adaptation hypothesis: habitat-dependent propagation can favour particular song features. Its support is qualified, with uncertainty about generality and differences across acoustic features. Chuff isolates one engineered pressure, high-frequency attenuation, rather than modelling habitat evolution. A lower average frequency among acknowledged calls can reflect transmission-based selection of observations as well as a learned shift in the emission policy; those explanations need separate controls.

### 2.4 Prediction error, surprise, and affect

Friston (2010) distinguishes sensory surprisal, defined through a generative model’s probability of observations, from variational free energy, which bounds that surprisal. A cosine distance between two embeddings is neither quantity without additional modelling assumptions. Chuff has no learned temporal likelihood model, posterior inference, or free-energy objective.

We therefore describe the implemented routine as an **embedding-mismatch heuristic**. It samples and interpolates memories, compares the result with an observation, and supplies a score to a hand-designed display controller. “Dreaming,” “surprise,” and “contentment” are names for software states. Their emotional legibility requires human evaluation, as distinct from verification that the controller follows its rules (Breazeal, 2003).

### 2.5 Bounded memory and retention

The directly tested problem is exemplar eviction from a bounded store. Chuff does not update neural-network parameters, so losing an exemplar through capacity pressure is not the same mechanism as catastrophic interference in a trained network. Continual-learning work nevertheless offers useful memory-management baselines: Chaudhry et al. (2019) investigate small episodic memories and replay, while Zhang et al. (2024) study adaptive replay allocation and representative sample selection. Neither establishes the superiority of Chuff’s particular retention rule.

Chuff’s “recency” policy refreshes whichever stored entry is the nearest neighbour of a query, even when the query is too distant to count as recognition. It is therefore more precisely a recency-of-nearest-match policy than a reward-weighted memory rule. Its benefits and failure modes should be measured against FIFO, random eviction, and reservoir sampling under the same stream.

### 2.6 Convention, information, and causal communication

Lewis (1969) and Skyrms (2010) provide the conceptual basis for signalling conventions. A stable mapping is an important observation, but it is not sufficient to demonstrate successful reference. Lowe et al. (2019) distinguish evidence that messages correlate with an agent’s state from evidence that they influence another agent’s behaviour. Rita et al. (2022) further analyse overfitting and co-adaptation in Lewis games, motivating held-out inputs and partner tests.

Efficient semantic categorisation and robust transmission also address different objectives. Zaslavsky et al. (2018) study information-bottleneck efficiency in colour naming; Vital, Sardinha, and Melo (2025) directly study redundancy under noisy emergent communication. These are compatible pressures: an economical category code can still require extra transmission redundancy. Chuff presently learns neither variable message length nor a receiver’s referent decoder, so it cannot yet evaluate this trade-off.

### 2.7 Acoustic representation and reusable programs

MFCCs have a long history in speech recognition (Davis & Mermelstein, 1980). Their suitability for Chuff’s tones, chirps, and later environmental recordings remains an empirical question. In particular, averaging across frames discards temporal order, making different frequency trajectories potentially similar under the current objective.

DreamCoder (Ellis et al., 2021) and Stitch (Bowers et al., 2023) provide established approaches to learning reusable program libraries. They motivate mining useful structures from evolved call programs, provided usefulness is assessed beyond the training fitness score. Multi-resolution spectrogram objectives, used in waveform generation by Yamamoto, Song, and Kim (2020), offer one candidate evaluation component; they would need validation for this task.

## 3. System Architecture

### 3.1 Perception, memory, and call selection

The Python implementation resides in python/src/chuff_mpu. Audio is sampled at 16 kHz; environmental windows are one second. The feature extractor computes 14 MFCCs, discards coefficient zero, and averages each remaining coefficient over frames to produce a 13-dimensional vector.

Coefficient zero captures the mean log-mel spectral level under this transform and is gain-sensitive; it is not a separately measured waveform-energy variable. Dropping it reduces one source of level dependence but does not guarantee invariance to noise, clipping, or frequency-dependent channel changes.

Memory stores at most 2,048 embeddings by default. Novelty is the nearest stored cosine distance,

~~~text
d(x, y) = 1 - (x · y) / (||x|| ||y||).
~~~

The empty-store score is assigned the sentinel value 1.0; nonzero-vector cosine distance can range from 0 to 2. Recognition uses distance below 0.2, and an environmental observation is added when novelty is at least 0.7. The intermediate region is not automatically stored.

Each agent starts with 12 independently drawn call frequencies between 400 and 3,000 Hz. Calls last 500 ms. The repertoire does not initially choose the acoustically nearest call: it scores each call using a small index preference plus weighted similarities to learned context exemplars. An untrained repertoire therefore favours index zero. Acknowledgement reinforces the selected association at rate 0.1; missing acknowledgement decays it at rate 0.05. A cosine-similarity gate of 0.5 controls generalisation between contexts.

Self-generated and external audio use the same MFCC extractor. This makes their acoustic vectors comparable; it does not establish a learned joint motor-perceptual representation or a mirror-neuron mechanism.

### 3.2 Channel and acknowledgement

The six environmental recipes comprise four tones and two linear chirps spanning approximately 350–3,200 Hz, with 15 Hz synthesis jitter and additive Gaussian noise of configured amplitude 0.03. The channel optionally applies a fourth-order Butterworth low-pass filter, then amplitude scaling (default 0.5) and fresh noise. It does not simulate a room impulse response, reverberant decay, microphone directivity, or calibrated physical distance.

A listener acknowledges a call when its MFCC is sufficiently close to something in its separate call-recognition memory. Otherwise it stores that call for possible future recognition. The acknowledgement is a fixed, pre-agreed 800 Hz tone lasting 200 ms, detected by an FFT-peak rule. Thus the repertoire mapping is initially unshared, but timing and the acknowledgement protocol are supplied by the experimenter.

### 3.3 Output limiting and expressive state

The ordinary agent response passes through a numerical limiter that sanitises non-finite samples, caps duration at three seconds, and imposes a peak-amplitude ceiling of 0.5. These are waveform constraints, not a calibration of loudspeaker power, acoustic sound-pressure level, or hearing safety. The simulated acknowledgement tones are constructed separately and passed directly to the channel; the claim that every synthesised waveform is limited is therefore too broad.

The controller defines nine mood labels and prioritises acknowledgement, embedding mismatch, idle time, and novelty signals in that order. Defining nine labels does not make all nine reachable through the current update function (Section 5.8).

### 3.4 Behavioural grading

The evaluation harness checks recognition, capacity, sample-pair availability, call-frequency bounds, deterministic choice when weights are unchanged, reinforcement, context separation, and decay. The original project reports stub and reference-solution checks scoring 0.0 and 1.0. During this revision, the existing grader was rerun against the production modules and returned 1.0.

This verifies those behavioural checks. It does not validate biological analogies, communicative understanding, statistical claims, hardware fidelity, or every code path.

## 4. Methods and Evidence Provenance

Each comparison uses a script under python/src/chuff_mpu/sim and a console entry in python/pyproject.toml. Important defaults are in config.py, but simulation-specific parameters and overrides also appear in scripts and command-line arguments. A reproducible experiment therefore needs the complete invocation and software revision, not config.py alone.

This is an exploratory pilot with heterogeneous sampling. Section 5.2 uses one seed; the babbling check uses 20 paired seeds and six labels; the original channel summary uses 15 seeds; memory and grounding summaries use ten seeds; the original mood summary describes five runs. Repeated labels within a seed share an agent initialisation and are not independent replicates. Seed-level paired differences are the appropriate unit for uncertainty estimates in those comparisons.

**Numerical provenance.** During the 14 October 2026 review, the grader, babbling comparison, and seed-5 acknowledgement summaries were rerun against code revision 77cbcb8e42b9e138b9efc3e510f42245f331b518. Other numerical aggregates below are retained from the original pilot manuscript and identified as such; they were not independently recomputed in this revision. Raw per-seed outputs and a complete experiment manifest are not supplied with those aggregates, so their precision, variability, and reproduction remain to be audited.

No confidence intervals, equivalence tests, or preregistered convergence criteria are available for the historical comparisons. We consequently report descriptive differences rather than statistical significance, proof of a null effect, or decisive mechanism identification. Agarwal et al. (2021) motivate this attention to uncertainty in evaluations based on small numbers of learning runs.

A further reproducibility limitation is that EmbeddingStore.sample_pair() constructs a fresh unseeded random generator. Passing a seed to the mood experiment does not fully determine its interpolation samples. This matters specifically for replaying mood/dream trajectories.

## 5. Exploratory Results

### 5.1 Integrated loop and numerical output constraints

The single-agent demonstration runs feature extraction, novelty scoring, memory insertion, call synthesis, output limiting, and noisy self-recognition. Its limiter check supplies a high-amplitude input containing NaN and infinity, then asserts finite output within the configured ceiling. This is a useful integration check for the tested input. It does not prove physical speaker safety or universal robustness to every malformed input.

### 5.2 Acknowledgement availability and call-selection dynamics

**Question.** Does call selection behave differently when recognition-triggered acknowledgement is available?

**Design.** In two_unit_demo.py, A hears an environmental sound and emits a call. B acknowledges a recognised call; A then reinforces or decays its context-call association. In the “passive” branch, B’s recognition/learning routine is bypassed and acknowledgement is forced absent.

**Verified result.** Rerunning seed 5 for 60 rounds with attenuation 0.5 and no filtering yielded an acknowledgement rate of 28/30 (0.933) in the first half and 30/30 (1.000) in the second. The original manuscript’s first-half value of 0.90 is replaced by this reproducible result. The passive condition yielded zero acknowledgements, as specified by its implementation.

The original account also describes rapid stabilisation in the contingent condition and continuing turnover in the passive condition. A formal time-to-convergence endpoint was not specified, so the earlier estimate of “approximately five rounds” is not retained as an established measurement.

**Interpretation.** This is a check of reinforcement with versus without feedback. It does not isolate contingency from feedback amount, hold realised call sequences fixed, or keep listener learning equivalent. Branch-dependent acknowledgement transmissions also consume random draws from the shared environment generator, so equal seeds do not guarantee matched later environmental sequences across conditions. A yoked-feedback comparison and separated random streams are needed before drawing a social-contingency conclusion.

### 5.3 Babbling as an explicit self-reinforcement warm start

**Question.** Does practising and self-reinforcing each call alter the first response to an external sound?

**Design.** babbling_experiment.py compares fresh agents with the same initial repertoire, with or without one babbling pass. That pass reinforces each call against its clean MFCC signature at rate 0.05. For each of six environmental labels and 20 seeds, the experiment measures absolute error between the chosen call frequency and a target characteristic frequency: the tone frequency or the midpoint of a chirp.

**Verified result.** The selected call changed in 89/120 seed-label cases (74%). Mean absolute frequency error fell from 1,021 Hz without babbling to 251 Hz with babbling, a descriptive reduction of 770 Hz.

**Interpretation.** Self-reinforcement supplies useful acoustic anchors compared with the untrained index preference. The 120 cases form 20 seed blocks, not 120 independent agent initialisations. The outcome is a frequency-proximity proxy, especially limited for chirps; it is not an imitation judgement or a measure of subsequent learning efficiency. The design also gives one condition extra reinforcement, so it cannot attribute the benefit specifically to producing sound rather than receiving equivalent feature anchors.

### 5.4 Filtering and the frequency of acknowledged calls

**Question.** Does low-pass filtering change which calls receive acknowledgement?

**Original pilot summary.** Across 15 seeds and 60 rounds per condition, mean frequency among acknowledged calls was reported as 1,903 Hz without filtering and 1,483 Hz with a 900 Hz cutoff, a difference of 420 Hz. Mean second-half acknowledgement rates were 0.99 and 0.86, respectively. Per-seed differences and intervals were not preserved in the manuscript.

These summaries are consistent with preferential transmission of lower frequencies in this channel. They do not distinguish learned adaptation from measurement selection: even a fixed, nonlearning policy could have a lower mean frequency *among acknowledged calls* after high-frequency attenuation. Filtering affects both the outbound calls and the 800 Hz return acknowledgement. Separate forward- and return-channel manipulations are needed.

The original development also exposed an index-frequency confound: sorting the initial repertoire placed low frequencies at preferred indices. Removing that sort prevents this specific confound, although index preference remains. Filter strength was adjusted during exploration, so the final comparison is not an independent confirmatory test of a previously fixed channel model.

### 5.5 Memory retention under capacity pressure

**Question.** Does recency-of-nearest-match eviction preserve recurring inputs better than FIFO?

The experiment interleaves a recurring 1,500 Hz tone with filler tones, adding observations only above the novelty threshold. The original ten-seed summaries are:

| Capacity : revisit gap | FIFO: recognised at end | Recency: recognised at end | Additional reported summary |
|---|---:|---:|---|
| 3 : 10 | 90% | 100% | No revisit-recognition estimate supplied |
| 4 : 6 | 100% | 100% | Revisit recognition 57% versus 71%; final novelty 0.005 versus 0.000 |
| 6 : 8 | 80% | 100% | Revisit recognition 78% versus 89% |

The direction favours recency on several reported measures, with ties on others. Ten seeds give coarse endpoint percentages, and these summaries do not establish statistical reliability or how the advantage changes monotonically with churn.

Several implementation details limit interpretation. The same rare-tone embedding is reused at each revisit; a forgotten exemplar can be re-added; and an acoustically similar filler can also yield recognition. End-of-run recognition therefore does not measure uninterrupted survival of the original memory. Filler frequencies are selected from a shuffled grid, but the experiment does not verify that all fillers are separated in MFCC space. A stronger retention experiment should track exemplar identity and evaluate fresh noisy instances at controlled times after a revisit.

### 5.6 Stable call pairings and the limits of “grounding”

Both agents in grounding_demo.py hear the same environmental event through independently added noise. Each emits a call, while a separate listener memory on the other side decides whether that call is familiar. After 150 rounds, the analysis selects each agent’s majority call for each label in the last 30% of that label’s occurrences. “Purity” is the fraction of that tail using the majority call; the pair-count metric counts distinct pairs of majority indices across labels.

**Original pilot summary.** Ten seeds (0–9) reportedly produced two distinct pairs across six labels and 100% tail purity. Most labels used index zero on both agents. Which labels occupied the minority pairing varied by seed. An initial claim that the split consistently followed acoustic frequency was withdrawn after this wider sweep.

This supports stable, coarse, path-dependent call selection in the sampled runs. It does not identify the causal mechanism conclusively or demonstrate a shared referential code. Both agents already observe the putative referent; their call-recognition memories do not decode environmental labels; and acknowledgement rewards familiarity rather than correct reference. Moreover, an (A-call, B-call) pair is an analyst’s joint summary, not a single message shown to support receiver discrimination. Tail purity also ignores earlier instability and depends on the number of observations per label.

### 5.7 Relevance-threshold sensitivity

**Prediction.** Raising the cross-context similarity gate would reduce generalisation and increase the number of distinct call pairs.

**Original pilot summary.** Four thresholds were evaluated across ten seeds each:

| Similarity threshold | Mean distinct pairs across six labels | Mean tail purity |
|---|---:|---:|
| 0.30 | 2.0 | 100% |
| 0.50 | 2.0 | 100% |
| 0.70 | 2.1 | 100% |
| 0.90 | 2.0 | 100% |

This range is 0.3–0.9, a threefold endpoint ratio, not four orders of magnitude. The predicted monotonic increase is absent in the reported means. These results do not demonstrate equivalence or rule out effects on other outcomes, longer runs, or different sound distributions.

The index preference is a plausible alternative contributor because it acts before any relevant exemplar exists. A controlled intervention is still required to establish its causal contribution. Randomising one fixed global index order alone would retain a globally preferred call; it may simply relabel the dominant choice.

### 5.8 Embedding mismatch and mood reachability

The mood demonstration uses sparse environmental presentations to accumulate simulated idle time. When the idle threshold is reached and at least two memories exist, it may interpolate two embeddings and immediately sample an additional synthetic sound for comparison. This wake observation is explicitly generated by the demonstration; it is not necessarily the next independently scheduled environmental event.

The original five-run account reports two to six interpolation events per 300 ticks at sound probability 0.03, with different subsequent “surprise” and “contentment” outputs. These event counts are not fully seed-reproducible because of unseeded memory-pair sampling.

The original claim that all nine mood labels were exercised is inconsistent with the current implementation. update_mood() never returns STRONG_NOVELTY: high novelty instead returns NEW_SOUND_LEARNED. The demonstration never passes ack_received=True, so it cannot emit ACKNOWLEDGEMENT. Thus it cannot exercise all nine labels, regardless of seed.

The defensible result is that the integration supports idle-triggered embedding comparison and corresponding state transitions. It does not establish forecasting skill, subjective affect, or a free-energy implementation. Those questions require distinct models and evaluations.

### 5.9 Evolved call programs and objective ambiguity

The call DSL combines a tone or chirp generator with optional amplitude modulation and envelopes. Evolution uses elitism, mutation, and crossover to optimise cosine similarity to a target’s time-averaged MFCC.

**Original pilot observations.** Tone and chirp examples reportedly reached fitness around 0.97–0.98 within roughly 15–20 generations, sometimes using a different generator type from the target. The manuscript does not provide a sufficient seed-level summary to support the stronger claim that this happens reliably.

A near-zero-bandwidth chirp may be acoustically equivalent to a tone despite a different syntax. Generator mismatch alone is therefore not failure. Conversely, a modulated tone that matches a chirp’s averaged spectrum may miss its temporal trajectory. This is an identifiability and objective-validity problem to investigate, rather than a demonstrated universal degeneracy. Acoustic fidelity, perceptual similarity, and recovery of the original program must be evaluated separately.

## 6. Relation to the Unpublished clebran-arc-agi Project

The original manuscript discusses clebran-arc-agi (Jones, 2026), an unpublished program-synthesis project described as using multiple search populations, constrained sharing, and reusable abstractions. That comparison is useful as a source of design hypotheses, but its experiments and numerical evidence are not presented or independently verified here. We therefore do not use its claimed performance advantage for a single learner as an established result supporting Chuff.

Published library-learning work provides the firmer methodological foundation (Ellis et al., 2021; Bowers et al., 2023). The relevant question is whether reusable call motifs improve held-out acoustic tasks at matched search cost. Comparing one learner with multiple communicating learners would additionally require matching total evaluations, memory, interaction opportunities, and any library-building budget. A successful two-agent interaction alone would not answer that efficiency question.

A scored, bandwidth-limited message admission scheme from the adjacent project could be explored if Chuff expands to larger populations. It is a design option, not yet an empirically justified choice for this acoustic task. The current presence/absence acknowledgement is a binary feedback variable within an agreed listening window; its physical channel capacity has not been measured.

## 7. Discussion

The strongest present conclusions concern implementation and experimental design. Self-reinforcement changes first-response selection, acknowledgement activates the intended reinforcement loop, and the existing grader confirms its specified memory and repertoire behaviours. The historical sweeps suggest useful patterns to reproduce, particularly coarse call-selection stability and a possible retention advantage.

The review also exposes how evaluation can overstate the phenomenon being tested. Familiarity recognition is not environmental decoding; stability is not proof of causal communication; a mismatch score is not probabilistic surprise; and a syntactically different program need not be acoustically wrong. In each case, a clearer operational definition leads to a more informative next experiment.

The corrected frequency-partition claim remains instructive. A few seeds appeared to support an appealing acoustic explanation that a larger sample did not sustain. The threshold sweep then failed to support the proposed remedy, but did not causally establish a replacement explanation. Negative or inconsistent evidence should constrain the claim at precisely the level measured.

Chuff is therefore best positioned as an inspectable platform for developing and testing small acoustic-agent mechanisms. Its connection to vocal learning, signalling theory, and robot affect is a source of experimental questions, rather than evidence that those broader phenomena have already been reproduced.

## 8. Limitations

All inputs and transmissions are synthetic. The sound inventory, fixed repertoire, synchronous schedule, Gaussian noise, and low-pass model leave substantial gaps to real acoustic environments. Both agents observe each event, and timing plus the acknowledgement convention are pre-agreed. No human participant study or physical audio experiment is reported.

The study is exploratory, with development-driven changes and incomplete historical run provenance. Several summaries lack exact invocations, per-seed outputs, and uncertainty estimates. The grader covers a limited behavioural contract. Randomness is not fully controlled in the interpolation routine, and common random seeds do not guarantee matched experience across feedback branches.

Feature averaging removes temporal order; call familiarity can be rewarded independently of referential usefulness; memory recognition can reflect relearning or feature collisions; and mood labels encode designer choices. None of the observed effects establishes subjective feeling, biological learning mechanisms, or superior sample efficiency from communication. These limitations define the next research stage.

## 9. Future Work

### 9.1 Establish a reproducible baseline before enlarging the model

The immediate priority is an immutable experiment manifest: code revision, dependency lockfile, complete arguments, independent random streams for environment, agents, channels, feedback and memory sampling, and raw per-round outcomes. Reproduce the historical channel, memory, grounding, threshold and mood summaries before treating them as a reference benchmark. Report all selected seeds, including runs with no acknowledgement, and account for missing frequency means in those runs.

For new comparisons, choose primary outcomes and practically meaningful effect sizes before running the final evaluation. Estimate uncertainty from independent seed blocks, with paired comparisons where experience is genuinely matched. Determine replication counts from pilot variability and desired precision; a nominal number of seeds is not by itself a reliability guarantee (Agarwal et al., 2021).

### 9.2 Isolate contingency and the effect of self-experience

Use contingent, yoked, randomly timed rate-matched, and absent-feedback conditions. Keep listener exposure and updates identical, and replay the same environmental inputs and channel-noise sequences. Measure call-switch rates, time to a predefined stability criterion, and held-out performance rather than acknowledgement rate alone.

The contingency prediction is that feedback tied to the current action improves these outcomes beyond feedback with the same rate and acoustic content. If yoked feedback performs similarly, the result supports feedback availability rather than contingency. This design follows the logic of the primary infant work without claiming biological equivalence (Goldstein et al., 2003).

For babbling, compare active self-production with playback of the same calls, feature-anchor insertion without playback, and equal-budget random reinforcement. Compare against a simple nearest-call acoustic baseline as well as the current index-biased cold start. A production-specific explanation requires benefits beyond matched anchors and extra updates; otherwise the simpler explanation is representational initialisation.

### 9.3 Test whether a receiver actually uses the signal

Introduce a referential task in which only the sender observes the environmental event. The receiver must choose a label or task action using the transmitted waveform, with neither shared event timing that reveals the answer nor a direct copy of the environmental sound. Reward correct decisions rather than familiarity.

Compare learned messages with silence, shuffled messages, fixed signals, and a designed frequency code. At evaluation, replace or permute messages while holding the receiver’s other inputs fixed. Evidence for functional communication requires the correct messages to improve held-out receiver decisions and interventions to change those decisions in the predicted direction (Lowe et al., 2019). A balanced six-class decoder has chance accuracy 1/6; also report the empirical majority-class baseline when event frequencies differ.

Evaluate novel acoustic instances, new noise levels, and unfamiliar partners. Measure receiver accuracy, confusion matrices, message-use entropy, and state-message dependence alongside causal interventions. Success with a familiar partner alone can reflect co-adaptation rather than a general code (Rita et al., 2022).

### 9.4 Identify the mechanism behind coarse partitions

Cross tie-breaking rules with relevance thresholds: the current index preference, a fixed random permutation, random choice among tied calls on each decision, and controlled exploration. Permute presentation order while holding the event multiset and repertoire frequencies fixed. Separate any effect of call frequency from the ranking used to resolve untrained choices.

A fixed permutation tests whether a preferred index merely changes identity; random tie selection and exploration test whether a universal initial favourite contributes to collapse. The causal prediction is a change in pair counts, category coverage, and receiver accuracy when the hypothesised mechanism is removed. If only the identity of the majority call changes, the proposed remedy has not solved coarse discrimination. Retain purity as a stability measure, not as the communication endpoint.

### 9.5 Evaluate acoustic fidelity and reusable call structure

Create held-out targets with matched average spectra but different temporal structure, including ascending versus descending chirps. Compare the present averaged-MFCC score with framewise features, multi-resolution spectrogram distances, frequency-trajectory error, and onset/envelope measures. Validate which metrics agree with task-relevant discrimination and, where appropriate, blinded listening judgements. Yamamoto et al. (2020) motivate time-frequency objectives, without establishing which combination is best for Chuff.

Define acoustic equivalence classes in the DSL so a tone and an effectively constant chirp need not count as different solutions. Evaluate structural recovery only for targets identifiable under the chosen observations and explicit program priors. Then test DreamCoder- or Stitch-inspired motif libraries against primitive-only search at equal synthesis-evaluation budgets, measuring held-out fidelity and total description length including library cost (Ellis et al., 2021; Bowers et al., 2023). This makes library reuse a measurable benefit rather than an assumption.

### 9.6 Separate channel selection, learning, and coding efficiency

Run frozen-policy and adaptive-policy agents through the same channels. Measure both the unconditional emitted-frequency distribution and frequency conditional on acknowledgement; estimate acknowledgement probability by frequency. A filtering effect in the frozen policy would quantify transmission selection, while a subsequent change in emissions would provide evidence for learning.

Vary forward-path and acknowledgement-path filtering separately, then add measured room impulse responses, real background recordings, dropouts, and different speaker/microphone responses. Once receiver decoding is established, permit duration, bandwidth, repetition, and amplitude to vary under explicit energy or airtime costs. Trace receiver error versus transmission cost across noise levels. This can test efficient representation and protective redundancy as complementary objectives (Zaslavsky et al., 2018; Vital et al., 2025).

### 9.7 Measure durable memory and predictive value

For memory, track exemplar identity separately from recognition and relearning. Use fresh jittered/noisy instances on every revisit, randomise the endpoint’s position within the revisit cycle, and vary event frequency and distribution shifts. Compare FIFO, current recency, recognition-gated recency, random eviction, and reservoir sampling at equal capacity. Report rare-event recall, false recognition, continuous retention duration, and lookup cost. This separates useful retention from accidental matches and endpoint effects.

For the idle routine, first make memory sampling seed-controlled and compare interpolation with persistence, random-memory, mean-memory, and a simple learned transition predictor. Use temporally structured worlds with held-out transitions. Lower held-out prediction error than these baselines would demonstrate predictive value; if no advantage appears, interpolation should remain described as memory recombination. Any future probabilistic-surprise or active-inference claim would require an explicit likelihood model and an appropriate objective (Friston, 2010).

### 9.8 Transfer to hardware and asynchronous interaction

Port the memory and feature pipeline with recorded reference vectors, comparing numerical parity, recognition performance, RAM, latency, and energy on the target board. Measure microphone gain and frequency response, speaker response, limiter behaviour, end-to-end delay, and acknowledgement false positives. Route every physical output, including acknowledgements, through the calibrated output path.

Progress from controlled loopback to physically separated devices with independent clocks. Evaluate missed windows, variable latency, collisions, self-echo, and simultaneous calls before treating synchronous simulation success as transferable. Turn-taking is itself a learning and coordination problem, as illustrated by Taillandier et al. (2023); start with an explicit scheduling baseline before attempting learned timing.

### 9.9 Evaluate affective legibility separately from learning

First resolve which mood labels the controller is intended to reach and verify those transitions. Then conduct a blinded, counterbalanced observer study using identical displays driven by learning-linked states, matched scripted sequences, and shuffled state sequences. Measure emotion-label agreement, confidence, and ability to identify learning events; report confusions between labels rather than only overall appeal.

If observers distinguish learning-linked sequences and interpret them consistently, that supports affective legibility and behavioural transparency. It does not establish felt emotion. Participant-facing claims should stay at that level, consistent with the distinction between internal control and perceived expression in sociable robotics (Breazeal, 2003).

An independently authored implementation can additionally be evaluated against held-out behavioural checks, numerical reference vectors, and hardware traces. This would extend the engineering evidence supplied by the current grader, rather than serve as a cognitive-science experiment.

## Acknowledgments and AI Assistance

The original implementation, exploratory runs, and manuscript were developed through iterative collaboration between Aidan Jones and Claude, acting as a coding and research assistant. Codex assisted with this revision by inspecting the implementation, checking primary literature, rerunning selected comparisons and the existing grader, and revising the claims and proposed experiments. The author retains responsibility for verifying the historical results and the final manuscript.

## References

Agarwal, R., Schwarzer, M., Castro, P. S., Courville, A. C., & Bellemare, M. G. (2021). Deep reinforcement learning at the edge of the statistical precipice. *Advances in Neural Information Processing Systems*, 34, 29304–29320. [Proceedings](https://proceedings.neurips.cc/paper/2021/hash/f514cec81cb148559cf475e7426eed5e-Abstract.html).

Boncoraglio, G., & Saino, N. (2007). Habitat structure and the evolution of bird song: A meta-analysis of the evidence for the acoustic adaptation hypothesis. *Functional Ecology*, 21(1), 134–142. [doi:10.1111/j.1365-2435.2006.01207.x](https://doi.org/10.1111/j.1365-2435.2006.01207.x).

Bowers, M., Olausson, T. X., Wong, L., Grand, G., Tenenbaum, J. B., Ellis, K., & Solar-Lezama, A. (2023). Top-down synthesis for library learning. *Proceedings of the ACM on Programming Languages*, 7(POPL), Article 41, 1182–1213. [doi:10.1145/3571234](https://doi.org/10.1145/3571234).

Breazeal, C. (2003). Emotion and sociable humanoid robots. *International Journal of Human-Computer Studies*, 59(1–2), 119–155. [doi:10.1016/S1071-5819(03)00018-1](https://doi.org/10.1016/S1071-5819(03)00018-1).

Chaudhry, A., Rohrbach, M., Elhoseiny, M., Ajanthan, T., Dokania, P. K., Torr, P. H. S., & Ranzato, M. (2019). On tiny episodic memories in continual learning. *arXiv preprint*, arXiv:1902.10486. [Author manuscript](https://arxiv.org/abs/1902.10486).

Davis, S., & Mermelstein, P. (1980). Comparison of parametric representations for monosyllabic word recognition in continuously spoken sentences. *IEEE Transactions on Acoustics, Speech, and Signal Processing*, 28(4), 357–366. [doi:10.1109/TASSP.1980.1163420](https://doi.org/10.1109/TASSP.1980.1163420).

Ellis, K., Wong, C., Nye, M., Sablé-Meyer, M., Morales, L., Hewitt, L., Cary, L., Solar-Lezama, A., & Tenenbaum, J. B. (2021). DreamCoder: Bootstrapping inductive program synthesis with wake-sleep library learning. *Proceedings of PLDI 2021*, 835–850. [doi:10.1145/3453483.3454080](https://doi.org/10.1145/3453483.3454080).

Friston, K. (2010). The free-energy principle: A unified brain theory? *Nature Reviews Neuroscience*, 11, 127–138. [doi:10.1038/nrn2787](https://doi.org/10.1038/nrn2787).

Goldstein, M. H., King, A. P., & West, M. J. (2003). Social interaction shapes babbling: Testing parallels between birdsong and speech. *Proceedings of the National Academy of Sciences*, 100(13), 8030–8035. [doi:10.1073/pnas.1332441100](https://doi.org/10.1073/pnas.1332441100).

Jones, A. (2026). *clebran-arc-agi* [Unpublished software project; comparison described in the original Chuff pilot manuscript, without a public archival identifier].

Kuhl, P. K., Tsao, F.-M., & Liu, H.-M. (2003). Foreign-language experience in infancy: Effects of short-term exposure and social interaction on phonetic learning. *Proceedings of the National Academy of Sciences*, 100(15), 9096–9101. [doi:10.1073/pnas.1532872100](https://doi.org/10.1073/pnas.1532872100).

Leitão, A., & Gahr, M. (2024). Babbling opens the sensory phase for imitative vocal learning. *Proceedings of the National Academy of Sciences*, 121(18), e2312323121. [doi:10.1073/pnas.2312323121](https://doi.org/10.1073/pnas.2312323121).

Lewis, D. K. (1969). *Convention: A Philosophical Study*. Harvard University Press.

Lowe, R., Foerster, J., Boureau, Y.-L., Pineau, J., & Dauphin, Y. (2019). On the pitfalls of measuring emergent communication. *Proceedings of AAMAS 2019*, 693–701. [Proceedings](https://ifaamas.org/Proceedings/aamas2019/pdfs/p693.pdf).

Oudeyer, P.-Y. (2005). The self-organization of speech sounds. *Journal of Theoretical Biology*, 233(3), 435–449. [doi:10.1016/j.jtbi.2004.10.025](https://doi.org/10.1016/j.jtbi.2004.10.025).

Rita, M., Tallec, C., Michel, P., Grill, J.-B., Pietquin, O., Dupoux, E., & Strub, F. (2022). Emergent communication: Generalization and overfitting in Lewis games. *Advances in Neural Information Processing Systems*, 35. [Proceedings](https://proceedings.neurips.cc/paper_files/paper/2022/hash/093b08a7ad6e6dd8d34b9cc86bb5f07c-Abstract-Conference.html).

Skyrms, B. (2010). *Signals: Evolution, Learning, and Information*. Oxford University Press.

Taillandier, V., Hupkes, D., Sagot, B., Dupoux, E., & Michel, P. (2023). Neural agents struggle to take turns in bidirectional emergent communication. *International Conference on Learning Representations*. [Proceedings](https://openreview.net/forum?id=GULFHQfgw0g).

Vital, F., Sardinha, A., & Melo, F. S. (2025). Implicit repair with reinforcement learning in emergent communication. *Proceedings of AAMAS 2025*. [Author manuscript](https://arxiv.org/abs/2502.12624).

Yamamoto, R., Song, E., & Kim, J.-M. (2020). Parallel WaveGAN: A fast waveform generation model based on generative adversarial networks with multi-resolution spectrogram. *Proceedings of ICASSP 2020*. [Author manuscript](https://arxiv.org/abs/1910.11480).

Yazaki-Sugiyama, Y. (2024). Tutor auditory memory for guiding sensorimotor learning in birdsong. *Frontiers in Neural Circuits*, 18, 1431119. [doi:10.3389/fncir.2024.1431119](https://doi.org/10.3389/fncir.2024.1431119).

Zaslavsky, N., Kemp, C., Regier, T., & Tishby, N. (2018). Efficient compression in color naming and its evolution. *Proceedings of the National Academy of Sciences*, 115(31), 7937–7942. [doi:10.1073/pnas.1800521115](https://doi.org/10.1073/pnas.1800521115).

Zhang, J., Fu, Y., Peng, Z., Yao, D., & He, K. (2024). CORE: Mitigating catastrophic forgetting in continual learning through cognitive replay. *arXiv preprint*, arXiv:2402.01348. [Author manuscript](https://arxiv.org/abs/2402.01348).

## Appendix A. Commands for the Selected Review Checks

Run from the python directory with the project environment. The review used its existing .venv/Scripts/python.exe interpreter; the following uv equivalents use the same project.

~~~powershell
uv run python ../eval/tasks/sound_memory_and_voice/grader.py --candidate-dir src/chuff_mpu
uv run chuff-sim-babble --seeds 20
uv run chuff-sim-2unit --rounds 60 --seed 5 --condition social --attenuation 0.5
uv run chuff-sim-2unit --rounds 60 --seed 5 --condition passive --attenuation 0.5
~~~

Observed review outputs were grader score 1.0; babbling 89/120 changed choices and rounded frequency errors 1,021 versus 251 Hz; contingent acknowledgement 28/30 then 30/30; absent-feedback acknowledgement 0/30 in each half. The last comparison was run through the same two_unit_demo.run() function with verbose output disabled. These selected checks are not a replication of the remaining historical sweeps.
