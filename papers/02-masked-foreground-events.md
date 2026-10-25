# Recovering Recurring Acoustic Events Beneath Louder Noise: Band-Relative Evidence and Conservative Memory Admission in Chuff

**Aidan Jones**

17 October 2026; updated 25 October 2026 — exploratory engineering study, with Codex assistance

## Abstract

A novelty-driven acoustic agent should discover recurring foreground patterns
without filling its memory with noise. Chuff's initial foreground learner
performed well on moderately noisy procedural audio but almost failed when
events were 6 dB below the background. We investigate this failure using a
bounded, unsupervised prototype learner and paired acoustic mixtures. Three
development iterations introduce frequency-local detection, compare raw and
background-relative feature representations, replace contaminated per-bin
noise estimates with shared quiet-frame estimates, and constrain temporal
smoothing so short impacts do not acquire artificial duration. After fixing
the method, ten new evaluation seeds per condition show mean correct repeat
recognition at −6 dB improving from 2.0% to 79.5% for tonal events and from
0.3% to 64.0% for an extended inventory containing broadband bursts and long
alarms. No nuisance memories were admitted by the selected method in the
main evaluation. More severe −9/−12 dB tests expose both reduced recognition
and occasional distractor admissions. A subsequent frozen-detector variance
study on ten fresh seeds reduces −6 dB repeat recognition from 73.7% in its
reference environment to 13.5% under joint source/acoustic variation and
exposes admissions on windows without current events or distractors.
Results support a practical improvement
on these procedural environments, not general source separation or semantic
importance detection. All development stages, the local analysis plan,
source snapshots, per-window decisions and seed-level uncertainty estimates
are archived for reproduction.

## 1. Motivation and scope

The preceding Chuff study found that whole-window acoustic similarity can
learn the room background rather than the event within it. A first selective
listener improved this by detecting foreground excess, preserving temporal
structure and requiring recurrence before memory admission. However, its
archived −6 dB stress test detected only 14.2% of events and correctly
recognised 0.6% of repeats. Those results motivated the present investigation.
They are historical figures from different seeds, not the paired baseline
used to measure the new improvement here.

The problem has two coupled objectives: recover true events despite masking,
and avoid converting nuisance activity into permanent memory. Ignoring every
input solves neither. Likewise, recognising a background prototype on every
window is not successful foreground recognition.

This study asks whether local spectral evidence and a less contaminated
background estimate can improve a small agent without supervised acoustic
labels or a pretrained classifier. It also examines how apparent improvements
can fail negative controls. The contribution is an inspectable engineering
comparison and reproducible experimental record, not a claim of a new general
theory of hearing or emotion.

## 2. Related work

Spectral subtraction estimates noise and removes its expected spectral
contribution before further processing (Boll, 1979). Its usefulness depends
on the estimate remaining representative. Cohen and Berdugo (2002) address this general
problem through minima-controlled recursive averaging, with subband-dependent
updates related to signal presence. The present quiet-frame estimator is a
simpler heuristic; it does not implement that algorithm or its probability
model.

Per-channel energy normalization (PCEN) combines temporal integration,
adaptive gain control and compression. Wang et al. (2017) evaluate it for
far-field keyword spotting; Lostanlen et al. (2019) analyse its behaviour and
parameters for noisy acoustic environments. These works motivate examining
evidence relative to each frequency band's background rather than only
relative to total energy. Our detector is not PCEN: it uses estimated power
ratios and thresholds, and the stored representation is selected through
an explicit ablation. We neither reproduce their trained systems nor compare
against them empirically.

Reliable small-run comparisons require uncertainty at the level of independent
learning runs, rather than treating correlated observations as independent
samples (Agarwal et al., 2021). Here, all methods receive the same mixture
within a seed, and uncertainty in their difference is computed with paired
seed resampling. These principles are applicable even though this learner
does not use deep reinforcement learning.

## 3. Environment and task

### 3.1 Acoustic scenes

Each run contains 120 one-second, 16 kHz mono windows. Background includes
coloured Gaussian noise, 100/200 Hz hum, gradual level drift and a fan-like
340 Hz component introduced halfway through. A seeded impulse response adds
early echoes and a sparse decaying tail, with overlap carried across window
boundaries. A persistent bandpass filter approximates microphone colouration.
These remain simplified, uncalibrated channel models.

The tonal inventory comprises harmonic double chimes and rising/falling
calls with envelope and frequency variation. The extended inventory adds
low-band and high-band filtered-noise bursts plus a modulated harmonic alarm
lasting approximately 0.78–0.84 seconds. Broadband waveforms are newly sampled
on each occurrence, so matching them requires recognising family-level
structure rather than identical sample sequences. They are not recordings
of rustling, speech, footsteps or physical alarms.

Events occur with probability 0.35 after six initial background windows.
Other windows may contain short broadband impacts or random three-note
sequences. The nuisance-only suite sets event probability to zero but retains
the same background changes and nuisance generation. A randomly chosen tonal
sequence is not guaranteed to be perceptually unrelated to every other one;
occasional acoustic collisions are part of the admission challenge.

Requested SNR is calibrated over the active source segment against that
segment's background before the common room/microphone channel. Thus −6 dB
means approximately one quarter of the background power over that segment.
It is not full-window SNR, measured microphone SNR or physical sound-pressure
level. Every method receives the same mixture. Labels, nuisance flags, event
times and true SNR are supplied only to the evaluator after the decision.

### 3.2 Learning and scoring

All compared methods use the same bounded prototype-memory mechanism:
32 admitted patterns, up to 32 pending candidates, cosine-distance matching
at 0.18, and two separated sightings before admission. Adjacent windows alone
do not confirm recurrence. Pending candidates expire after 40 windows. An
admitted prototype remains fixed, while successful matches refresh its
retention priority. No environmental class label trains these memories.

The evaluator assigns each admitted ID the label present at admission.
Correct repeat recognition requires subsequently selecting an ID admitted
on that same class. Its denominator includes all events after the first two
ground-truth occurrences of a class, including events a detector misses.
This exposes both detection and recognition failures. Merely acquiring a
prototype during a labelled window does not establish a useful representation.

False admissions count permanent memories created on nuisance or background
windows. Background attention also counts provisional candidates and familiar
matches on such windows. These are distinct costs: momentary curiosity does
not necessarily become a stored memory.

## 4. Frontends and development iterations

### 4.1 Original selective learner

The original frontend computes a short-time power spectrum, estimates a
per-bin floor from a corrected lower quantile, and combines that floor with
recent history. It subtracts estimated noise, then requires sufficient excess
energy over the entire mel spectrum. Duration and spectral-flatness gates
reject many short or noise-like events. Three ordered spectral summaries form
the 96-dimensional feature vector. Its algorithmic behaviour is retained as
the comparison, including its original thresholds.

A weak narrowband signal can have substantial evidence in a few bands while
contributing little to the whole-spectrum ratio. Conversely, a long event
can occupy enough frames to contaminate a per-bin lower quantile and become
part of its own estimated background.

### 4.2 Band-local evidence

Let P(f,t) be power, H(b,f) the mel filters and N(f) the estimated noise.
The detector computes the background-relative band power

~~~text
R(b,t) = sum_f H(b,f) P(f,t) / max(sum_f H(b,f) N(f), epsilon).
E(t) = max(max_b smooth(R(b,t)) / 3.5,
           mean_b smooth(R(b,t)) / 1.5).
~~~

The maximum-band branch supports narrowband events; the mean-band branch
supports broader changes. A five-frame moving average stabilises evidence.
Short gaps are closed, and the highest integrated-evidence eligible region
is selected. This implementation selects at most one region per window; it
does not separate multiple overlapping foregrounds.

The band variants remove the flatness rejection and allow regions up to
0.95 seconds. Therefore the original-to-band comparison changes a frontend
configuration, not a single isolated threshold. Comparisons among band
variants isolate the feature/noise-estimation choices more closely.

### 4.3 Three development stages

Development used seeds 0–2 at −6/0 dB. Separate regression fixtures used seeds
42–45; none was reused for the main evaluation or stress estimates.

1. **Local detection and feature ablation.** Band-local evidence recovered many
   masked events. Replacing raw excess features with background-relative
   features did not improve recognition consistently. We retained that
   variant as a negative ablation rather than selecting it on intuition.
2. **Shared quiet frames.** To reduce long-event contamination, select the
   quietest 20% of time frames using smoothed total power, estimate each
   frequency's mean power over those same frames, and combine it with the
   historical estimate by taking the larger value. This improved long-event
   discovery. It still assumes that a window contains informative quiet
   frames and can fail when foreground fills the window.
3. **Unsmoothed support.** A nuisance-only regression fixture revealed three
   admitted impact patterns. Smoothing can extend short impulses enough
   to pass a duration gate. The selected method requires at least 80 ms
   of evidence in the unsmoothed ratios within an eligible region. This
   removed admissions in that fixture, at a cost in weak-event recall.

| Development stage | Extended −6 dB mean repeat recognition | Important qualification |
|---|---:|---|
| Local gate with raw excess features | 66.2% | Long alarms still undetected |
| Shared quiet-frame estimate | 78.3% | Separate impact fixture admitted three nuisance patterns |
| Added instantaneous-support requirement | 70.3% | Impulse fixture passed; method then frozen |

These stage summaries describe adaptive development, not independent
confirmatory comparisons. Stage source snapshots and outcomes are retained
so the trade-off is visible.

### 4.4 Selected method and ablations

The selected **quiet-band** method uses shared quiet-frame noise estimates,
band-local detection, instantaneous temporal support, and raw residual
mel-band features. Within each of three temporal sections, features are
normalised, square-root compressed and concatenated into a unit vector.

**Band-gate/raw** uses the same final band-detector and temporal-support
configuration but retains per-bin quantile noise estimation. **Band-relative**
uses that estimator and stores background-relative excess instead of raw
excess. The latter is a simple normalization ablation, not the PCEN transform.
All methods keep the same memory capacity, match threshold and confirmation
count. No wall-clock, energy or MCU-resource superiority is claimed.

## 5. Evaluation protocol and uncertainty

After development, the [local analysis plan](results/masked-events/protocol.json)
fixed settings and endpoints before inspection of new evaluation outcomes.
This was not external preregistration. Evaluation uses seeds 300–309 at
−6/0/+6 dB; stress testing uses seeds 400–409 at −12/−9 dB. Each condition
includes tonal, extended and nuisance-only suites. No detector parameters
were subsequently retuned using these results.

Reported percentages are means of per-seed rates. Intervals are percentile
intervals from 10,000 paired bootstrap resamples of ten seeds within a suite
and SNR. They are descriptive estimates for this generator, not
multiplicity-adjusted confirmatory tests or distributions over all real
acoustic environments. SNR conditions share schedules within a seed; we do
not pool their windows as independent observations.

Class-level summaries omit a seed only when that class has no eligible repeat
events; eligible counts are recorded. A zero-width bootstrap interval for
zero observed nuisance admissions does not prove the true admission risk is
zero. The more severe tests provide a concrete counterexample to that reading.

## 6. Results

### 6.1 Main paired evaluation

| Inventory | SNR | Original repeat recognition | Quiet-band repeat recognition | Quiet-band detection |
|---|---:|---:|---:|---:|
| Tonal | −6 dB | 2.0% | 79.5% | 88.0% |
| Tonal | 0 dB | 89.3% | 94.6% | 99.3% |
| Tonal | +6 dB | 96.0% | 96.3% | 100.0% |
| Extended | −6 dB | 0.3% | 64.0% | 78.3% |
| Extended | 0 dB | 70.0% | 92.9% | 98.4% |
| Extended | +6 dB | 80.5% | 94.1% | 100.0% |

At −6 dB, the tonal recognition improvement is **77.5 percentage points**
(paired bootstrap interval 71.3–83.1 points); the selected method's absolute
rate has interval 72.8–85.1%. For the extended inventory, improvement is
**63.7 points** (59.0–68.2 points), with absolute rate interval 59.6–68.2%.
At +6 dB on tonal events, methods are already close; the small difference
does not motivate a broad superiority claim in easy conditions.

The selected method admitted **zero nuisance memories across the main
evaluation**: 90 streams spanning the three suites and three SNRs. This is
an observed count, not 90 independent Bernoulli trials or a guarantee. In
the nuisance-only suite, provisional background attention averaged about
5.8% of windows. Most such attention did not survive confirmation.

### 6.2 What each change contributes

For the extended inventory at −6 dB:

| Frontend | Detection | Correct repeat recognition |
|---|---:|---:|
| Original selective frontend | 6.0% | 0.3% |
| Band-gate/raw | 63.1% | 49.0% |
| Band-relative feature ablation | 63.1% | 47.9% |
| Quiet-band, selected | 78.3% | 64.0% |

The background-relative feature ablation offers no improvement here despite
using the same detector. Shared quiet-frame estimation adds benefit beyond
the final band gate alone. The results support separating detection-time
normalization from the choice of stored representation; they do not show
that all forms of feature normalization or PCEN are ineffective.

### 6.3 Aggregates conceal a remaining long-event failure

Within the extended inventory at −6 dB, selected-method mean repeat
recognition is 91.7% for high-band bursts, 60.5% for low-band bursts, but
only **16.1% for long alarms**. Long-alarm detection is 45.6%, so failures
include both missed evidence and unstable recognition. At 0 dB, long-alarm
recognition rises to 74.1%, compared with zero for the original frontend.

The high-band burst rate uses nine eligible seeds, because one seed lacks
eligible repeats of that class; the other two reported classes use ten.
The separate class archive records both event counts and eligible seeds.
Six-class aggregate recognition must not be read as solving every class.

### 6.4 More severe masking

| Inventory | SNR | Quiet-band detection | Quiet-band repeat recognition | Nuisance admissions, total over ten runs |
|---|---:|---:|---:|---:|
| Tonal | −9 dB | 78.7% | 66.9% | 1 |
| Tonal | −12 dB | 64.9% | 35.3% | 1 |
| Extended | −9 dB | 48.6% | 31.2% | 0 |
| Extended | −12 dB | 32.6% | 16.1% | 0 |

The original frontend has zero correct repeat recognition in these stress
conditions. Improvement therefore persists, but selected-method performance
is far from satisfactory. Nuisance-only stress streams admit two distractors
at −12 dB and one at −9 dB across ten runs each. Audited false admissions
occurred on generated distractors, not ambience-only windows. Recurrence
confirmation reduces accidental storage; it does not eliminate collisions
between acoustically similar nuisance sequences.

## 7. Discussion

Three engineering lessons follow. First, a signal can be weak globally and
strong locally: the original energy rule discarded useful narrowband
evidence before memory could help. Second, an estimator can absorb the event
it is supposed to reveal, especially when that event occupies most of a
window. Selecting shared quiet frames helped here, with clear assumptions
and remaining long-alarm failures. Third, temporal processing changes the
meaning of a duration threshold. Smoothing improves robustness but must not
be mistaken for longer physical evidence.

The last point also explains why simply maximising development recall was
the wrong objective. The second iteration had higher recall than the
selected method on development seeds, yet failed an impulse-only control.
Accepting a recall reduction was justified by the admission objective and
was checked on new seeds rather than hidden by selecting a favourable run.

Novelty here means distance to admitted acoustic prototypes. Neither that
distance nor frequency of recurrence establishes human importance. A repeated
fan, a recurring impact and an alarm may require different responses even
when each is acoustically learnable. The present duration and stationarity
rules encode conservative task assumptions, including rejection of many
events shorter than 80 ms. They may reject meaningful clicks or footsteps.
This is a limitation to evaluate, not a semantic distinction the model has
discovered.

## 8. Reproducibility and software checks

The result archive contains the plan, three development stages, evaluation
and stress manifests, exact source snapshots, per-seed summaries, aggregates,
class-level results and compressed decisions. The main evaluation contains
43,200 window/method decisions and the stress evaluation 28,800. These are
not counts of independent statistical replicates. Each snapshot can be
overlaid on the base repository revision named in its manifest; dependency
versions and hashes are recorded.

An independent audit recomputes prototype-label scoring, class counts,
false-admission counts and bootstrap aggregates from saved decisions, and
checks archived source hashes. All 72,000 decisions pass that audit. The
source snapshots also retain the earlier, less conservative development
variants rather than substituting the final code retrospectively.

The 26-test suite passes, including masked recovery fixtures, broadband and
long-event recognition, nuisance-only rejection, paired-resampling behaviour,
saved-schema isolation, silent replay, controller transitions and mocked
devices. The original memory/repertoire grader remains 1.0. These checks
establish behaviour on their fixtures; they do not validate new field audio
or physical hardware.

The new frontend is available experimentally through
`chuff-selective --frontend band`. It has a separate saved feature schema;
the original frontend remains the default, and their memories cannot be
cross-loaded. No automatic legacy voice/dream feature mixing is introduced.

## 9. Limitations and next iteration

The next experiments are tracked in Beads; this table describes their
scientific purpose rather than replacing the tracker. Each subsequent
detector revision needs fresh evaluation examples, since the present
evaluation and stress seeds are now observed.

| Issue | Experiment | Evidence needed before claiming improvement |
|---|---|---|
| ch-6w4.8 | Independently vary procedural sources and acoustics | Frozen detectors; individual factors and joint variation; paired seeds and nuisance-only controls. Results below. |
| ch-6w4.10 (P1) | Stable identities and rejection of modulated ambience | Diagnose admission versus identity errors; improve pitch/time robustness without merging distinct classes; pure ambience controls and fresh held-out seeds. |
| ch-6w4.6 | Sustained events without useful quiet frames | Separate detection failures from unstable stored identities; continuous and boundary-spanning events; stationary fan/hum controls. |
| ch-6w4.9 | Measure masking after the channel; separate pitch and duration | Evaluator-only filtered source/background stems; matched effective SNR; independent pitch/time factors with explicit class-identity assumptions. |
| ch-6w4.1 | Disjoint recorded-audio benchmark | Licensed or owned recordings with source-disjoint splits; independently assigned rooms/backgrounds; speech, footsteps, alarms and overlap. |
| ch-6w4.7 | PCEN and adaptive noise-estimation baselines | Matched memory and tuning budgets on development recordings, then sealed evaluation; isolate detection and representation changes. |
| ch-6w4.3 | Recurrence, learning progress and one-off importance | Admission delay, nuisance memory, missed rare events and semantic relevance measured separately; novelty alone is insufficient. |
| ch-6w4.5 | Physical capture and cue feedback | Measured audio gaps, latency, cue self-echo, CPU/RAM and energy with real hardware. |

The next important test is a disjoint recorded-audio benchmark, already
tracked as ch-6w4.1. It should vary foreground recordings, backgrounds, room
responses and microphone characteristics independently, including overlap
and classes not used to design these heuristics. Synthetic broadband bursts
do not substitute for recorded speech or footsteps.

Long alarms under masking remain a specific target: reduce contamination
when no quiet frames exist, and distinguish detection from recognition
instability. Compare shared-quiet estimation with principled adaptive noise
estimators and a direct PCEN baseline at matched memory/evaluation budgets.
Such comparisons should use development audio first and a new sealed
evaluation split, rather than repeatedly tuning on seeds 300–409.

Rare distractor admissions under severe masking require an additional
admission/uncertainty experiment. More confirmations, temporal template
consistency, or learning-progress signals may reduce them but can delay
important one-off responses. Report that trade-off and the probability of
missing useful events, not only average recall.

Finally, measure capture latency, missed audio and cue self-echo on physical
hardware (ch-6w4.5), alongside CPU, RAM and energy. A statistically better
procedural frontend is not yet a deployable low-power auditory system.

## 10. Follow-up: independently varied procedural acoustics

### 10.1 Design and scope

After the masked-event analysis, ch-6w4.8 tested whether the fixed method
survives broader procedural variation. The local prospective plan is
`results/masked-events/variance-protocol.json`. Neither frontend was retuned.
Ten fresh event seeds, 600–609, were paired across −6 and 0 dB and seven
conditions, with environment seeds equal to 100000 plus the event seed.
Each stream contains 120 one-second windows. The six foreground classes,
event probabilities, recurrence rules and scoring remain unchanged.

The reference is byte-identical to `ChallengeScene` for the same seed.
Individual factors change the following, with joint variation combining all:

| Factor | Variation |
|---|---|
| Source | Per-occurrence waveform resampling with duration multiplier 0.75–1.3, capped at 0.94 s; pitch and duration change together. Soft boundaries are reapplied. |
| Room | Response length 0.03–0.5 s and 24 randomized sparse signed echoes with a decaying envelope. These are filter response lengths, not calibrated RT60 measurements. |
| Microphone | Second-order bandpass with lower cutoff 60–450 Hz, upper cutoff 2.8–7.2 kHz at 16 kHz sampling, and gain −12 to +12 dB. |
| Background | AR noise pole 0.35–0.97; hum 45–180 Hz; 0.17 Hz amplitude modulation of depth 0.45; halfway fan addition at 220–950 Hz and level change −6 to +9 dB. |

Room and microphone parameters are fixed per run, while source transforms
vary per occurrence. This measures learning within varied environments,
not transfer of saved memories between microphones. Background and channel
parameters have separate RNG streams; changing the environment does not
change event scheduling. The learner receives only the mixed waveform.
Active SNR is matched before the common channel, as in the original study;
filtering can change effective masking differently for different classes.
The design therefore tests robustness to these transformations, not isolated
channel invariance at equal post-channel SNR. Resampling also changes pitch
and duration together and can create class-identity ambiguity. These limits
motivate ch-6w4.9.

The seventh condition uses joint variation with zero foreground probability
but retains nuisance distractors. It is a no-foreground control, not a pure
ambience stream. All conditions reuse the same paired seeds and must not be
counted as independent environmental replicates. Bootstrap intervals resample
the ten seed pairs within each condition/SNR, with 10,000 draws.

### 10.2 Results and failure diagnosis

Mean correct repeat recognition, including missed events in the denominator:

| Condition | Original, −6 dB | Quiet-band, −6 dB | Original, 0 dB | Quiet-band, 0 dB |
|---|---:|---:|---:|---:|
| Reference | 0.3% | 73.7% | 75.8% | 92.8% |
| Source variation | 0.0% | 37.2% | 39.0% | 44.9% |
| Room variation | 1.1% | 68.8% | 80.8% | 92.5% |
| Microphone variation | 5.1% | 76.5% | 75.5% | 92.8% |
| Background variation | 0.8% | 33.1% | 53.0% | 71.4% |
| Joint variation | 0.7% | 13.5% | 20.3% | 33.3% |

The new reference differs from the earlier 64.0% extended-inventory result
because these are different seeds; it is not a further method improvement.
On the same fresh seeds, source variation reduces quiet-band recognition
at −6 dB by 36.5 percentage points (paired descriptive 95% bootstrap interval
−43.6 to −29.5). Joint variation reduces it by 60.2 points (−70.3 to −48.5).
Absolute joint recognition is 13.5% (7.6–19.8), compared with the reference
73.7% (67.2–79.4). These are small-sample descriptive intervals, without
multiple-comparison adjustment.

Detection is less degraded than identity recognition: at −6 dB it is 84.8%
in the reference, 83.1% with source variation and 70.2% jointly. At 0 dB,
joint detection is 91.8%, but correct repeat recognition is only 33.3%.
These gaps indicate a failure after or alongside detection, but do not alone
distinguish unstable embeddings, failed candidate confirmation, prototype
fragmentation or incorrect matches. Joint −6 dB long-alarm repeat recognition
is 4.3%, with 46.4% detection. Temporal alignment or multiple exemplars might
help; this study has not tested that hypothesis.

Noise rejection also fails under the broader background process:

| Condition | Quiet-band false admissions, −6 dB | Quiet-band false admissions, 0 dB |
|---|---:|---:|
| Reference, source, room or microphone individually | 0 each | 0 each |
| Background variation | 8 | 9 |
| Joint variation | 9 | 9 |
| Joint no-foreground control | 12 | 12 |

Counts are totals over ten streams per cell. In the no-foreground control,
all 12 admissions at each SNR occur on windows without a current distractor.
In foreground-bearing joint streams, seven of nine admissions at −6 dB and
eight of nine at 0 dB likewise have no current event or distractor. Persistent
room/filter tails can carry earlier sounds across windows, so these labels
do not by themselves prove a purely ambient causal origin. The original
method admits one distractor in the 0 dB no-foreground control and none
elsewhere in this study, while missing many foreground events.
Quiet-band background attention rises from 5.2% in the reference to about
24–27% under varied backgrounds/joint conditions. The earlier absence of
ambience-only admissions therefore does not generalize to this simulator.

This is a useful negative result: better masked-event detection has not
solved stable novelty memory or background rejection. The next priority,
ch-6w4.10, is to diagnose those errors and compare temporal alignment,
bounded multiple exemplars and background/confirmation policies without
simply broadening matching until unrelated events merge. Add pure ambience
controls, report detection/admission/identity separately, and preserve tests
for rare useful events. Seeds 600–609 are now observed evidence and must
not be reused as a sealed evaluation set for that work.

### 10.3 Evidence and validation

`results/masked-events/variance-evaluation/` contains exact source snapshots,
sampled environment parameters, per-class counts, summaries and all 33,600
window/method decisions from 140 streams. `variance-analysis.json` records
class statistics and paired contrasts against reference acoustics. The
variance audit checks source hashes, scores and intervals, the complete
condition/seed matrix, and identical event/nuisance schedules across factors,
SNRs and methods. All checks pass. Thirty software tests pass, including
reference equivalence, determinism, finite/window-bounded audio, matched
pre-channel SNR and scheduling independence. The legacy grader remains 1.0.
No claim of field-audio validity, general sound separation, measured-room
fidelity or semantic importance follows from these procedural tests.

## 11. Follow-up: memory error attribution and admission-policy trade-offs

### 11.1 Fixed diagnostic design (25 October 2026)

The next experiment, ch-6w4.10.1, partitions the failures exposed in section
10 before changing the representation. The prospective local plan is
`results/masked-events/admission-protocol.json`. Five fixed policies receive
identical waveform windows: the original frontend, the existing quiet-band
default, quiet-band requiring four confirmations, quiet-band with cosine
matching distance increased from 0.18 to 0.28, and both changes together.
The broader threshold affects candidate association as well as admitted
prototype recognition. No frontend, representation or noise-estimation
threshold is otherwise changed. This is a policy sensitivity experiment,
not evidence that either change should become the production default.

Fresh seeds 700–709 are paired across reference, source-only and joint
variation, plus joint no-foreground streams with distractors and separate
pure-ambience streams with neither events nor distractors. Each run has
120 windows at nominal −6 or 0 dB before the channel. Pure ambience has
no applicable event SNR and is byte-identical across those nominal settings;
its repeated renderings must not be counted as additional independent
controls. There are ten unique pure-ambience streams.

For every policy, repeat eligibility begins after two true occurrences of
the evaluator class, including missed detections. Requiring four confirmed
sightings therefore incurs its actual delay rather than receiving a later
scoring denominator. Each repeat window has exactly one evaluator outcome:
missed detection, pending candidate, first admission, duplicate admission,
wrong admitted identity or correct admitted identity. A prototype receives
its evaluator label at admission time and is never retrospectively relabelled
to favour subsequent matches. Labels never reach the learner.

Pending candidates can reflect insufficient confirmations, changing features,
expiry or capacity pressure; that category does not diagnose the cause by
itself. Duplicate admission means an additional prototype for an already
admitted procedural class, which is a fragmentation proxy rather than proof
that the acoustic instances should share one identity. First admission is
also recorded by true occurrence number, counting earlier missed events;
classes never admitted remain explicitly absent rather than being assigned
a finite learning delay. False admissions distinguish current distractors
from windows without a current source. Only the pure-ambience control rules
out earlier event tails as their cause.

### 11.2 Results: confirmation count is not a solution to ambience learning

The following are mean per-seed correct repeat-recognition rates. Differences
from section 10 use new seeds and must not be interpreted as changes to the
unchanged quiet-band algorithm.

| Scene / SNR | Original | Quiet-band | Four confirmations | Broader matching | Both changes |
|---|---:|---:|---:|---:|---:|
| Reference / −6 dB | 0.3% | 65.3% | 22.9% | 64.6% | 27.9% |
| Reference / 0 dB | 76.5% | 92.0% | 38.0% | 91.2% | 40.8% |
| Source variation / −6 dB | 0.0% | 32.5% | 5.2% | 39.4% | 11.4% |
| Source variation / 0 dB | 38.3% | 45.9% | 13.0% | 51.6% | 17.5% |
| Joint variation / −6 dB | 0.3% | 7.8% | 0.0% | 7.5% | 1.1% |
| Joint variation / 0 dB | 17.2% | 29.6% | 4.9% | 24.9% | 3.7% |

Broader matching improves source-only recognition at −6 dB by 6.9 percentage
points relative to quiet-band (paired seed bootstrap 95% interval 2.5–11.7).
It does not establish robust identity: on joint variation at −6 dB the change
is −0.3 points (−2.3 to +2.0), and at 0 dB mean recognition decreases from
29.6% to 24.9%. Four confirmations reduce source-only −6 dB recognition by
27.3 points (−30.5 to −23.4). These are descriptive intervals from ten seed
pairs, without adjustment for the multiple comparisons. Longer streams
could change the cost of delayed admission; this study measures the fixed
120-window budget rather than asymptotic learning.

Pure ambience decisively resolves one ambiguity from the earlier study:

| Policy | False prototypes across ten unique pure-ambience streams | Streams with any false admission |
|---|---:|---:|
| Original | 0 | 0/10 |
| Quiet-band | 13 | 10/10 |
| Four confirmations | 10 | 10/10 |
| Broader matching | 10 | 10/10 |
| Both changes | 10 | 10/10 |

The nominal −6/0 dB pure-ambience renderings are identical, so the table
counts each stream once. Requiring four confirmations changes the mean false
prototype count by −0.3 per stream (paired bootstrap interval −0.6 to 0.0),
while leaving the observed affected-stream fraction unchanged. Fewer
prototypes alone would therefore be a misleading success criterion. Every
quiet-band stream acquires a false memory before the halfway background
change; first admissions occur at zero-based windows 5–18. This establishes
failure on the initial modulated background, not only on the later fan/level
transition. The original frontend rejects these ten controls but also misses
many masked foregrounds. No field risk estimate follows from ten procedural
backgrounds.

### 11.3 Where the repeat errors occur

Each foreground-bearing scene has 257 eligible repeat windows pooled across
the ten seeds. The following counts use that pooled denominator; they are
not the equally weighted per-seed means in the recognition table above.

| Quiet-band at −6 dB | Reference | Source variation | Joint variation |
|---|---:|---:|---:|
| Missed detection | 42 | 31 | 69 |
| Pending candidate | 20 | 88 | 87 |
| First admission on this repeat | 23 | 31 | 20 |
| Duplicate class admission | 2 | 16 | 9 |
| Wrong admitted identity | 0 | 5 | 49 |
| Correct admitted identity | 170 | 86 | 23 |
| Total | 257 | 257 | 257 |

Source variation increases pending candidates and duplicate class admissions
despite detection remaining high (84.4% across all event windows). Joint
variation adds missed detections and wrong matches. Broader matching reduces
joint −6 dB pending candidates from 87 to 63, but wrong identities rise from
49 to 79, with correct identities changing from 23 to 22. On source-only
variation, it raises correct identities from 86 to 105 while wrong identities
also rise from 5 to 18. Simply widening the acceptance radius is therefore
not a general solution to the observed identity problem.

Four confirmations on joint −6 dB variation produce 136 pending candidates,
49 wrong identities and zero correct identities. Admission delay and
background prototypes both remain relevant; confirmation count does not
create an event/background distinction when the background itself recurs.
The diagnostics are evaluator bookkeeping rather than a causal attribution
of each internal failure, but now separate the main observable failure modes.

### 11.4 Next experiment and reproducibility

Keep the existing defaults: none of these fixed policy changes resolves the
combined identity/noise problem. The next experiment, ch-6w4.10.2, should isolate why
modulated ambience passes the noise estimator and gate, including modulation
depth/rate, stationary matched spectra and foreground-positive controls.
Evaluate background rejection before increasing memory tolerance; otherwise
an invariant representation could make nuisance memories still easier to
match. Compare any improved gate with the unchanged quiet-band learner on a
new frozen split, with both affected-stream counts and total false admissions.
Ch-6w4.10 remains open for this broader robustness problem; this diagnostic
subtask does not claim to solve semantic relevance or importance.

`results/masked-events/admission-evaluation/` contains the 100-run matrix,
source snapshots, sampled parameters, class metrics, diagnostic partitions
and 60,000 waveform-window/method decisions. `admission-contrasts.json`
contains paired seed comparisons to quiet-band. The audit verifies source
hashes, raw-decision scoring, exhaustive repeat partitions, fixed denominators,
the complete matrix, matched schedules and source-free control truth. All
checks pass. Thirty-four software tests pass; the legacy grader remains 1.0.
Seeds 700–709 are now observed evidence and cannot be reused as a sealed
test set for a method designed from these results.

## 12. Subsequent controlled study

For the subsequent controlled modulation/noise-estimator experiment, see
[paper 03](03-modulation-and-noise-estimation.md). It establishes the
slow-modulation background-admission failure and shows why window-level
detection and zero false-prototype counts can both be misleading under
fast modulation. Historical detection values above retain their original
definition; event localization is a new tracked requirement (ch-6w4.10.3).

## AI assistance and responsibility

Codex assisted with implementation, experimental design, source inspection,
primary-literature checks, execution, auditing and manuscript drafting.
Aidan Jones retains responsibility for verifying and interpreting the
results. This is an exploratory project report, not a peer-reviewed paper.

## References

Agarwal, R., Schwarzer, M., Castro, P. S., Courville, A. C., & Bellemare, M. G.
(2021). Deep reinforcement learning at the edge of the statistical precipice.
*Advances in Neural Information Processing Systems*, 34, 29304–29320.
[Proceedings](https://proceedings.neurips.cc/paper/2021/hash/f514cec81cb148559cf475e7426eed5e-Abstract.html).

Boll, S. F. (1979). Suppression of acoustic noise in speech using spectral
subtraction. *IEEE Transactions on Acoustics, Speech, and Signal Processing*,
27(2), 113–120. [Article](https://course.ece.cmu.edu/~ece491/homework/Boll79.pdf).

Cohen, I., & Berdugo, B. (2002). Noise estimation by minima controlled recursive
averaging for robust speech enhancement. *IEEE Signal Processing Letters*,
9(1), 12–15.
[Article](https://www.ee.technion.ac.il/Sites/People/IsraelCohen/Publications/SPL0102.pdf).

Lostanlen, V., Salamon, J., Cartwright, M., McFee, B., Farnsworth, A., Kelling, S.,
& Bello, J. P. (2019). Per-channel energy normalization: Why and how.
*IEEE Signal Processing Letters*, 26(1), 39–43.
[Author manuscript](https://www.justinsalamon.com/uploads/4/3/9/4/4394963/lostanlen_pcen_spl2018.pdf).

Wang, Y., Getreuer, P., Hughes, T., Lyon, R. F., & Saurous, R. A. (2017).
Trainable frontend for robust and far-field keyword spotting.
*Proceedings of ICASSP 2017*, 5670–5674.
[doi:10.1109/ICASSP.2017.7953242](https://doi.org/10.1109/ICASSP.2017.7953242).
