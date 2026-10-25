# Foreground novelty under changing backgrounds

*Exploratory simulation report, 25 October 2026. Companion to papers
[04](04-shared-spectral-gain.md) and
[05](05-channel-calibration-and-source-interventions.md).*

## Question and protocol

Does a listener that distinguishes recurring foregrounds from modulated
ambience retain that ability when the environment changes after startup?
Paper 04's calibration-derived frontend freezes a raw/compensated power
choice from its first three audio windows. That protects stationary recall
on its controlled split, but cannot reconsider an unrepresentative start.
The numerical silent-profile correction in that paper does not fix the
frozen choice. We evaluate that limitation directly before adding adaptation.

The transition benchmark uses ten cases: stationary, slow and fast constant
backgrounds; stationary→fast, fast→stationary, stationary→slow and
slow→stationary changes; silence→fast; shallow slow modulation; and a
background spectrum change with fast modulation held fixed. Slow and fast
amplitude modulation have depth .45 at .17 and 1.3 Hz respectively. The
shallow case uses depth .15 at .07 Hz. The spectrum-change case switches
from the varied spectrum to the reference spectrum used in paper 03.
These are declared stressors, not a distribution of natural environments.

Every stream has 120 one-second windows. Six initial windows contain no
target; in the silent-start case those six windows are exactly silent.
Ordinary changes occur at zero-based window 60, with continuous room and
microphone filter state. Silence ends at window 6. Constant conditions are
split at 60 for reporting only. Changes are abrupt, so the first changed
window can contain a transition transient. Room/microphone settings vary
between seeds and stay fixed within a stream.

Each case has positive and no-target control streams. Positive cases share
the same seeded target labels and placements; their probability is .35 after
warmup. Six fixed procedural source factories are reused without independent
pitch/time transforms. No extra nuisance or simultaneous masker is injected.
The current source is calibrated to -6 dB against the residual mixture
**after** the physical channel, over its active source interval. Residual
includes background and prior source responses. Component sums and target
SNR are checked during execution. Learners receive only mixture waveforms.

The three existing methods are quiet-band/raw, always spectral gain, and the
frozen calibrated choice, using their existing settings. Development uses
seeds 1900–1901; the unchanged-method evaluation uses fresh seeds 2000–2009.
The protocol was written before evaluation; exact sources, dependencies,
settings, observations and per-seed summaries are archived. The evaluation
contains 200 streams and 72,000 actual decisions, plus conditional
source-deletion probes. Windows and decisions are not independent trials.

The primary score is correct repeat identity localized at IoU ≥.25 against
the pre-channel active source interval, reported after the declared change.
Occurrence counts and prototype admission labels carry across the change;
the evaluator does not reset them at the phase boundary. Means weight seeds
equally; intervals use 10,000 paired seed bootstrap draws with RNG seed 731.
They are descriptive, without multiplicity adjustment. Source-removal
branches start from the identical prior learner state, remove only the
current source, and discard their state after observation.

## Frozen-choice evaluation

After-phase localized repeat identity, averaged across ten seeds:

| Case | Raw | Always compensated | Frozen choice |
|---|---:|---:|---:|
| Stationary | 42.2% | 28.5% | 42.2% |
| Constant slow | 13.0% | 29.7% | 29.7% |
| Constant fast | 0.0% | 46.4% | 46.4% |
| Stationary→fast | 11.4% | 38.0% | 11.4% |
| Fast→stationary | 14.5% | 29.6% | 29.6% |
| Stationary→slow | 19.5% | 28.7% | 19.5% |
| Slow→stationary | 26.4% | 30.3% | 29.9% |
| Silence→fast | 0.3% | 43.8% | 0.3% |
| Shallow slow | 39.9% | 29.4% | 39.9% |
| Spectrum change, fast | 0.0% | 42.9% | 43.6% |

After stationary→fast or silence→fast, the frozen choice attends to **100%**
of no-target control windows. Always-compensated rates are .2% and 0.0%
respectively. After stationary→slow, frozen-choice control attention is
27.8%, with 11 newly admitted background prototypes across ten controls;
always-compensated attention/admissions are zero. The constant slow case
also has raw attention without new after-phase admissions; absence of new
admissions alone would miss an attention failure.

The frozen selector's stationary→fast apparent localized repeat identity is
11.4%, while the same prior-state source-deletion branch recognizes the
current class on 11.2% of eligible repeat windows without the current source.
These endpoints have different localization requirements and must not be
treated as identical measures or independent estimates. They show that the
apparent identity score is weak evidence of tracking the new target. The
deletion branch retains prior physical source tails, so it is not an
all-source-free or semantic-error oracle. Always-compensated repeat identity
without the current source is zero in every condition in this evaluation.
Zero observations do not establish zero future error risk.

Always enabling compensation retains a stationary cost: 28.5% versus 42.2%
localized repeat identity. Shallow slow modulation also favors the raw path.
Thus enabling compensation permanently does not resolve the trade-off.
Scores differ from earlier papers because this study uses new seeds,
post-channel SNR matching and phase-specific scoring.

## Persistent-evidence adaptation

The experimental `AdaptiveGainForegroundLearner` retains the calibrated
initial choice, then gathers a rolling nine-window history of the common
band-gain coefficient of variation. Seven values ≥.12 request compensation;
seven values ≤.09 request the raw path. A strict majority and separate
thresholds resist isolated bursts and rapid toggling, but do not guarantee
that persistent foreground cannot be confused with an environment change.
Evidence comes only from waveform statistics, without truth, phase, SNR or
stem inputs. The spectral profile remains fixed after usable calibration.

A requested switch applies to the next window. It clears the background
estimate and pending recurrence confirmations, keeps admitted vectors and
IDs, clears selection history, and suppresses admissions for three settling
windows. Keeping a vector does not prove that matching it across modes
retains identity. Saved adaptive memories have a distinct schema; load
retains admitted memories and restarts calibration/selection state.

On development seeds, the selector reduces false attention after fast
changes and improves recall after return to stationary conditions. It loses
some identity recall in stationary→slow conditions while reducing control
attention. We keep these settings unchanged and evaluate on fresh seeds
2200–2209, separately from the frozen-choice evaluation above. This actor
remains a research implementation, without a live frontend option.

## Adaptive evaluation on fresh seeds

Seeds 2200–2209 produce another 200 streams, with all four methods observing
the same mixtures: 96,000 actual decisions plus source-removal probes.
Adaptive development contains 19,200 decisions on seeds 1900–1901. Method
settings were frozen before the new evaluation. Comparisons below use the
fixed selector rerun on **these same seeds**, rather than borrowing scores
from the earlier table.

| Case | Frozen choice identity | Adaptive identity | Frozen / adaptive no-target attention |
|---|---:|---:|---:|
| Stationary | 50.1% | 50.1% | 0.0% / 0.0% |
| Constant slow | 32.4% | 32.4% | 0.0% / 0.0% |
| Constant fast | 37.7% | 37.7% | 0.0% / 0.0% |
| Stationary→fast | 9.8% | 39.2% | 100.0% / 11.7% |
| Fast→stationary | 32.2% | 33.4% | 0.0% / 0.0% |
| Stationary→slow | 29.3% | 30.8% | 25.3% / 5.7% |
| Slow→stationary | 30.9% | 33.0% | 0.0% / 0.0% |
| Silence→fast | 0.0% | 28.4% | 100.0% / 9.6% |
| Shallow slow | 41.7% | 41.7% | 0.0% / 0.0% |
| Spectrum change, fast | 38.2% | 38.2% | 0.5% / 0.5% |

Stationary→fast improves localized repeat identity by 29.4 percentage points
(paired seed-bootstrap 95% interval +14.0 to +42.3); silence→fast improves by
28.4 points (+24.6 to +31.2). Stationary recall is preserved seed by seed on
this split, without observed stationary mode switches. This is conditional
preservation in these fixtures, not a general equivalence or robustness
claim. Constant slow/fast, shallow and spectral-change primary scores also
equal the frozen selector; a change in spectrum does not itself cause a
gain-mode change in these cases.

The return-to-stationary improvements are small and uncertain: fast→stationary
+1.2 points (−4.6 to +6.8), slow→stationary +2.0 (−2.2 to +6.3). The
stationary→slow estimate is +1.6 (−6.9 to +9.4). Its negative recall difference
on two development seeds does not reproduce as a clear evaluation loss;
neither result supports a uniform benefit. Always-compensated stationary
identity is 32.6%, below the adaptive/raw 50.1%. Always-compensated
silence→fast identity is 33.5%, above adaptive 28.4%: delayed selection and
different pre-switch learning/history still have costs.

In positive streams the new mode is reached after 5–7 windows for
stationary→fast, 6–26 for stationary→slow, 7–11 for either return to
stationary, and 11 for silence→fast. These are evaluator-defined delays
until five consecutive windows have the expected mode. They measure mode
selection, not event recognition recovery; the first three switched windows
also suppress admission. There is one switch per changing-modulation
positive stream, with no later reversal in this evaluation. Foreground
content can affect switch timing; it is not an oracle background-only clock.

Residual errors matter. Stationary→slow controls admit **seven background
prototypes in seven of ten streams**, versus twelve prototypes for the
frozen selector. The adaptive positive streams admit three in three of ten
streams. Stationary→fast positive streams admit four in three of ten
streams, despite zero admissions in its pure no-target controls. Every one
of these adaptive admissions occurs while the old raw mode is still active,
before the switch. The policy preserves all admitted prototypes, including
these ambiguous no-target-window admissions. In mixed streams, the stored
candidate vector can originate before its admission window, potentially in
a foreground episode; admission-time labels alone do not establish that
vector's origin. Pure no-target controls have no injected foreground and
give stronger evidence of ambient learning. Ch-6w4.10.7 tracks this
provenance distinction without replacing the historical scoring convention.
The policy improves attention and identity without solving safe admission
during environment uncertainty.

Adaptive stationary→fast source-deletion repeat identity remains 2.1% over
the full after phase, versus 10.8% for the fixed selector. All such adaptive
current-class deletion recognitions occur before compensation is enabled;
none occur in its compensated portion. Other adaptive conditions have zero
observed source-deletion repeat identity. These conditional results retain
the previous-tail caveat and do not establish source separation.

Ch-6w4.10.4.2.2.3 evaluates admission uncertainty handling in the follow-up
below. Reducing switch delay alone could react to foregrounds or erase
meaningful novelty, so the new policy retains the original mode selector.
The broader adaptation task remains open.

## Conditional corroboration before admitting new memories

The guard leaves sensory extraction, mode selection and recognition of
established prototypes unchanged. During uncertain raw-mode conditions,
committing a **new** prototype additionally requires supporting evidence
from an always-compensated frontend observing the same mixture. This shadow
frontend keeps its own background/profile state but no candidate/admitted
prototype store. Its current event embedding must be within the existing
cosine radius .18 of the proposed candidate's stored vector. A held proposal
still returns candidate attention, retains novelty/salience, expires under
the normal bounded policy, and can be admitted later if supported. Mode
switches clear pending candidates as before. No ground-truth or stem enters
the actor.

Always requiring this support was too conservative: on development seeds
2200–2201, stationary after-phase identity fell from 48.7% to 25.2%. That
version and its 24,000 decisions remain archived as
`guarded-transition-development/`. The conditional version activates the
check after two high-CV (≥.12) values in a rolling three-window raw-mode
history. Three consecutive low-CV (≤.09) windows, or entering compensated
mode, clear uncertainty. Outside uncertainty, ordinary admission proceeds.
Its separate schema is `guarded-gain-band.v2`; all shadow/uncertainty state
resets on load, while established memories persist.

The conditional development archive has another 24,000 decisions on the same
two seeds. Its parameters are then frozen for fresh evaluation seeds
2400–2409: 200 streams and 120,000 actual decisions across five methods.
All existing methods are rerun on these same mixtures. There is no change
to the live default or automatic exposure of the guard as a live frontend.

| Case | Adaptive / guarded localized repeat identity | Control after-phase admissions, adaptive / guarded | Mixed-stream no-target admissions, adaptive / guarded |
|---|---:|---:|---:|
| Stationary | 46.5% / 44.9% | 0 / 0 | 0 / 0 |
| Stationary→fast | 34.5% / 33.4% | 0 / 0 | 5 / 3 |
| Stationary→slow | 30.1% / 29.6% | 10 / 0 | 5 / 0 |
| Shallow slow | 41.3% / 40.8% | 0 / 0 | 0 / 0 |
| Spectrum change, fast | 45.7% / 45.7% | 1 / 1 | 0 / 0 |

Other primary condition scores equal the current adaptive selector. The
stationary paired difference is −1.6 percentage points (seed-bootstrap 95%
interval −3.8 to 0.0); stationary→fast is −1.1 (−3.2 to 0.0), and
stationary→slow is −.5 (−1.6 to 0.0). Source-deleted repeat identity is zero
in all guarded conditions on this split, versus .5% for adaptive in both
stationary→fast and stationary→slow. Those zero observations retain the
source-tail/labeling caveats above; they do not prove zero field error.

False attention and localization boundaries are unchanged: withholding
admission still allows candidate attention. The audit verifies exact paired
salience, region timing, gain CV and mode trajectories; a waveform unit test
also checks feature equality. The guard targets memory commitment, not
suppression of all sensory responses to noise.

Small primary recall differences conceal some learning costs. Across the 60
observed class/seed combinations in stationary streams, admitted classes
fall from 46 to 44; both missing classes are high rustle. In
stationary→slow, coverage falls from 35 to 31, including high/low rustle and
long alarm. Among classes admitted by both methods, the equal-seed mean
extra first-admission delay is .07 occurrences for stationary and .03 for
stationary→slow. These conditional delays exclude never-admitted classes;
the lost/gained cohorts are reported separately in `guarded_analysis.json`
so missing memories cannot masquerade as fast learning.

Three remaining mixed-stream stationary→fast admissions occur at window
60, the first changed window, before two observations can establish
uncertainty. They have no shadow support, but the guard has not activated.
The remaining spectrum-change control admission occurs at window 62 of
seed 2409 while compensation is already enabled; the shadow also detects
an event. These are distinct unresolved mechanisms, tracked as
ch-6w4.10.4.2.2.4 and ch-6w4.10.4.2.4. Tightening thresholds uniformly would
not resolve the demonstrated broadband learning cost.

## Single-episode novelty check

An additional experiment inserts exactly one source episode at window 66
in each 120-window stream. Each of six procedural factories is crossed
with stationary and stationary→slow backgrounds and ten fresh seeds
2400–2409: 120 streams, 120 true episodes and 28,800 actual decisions for
adaptive/guarded. The separate development archive has 5,760 decisions on
2200–2201. The episode is calibrated to −6 dB after the channel. This is
not a one-shot learning or importance benchmark: the existing admission
rule is recurrence-based, and these targets do not recur.

Stationary localized novel attention is 61.7% for both methods. During slow
transitions, localized sensory attention is 43.3% for both; localized
**novel** attention is 35.0% for adaptive and 43.3% for guarded. Five of 60
changed-background episodes gain novel status, having previously matched a
prototype admitted on a no-target window. The equal-seed paired difference
is +8.3 points (95% interval 0.0 to +25.0), after averaging six factory
outcomes within each seed. All five gains occur in seed 2409; there are no
paired losses, but the apparent benefit is concentrated in one environment.
This small, uncertain result does not establish
better semantic novelty. It does show why memory contamination and novel
attention should be assessed separately from basic detection.

The guard preserves the detected source boundaries and features, but fails
to detect many single episodes under both policies. Withholding persistent
memory is compatible with reporting candidate novelty; deciding whether an
important one-off deserves persistent memory remains ch-6w4.3.

## Remaining scope

Beads ch-6w4.10.4.2.3 tracks deliberately foreground-contaminated startup and
independently scheduled overlapping sources. Initial calibration in this
benchmark is target-free; after silent startup, late usable-profile fitting
can encounter ordinary scheduled targets, but this does not isolate a
controlled contamination effect. Ch-6w4.1 tracks recorded, source-disjoint
audio. A [pinned ESC-10 input catalog](data/esc10-pilot/README.md) now verifies
20 clips, attribution, checksums and disjoint original sources. Paper
[07](07-recorded-replay-and-source-exclusion.md) adds the first full-duration
recorded-replay evaluation and source-exclusion scores. Its many background
admissions limit transfer of the procedural results above. Missing
speech/footstep/alarm coverage remains ch-6w4.1.3. Ch-6w4.10.6 tracks pitch-tolerant identity with matched masking and
source-deletion controls. Physical capture/cue-echo measurements remain
ch-6w4.5 and require bench access.

This is a common-level normalization and foreground-admission experiment,
not source separation, semantic novelty, importance learning, PCEN, or a
state-of-the-art audio benchmark. The literature context and PCEN distinction
remain in paper 04; the physical-component intervention method is detailed
in paper 05. The policy's practical question is whether it can reject
changing ambience while retaining recurring foreground identity.

## Reproduction

`python/src/chuff_mpu/sim/MASKED.md` documents the runner and audit commands.
Archives are `results/masked-events/transition-development/` and
`transition-evaluation/`; the adaptive development/evaluation use the
`adaptive-transition-` prefix. Adjacent protocol JSON files declare the
splits. Hash checks, complete matrices, target schedules, stored post-SNR,
phase counts and bootstrap estimates are independently recomputed. The
archive retains observations and exact generating sources, not every audio
sample; component checks are runtime assertions rather than a retained
whole-waveform audit. Earlier snapshots remain unchanged.

Both fresh evaluation audits pass. Adaptive audits additionally recompute
latency and mode counts, check that switches suppress admission, and verify
that memory size is retained at switches. Unit tests check vector/ID
retention, reset semantics, real waveform transitions, isolated evidence
bursts, latency scoring and saved-schema isolation. Guarded and one-off
audits also verify admission checks, matched class cohorts, the single-target
schedule and seed-level uncertainty. Their exact sources and frozen protocol
are retained under `guarded-transition-*` and `guarded-one-off-*`. Probe
isolation tests check that the shadow frontend is not modified by discarded
branches. All 98 project tests pass; the original grader remains 1.0.
No microphone/speaker hardware is exercised. Seeds 1900–1901, 2000–2009,
2200–2209 and 2400–2409 are now observed evidence;
future policy tuning requires another frozen evaluation split.

Codex assisted with implementation, execution, auditing and drafting. Aidan
Jones retains responsibility for interpretation. This report has not been
peer reviewed.
