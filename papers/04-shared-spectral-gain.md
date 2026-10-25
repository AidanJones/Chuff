# Shared Spectral Gain Reduces Modulation-Driven Acoustic Memories, with a Stationary-Recall Cost

**Aidan Jones — 25 October 2026, exploratory engineering report with Codex assistance**

## Result and scope

Chuff's quiet-band listener can mistake modulated background for recurring
foreground. Three development stages compare common-gain compensation before
its unchanged gate and prototype memory. On ten fresh evaluation seeds,
correct repeat identity with IoU ≥0.25 increases from 1.2% to 55.1% under
fast modulation and from 26.9% to 45.6% under slow modulation. Fast-background
attention falls from 97.5% to 0.1%, without background prototype admissions
by the new candidate. Stationary-background recognition falls from 67.3% to
45.5%. This is a useful procedural trade-off, not uniform robustness or
semantic importance detection.

A separate follow-up below selects raw versus compensated processing from
unlabelled calibration audio. On its own fresh split it preserves stationary
recognition while improving the modulated cases; this conditional result does
not remove the need to test calibration and later environment changes.

[Paper 03](03-modulation-and-noise-estimation.md) supplies the controlled
modulation environment and explains why detection, identity, false attention
and chance overlap need separate measurement. The present task is ch-6w4.10.4.

Gain control is established frontend practice. Wang et al. (2017) introduced
PCEN using frequency-channel gain control and dynamic compression for keyword
spotting; that supervised task and learned frontend differ from our
unsupervised prototype-memory experiment.
[Primary paper](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/45911.pdf).
Our shared-gain transform is not PCEN and does not demonstrate superiority
over it. Direct published-method baselines remain ch-6w4.7. The
[librosa 0.11 PCEN reference](https://librosa.org/doc/0.11.0/generated/librosa.pcen.html)
specifies input scaling and streaming filter-state handling that must be
configured deliberately for that comparison.

## Design and development

Each run contains 120 one-second windows, varied procedural background spectrum
and room/microphone filters, and stationary, slow (0.17 Hz) or fast (1.3 Hz)
background modulation. Nonzero modulation depth is 0.45. Each condition has
pure-ambience and six-class foreground-positive streams, at active −6 dB SNR
before the common channel. There are no distractors, halfway spectral changes
or expanded source pitch/duration transforms. Every method receives identical
audio and retains the same matching radius, two separated confirmations,
prototype budget and repeat denominator after two true class occurrences.

Development seeds are 1100–1102; untouched evaluation seeds are 1200–1209.
The local prospective records under `results/masked-events/` include
`gain-protocol.json`, the second/third development protocols,
`gain-selection.json` and `gain-frozen-evaluation.json`. These are not external
preregistration. Failed stages and exact sources are preserved.

Stage one normalizes the waveform by the square root of a sample-by-sample
exponential power average, before the existing band learner. Time constants
0.05, 0.2 and 0.5 s all fail the predeclared ≤10% pure-ambience attention
criterion under fast modulation. Stage two tests 0.005, 0.01 and 0.02 s.
All reject the development backgrounds, but foreground detection falls to
approximately 28–33% for the best eligible RMS candidate, 0.02 s. Its mean
primary localized repeat recognition across the three positive cells is
19.9%. Turning down all attention is not foreground discovery.

Stage three uses the existing 32-band mel filterbank. Let B[m,t] be observed
band power and P[m] the mean band profile of the first three unlabelled
calibration windows. Compute `median_m(B[m,t] / max(P[m], 1e-12))`, smooth that
common level with a fixed 0.02 s exponential filter, and divide each STFT
power frame by the smoothed scalar, floored at 1e-6. The original quiet-frame
noise estimator, excess features, gate and memory then process that power.
The profile freezes after warmup. The gain filter carries state across
windows; the complete one-second-window frontend is not claimed to deliver
sample-by-sample real-time decisions.

A localized foreground can affect a few bands without dominating the median.
This is a hypothesis, not a guaranteed clean-noise estimate: broadband sources,
spectral drift and contaminated warmup can violate it. The fixed candidate
has no development background attention/admissions and 49.7% mean primary
localized repeat recognition, but loses stationary recall. It is frozen before
evaluation. All six RMS times are evaluated too; there is no post-evaluation
parameter change or retrospective replacement of unsuccessful variants.

## Fresh evaluation

Mean per-seed correct repeat identity **and** IoU ≥0.25, including missed events:

| Method | Stationary | Slow modulation | Fast modulation |
|---|---:|---:|---:|
| Original | 17.0% | 9.5% | 3.1% |
| Quiet-band | 67.3% | 26.9% | 1.2% |
| Binwise noise ablation | 46.2% | 28.2% | 6.8% |
| RMS 0.005 s | 17.7% | 18.1% | 17.2% |
| RMS 0.01 s | 17.8% | 18.8% | 18.1% |
| RMS 0.02 s | 20.3% | 20.4% | 22.1% |
| RMS 0.05 s | 28.3% | 29.2% | 13.0% |
| RMS 0.2 s | 49.4% | 49.5% | 5.0% |
| RMS 0.5 s | 57.5% | 47.8% | 3.4% |
| Shared spectral gain | 45.5% | 45.6% | 55.1% |

Spectral-gain versus quiet-band differences are +53.9 percentage points under
fast modulation (paired seed-bootstrap 95% interval +44.8 to +64.9), +18.7
under slow modulation (+10.6 to +25.0), and **−21.8 on stationary backgrounds**
(−28.4 to −15.3). Absolute fast recognition is 55.1% (45.4–66.3).
Intervals use 10,000 paired seed draws per cell, are descriptive, and have
no multiplicity adjustment. Conditions share ten seeds; these are not thirty
independent seed replicates. There is no uniformly preferred frontend.

Pure-ambience controls:

| Method | Slow attention | Fast attention | Slow false prototypes / affected streams | Fast false prototypes / affected streams |
|---|---:|---:|---:|---:|
| Quiet-band | 26.2% | 97.5% | 10 / 10 of 10 | 0 / 0 of 10 |
| Binwise | 0.0% | 97.5% | 0 / 0 of 10 | 1 / 1 of 10 |
| RMS 0.02 s | 0.0% | 0.0% | 0 / 0 of 10 | 0 / 0 of 10 |
| Shared spectral gain | 0.0% | 0.1% | 0 / 0 of 10 | 0 / 0 of 10 |

All four methods have zero stationary-control attention and admissions.
Spectral-gain fast attention is one of 1,200 windows, and it has no false
admissions in the foreground-bearing streams either. These observations
do not guarantee zero field risk. The 0.5 s RMS candidate admits six slow
background prototypes across six streams. Successful rejection by the faster
RMS variants must be considered alongside their low foreground recall.

## Localization and overlap controls

Spectral-gain localized detection is 57.3%, 58.1% and 71.6% on stationary,
slow and fast positives. Detection falls relative to quiet-band even when
correct identity improves: fewer decisions can coexist with better memories.
Recall and identity cannot substitute for one another.

On fast positives its 71.6% IoU-0.25 overlap compares with 0.0% from matched
background-only regions and 54.0% from length-preserving alternative event
positions. Excess over alternatives is 17.6 points (15.9–19.5). As in paper
03, alternative placements hold selected regions fixed, while matched pure
background trajectories have different learning histories. Neither is a causal
attribution/significance test. Improved correct identity and low matched
background overlap are stronger evidence than near-continuous attention with
coincidental overlap. All IoU thresholds and duration controls are archived.

## Limits and next work

Stationary regression remains unresolved. Test adaptation or evidence fusion
without restoring background attention, and report foreground losses per
class. The frozen profile also needs tests for silence or foreground during
calibration, background spectral changes, overlapping sources, and wider
pitch/duration variation. None is established by this study.

Recorded source-disjoint audio (ch-6w4.1), PCEN/adaptive-noise baselines
(ch-6w4.7), post-channel SNR and identical-state probes (ch-6w4.9), and physical
timing/echo checks (ch-6w4.5) remain necessary. No recorded audio was available
in this repository for this experiment. There is no new claim of semantic
importance, source separation or hardware energy savings.

The original capture default stays unchanged. The opt-in
`chuff-selective --frontend spectral-gain` uses a separate memory schema and
state file. Provide background-only initial calibration when examining it;
profile, background and pending candidates reset on load while admitted
prototypes survive. WAV replay remains silent. This supports inspection of
the trade-off, not field validation.

## Follow-up: selecting compensation from calibration audio (ch-6w4.10.4.1)

The fixed transform's stationary loss motivates an audio-derived selection
rule. Development telemetry shows why selecting from each event window is
unsafe: foregrounds themselves can raise common-level variation. Instead,
collect the coefficient of variation of the smoothed shared band level in
the first three unlabelled calibration windows, discarding the first three
gain time constants of each window. Enable compensation when their mean is
at least 0.12; otherwise use raw power. Both modes feed raw power to the
original background estimator during warmup. The choice freezes afterward.
Neither event labels nor simulation-condition names reach the learner.

On observed development seeds 1100–1102, calibration means span 0.061–0.068
for stationary backgrounds, 0.167–0.195 for slow modulation and 0.545–0.572
for fast modulation. Threshold 0.12 and the complete policy are frozen in
`calibrated-gain-protocol.json` before fresh evaluation on seeds 1400–1409.
The earlier 1200–1209 outcomes are now observed context that motivated this
new experiment. Its numerical threshold is chosen from the 1100-series
development telemetry, and the 1400-series evaluation remains separate.

Fresh mean correct repeat identity and IoU ≥0.25:

| Method | Stationary | Slow modulation | Fast modulation |
|---|---:|---:|---:|
| Quiet-band | 69.3% | 22.2% | 5.2% |
| Always spectral gain | 44.9% | 46.2% | 49.3% |
| Calibration-derived choice | 69.3% | 46.2% | 48.8% |

The new policy chooses raw power in all ten stationary cases and compensation
in all ten cases of each modulation type. These decisions come from audio
statistics, not an oracle switch. Evaluation calibration means are
0.057–0.075, 0.150–0.243 and 0.542–0.578 respectively. Positive and pure
background streams make identical calibration decisions, as expected from
their identical initial audio. The audit checks this pairing and recomputes
the decision from the saved CVs and threshold.

Stationary primary scores equal quiet-band seed by seed (paired difference
zero). A separate waveform/feature test verifies exact baseline decisions
on the raw path. This is conditional preservation through the raw branch,
not a proof of non-regression on arbitrary recordings or a general
statistical equivalence claim. Slow-modulation gain versus quiet-band is
24.0 percentage points (paired bootstrap 95% interval +16.6 to +30.8);
fast-modulation gain is 43.6 points (+33.4 to +53.3). Absolute fast recognition
is 48.8% (39.6–58.0). These remain descriptive ten-seed intervals without
multiplicity adjustment. Differences from the previous table use new seeds.

Pure-ambience attention is 0.0%, 0.0% and 0.2% for stationary, slow and fast
backgrounds; fast attention is two of 1,200 windows. No control stream admits
a background prototype. Quiet-band attention is 0.0%, 26.9% and 97.5% on the
same streams. All overlap thresholds and matched/alternative-position controls
remain available in the archive; zero observed admissions does not prove
zero risk outside these cases.

The calibration-derived frontend is also opt-in:
`chuff-selective --frontend calibrated-gain`. It has a distinct saved schema,
resets the mode/profile on load and reports whether compensation was selected.
The original default stays unchanged. Tests verify that later input cannot
silently change the frozen choice, and that loading begins a new calibration.

This resolves the stationary-regression task on the controlled split, while
calibration and environment generalization remain ch-6w4.10.4.2. Silence,
foreground during calibration, slower/shallower modulation, a mode that changes
mid-run, spectral shifts and overlapping recordings are not established by
this result. A frozen three-window choice can be wrong when initialization
does not represent later conditions; it is not ongoing environment adaptation.

`calibrated-gain-development/` contains 12,960 decisions and
`calibrated-gain-evaluation/` 43,200 decisions, with exact sources, raw timings,
CVs, mode decisions and all scores. Their source, matrix, pairing, timing,
calibration and bootstrap audits pass. This follow-up's software suite has
60 passing tests and the original grader remains 1.0. Seeds 1400–1409 are
now observed evidence and must not remain a sealed future test set.

## Later numerical correction: silent initialization

Ch-6w4.10.4.2.1 verifies a separate startup bug: exact silence could freeze
the fixed spectral template at its numerical floor. The corrected method
uses raw power while no usable profile exists, fits its first three usable
windows, and rebases late background/pending state without deleting admitted
prototypes. Late fitting windows suppress admission. Normal calibration is
unchanged; the calibrated frontend's raw mode remains baseline-equivalent.
The correction does not make its frozen mode adapt to later changes.

Archived/current comparison on seeds 1800–1809 contains 4,800 saved decisions
following the same audio, with or without three zero prefix windows. Of 372
event windows, the archived silent-start model detects 81 versus 197 for its
clean-start counterpart. The corrected silent-start model detects 197 and
matches the corrected clean trajectory window by window. Corrected clean
trajectories also equal archived clean trajectories, with feature/salience
equality asserted during execution. There are no false admissions in these
fixtures. These are pooled regression counts, not independent statistical
trials, field performance or an invariant-source benchmark.

The archive `results/masked-events/startup-recovery/` contains both source
versions, observations and summaries; its independent stored-decision/hash
audit passes. Tests cover late-calibration memory retention and calibrated
raw-path parity. Broader startup contamination, overlap and changing-room
tests remain ch-6w4.10.4.2. Earlier evaluation snapshots are preserved rather
than retrospectively replacing their model code.

## Evidence and checks

Paper [06](06-changing-backgrounds.md) now measures later environment
changes, silent startup and shallower modulation with post-channel SNR and
same-state current-source-removal controls. It confirms that numerical
silent-profile recovery and a frozen mode choice are separate problems.
Its adaptive-policy results use another fresh split; the original
calibration experiment above retains its exact earlier sources and scope.

Development archives contain 12,960, 12,960 and 10,800 decisions;
`gain-evaluation/` contains 72,000 decisions from 60 streams and ten methods.
These are not independent statistical sample sizes. Exact sources, dependency
versions, parameters, raw timing, class/diagnostic counts, per-seed results,
paired intervals and overlap controls are retained. Audits recompute scores,
complete matrices, source hashes, pairing, source-free truth, timing coverage
and uncertainty. Evaluated models are not substituted in their snapshots.
Seeds 1200–1209 are now observed evidence, not future untouched evaluation.

At the initial shared-gain milestone, 58 software tests passed, including RMS chunk/prefix causality,
invalid-input state preservation, spectral-profile freeze, silent experimental
replay and saved-schema isolation. The original grader remains 1.0. Tests do
not validate unmeasured field acoustics or physical hardware.

Codex assisted with implementation, literature verification, iteration,
execution, auditing and drafting. Aidan Jones retains responsibility for
interpretation. This is an exploratory project report, not a peer-reviewed paper.

## References

Wang, Y., Getreuer, P., Hughes, T., Lyon, R. F., & Saurous, R. A. (2017).
Trainable frontend for robust and far-field keyword spotting. *ICASSP*,
5670–5674. [Primary paper](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/45911.pdf).

librosa contributors. *librosa.pcen, version 0.11.0 documentation*.
[Implementation reference](https://librosa.org/doc/0.11.0/generated/librosa.pcen.html).
