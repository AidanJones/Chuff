# When Recurring Background Becomes a Memory: A Controlled Modulation Study

**Aidan Jones — 25 October 2026, exploratory engineering report with Codex assistance**

## Result

Slow background amplitude modulation is sufficient to reproduce Chuff's
quiet-band ambience-admission failure, without source variation, distractors
or a halfway environment change. In every slow-modulation evaluation cell,
all ten pure-ambience streams acquire a memory. Matched stationary controls
have no admissions. The existing binwise noise-estimation ablation rejects
those controls but loses foreground sensitivity. Fast modulation exposes a
different failure: almost continuous attention without permanent admission,
and very poor repeat identity. A zero false-prototype count alone is therefore
insufficient evidence that the agent has separated signal from noise.

## Design

This follows [the masked-event report](02-masked-foreground-events.md),
especially its sections 10–11. Beads task ch-6w4.10.2 isolates factors that
were combined in the preceding variance simulation. The local prospective
plan is [modulation-protocol.json](results/masked-events/modulation-protocol.json).
Three development seeds (800–802) precede ten fresh evaluation seeds
(900–909); all three methods are existing, frozen configurations. No method
is tuned on either set.

The factorial design crosses two background spectra, two channel conditions,
three modulations and two source conditions:

- Reference spectrum: coloured plus white noise and 100/200 Hz hum; varied
  spectrum: independently seeded AR pole and hum frequency, as in paper 02.
- Reference channel: the earlier fixed room/microphone approximation; varied
  channel: independently seeded sparse room echoes and microphone cutoffs/gain.
- Stationary amplitude, or depth 0.45 modulation at 0.17 Hz (slow) or 1.3 Hz
  (fast), multiplying the same underlying background waveform. Modulation
  depth zero has no meaningful rate, so it is represented once.
- Pure ambience, or six recurring procedural foreground classes with active
  SNR −6 dB before the common channel. Neither condition contains nuisance
  distractors, halfway background changes or expanded source resampling.

Each stream contains 120 one-second windows. Independent random streams
preserve event schedules and underlying background draws across factors.
The channels stay fixed within a stream; they are signal-processing
approximations, not calibrated acoustic rooms. The learner receives only the
mixture. Source timing and labels are not supplied to it.

Methods are the original frontend, quiet-band with shared quiet-frame noise
estimation, and its existing binwise ablation (`noise_mode=0`). The ablation
estimates each frequency's noise using its within-window 30th power percentile,
divided by −log(0.7), following the exponential-periodogram approximation
already described in paper 02. It keeps the band evidence thresholds and
prototype memory unchanged. That approximation is a heuristic under these
nonstationary mixtures, not a known true noise estimate. No direct PCEN or
MCRA implementation is evaluated here; those literature baselines remain
tracked separately as ch-6w4.7.

## Background admission and attention

Evaluation false-prototype totals, with affected-stream counts in parentheses:

| Spectrum / channel | Quiet-band stationary | Quiet-band slow | Quiet-band fast | Binwise, all three rates |
|---|---:|---:|---:|---:|
| Reference / reference | 0 (0/10) | 12 (10/10) | 0 (0/10) | 0 each (0/10 each) |
| Reference / varied | 0 (0/10) | 11 (10/10) | 0 (0/10) | 0 each (0/10 each) |
| Varied / reference | 0 (0/10) | 13 (10/10) | 0 (0/10) | 0 each (0/10 each) |
| Varied / varied | 0 (0/10) | 10 (10/10) | 0 (0/10) | 0 each (0/10 each) |

The original frontend has two admissions in the reference-spectrum,
varied-channel fast-modulation cell (two affected streams), and none in
the other pure-ambience cells. The four spectrum/channel cells share seeds;
they are not forty independent background replications. Zero observed
admissions in these small procedural controls does not establish zero risk.

The binwise-minus-quiet-band false-prototype difference in the reference
spectrum/channel slow cell is −1.2 per stream, with a descriptive paired
seed-bootstrap 95% interval of −1.5 to −1.0. There are 10,000 bootstrap draws
within each cell and no multiple-comparison adjustment.

Fast modulation is not a successful rejection condition: both band methods
attend to **97.5%** of pure-ambience windows in every spectrum/channel cell.
That is all 117 windows after their three-window calibration period, despite
zero permanent admissions. Quiet-band slow-modulation attention is 24.6–26.7%
across cells; stationary attention is zero. Confirmation and feature matching
must be distinguished from the initial detection gate. The absence of memory
does not imply that the input was recognized as background or ignored.

## Foreground cost

Mean per-seed correct repeat recognition at −6 dB, with eligibility fixed
after two true class occurrences and missed events included:

| Spectrum / channel | Stationary quiet / binwise | Slow quiet / binwise | Fast quiet / binwise |
|---|---:|---:|---:|
| Reference / reference | 70.8% / 47.1% | 24.4% / 31.9% | 3.6% / 6.0% |
| Reference / varied | 74.9% / 49.4% | 38.2% / 33.3% | 3.5% / 4.2% |
| Varied / reference | 64.1% / 39.6% | 22.5% / 20.5% | 0.0% / 1.2% |
| Varied / varied | 64.4% / 40.4% | 25.5% / 24.4% | 0.0% / 1.7% |

The original frontend's recognition ranges from 0.0% to 15.7% across these
cells. The binwise ablation costs roughly 24–25 percentage points of mean
recognition on stationary backgrounds. Under varied-spectrum/varied-channel
slow modulation, its window-level detection falls from 68.1% to 43.2%:
a paired −24.9 points (bootstrap interval −28.8 to −21.6). Correct repeat
recognition changes by −1.2 points (−6.0 to +4.2); that interval is not an
equivalence test. Rejecting ambience and detecting weak sources remain
competing engineering objectives.

## A measurement failure and the next experiment

Both band methods report 100% window-level event detection under fast
modulation, despite only 0–6% correct repeat recognition and 97.5% attention
in the corresponding no-source streams. The historical detection metric
counts any candidate, admission or recognition on an event-bearing window;
it does **not** establish that the selected spectral region corresponds to
the true event. A background-triggered decision can therefore be counted as
an event detection by coincidence. Earlier detection figures should be read
with this limitation; their archived definitions/results are retained.

Ch-6w4.10.3 adds evaluator-only source intervals and detector-region telemetry,
with overlap-based scoring and positive/negative fixtures. No timing truth
may enter the learner. Ch-6w4.10.4 then tests time-varying noise estimation,
shared-quiet/binwise combinations or principled adaptive baselines against
the fixed current models. It must measure event localization, false attention,
affected streams and identity alongside memory count, on fresh evaluation
seeds. Widening memory tolerance before this gate problem is resolved risks
making background prototypes easier to reuse.

The present experiment demonstrates sensitivity to controlled modulation;
it does not fully attribute the internal mechanism, identify semantic
importance, validate field audio or demonstrate source separation. The
quiet-frame selector can be biased by level changes, but temporal gating
and consecutive-window confirmation also affect the observed outcomes.
Those mechanisms need separate evidence rather than inference from one
aggregate score. Defaults are unchanged.

## Follow-up: timing-aware measurement (ch-6w4.10.3)

Passive detector-region telemetry and evaluator-only source onset/offset
annotations now make the coincidence issue measurable. Adding annotations
preserves the original generator audio bit for bit on a 25-window fixture.
Comparing archived and current learners on 120 paired frames spanning all
three modulation conditions preserves statuses, salience, novelty, prototype
IDs and feature vectors exactly. Timing is returned after the waveform-only
decision and does not influence learning.

The prospective `localization-protocol.json` specifies fresh seeds
1000–1009, varied spectrum/channel, three modulation conditions and matched
ambience/foreground streams: 60 runs and 21,600 window/method decisions.
IoU uses the pre-channel active source interval and the full sample support
of the selected STFT frames, including FFT width. Primary IoU is 0.25;
0.1 and 0.5 are sensitivity checks. Missing timing is unknown rather than
being counted as a failure or success. Previous-source finite room-IR
support is flagged separately; microphone IIR residuals have no exact finite
bound and are not asserted to be absent when the room-tail flag is zero.

Mean per-seed foreground localization rates on these new seeds:

| Method / background | Any detection in event window | IoU ≥0.1 | IoU ≥0.25 | IoU ≥0.5 | Correct repeat identity and IoU ≥0.25 |
|---|---:|---:|---:|---:|---:|
| Quiet-band / stationary | 81.5% | 81.5% | 78.9% | 50.9% | 67.3% |
| Quiet-band / slow | 73.8% | 71.6% | 67.2% | 44.3% | 30.3% |
| Quiet-band / fast | 100.0% | 88.2% | 83.9% | 52.7% | 2.9% |
| Binwise / stationary | 50.7% | 50.7% | 49.7% | 33.9% | 41.3% |
| Binwise / slow | 44.5% | 44.5% | 43.2% | 24.4% | 28.4% |
| Binwise / fast | 100.0% | 73.6% | 65.6% | 44.3% | 5.9% |

Quiet-band fast-modulation localization at IoU 0.25 is 83.9% (seed-bootstrap
95% interval 79.5–87.8), falling to 52.7% (49.2–55.9) at IoU 0.5. This
quantifies the sensitivity of the apparent recovery to temporal alignment.
It still does not prove that a decision was caused by the foreground:
background-triggered wide intervals may overlap events by chance, and
pure-ambience fast-modulation attention remains 97.5%. Among foreground-stream
background windows with no flagged finite room tail, both band methods attend
on 645 of 675 windows under fast modulation. This is a pooled descriptive
count, not 675 independent trials. Under slow modulation those counts are
157/675 for quiet-band and 0/675 for binwise; under stationary backgrounds,
both are 0/675. Filtering residuals and model history remain limitations.

The next measurement task should compare observed overlap against shifted
or counterfactual event intervals, preserving interval lengths and matched
backgrounds. This prevents replacing window coincidence with a different
chance-overlap success criterion. The new localization metric corrects the
definition of detection evidence without establishing source attribution.

The localization audit recomputes all overlap scores and seed intervals from
raw intervals, checks complete timing coverage and the paired matrix, and
retains the original window-level metric. All checks pass. Forty-seven
software tests pass and the original grader remains 1.0. New evidence is
archived under `results/masked-events/localization-evaluation/` with
`localization-analysis.json`; seeds 1000–1009 are now observed evidence.

## Follow-up: overlap against matched controls (ch-6w4.10.5)

This post-hoc diagnostic reuses the observed localization archive; it is not
a fresh held-out experiment. Two controls quantify how much raw overlap can
arise without precise alignment. First, use the detector region from the
matched pure-ambience trajectory at the same seed, window and method, and
score it against the positive stream's actual source interval. Background
draws/channel parameters match, but the complete learner histories differ
because one trajectory has never heard foreground sources. This is a
sequence-level comparison, not a current-event intervention at fixed state.

Second, hold the positive stream's selected region fixed and integrate overlap
over alternative source positions with the actual source length. Two hundred
midpoints cover the generator's legal placement range, including its margin
rule, with no wrapping across window boundaries. An interval of length L
still has length L at every alternative position. Because the selected
region depends on the original mixture, this is alignment sensitivity rather
than a valid causal permutation/significance test. All three IoU thresholds
are retained. Seed-bootstrap intervals preserve the paired differences.

Mean rates at primary IoU 0.25:

| Method / modulation | Observed overlap | Matched background-region overlap | Alternative-position overlap |
|---|---:|---:|---:|
| Quiet-band / stationary | 78.9% | 0.0% | 58.3% |
| Quiet-band / slow | 67.2% | 4.2% | 48.4% |
| Quiet-band / fast | 83.9% | 72.0% | 77.2% |
| Binwise / stationary | 49.7% | 0.0% | 33.6% |
| Binwise / slow | 43.2% | 0.0% | 28.9% |
| Binwise / fast | 65.6% | 55.5% | 62.9% |

Quiet-band fast-modulation overlap exceeds the matched-background control by
11.9 percentage points (descriptive paired bootstrap 95% interval 9.5–14.1)
and the alternative-position control by 6.7 points (4.0–8.5). These contrasts
do not identify a causal fraction of events: the counterfactual histories
and selected-region dependence limit that interpretation. The large control
rates explain why raw 83.9% overlap greatly overstates evidence of precise
event alignment, even before considering the 2.9% localized repeat identity.

Duration is a particularly strong confound. The 70 fast-modulation long-alarm
windows average 0.812 s of source activity. Quiet-band observed overlap at
IoU 0.25 is 98.6%, versus 100% from the matched background regions and 98.6%
under alternative placements. These are pooled class-level descriptions,
not seed-independent trials or semantic recognition scores. The class table
in `localization-controls/analysis.json` retains shorter sources too.

Analysis code/source hashes, input-manifest and raw-decision digests, interval
rules, class duration sensitivity and all seed intervals are archived in
`results/masked-events/localization-controls/`. The underlying 21,600 decisions
were re-audited before analysis. Tests verify that whole-window regions can
have perfect IoU-threshold success with zero excess over either control,
and that legal margins, lengths and pairing are preserved. Fifty software
tests pass; the original grader remains 1.0.

The gate experiment ch-6w4.10.4 should report these controls alongside localized
identity and false attention. Ch-6w4.9's separated source/channel stems can
support a stronger future experiment: compare foreground-present and
foreground-removed probes from identical prior learner state, retaining
previous-source tails. This controls the history difference that the present
matched-trajectory diagnostic does not remove.

## Reproduction and evidence

The [experiment guide](../python/src/chuff_mpu/sim/MASKED.md) provides commands.
Archives under `results/masked-events/modulation-development/` and
`modulation-evaluation/` retain exact source snapshots, dependency versions,
sampled parameters, class/diagnostic counts, summaries, intervals and all
25,920 development plus 86,400 evaluation decisions. These window/method
counts are not independent sample sizes. The audit verifies hashes, raw
scores, the complete factorial matrix, paired schedules, absence of nuisance
events and halfway changes, source-free controls and diagnostic partitions.
All checks pass. The simulation fixtures verify exact modulation of paired
backgrounds and independence of source rendering/scheduling from the factors.
Evaluation seeds 900–909 are now observed evidence, not a future sealed test.

Codex assisted with implementation, execution, auditing and drafting. Aidan
Jones retains responsibility for interpretation. This is an exploratory
project report, not a peer-reviewed paper.
