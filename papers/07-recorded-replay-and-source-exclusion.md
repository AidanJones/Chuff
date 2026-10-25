# Recorded replay identity amid changing ambience

*Exploratory project report, 25 October 2026. Follow-up to
[paper 06](06-changing-backgrounds.md).*

## Question and inputs

The procedural studies showed that gain compensation can reject common
amplitude modulation. This pilot replaces synthesized foregrounds/noise
with recorded inputs. It tests whether recognition of a later replay
depends on the injected recording, rather than background attention.
It is a recording-identity task, not semantic classification or natural-event
source separation.

The [pinned catalog](data/esc10-pilot/README.md) selects twenty ESC-10 files
at revision `33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6`: ten development
files from fold 1 and ten evaluation files from fold 5. Original Freesound
source IDs are disjoint, including alternate excerpts/takes. Source identity,
attribution, immutable URL, format and SHA-256 are retained. Upstream licenses
ESC-10 under CC BY 3.0; the project's code license does not replace those
data terms. Derived examples retain input attribution.

Dog, rooster, crying baby, sneezing and clock tick supply foregrounds;
rain, sea waves, fire, helicopter and chainsaw supply the background.
These roles define intended targets, not intrinsically meaningful versus
irrelevant sounds. Clips contain their own ambience and lack natural-event
onset/offset annotations. Each category has only one file per split.

## Continuous full-duration protocol

Each complete five-second clip is resampled from 44.1 to 16 kHz with
`resample_poly`; foreground injection edges receive 10 ms fades. No
energy-selected patch or duration truncation is used. Backgrounds overlap
complete five-second grains with periodic Hann weights, 50% overlap and
independent ±3 dB levels. A second condition adds common amplitude
modulation of depth .45 at 1.3 Hz. This synthesizes ambience from recordings;
it is not one continuous natural field recording.

Streams last 210 seconds: ten seconds of initial background, then twenty
five-second episodes separated by five-second no-target gaps. Four shuffled
balanced cycles replay all five recording identities. Actors observe
one-second windows. Independent seeded room responses and microphone
filters/gains stay continuous across boundaries. Actors receive only audio,
never recording IDs, episode times, labels, SNR or clean components.

Whole-episode current-recording/residual SNR is matched **after** the linear
channel at 0 or −6 dB. Residual retains background and previous source
responses. Component sums, calibration and matched positive/control
background outputs are checked. Source-free windows can retain room tails;
tail strata are retained without claiming the IIR microphone has finite
support.

Original selective, quiet-band, calibrated gain, adaptive gain and guarded
gain keep their existing settings. Development seeds 2600–2601 use fold 1
and contain 16,800 actual decisions. The frozen evaluation uses fresh
2700–2709 and fold 5: eighty streams and 84,000 actual decisions, plus
discarded source-removal probes. No learner tuning or source selection uses
evaluation outcomes.

## Recognition versus source dependence

Five adjacent windows remain **one episode**. Only the third/fourth replay
is eligible for repeat recognition. A matching prototype must have been
admitted before the current episode; learning in its first window and
recognizing in its later windows earns no repeat credit. Admission-time
recording labels retain the convention and candidate-origin ambiguity
tracked in ch-6w4.10.7.

The prespecified source-conditioned score requires at least one correct
recognition window whose matching source-removed probe does not recognize
that recording. Each probe copies actual prior learner state, removes the
entire current episode's physical channel contribution for that window
(including current-episode echoes), preserves prior episodes and discards
its branch state.

A stricter **secondary** diagnostic, added after development, also requires
no correct current-recording recognition in *any* of the five removed-source
probes. The planned score is retained. Both endpoints concern the entire
injected recording, including its ambience; neither proves that a bark or
voice inside it was isolated. Probe priors retain actual earlier learner
exposure to the episode: these are independent window interventions, not
one counterfactual history with the whole episode absent.

Means weight seeds equally. Descriptive 95% intervals use 10,000 paired-seed
bootstrap draws, RNG 731, without multiplicity adjustment. Uncertainty is
conditional on these fixed recordings, not new independently sampled source
corpora. The two nominal SNR controls have identical no-target waveforms and
decisions and are not independent background replications.

## Held-out results

Eligible replay episodes satisfying the planned source-conditioned score;
stricter secondary exclusion appears in parentheses:

| Background / whole-recording SNR | Original | Quiet-band | Calibrated / adaptive / guarded gain |
|---|---:|---:|---:|
| Recorded / 0 dB | 38% (35%) | 52% (48%) | 53% (49%) |
| Recorded / −6 dB | 14% (13%) | 17% (16%) | 43% (38%) |
| Modulated / 0 dB | 22% (19%) | 31% (29%) | 55% (50%) |
| Modulated / −6 dB | 10% (9%) | 16% (15%) | 38% (32%) |

The three gain policies have identical saved decision, prototype, salience,
timing and probe trajectories. Extra adaptation/guarding has no measured
benefit over the frozen policy on this split; equality does not establish
general mode-selection robustness.

At recorded/−6 dB, stricter gain recognition is 38% (34–43%), versus 16%
(8–25%) for quiet-band: paired difference +22 points (+11 to +32). The
modulated/−6 dB difference is +17 (+6 to +28). Recorded/0 dB differs by only
+1 (−9 to +11). These exploratory fixed-corpus comparisons do not establish
superiority to published audio frontends.

Detection alone is weak evidence. Gain detects something in 99.5% of
injected episodes in every cell; quiet-band detects 97.5–100%. Gain also
recognizes the current recording in removed-source probes on 9–16% of
eligible episodes. Correct and wrong identities can occur in different
windows of one episode: at recorded/−6 dB, ordinary gain repeat identity is
53%, while 77% include a wrong identity. These rates are not complements.

Stricter hits per held-out recording at recorded/−6 dB:

| One fixed recording | Original | Quiet-band | Gain |
|---|---:|---:|---:|
| Crying baby | 0 / 20 | 2 / 20 | 13 / 20 |
| Sneezing | 2 / 20 | 4 / 20 | 0 / 20 |
| Rooster | 3 / 20 | 4 / 20 | 10 / 20 |
| Clock tick | 2 / 20 | 0 / 20 | 2 / 20 |
| Dog | 6 / 20 | 6 / 20 | 13 / 20 |

Each row contains twenty eligible replays of one file, two per seed.
It measures repeated exposure to that recording. Gains are uneven across
the selected files.

## The unresolved background problem

Pure-background attention and persistent prototype commitments remain high:

| Method | Recorded attention | Modulated attention | Recorded / modulated admissions across ten controls |
|---|---:|---:|---:|
| Original | 22.6% | 23.3% | 27 / 24 |
| Quiet-band | 60.0% | 89.0% | 45 / 52 |
| Gain policies | 47.7% | 49.3% | 68 / 74 |

Attention is per one-second window; admissions are commitments, not counts
of distinct semantic nuisances. These controls have no injected target, so
the background-learning evidence is stronger than mixed-stream admission
labels. Improved source-conditioned recall coexists with frequent background
attention and memory formation. Near-zero admissions on procedural modulated
ambience do not generalize to this recorded bed.

Whole-recording SNR conceals within-episode variation. At the −6 dB target,
one-second ratios span approximately −100 to +13 dB, with median about
−6.6 dB. Intrinsic pauses concentrate energy into audible portions. This
does not establish uniform recognition of natural events masked at −6 dB.
Ch-6w4.1.4 tracks active-support calibration without cropping full episodes;
masks/annotations must be independently justified, not favorable patches
selected from held-out waveforms.

## Next work, evidence and limits

Ch-6w4.7 compares carefully configured published PCEN/adaptive-noise
baselines with the same memory budget. Current gain compensation is not
PCEN. Source-disjoint inputs and source-removal controls now make that
comparison practical. Speech, footsteps, sustained alarms and more source
recordings remain ch-6w4.1.3; relevance/one-off importance learning remains
ch-6w4.3. Physical capture/cue-echo work still needs the bench access in
`HUMAN_TODO.md`. The broad recorded benchmark parent remains open.

[Paper 08](08-published-frontends-and-recorded-novelty.md) now records the
completed PCEN/MCRA adapter comparison. It uses new fold-2/fold-4
sources excluding every pilot source ID, fixes equal finite tuning grids
and requires foreground retention when selecting for fewer background
admissions. Its separate held-out evaluation finds fewer background
commitments but missed episodes for selected PCEN, and low repeat identity
for the tested MCRA adapter. These findings do not resolve background
learning or establish semantic relevance.

This pilot establishes replay/scoring infrastructure and exposes limitations;
it does not establish missing-category coverage, clean natural-event stems,
calibrated physical acoustics, semantic relevance or field robustness.

`results/masked-events/recorded-development/` and `recorded-evaluation/`
retain exact sources, settings/dependencies, pinned input catalog/attribution,
processed audio hashes, raw observations, per-recording counts and attributed
full-duration component examples. Separate `*-exclusion/` archives retain
the secondary scorer, both endpoints and input/output hashes. Derived audio
keeps the input terms in each archive's `input/` records.

Audits verify complete matrices, hashes, full episode duration, post-channel
calibration, matched backgrounds, prior-episode scoring and seed estimates.
Tests cover rendering, probe isolation, excluding same-episode learning and
the stricter diagnostic. All 102 software tests pass; the original grader
remains 1.0. No microphone/speaker hardware is exercised. Seeds 2600–2601
and 2700–2709 are observed evidence; future tuning requires a new frozen
evaluation split.

## References

Piczak, K. J. (2015). *ESC: Dataset for Environmental Sound Classification*.
ACM Multimedia, 1015–1018. [DOI](https://doi.org/10.1145/2733373.2806390).
[Pinned official repository](https://github.com/karolpiczak/ESC-50/tree/33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6).

Wang, Y., Getreuer, P., Hughes, T., Lyon, R. F., & Saurous, R. A. (2017).
*Trainable frontend for robust and far-field keyword spotting*. ICASSP,
5670–5674. [Primary paper](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/45911.pdf).
This motivates a future direct comparison; this study does not evaluate
PCEN or reproduce its keyword-spotting task.

Codex assisted with implementation, execution, auditing and drafting. Aidan
Jones retains responsibility for interpretation. This is an exploratory
project report, not a peer-reviewed paper.
