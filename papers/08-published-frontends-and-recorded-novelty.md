# Published frontends for recorded novelty: a frozen comparison protocol

**Status:** Development selection completed and audited. No held-out
evaluation result is available. The final configuration list is frozen.

## Question

The recorded pilot in [paper 07](07-recorded-replay-and-source-exclusion.md)
found frequent background attention and prototype commitments even when
gain compensation improved recording-specific recall. The question here is
whether established normalization and adaptive noise estimation primitives
improve this trade-off in the same bounded discovery system. The desired
behavior is retaining novel and recurring foreground structure while
reducing learning of the background bed. A detector that rejects everything
does not satisfy that objective.

The experiment compares adapters around PCEN and MCRA with the existing
quiet-band method. It does not reproduce their original keyword-spotting or
speech-enhancement systems, or establish a state-of-the-art comparison.

## Published methods and the discovery backend

Wang et al. introduced PCEN for robust keyword spotting and also trained its
parameters jointly with a recognizer. This study uses fixed PCEN parameters
and no keyword recognizer or supervised acoustic model [1]. Direct
`librosa.pcen` 0.11 calls retain filter state across observations. Input uses
magnitude mel bands, a 2**31 amplitude scale and restored Hann-window scale.
Our 32 HTK bands and 32 ms windows differ from the original frontend; the
hop is 10 ms. An additional quiet-frame estimate converts PCEN values into
detector evidence [2].

Cohen and Berdugo's MCRA estimates noise power using smoothed spectral
minima and signal-presence-dependent updates [3]. Our adapter uses their
16 kHz, 512-sample FFT, 128-sample hop, .8/.95/.2 smoothing, threshold 5 and
125-frame minimum-block settings. Frequency weights [.25, .5, .25] and
nearest-edge extension are declared choices. The first power frame
initializes noise; the current estimate precedes the next-frame update.
This is MCRA rather than IMCRA, and no OM-LSA enhancement stage is included.

All four families use the same recurrence confirmation, candidate lifetime,
96-dimensional three-part feature representation and capacity of 32 admitted
prototypes. The quiet-band method is run at both 160- and 128-sample hops,
providing the MCRA geometry control. An untuned adaptive-gain actor remains
an anchor. Detection and stored features are separate: feature mode 0 uses
raw spectral excess; mode 1 uses relative detector features. In particular,
PCEN detection with mode 0 does not mean that PCEN features are stored.

Each one-second observation computes an unpadded STFT separately. The
PCEN/MCRA spectral state persists, but boundary frames omitted by this API
are not reconstructed. Streaming parity tests concern the supplied spectral
frames, not an equivalence to continuous waveform framing. All methods
share this observation API. Sustained sources and contaminated startup
remain relevant limitations.

## Inputs and split reservation

The previous pilot's folds 1 and 5 are already observed. A new
[metadata-only reservation](data/esc10-published-reservation/README.md) uses
fold 2 for development and fold 4 for evaluation. It archives the prior
catalog and excludes every original Freesound source ID in it, including
alternative takes. Twenty newly selected source IDs are mutually distinct
and disjoint from the twenty pilot sources. Selection uses pinned upstream
metadata and lexical filename order, without choosing favorable waveforms.

The reservation precedes new audio materialization. Only its ten development
recordings have been downloaded: five foreground categories and five
background categories, each a complete five-second recording. Evaluation
materialization is deferred until development selection is archived. The
catalog builder can enforce separate partition downloads and verifies WAV
format, file hashes and individual attribution. ESC-10 data retain their
upstream CC BY 3.0 terms and source attribution [4].

Source-ID separation does not establish different creators, equipment or
recording sessions. There is one recording per category and partition, and
category labels supply neither precise event boundaries nor clean stems.
An injected recording can itself contain unwanted sounds.

## Frozen tuning and selection

[published-protocol.json](results/masked-events/published-protocol.json)
pins primitive parameters, method-source hashes, dependencies, sources,
seeds, grids and selection rules before development decisions are observed.
Each family has exactly 12 configurations:

| Factor | Values |
|---|---|
| Family | quiet160, quiet128, PCEN160, MCRA128 |
| Local detection threshold | 1.5, 2.5, 3.5 |
| Broad detection threshold | 1.15, 1.5 |
| Stored feature mode | raw excess, relative evidence |

Thus there are 48 candidate configurations and one untuned anchor. Published
smoother parameters are fixed; there is no additional hidden PCEN/MCRA
search. Development uses seeds 2800–2801; final evaluation reserves
2900–2909. Each seed has four positive suites (recorded/modulated ambience
at whole-episode 0/−6 dB) and two pure-background controls. A no-target
control has no target SNR and is run once per ambience condition, avoiding
duplicated controls masquerading as independent evidence.

A configuration is feasible only if it achieves all of:

- At least 80% episode detection in **each** positive suite, averaged across
  the two development seeds.
- At least 60% mean first-episode novel attention across positive suites.
- At least 20% mean strict source-exclusion repeat identity across positive
  suites.

These are declared engineering screening thresholds, not statistical
guarantees or deployment requirements. Among feasible configurations,
selection minimizes mean pure-control admissions, then maximizes strict
repeat identity, minimizes wrong identity and breaks ties by configuration
ID. Selection is reported overall and within each stored-feature mode,
using the same existing grid. The full grid remains visible.

If a group is infeasible, its selected configuration is null. The report
also identifies its highest-strict diagnostic candidate, with background
admissions and wrong identity breaking ties. Such a candidate remains
explicitly infeasible; constraints are not relaxed. Final evaluation retains
the declared defaults and the deduplicated feasible or diagnostic choices,
alongside the untuned anchor. Development scores will not be presented as
held-out performance.

## Replay, exclusion and runtime evidence

Each 210-second stream contains twenty complete five-second foreground
episodes, four replays per recording, separated by background gaps. The
continuous room/microphone channel and recorded overlap-added bed match
across actors and positive/control suites. Whole-episode post-channel SNR
is checked against the current episode divided by residual audio, including
earlier episode tails. One-second source/residual RMS and SNR remain in raw
rows, since intrinsic source pauses invalidate a uniform masked-event SNR
interpretation.

The repeat denominator is the third/fourth physical replay, not adjacent
windows from one clip. A correct identity must use a prototype admitted
before the current episode. The primary strict endpoint additionally rejects
an episode if any matching source-removal probe still recognizes that
recording. Probes remove the entire current episode's physical channel
contribution, copy actual prior learner state for each window and discard
their updates. They retain earlier actual exposure; they are not a complete
episode-deleted learning history. The less strict, same-window-conditioned
endpoint and wrong-identity rates are retained as separate diagnostics.

Runtime uses `perf_counter_ns` around each actual `observe`, excluding
rendering, branch copying and probe observations. Imports/JIT are warmed
using disposable actors before measurements; the first three measured
warmup windows are excluded from summaries. Per-window durations and host,
dependency and thread settings are retained. These timings concern this
Python implementation on a shared desktop, without CPU isolation or
hardware energy measurements; they do not establish algorithmic efficiency.

Source/input/output hashes, attributed full-duration component examples,
every observation and source-removal result, per-recording counts and the
complete development selection are archived. An independent offline audit
checks the expected matrix, SNR/components, paired backgrounds, source
separation, method parameters, metrics, timings and exact selection rule.

## Development findings: selection data

The completed archive at
[`published-development/`](results/masked-events/published-development/)
contains 123,480 actual observation decisions: 49 actors × 210 windows ×
6 suites × 2 seeds. It additionally records 39,200 source-removal probe
decisions. The offline audit verified the complete matrix, physical
components/SNR, source/method/input/output hashes, timings and exact
selection from raw rows. All 121 software tests pass; the original grader
scores 1.0. No hardware is exercised.

| Family's overall choice | Strict repeat identity | Mean control admissions per 210 s | Wrong repeat identity | Feasible settings / 12 |
|---|---:|---:|---:|---:|
| Quiet160, raw, local 3.5 / broad 1.5 | 47.5% | 8.50 | 83.75% | 12 |
| Quiet128, raw, local 3.5 / broad 1.5 | 42.5% | 8.50 | 85.00% | 10 |
| PCEN160, raw, local 2.5 / broad 1.5 | 31.25% | 5.25 | 56.25% | 5 |
| MCRA128 diagnostic, raw, local 3.5 / broad 1.5 | 13.75% | 6.25 | 71.25% | 0 |

Strict identity and wrong identity are averaged over the four positive
suites and two seeds, with 80 eligible repeat episodes per configuration.
Correct and wrong matches can occur in different windows of the same
episode, so their rates are not complements. Background commitments are
averaged over four unique no-target streams. These are development values
used for selection, with only five foreground recordings.

The selected PCEN adapter reduces background commitments relative to the
quiet-band choice on this development split, while retaining fewer strict
repeat identities. This is a trade-off rather than a broad improvement.
Its lowest suite-level episode detection is 87.5%, with 92.5% mean
first-episode novel attention. All MCRA settings fail the declared strict
retention requirement; even the best diagnostic retains only 13.75%.
Detection and novelty are high in that diagnostic (95% each), demonstrating
that detecting change alone does not establish stable source identity.

The full grid, feature-specific selections, infeasibility labels and
deduplicated evaluation list are in `selection.json`. Final evaluation will
use nine configurations plus the untuned adaptive-gain anchor. The raw PCEN
default is retained alongside its tuned choice; its development detection
and strict recall do not meet the declared screening requirements. Both
raw and relative feature choices remain represented. No additional tuning
or threshold relaxation follows these outcomes.

## Remaining scope

The comparison and its follow-through are tracked in Beads ch-6w4.7.2.
Active-support calibration without cropping remains ch-6w4.1.4; broader
categories and more original recordings remain ch-6w4.1.3. Importance and
one-off relevance learning remain ch-6w4.3. These methods can emphasize
spectral change without establishing that a learned sound matters to the
robot. Neither recording replay nor source exclusion supplies semantic
relevance labels. Physical capture and cue-echo validation still require
the bench work recorded in `HUMAN_TODO.md`.

## References

1. Wang, Y., Getreuer, P., Hughes, T., Lyon, R. F., & Saurous, R. A. (2017).
   *Trainable frontend for robust and far-field keyword spotting*. ICASSP,
   5670–5674. [Primary publication](https://research.google/pubs/trainable-frontend-for-robust-and-far-field-keyword-spotting/).
2. librosa 0.11.0. [PCEN implementation and streaming/input-scale documentation](https://librosa.org/doc/0.11.0/generated/librosa.pcen.html).
3. Cohen, I., & Berdugo, B. (2002). *Noise Estimation by Minima Controlled
   Recursive Averaging for Robust Speech Enhancement*. IEEE Signal Processing
   Letters, 9(1), 12–15. [Author manuscript](https://webee.technion.ac.il/Sites/People/IsraelCohen/Publications/SPL_Jan2002.pdf).
   [DOI](https://doi.org/10.1109/97.988717).
4. Piczak, K. J. (2015). *ESC: Dataset for Environmental Sound Classification*.
   ACM Multimedia, 1015–1018. [DOI](https://doi.org/10.1145/2733373.2806390).
   Exact upstream attribution and metadata accompany the reserved catalog.

Codex assisted with implementation, execution, auditing and drafting. Aidan
Jones retains responsibility for interpretation. This is an exploratory
project report, not a peer-reviewed paper.
