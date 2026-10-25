# Can background reproduce a candidate's stored feature?

**Status:** Full observed-data replay and independent vector audit completed.
Foreground removal leaves a within-radius stored-feature match at 152 of
853 Quiet160 candidate births where the injected foreground actually changes
the audio. Selected PCEN gives 85/630. These are conditional feature-pipeline
diagnostics; neither recording labels nor feature matches establish semantic
importance. No learner policy or original accuracy endpoint changes.

## Question and relation to prior work

[Paper 09](09-candidate-birth-and-admission-labels.md) showed that a prototype's
birth and admission can occur in differently labelled windows. Its persistent
vector comes from birth. This experiment asks whether the background bed,
processed from the same prior learner state, can still produce a selected
feature within the stored vector's ordinary matching radius.

This question is narrower than novelty or relevance. Oudeyer, Kaplan and
Hafner distinguish progress in learning from permanently unpredictable
situations [1]. Pathak et al. predict action consequences in an inverse-dynamics
feature space [2]; their action-conditioned formulation does not transfer
directly to this passive microphone. Random network distillation predicts
fixed random observation features [3], providing a different novelty signal
from temporal background prediction. None supplies semantic relevance labels
for these recordings. A promising future comparison therefore separates
feature rarity, temporal prediction error and improvement in prediction,
while retaining one-off relevant events as a separate requirement.

The present experiment tests none of those alternative learners. Its purpose
is to locate an ambiguity in the existing candidate memory before designing
another admission rule. [Paper 08](08-published-frontends-and-recorded-novelty.md)
documents the PCEN and MCRA primary sources, ESC-10 provenance and frozen
source split used here.

## Intervention and coverage

We replay all sixty evaluation streams: four positive suites and two
no-target controls, ten seeds, ten correlated configurations, 210 one-second
observations per stream. Each positive stream injects twenty complete
five-second recorded episodes. All 126,000 original decisions, their 40,000
legacy source-removal checks, physical metadata and complete candidate
histories reproduce exactly, excluding observation timing.

The outer evaluator probes **every** archived candidate observation, including
never-admitted candidates: 28,813 windows and 5,940 eventual admission pairs.
At each scheduled window it retains the prior learner state. Independent
copies observe the mixture, the current-episode removal and the bed with
all injected foreground contributions removed, including older room tails.
Only the actual mixture advances the actual learner. Clean stems, labels
and probe outcomes never enter its state. Earlier actual exposure remains
in every branch; this is not an entirely source-deleted learning history.

Each record links actual, stored, current-removed and bed float32 vectors;
hashes, norms, cosine distances, detection status and selected regions are
saved separately. The existing strict distance threshold, 0.18, is unchanged.
An actual feature must first match its stored vector before the probe is
classified. A missing probe feature, an outside-radius feature and a
within-radius feature remain separate outcomes. Detected zero features are
distinct from absent features: identical zero bytes do not establish a
cosine match. No actual zero features occur in this replay; synthetic tests
exercise that case.

Birth, confirmation and adjacent observations partition the candidate slots;
admission overlaps confirmation. The auditor reconstructs lifecycle coverage,
recomputes all vector comparisons, checks configured dimensions/radii and
verifies identical-audio controls. Original scoring code remains unchanged.

## Target windows and actual audio removal

Scheduled recording labels include source pauses and edges. Removing a
foreground can therefore leave identical float32 input bytes. Quiet160 has
49 such no-op births among 902 recording-labelled births; selected PCEN has
82/712, MCRA raw 68/603 and adaptive gain 65/1,015. The table reports the
overlapping subset where foreground removal **changes the delivered audio**.
This excludes exact no-ops; it is not an annotated event-support mask or a
claim that all remaining source energy belongs to the named event.

| Pipeline | Bed-reproducible births / eligible | Equal-seed birth % [95% interval] | Bed-reproducible admissions / eligible | Equal-seed admission % [95% interval] |
|---|---:|---:|---:|---:|
| Quiet160 raw | 152/853 | 17.71 [15.44, 19.73] | 73/203 | 35.78 [30.00, 42.42] |
| PCEN selected raw | 85/630 | 13.80 [11.85, 15.43] | 45/156 | 26.70 [18.70, 33.59] |
| MCRA raw diagnostic | 165/535 | 31.13 [27.45, 35.04] | 65/107 | 60.70 [49.50, 72.86] |
| Adaptive-gain anchor | 109/950 | 11.29 [9.93, 12.54] | 39/205 | 18.12 [8.33, 28.43] |

Counts pool suites and seeds. Estimates first aggregate counts across the
four positive suites within each seed, then equally weight ten seed ratios.
Intervals use 10,000 seed-bootstrap draws, conditional on five fixed
foreground files. All shown denominators are defined; the report returns
null means/intervals rather than silently dropping undefined seeds. These
exploratory intervals do not establish a ranking or generalization across
recordings, and no multiplicity adjustment is applied.

Including no-op labelled windows gives Quiet160 201/902 and selected PCEN
167/712 bed-reproducible births. Quiet160's other 701 births split into 243
without a detected bed feature and 458 with an outside-radius bed feature.
Selected PCEN splits its other 545 into 266 and 279 respectively. Reporting
only a single failure fraction would conceal this detector/feature distinction.

The higher admission fraction is conditional on candidates that survived to
admission; it does not mean an individual candidate becomes more background
dependent. Current admission features and persistent birth vectors remain
different objects.

## Birth/admission pairs and selected regions

Across all mixed-stream admissions, including no-target gaps:

| Pipeline | Both ends bed-reproducible | Birth only | Admission only | Neither | Total |
|---|---:|---:|---:|---:|---:|
| Quiet160 raw | 221 | 15 | 6 | 115 | 357 |
| PCEN selected raw | 168 | 10 | 5 | 101 | 284 |
| MCRA raw diagnostic | 261 | 11 | 3 | 31 | 306 |
| Adaptive-gain anchor | 210 | 18 | 11 | 149 | 388 |

These totals include obvious background windows and must not be read as
target-window percentages. For Quiet160's 69 no-target-to-target admissions,
63 are bed-reproducible at both ends and six at birth only. Yet only four
of its 116 same-recording birth/admission pairs are reproducible at both
ends; 102 are reproducible at neither. This supports retaining both temporal
context and the intervention diagnostic instead of assigning one causal
label to every prototype.

Reselection matters. Of Quiet160's 201 bed-reproducible recording-window
births, 92 select changed region boundaries and eight select disjoint
regions. At admission the corresponding counts are 53 and one among 91.
The intervention reruns the entire selected-feature pipeline, including
noise-estimator processing. It cannot attribute changes to a fixed acoustic
patch. Failure to reproduce a match also does not prove semantic relevance,
clean-source identity or successful recognition.

Older-tail removal is a distinct intervention. In selected PCEN's no-target
gaps with a prior-source tail, all 89 birth features still match after
removing the current episode (there is none), but only 87 match after
removing all foreground contributions. Adaptive gain gives 33 versus 32.
The archive keeps these cases separate from recording-labelled windows and
tail-free gaps; a current-episode-only probe can miss this residual effect.

All 1,746 pure-control admissions across the ten configurations reproduce
the stored match at both ends with identical actual/bed packets. They are
negative controls for the instrument and direct evidence of commitments to
the experimental background role, not a foreground-deletion success rate.
A recorded bed can itself contain meaningful sounds.

## Next experiments and limits

Beads ch-6w4.10.8.2 records this completed replay. Fixed-region extraction
alongside reselection is tracked in ch-6w4.10.8.3. It should preserve original
state boundaries and absent/zero controls before interpreting feature changes.

The next learner investigation, ch-6w4.10.9, should measure whether a
mixture-only background predictor or feature-compatibility signal identifies
these ambiguous candidates. It must receive no clean stems or evaluator
labels. Learning progress is a hypothesis worth comparing with prediction
error [1]; predictable alarms and unique important sounds remain necessary
counterexamples. A post-hoc oracle that rejects every bed-reproducible
candidate would not be a deployable result.

Active-support calibration remains ch-6w4.1.4: the no-op subset here explains
why scheduled windows need a physical-support audit, but does not define
that support. Continuous STFT framing, broader source groups and the MCRA
state investigation from paper 08 remain separate. Any changed admission
policy or representation needs a fresh frozen source evaluation. These
observed recordings cannot become held-out evidence again by choosing new
simulation seeds. Hardware dependencies remain in `HUMAN_TODO.md`; this
diagnostic required no human input.

## Reproduction and sources

[`published-feature-dependence/`](results/masked-events/published-feature-dependence/)
contains raw probe packets, sixty vector bundles, admission pairs, copied
original evidence and exact source/input/output/vector hashes.
[`published-feature-dependence-report/`](results/masked-events/published-feature-dependence-report/)
contains all ten configurations, context/stage counts, per-recording counts,
seed ratios and independently recomputable intervals. Their READMEs give
offline audit and replay commands. Instrument details are in
`python/src/chuff_mpu/sim/FEATURE_PROBES.md`.

1. Oudeyer, P.-Y., Kaplan, F. and Hafner, V. V. (2007). *Intrinsic Motivation
   Systems for Autonomous Mental Development*. IEEE Transactions on
   Evolutionary Computation 11(2), 265–286.
   [Author-hosted paper](https://adapt.informatik.hu-berlin.de/pub/papers/ims.pdf),
   [DOI](https://doi.org/10.1109/TEVC.2006.890271).
2. Pathak, D., Agrawal, P., Efros, A. A. and Darrell, T. (2017).
   *Curiosity-driven Exploration by Self-supervised Prediction*. ICML.
   [Primary manuscript](https://arxiv.org/abs/1705.05363).
3. Burda, Y., Edwards, H., Storkey, A. and Klimov, O. (2018).
   *Exploration by Random Network Distillation*.
   [Primary manuscript](https://arxiv.org/abs/1810.12894).

Codex assisted with implementation, execution, auditing and drafting. Aidan
Jones retains responsibility for interpretation. This is an exploratory
engineering report, not a peer-reviewed paper.
