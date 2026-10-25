# Masked-event research and reproduction

This follow-up compares the original selective frontend, a band-local gate
with raw features, a relative-feature ablation, and the selected quiet-band
frontend. It adds procedural broadband bursts and long alarms, nuisance-only
streams, paired seed uncertainty, exact source snapshots, and a decision audit.
See [the paper](../../../../papers/02-masked-foreground-events.md).

From the python directory, use new output directories:

```powershell
uv run chuff-sim-masked --phase development --seed-start 0 --seeds 3 --snr-db -6 0 --output ../eval/results/my-masked-dev
uv run chuff-sim-masked --phase evaluation --seed-start 300 --seeds 10 --snr-db -6 0 6 --output ../eval/results/my-masked-eval
uv run chuff-sim-masked --phase stress --seed-start 400 --seeds 10 --snr-db -12 -9 --output ../eval/results/my-masked-stress
uv run python -m chuff_mpu.sim.audit_masked ../eval/results/my-masked-eval ../eval/results/my-masked-stress --output ../eval/results/my-class-results.json
```

Existing paper results live under papers/results/masked-events and are not
overwritten. The phase flag documents purpose; it does not enforce scientific
independence. Seeds 300–409 are now observed evidence and must not be reused
as an untouched test set after tuning a new method.

The source and schedule are shared across methods within a run. The evaluator
uses event labels only after waveform-only decisions. Aggregate intervals use
paired bootstrap resamples of seeds, never individual windows. False-admission
counts and missed-event denominators matter as much as recognition. A zero
bootstrap range after zero observed admissions does not prove zero risk.

Core settings live in BandAttentionConfig. The selected default uses local
ratio threshold 3.5, broad ratio threshold 1.5, five-frame smoothing, shared
quiet-frame noise estimation, raw excess features, and 80 ms of instantaneous
support. Smoothing alone cannot provide that support. Same-memory ablations
change only the band feature/noise-estimation modes; the original frontend has
its original duration, flatness and energy rules.

The experimental frontend is opt-in for capture/replay:

```powershell
uv run chuff-selective --frontend band --input-wav path/to/16khz-recording.wav --state ../eval/results/band-state.npz
```

Use a separate state file. Original and band memory schemas are different,
even though default vector lengths are both 96. Both remain incompatible with
the 13-D legacy voice/dream representation. `--frontend original` remains
the capture default; this study does not validate physical microphone audio.

Important limits: one selected event region per window, no clean-waveform
reconstruction, no semantic importance labels, poor long-alarm recognition at
−6 dB, and rare distractor memories at −9/−12 dB. The quiet-frame assumption
fails when no useful quiet part of a window exists. Improving those cases
needs new development examples and new held-out recordings, not favourable
reruns of the present evaluation seeds.

## Factorized variance study

`VariedScene` extends the six procedural classes without changing the earlier
generator or either detector. It supports reference, source, room, microphone,
background and joint conditions. Source resampling couples pitch and duration
(0.75–1.3 duration multiplier, capped at 0.94 s). Room response lengths span
0.03–0.5 s with randomized sparse echoes; these are not measured RT60 values.
Microphone cutoffs span 60–450 Hz and 2.8–7.2 kHz at 16 kHz, with ±12 dB gain.
Backgrounds vary their colour, hum, fan, modulation and halfway level change.
Room and microphone stay fixed within a run; source variation occurs per event.
There are no simultaneous events or recorded sounds in this study.

From `python`, reproduce into a new directory:

```powershell
uv run python -m chuff_mpu.sim.variance_experiment --seed-start 600 --seeds 10 --output ../eval/results/my-variance
uv run python -m chuff_mpu.sim.audit_variance ../eval/results/my-variance --output ../eval/results/my-variance-analysis.json
```

The extra `joint_background` condition has no foreground events but retains
nuisance distractors. Environment seeds equal 100000 plus the event seed;
schedule RNG draws are independent of environment rendering. Both detectors
receive identical audio. Active SNR is matched **before** room/microphone
filtering; post-channel masking can differ by class. This is a robustness
probe, not an acoustically calibrated benchmark. Follow-up ch-6w4.9 addresses
post-channel SNR and uncoupled pitch/time transformations. Seeds 600–609 become
observed examples after this evaluation and cannot remain a sealed test set.

## Memory failure diagnostics and admission policies

The admission experiment tests the unchanged original and quiet-band learners,
quiet-band with four confirmations, broader matching (cosine distance 0.28
instead of 0.18), and both changes together. It is a fixed sensitivity study,
not an automatically selected replacement frontend. From `python`:

```powershell
uv run python -m chuff_mpu.sim.admission_experiment --seed-start 700 --seeds 10 --output ../eval/results/my-admission
uv run python -m chuff_mpu.sim.audit_admission ../eval/results/my-admission --output ../eval/results/my-admission-contrasts.json
```

Reference, source-only and joint variation share event schedules. Joint
no-foreground controls retain distractors; separate pure-ambience controls
set both event and nuisance probabilities to zero. Pure ambience is identical
across nominal SNRs, so those repetitions do not provide independent evidence.
SNR remains measured before the channel for source-bearing windows.

`memory_diagnostics.diagnose` partitions repeats into missed detection,
pending candidate, first admission, duplicate admission, wrong identity and
correct identity. Repeat eligibility is fixed after two true class occurrences
for **all** policies; requiring four confirmations does not move the goalposts.
First admission is reported as the occurrence number of that class, counting
missed detections too. An extra admission for a class is a proxy for memory
fragmentation; the broad procedural class labels do not prove every instance
should be represented by a single prototype. False admissions are split by
whether a nuisance is present in the current window; only source-free controls
exclude earlier event tails as an explanation. Labels are used after the
waveform-only decisions and are never passed into any learner.

The audit recomputes counts and partitions, checks the complete run matrix,
source hashes, pairing, source-free control truth, and paired seed intervals.
Seeds 700–709 become observed evidence after this study and cannot be reused
as untouched evaluation examples for an algorithm designed from its results.

## Isolating background modulation

`ModulationScene` excludes halfway changes, nuisance events and widened source
pitch/duration variation. It crosses two background spectra, two channel
conditions (reference versus varied room/microphone), and modulation at zero
depth, depth 0.45 at 0.17 Hz, or depth 0.45 at 1.3 Hz. Each cell has pure
ambience and foreground-positive runs. Source SNR is −6 dB before filtering;
there is no event SNR for pure ambience. Equal seeds preserve underlying
noise draws and event schedules across the factors.

The fixed methods are original, quiet-band, and the existing binwise
noise-estimation ablation (noise_mode=0). Detection and memory settings are
otherwise unchanged. Reproduce into new directories from `python`:

```powershell
uv run python -m chuff_mpu.sim.modulation_experiment --phase development --seed-start 800 --seeds 3 --output ../eval/results/my-modulation-dev
uv run python -m chuff_mpu.sim.modulation_experiment --phase evaluation --seed-start 900 --seeds 10 --output ../eval/results/my-modulation-eval
uv run python -m chuff_mpu.sim.audit_modulation ../eval/results/my-modulation-eval --output ../eval/results/my-modulation-analysis.json
```

The audit checks raw scores and hashes, the complete factor/seed matrix,
matched schedules, source-free controls, absence of halfway changes and
diagnostic partitions. Report affected-stream counts as well as prototype
totals, and foreground recall alongside background rejection. Modulation
conditions share seeds and are not independent replicates. Evaluation seeds
900–909 become observed evidence after this study.

## Detector timing and event localization

`Observation.region_start_s/region_stop_s` report the sample support of the
selected STFT frames. They do not affect features or memory decisions.
`SceneFrame.event_start_s/event_stop_s` are evaluator-only, pre-channel source
intervals. `previous_source_tail_s` flags finite room-IR support continuing
from a previous source; the IIR microphone filter has no exact finite tail.
Neither interval nor tail annotation is passed to `observe(audio)`.

Window-level detection remains available for historical comparison. The new
scorer also requires intersection-over-union (IoU) between the selected
region and the source interval: primary threshold 0.25, sensitivity thresholds
0.1 and 0.5. Localized repeat recognition additionally requires a correctly
admitted identity. Missing timing produces unknown metrics, not zero success.
Background attention is split by the finite room-tail flag. Wide regions may
overlap by chance, so these metrics do not prove source attribution.

The focused six-cell evaluation uses varied spectrum/channel, three rates,
and ambience/foreground conditions at −6 dB. Reproduce into a new path:

```powershell
uv run python -m chuff_mpu.sim.modulation_experiment --phase evaluation --localization --seed-start 1000 --seeds 10 --output ../eval/results/my-localization
uv run python -m chuff_mpu.sim.audit_localization ../eval/results/my-localization --output ../eval/results/my-localization-analysis.json
```

Seeds 1000–1009 become observed evidence after evaluation. The archive retains
the original window metric and raw timing so thresholds can be independently
audited. Historical archives are preserved with their original definitions.

For post-hoc overlap controls on an existing localization archive:

```powershell
uv run python -m chuff_mpu.sim.overlap_controls ../eval/results/my-localization --output ../eval/results/my-localization-controls
```

The new output directory contains analysis and source/input hashes. Controls
compare positive-stream regions with uniformly spaced, length-preserving
alternative event positions (200 quadrature points within the generator's
legal placement margins), and pure-background regions at matched seeds/times.
The latter uses entire no-foreground trajectories, so learner histories differ.
Neither diagnostic is a causal intervention or a permutation significance
test. All three IoU thresholds and per-class duration summaries are retained.

## Shared-gain development and frozen comparison

Paper 04 documents three development stages and the fixed shared spectral-gain
candidate. RMS times 0.05/0.2/0.5 s failed fast-attention controls; faster
0.005/0.01/0.02 s rejected controls but lost foregrounds. The spectral method
uses median mel-band ratios to a three-window warmup template and a fixed
0.02 s gain filter. The template freezes afterward. Neither method is PCEN.

```powershell
uv run python -m chuff_mpu.sim.modulation_experiment --phase development --localization --gain-study --gain-time-s .005 .01 .02 --seed-start 1100 --seeds 3 --output ../eval/results/my-fast-gain-dev
uv run python -m chuff_mpu.sim.modulation_experiment --phase development --localization --gain-study --gain-time-s .02 --spectral-gain --seed-start 1100 --seeds 3 --output ../eval/results/my-spectral-gain-dev
uv run python -m chuff_mpu.sim.modulation_experiment --phase evaluation --localization --gain-study --gain-time-s .005 .01 .02 .05 .2 .5 --spectral-gain --seed-start 1200 --seeds 10 --output ../eval/results/my-gain-eval
uv run python -m chuff_mpu.sim.audit_localization ../eval/results/my-gain-eval --output ../eval/results/my-gain-analysis.json
uv run python -m chuff_mpu.sim.overlap_controls ../eval/results/my-gain-eval --output ../eval/results/my-gain-controls
```

This is a procedural trade-off: improved modulated-background identity and
rejection, but lower stationary recall. Seeds 1200–1209 are now observed.
The `spectral-gain` capture/replay frontend is opt-in, requires a separate state
file and initial background-only calibration, and resets its profile on load.
The original frontend remains the default. Paper 04 covers the stationary loss,
warmup assumptions and untested generalization conditions.

`--calibrated-gain` adds a waveform-derived choice between raw and compensated
power: mean calibration gain CV ≥0.12 enables compensation; otherwise raw
power preserves the original learner. The first three unlabelled windows
make the fixed choice; the learner never receives condition labels. This
preserves stationary scores on its separate controlled evaluation. Mid-run
changes are measured separately in the changing-background experiment below;
deliberately contaminated initialization remains open.

```powershell
uv run python -m chuff_mpu.sim.modulation_experiment --phase evaluation --localization --gain-study --gain-time-s .02 --spectral-gain --calibrated-gain --seed-start 1400 --seeds 10 --output ../eval/results/my-calibrated-eval
uv run python -m chuff_mpu.sim.audit_localization ../eval/results/my-calibrated-eval --output ../eval/results/my-calibrated-analysis.json
uv run python -m chuff_mpu.sim.overlap_controls ../eval/results/my-calibrated-eval --output ../eval/results/my-calibrated-controls
```

For capture/replay use `--frontend calibrated-gain` with its own state file.
Observation JSON reports `gain_compensation_enabled` (null during calibration,
then the fixed selection). Loading resets that decision and the gain profile.
Paper 04 reports the fresh results and limitations. Seeds 1400–1409 are now
observed examples, not an untouched future evaluation split.

## Physical-channel components and source-deletion probes

Paper 05 adds optional `SceneConfig.collect_stems` and the SNR reference
`input` (default) or `post_residual` (requires stems). Components share the
same continuous linear room/microphone channel. `source_audio` includes all
current/prior injected-source responses; `current_source_audio` is the current
injection alone. `without_current_source_audio` retains background and prior
source echoes, including IIR state. Ratios use the original active source
interval, not whole-tail energy; post SNR is measured before learner gain
normalization. Components are evaluator-only and are never passed to learners.

Independent source factors use separate pitch/time library operations and
RNGs, with per-event requested/applied parameters. They may introduce artifacts
and are not natural-source or semantic equivalence guarantees.

```powershell
uv run python -m chuff_mpu.sim.channel_experiment --phase development --seed-start 1500 --seeds 2 --output ../eval/results/my-channel-dev
uv run python -m chuff_mpu.sim.channel_experiment --phase evaluation --seed-start 1600 --seeds 10 --output ../eval/results/my-channel-eval
uv run python -m chuff_mpu.sim.audit_channel ../eval/results/my-channel-eval --contrasts-output ../eval/results/my-channel-contrasts.json
```

`evaluate(..., counterfactual_probes=True)` deep-copies each learner before
observing a source-removed frame and discards that branch. The real trajectory
always receives the mixture and is verified unchanged. Eight compressed
component examples support direct recombination/SNR audits. Evaluation seeds
1600–1609 are now observed evidence. Existing archives keep their earlier
input-SNR convention and exact sources.

## Silent-start numerical regression

The fixed spectral template now waits for usable median band energy rather
than fitting exact-zero calibration windows to a floor-valued profile. It
falls back to raw power while unavailable, then fits three usable windows;
late gain calibration clears background/pending estimates, retains admitted
prototypes, and prevents admissions during fitting. The calibrated raw choice
remains frozen and baseline-equivalent. This does not solve later environment
changes or foreground-contaminated calibration.

```powershell
uv run python -m chuff_mpu.sim.startup_recovery --reference-source ../papers/results/masked-events/channel-evaluation/source/python/src/chuff_mpu --seed-start 1800 --seeds 10 --output ../eval/results/my-startup-recovery
uv run python -m chuff_mpu.sim.audit_startup ../eval/results/my-startup-recovery
```

The regression compares archived/current learners with and without three
initial zero windows. Runtime assertions check full feature/salience equality;
the saved-decision audit checks counts, timing matrix, IDs and novelty. It is
a compatibility fixture sweep, not a new field-accuracy benchmark.

## Changing-background experiment

`TransitionScene` adds paired stationary/slow/fast transitions, six-window
silent startup followed by fast modulation, shallower/slower modulation, and
a fixed-fast-modulation background spectrum change. Pure no-target controls
share the background/channel draws. All positive cases share target identities
and placements. The continuous physical filter state survives changes; abrupt
transitions can themselves produce transients. Sources use fixed procedural
factories without the independent pitch/duration transforms in paper 05.

The target is -6 dB current-source/residual **after** the channel. Each frame
is checked for component recombination and target SNR. Actors receive only
the mixture; phase labels and source-deletion stems remain evaluator-only.
Phase metrics carry prior occurrences/admission labels across the change.
Resetting the evaluator at the change would incorrectly discard that history.

```powershell
uv run python -m chuff_mpu.sim.transition_experiment --phase evaluation --seed-start 2000 --seeds 10 --output ../eval/results/my-transitions
uv run python -m chuff_mpu.sim.audit_transition ../eval/results/my-transitions
```

The runner snapshots sources/settings/dependencies and hashes its outputs.
Its audit verifies schedules, matrices, frozen gain choices, stored post-SNR,
phase counts and seed intervals. Component checks happen during execution;
this archive does not retain every waveform for independent acoustic replay.
Re-run its source snapshot to regenerate them. Reproduction uses observed
seeds; a new method needs a new evaluation split after development.

`--adaptive` additionally includes `AdaptiveGainForegroundLearner`, currently
an experimental research actor. It requires 7/9 windows with CV >=.12 to
enable compensation or <=.09 to disable it. Switches apply on the next
window, clear background/pending estimates, preserve admitted vectors/IDs,
and suppress admissions for three settling windows. Its own saved schema
resets all mode/history/profile state on load. Persistent foreground can
still fool this rule; retaining a vector does not establish cross-mode
identity matching. It is not yet exposed as a live/replay frontend.

```powershell
uv run python -m chuff_mpu.sim.transition_experiment --adaptive --phase development --seed-start 1900 --seeds 2 --output ../eval/results/my-adaptive-dev
uv run python -m chuff_mpu.sim.transition_experiment --adaptive --phase evaluation --seed-start 2200 --seeds 10 --output ../eval/results/my-adaptive-eval
uv run python -m chuff_mpu.sim.audit_transition ../eval/results/my-adaptive-eval
```

Paper 06 reports the frozen fresh-split result and its remaining transition
admission errors. The adaptive audit also checks memory retention at switches
and recomputes evaluator-only mode latency. Seeds 2200–2209 are observed;
new policy variants need another held-out evaluation after development.

## Conditional admission and one-off attention

`--adaptive --guarded` adds a research actor that conditionally holds new
admissions during raw-mode gain uncertainty. Two high-CV values in a rolling
three-window history activate the guard; three low-CV windows or compensation
clear it. A memory-free, always-compensated shadow frontend must then detect
a vector within the ordinary cosine radius of the proposed stored candidate.
Recognition of established memories and sensory extraction continue unchanged.
Held confirmations still report candidate attention; a smaller admission count
does not imply smaller false attention. See paper 06 for the residual
first-changed-window error and broadband/class coverage costs.

```powershell
uv run python -m chuff_mpu.sim.transition_experiment --adaptive --guarded --phase evaluation --seed-start 2400 --seeds 10 --output ../eval/results/my-guarded
uv run python -m chuff_mpu.sim.audit_transition ../eval/results/my-guarded
uv run python -m chuff_mpu.sim.one_off_experiment --phase evaluation --seed-start 2400 --seeds 10 --output ../eval/results/my-one-offs
uv run python -m chuff_mpu.sim.one_off_experiment --audit ../eval/results/my-one-offs
```

The one-off experiment has one target at window 66 per stream, six factories
and two backgrounds. It measures localized attention and novel status, with
factory outcomes averaged within seed before bootstrap. It does not test
importance learning or one-shot persistent memory. Guarded analysis reports
lost/gained classes alongside conditional admission delays and uses paired
differences against the current adaptive selector. These evaluation seeds
are observed; changing the policy requires another frozen split.

The raw `held_admissions` counter counts blocked commitment attempts, not
distinct lost classes. Mixed-stream no-target-window admissions use the
historical admission-time label convention; candidate birth may precede
that window. Ch-6w4.10.7 tracks evaluator-only provenance diagnostics.

## Recorded input preparation

`papers/data/esc10-pilot/README.md` describes the pinned attributed catalog,
ignored audio cache, fetch command and offline checksum/source-disjointness
audit. Paper 07 adds full-duration recorded replay; the procedural results
above remain distinct from those recorded-input outcomes.

## Full-duration recorded replay

Complete five-second ESC-10 recordings are injected into continuous recorded
backgrounds under independent room/mic simulation. Actors observe one-second
windows; only third/fourth replay episodes count as repeat opportunities,
and same-episode admissions cannot earn repeat credit. These are recording
identities with embedded ambience, not annotated natural events or semantic
categories.

```powershell
uv run python -m chuff_mpu.sim.recorded_experiment --phase development --partition development --seed-start 2600 --seeds 2 --catalog ../papers/data/esc10-pilot --cache ../data/cache/esc10 --output ../eval/results/my-recorded-dev
uv run python -m chuff_mpu.sim.recorded_experiment --phase evaluation --partition evaluation --seed-start 2700 --seeds 10 --catalog ../papers/data/esc10-pilot --cache ../data/cache/esc10 --output ../eval/results/my-recorded-eval
uv run python -m chuff_mpu.sim.audit_recorded ../eval/results/my-recorded-eval
uv run python -m chuff_mpu.sim.recorded_exclusion ../eval/results/my-recorded-eval --output ../eval/results/my-recorded-exclusion
uv run python -m chuff_mpu.sim.recorded_exclusion ../eval/results/my-recorded-eval --output ../eval/results/my-recorded-exclusion --audit
```

Fetch the ignored cache with `recorded_catalog fetch` first. Phase and source
partition must match. Each stream is 210 seconds, with twenty full episodes.
The two nominal SNR controls are identical without targets. SNR measures the
whole-recording post-channel current/residual ratio; raw rows retain large
within-window variation caused by source pauses.

The planned source-conditioned endpoint requires one correct recognition
window whose matching source-removal probe fails. The stricter secondary
endpoint also rejects episodes if any probe recognizes the recording.
Per-window probes copy actual prior state, remove the full current-episode
physical response and discard updates. They retain earlier actual exposure
and are not one all-episode counterfactual history. Neither endpoint proves
which natural event within the noisy recording was encoded.

Source/config/input hashes, per-recording metrics and attributed full
component examples support audits. Secondary diagnostics preserve their
sources and raw-input hashes. Original source IDs are disjoint across
partitions; uncertainty is conditional on these fixed clips. Existing
learners remain unchanged. Broader category coverage, active-support SNR
and published frontend baselines remain open Beads tasks.

The research adapters in `published_attention.py` now provide stateful PCEN
and MCRA primitives with the same candidate, prototype and memory backend.
PCEN calls librosa 0.11 with persistent filter state on magnitude mel bands,
undoes SciPy's Hann normalization and uses an amplitude scale of 2**31.
It retains this project's 32 HTK bands, 512-sample FFT and 160-sample hop;
this differs from the original keyword-spotting frontend. An additional
quiet-frame background estimate converts PCEN output to detector evidence.
Warmup processes the filter state. See the
[librosa PCEN reference](https://librosa.org/doc/0.11.0/generated/librosa.pcen.html).

MCRA follows the recursions and defaults in Cohen and Berdugo (2002),
[Noise Estimation by Minima Controlled Recursive Averaging for Robust Speech Enhancement](https://doi.org/10.1109/97.988717):
16 kHz, FFT 512, hop 128, smoothing coefficients .8/.95/.2, threshold 5
and minimum block length 125 frames. The three-bin frequency weights
[.25, .5, .25] and nearest-edge extension are explicit implementation choices.
The first power frame initializes the noise estimate; the estimate for the
current frame precedes its update for the next frame. Sustained sources can
eventually enter the noise estimate. These adapters do not implement full
keyword spotting or OM-LSA speech enhancement.

Both adapters separate detection normalization from stored features:
`feature_mode=0` stores raw spectral excess, while mode 1 stores relative
detector features. Saved memories use distinct schemas and reload with fresh
frontend state. Reference, chunk-parity and frozen-default trajectory tests
cover these implementations. Their comparative accuracy and runtime have
not yet been established; the fresh-recording comparison is tracked in
Beads ch-6w4.7.2.

The published-frontend comparison has a separate metadata-only source
reservation in `papers/data/esc10-published-reservation`. It uses folds 2/4
and excludes all original source IDs from the observed folds 1/5 pilot.
`recorded_catalog reserve` archives the deterministic rule, exclusions and
pinned metadata without accessing audio. `prepare --reservation ...
--partition development` materializes only development inputs; evaluation
can be materialized separately after selection. Version 2 partition catalogs
retain the reservation and their own audio hashes; the auditor still accepts
the frozen version 1 catalogs. See the reservation README for exact commands.

`published_protocol.py` declares 12 configurations per frontend family and
selects for reduced background admissions only when detection, first-episode
novelty and strict source-exclusion repeat retention constraints hold.
Infeasible groups retain a null selection and a separately labelled
diagnostic candidate. The full grid and untuned defaults remain available.
`published_experiment.py` runs the frozen development/evaluation matrix,
archives source-removal probes and warmed per-observation timings, and keeps
one no-target control per ambience condition/seed. `audit_published.py`
recomputes the raw-row metrics, runtime summaries and selection and checks
the complete matrix, source/input/output hashes and physical evidence.
Paper 08 and Beads ch-6w4.7.2 document this comparison's status and scope.

That frozen comparison is complete: development retains 123,480 actual
decisions and evaluation 126,000, with independent audits of raw/probe
matrices and exact selection. `published_report.py` derives paired-seed
aggregate and contrast intervals while keeping suites correlated within
each seed. It preserves the original evaluation and labels post-run
contrasts as exploratory. `plot_published.py` is a standalone optional
matplotlib renderer; it does not change the experiment dependencies.
Paper 08 reports the foreground/background trade-offs and source-specific
failures, including admission-label ambiguity. New methods need fresh
frozen evaluation sources; these recordings are now observed evidence.

Evaluator-only candidate tracing is available in `memory_provenance.py`;
see `PROVENANCE.md` for its fresh-actor API and event semantics. It preserves
exact decisions and memory vectors in cross-family tests, logs stored and
current feature hashes separately, and isolates source-removal branch
histories. It supplies no labels to actors. The observed-data provenance
replay and separate birth-label scorer now reproduce all 126,000 original
decisions and 40,000 probes, validating 121,767 lifecycle events and 5,940
admissions. `audit_provenance.py` checks exact original links and recomputes
lifecycle/context and label scores. `provenance_report.py` retains suite
correlation in paired-seed sensitivity intervals. Paper 09 reports results
and the birth-label causality limits; existing published scores remain
unchanged. Feature dependence at birth/confirmation is ch-6w4.10.8.

The probe engine is available in `feature_probes.py` with an outer
`InterventionScene`/`FeatureProbeEvaluator` facade; see `FEATURE_PROBES.md`.
It preserves the legacy evaluator and actual learner state while comparing
current-episode removal and all-foreground removal from the same prior
state. Archived vectors support independent numerical checks. Identical-
audio, absent-feature, zero-feature and moved-region controls distinguish
selected-match dependence from acoustic identity. The full recorded study
is ch-6w4.10.8.2; no empirical dependence result is claimed from unit tests.
