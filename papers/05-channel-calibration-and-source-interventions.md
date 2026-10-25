# Calibrating Acoustic Masking After the Channel and Testing Current-Source Dependence

**Aidan Jones — 25 October 2026, exploratory engineering report with Codex assistance**

## Result

Earlier Chuff experiments set foreground SNR before room and microphone
filtering. This is not generally the SNR the listener receives. With a nominal
−6 dB input target, this study's reference evaluation spans **−11.30 to +0.33 dB**
after the channel, with median −4.49 dB. Matching current-source/residual SNR
to −6 dB after filtering lowers calibrated-gain localized repeat recognition
from 46.5% to 34.2%. Independent pitch and duration transforms further expose
identity instability. Identical-prior-state source-removal probes distinguish
new-source dependence from background attention and earlier echoes.

This completes the measurement work in ch-6w4.9. It does not establish source
separation in recorded audio or semantic importance. Models are unchanged;
the simulator and evaluation become more explicit.

## Linear components and SNR definitions

`SceneConfig.collect_stems=True` optionally renders ambient background and
injected sources separately through the **same** room impulse response and
microphone IIR filter. Each component retains its own continuous overlap-add
tail and filter state. Their sum matches the original mixture to float32
precision. Enabling collection leaves the ordinary mixture bit-identical on
the tested fixture. No component reaches the learning interface.

The simulator also renders the current source's response from zero channel
state and subtracts it from the full mixture. This removes only the current
emission while retaining every earlier source and background response,
including microphone IIR history. It differs from replaying an all-background
trajectory and avoids assuming that a finite room-tail flag bounds all echoes.

Both SNR definitions use the original active source interval, before any
tail extension. **Input SNR** is current source versus ambient background
before filtering. **Post-residual SNR** is the current source's filtered
response versus the source-removed residual after filtering, before learner
preprocessing. The residual includes earlier source tails. A separate
post-channel source-versus-ambient SNR is recorded too. None is whole-response
energy over an infinite IIR tail or a calibrated physical-room measurement.

For `snr_reference="post_residual"`, linearity permits scaling the current
source to the target using its filtered response and the already-existing
residual. This does not update channel history twice. The real mixture is
then rendered once with that source level. It is a synthetic mixture-design
operation, not an oracle supplied to the learner. Ideal floating-point channels
do not model microphone saturation, acoustic nonlinearities or power limits.

## Independent source factors

The earlier resampling intervention changed pitch and duration together.
Here reference, pitch-only, duration-only and independent pitch+duration cases
use separate parameter streams. Pitch ratio is log-uniform 0.85–1.15;
duration multiplier is uniform 0.8–1.2, capped so the source fits within
0.94 s. Requested and applied factors are both retained, since the cap changes
the duration distribution for long alarms.

The implementation uses [librosa 0.11 pitch shifting](https://librosa.org/doc/0.11.0/generated/librosa.effects.pitch_shift.html)
and [time stretching](https://librosa.org/doc/0.11.0/generated/librosa.effects.time_stretch.html),
with FFT 1024 and hop 256. Pitch shifting preserves sample count; stretch rate
is the reciprocal of the desired duration multiplier. Sinusoidal fixtures
verify shifted frequency at unchanged duration and stretched duration at
unchanged frequency. Phase-vocoder/resampling artifacts are part of these
interventions; they do not reproduce natural differences between speakers,
animals or objects. Factory labels remain evaluator class proxies, and large
transforms may make their identity assumptions ambiguous.

## Identical-state source-deletion probes

Immediately before each foreground observation, the evaluator deep-copies the
learner. The copy receives the current-source-removed waveform, then is
discarded. The real learner receives the original mixture and continues.
Both begin from identical background estimates, candidate/prototype memories,
gain profile and mode choice. Tests verify that enabling probes does not
change the complete real decision trajectory.

Saved probe outcomes include attention, selected region, prototype ID and
whether the current evaluator class is recognised without the new emission.
The latter can reflect earlier same-class echoes or a background match;
it is not automatically semantic error. Status/ID change also does not capture
every possible internal feature change. This is a current-source intervention
in the explicit linear simulator, not a field causal-importance claim.

## Fixed evaluation

Development seeds 1500–1501 validate the pipeline; fresh seeds 1600–1609
evaluate unchanged original, quiet-band and calibrated-gain learners. Each
stream has 120 windows, varied room/microphone and background parameters,
slow modulation and a halfway fan/level change. Eight positive conditions
cross the four source transforms with the two −6 dB SNR references; a ninth
condition is pure background. Three methods receive each identical mixture.
All source labels/schedules match across positive conditions. There is no
method selection or tuning on these evaluation seeds.

The local plan and freeze record are `channel-protocol.json` and
`channel-frozen-evaluation.json`. An inherited, unused coupled-resampling
metadata field in development is explicitly corrected in evaluation; actual
development waveform transforms and per-event parameters were already correct.
Development snapshots are preserved. These are local prospective records,
not external preregistration.

There are 395 source windows per positive condition across the ten seeds.
For reference/input, post-residual SNR varies by class and channel:

| Factory class | Windows | Mean post-residual SNR | Range |
|---|---:|---:|---:|
| Double chime | 72 | −4.21 dB | −7.87 to −0.95 |
| Falling call | 73 | −4.36 dB | −7.07 to −2.59 |
| High rustle | 56 | −5.78 dB | −7.93 to −4.09 |
| Long alarm | 71 | −4.24 dB | −11.30 to +0.33 |
| Low rustle | 56 | −4.92 dB | −6.65 to −3.40 |
| Rising call | 67 | −4.20 dB | −7.01 to −2.19 |

These are pooled stimulus descriptions, not independent statistical trials.
Matched post-residual cases achieve −6 dB on every event within the audit
tolerance. Their required input levels vary instead. This does not invalidate
earlier nominal-input measurements; it narrows their interpretation and shows
why claims about received masking require a specified channel/reference.

Mean per-seed correct repeat identity with IoU ≥0.25:

| Source factor | Quiet-band input / post target | Calibrated-gain input / post target |
|---|---:|---:|
| Reference | 32.6% / 19.4% | 46.5% / 34.2% |
| Pitch only | 18.2% / 9.0% | 23.4% / 17.3% |
| Duration only | 25.4% / 15.3% | 41.6% / 26.6% |
| Pitch and duration | 13.1% / 7.4% | 20.8% / 18.8% |

The original frontend ranges from 0.4–2.3% in the input-target conditions
and 0.0% in all four post-target conditions. For calibrated gain, changing
reference/input to reference/post-residual reduces recognition by 12.3
percentage points (paired seed-bootstrap 95% interval −17.5 to −6.4).
At matched post SNR, pitch-only reduces it by 16.9 points (−25.7 to −8.6)
and duration-only by 7.7 points (−14.1 to −2.2), relative to reference.
These descriptive intervals use 10,000 paired seed draws, without multiplicity
adjustment. Conditions share ten seeds and are not independent replications.

Calibrated-gain combined-transform recognition is 18.8% (13.3–24.0) at matched
post SNR. This remains a substantial identity limitation, despite its advantage
over these baselines. Transformation artifacts and class assumptions prevent
treating these differences as natural acoustic invariance measurements.

## What changes when the current source is removed

On reference/input events, quiet-band still attends on 21.2% of source-deleted
probes, overlaps the removed source interval on 3.8%, and recognises the current
class on 2.0% of eligible repeats without its new emission. Calibrated gain
attends on 0.2% of the corresponding probes, with zero observed overlap or
current-class recognition. The latter's post-target attention is also 0.2%.
These rates are conditional on foreground-bearing times and averaged over
seeds, not unconditional background-window rates.

The calibrated method has no false admissions in any evaluation condition.
Quiet-band admits ten background prototypes in the pure-background control
and seven to eight per positive condition, across ten streams per condition.
The original admits none but detects very few targets. Zero observed
admissions/identity in these controls is not proof of zero field risk.

The source-deletion results strengthen evidence that calibrated-gain decisions
depend on the new acoustic source, while confirming that identity is still
unstable under variation. Earlier matched all-background trajectories remain
useful diagnostics, but do not control prior history in the same way.

## Evidence, limits and next steps

Development archives hold 6,480 real decisions; evaluation holds 32,400 real
decisions, plus discarded source-deletion branch outcomes on foreground windows.
These are not independent sample sizes. Raw rows, all seed/class summaries,
source parameters, pre/post SNR, exact sources, dependency lock and eight
component examples are archived under `results/masked-events/channel-*`.
Maximum observed component-recombination error is 5.96e−8 in float audio units.
The audit checks source/example hashes, components, SNR targets, complete
matrices, timing, transform independence, metrics and paired intervals.
Derived condition contrasts retain exact analysis-source copies and input
summary hashes. All checks pass.

Sixty-seven software tests pass, including analytic SNR, persistent components,
old-tail retention, transform independence and unchanged real trajectories.
The original grader remains 1.0. No camera, microphone or speaker is exercised.
No detector settings are changed by this measurement task.

Paper [06](06-changing-backgrounds.md) now uses these component/deletion
tools for silent startup, changing modulation and spectral shifts. Deliberate
startup contamination and independent source overlap remain
ch-6w4.10.4.2.3. Bounded pitch/duration-tolerant identity matching without
merging distinct classes remains ch-6w4.10.6. Recorded evaluation and direct published
baselines remain ch-6w4.1/.7. Seeds 1600–1609 are now observed evidence, not a
future sealed test set. Source separation, semantic importance and field
robustness are not established by this procedural study.

Codex assisted with implementation, documentation checks, execution, auditing
and drafting. Aidan Jones retains responsibility for interpretation. This is
an exploratory project report, not a peer-reviewed paper.
