# Foreground discovery amid changing acoustic background

**Implementation and exploratory follow-up, 15 October 2026**

This extends the pilot paper's proposed work on acoustic robustness and
meaningful novelty. The practical goal is to remember recurring foreground
structure while habituating to a changing room, rather than treating every
unfamiliar waveform as a useful memory. It is an input-learning experiment;
there is no learned imitation voice or human-importance model yet.

## Implementation

The new scene contains coloured noise, low-frequency hum, level drift, a
mid-run fan-like change, nuisance impacts and one-off tonal sequences,
irregular foreground events, an echo/decay response, and continuous microphone
filtering. The three recurring event families are a harmonic double chime,
an upward call, and a downward call. These are procedural sound designs,
not animal or appliance recordings.

The new ForegroundLearner receives only the mixture. It estimates spectral
ambience, detects short excess-energy events, retains three ordered spectral
summaries, and admits a bounded prototype only after separated recurrences.
New patterns initially remain candidates. Subsequent matches have lower
novelty and are treated as familiar. Background and broadband/very short
events are conservatively filtered before admission. A repeating nuisance
could still be learned: this policy distinguishes recurring acoustic
structure from unpredictable input, not useful from useless meanings.

The design borrows the general noise-estimation idea of spectral subtraction
from [Boll (1979)](https://course.ece.cmu.edu/~ece491/homework/Boll79.pdf).
The distinction between unpredictable novelty and worthwhile learning also
motivates learning-progress approaches such as
[Oudeyer, Kaplan & Hafner (2007)](https://www.pyoudeyer.com/ims.pdf).
The current recurrence heuristic is not an implementation of their curiosity
algorithm or a general solution to acoustic source separation.

## Evaluation

One development seed (0) was used to debug the frontend. Main evaluation
uses five separate seeds (100–104), 120 one-second windows per seed and SNR,
at 0, +6, and +12 dB. An additional masked-signal stress test uses seeds
200–204 at -6 dB. Settings were not retuned after viewing these summaries.
The noise-quantile correction and separated-recurrence rule were fixed before
the archived main sweep. All comparisons receive identical mixtures.

SNR is the source's active-segment RMS ratio before the common acoustic
channel. It is not full-window SNR or a calibrated microphone SNR. The
evaluation uses no clean waveform, timing, label, or SNR as learner input.

The primary recognition metric counts a correct prototype-label association
on events after the first two occurrences of each class, including missed
events in the denominator. The evaluator associates IDs with labels only
after admission. A secondary measure counts admissions on background/nuisance
windows; zero admissions alone would be trivial if a detector ignored
everything, so detection and recognition must be reported alongside it.

The whole-clip comparator uses the original time-averaged MFCC/cosine
admission and recognition thresholds, at the same 32-entry committed-memory
capacity. The single-sighting ablation uses the new frontend but removes the
recurrence requirement. Total state and computational cost are not matched:
the full learner also has a bounded candidate stage and a noise estimate.

## Results

Percentages below are means of seed-level rates. Ranges are observed
minimum–maximum across five seeds, not confidence intervals.

| Active-segment source SNR | Foreground detection | Correct repeat recognition | Recognition range | False foreground-memory admissions, summed over five seeds |
|---|---:|---:|---:|---:|
| -6 dB, stress seeds | 14.2% | 0.6% | 0.0–3.1% | 0 |
| 0 dB | 99.0% | 91.1% | 88.1–97.1% | 0 |
| +6 dB | 100.0% | 96.5% | 90.5–100.0% | 0 |
| +12 dB | 100.0% | 97.0% | 92.9–100.0% | 0 |

All three recurring classes acquired an admitted prototype in every main
evaluation seed at 0/+6/+12 dB. This category-coverage observation is weaker
than correct recognition: a prototype can be fragmented or contaminated
despite being admitted during an event. At -6 dB, category coverage was
1, 0, 2, 3 and 2 classes across seeds, but almost none of their repeats
were correctly recognised. Detection, not just memory policy, fails there.

The single-sighting ablation admitted **14, 18 and 19 nuisance patterns** in
total at 0/+6/+12 dB, respectively, compared with zero for confirmation. Its
repeat recognition was higher: 95.7%, 99.0%, and 99.0%. Thus requiring
recurrence exchanges some speed/recall for resistance to one-off distractors.
It does not make initial acoustic attention disappear: roughly 3.9–5.0% of
non-event windows still attracted candidate-level attention in the full
learner, which is different from committing them to permanent memory.

At +6 dB, the full learner's mean early novelty was 0.859, versus 0.057 on
later repetitions. Early means include the first two occurrences, rather
than only a first-ever sighting. This decline supports habituation for these
patterns; it is not a measure of semantic interest or active exploration.

The unchanged whole-clip rule admitted the background on its first window
in each seed, discovered none of the three event classes, and scored 0%
correct repeat recognition at all tested SNRs. It calls virtually everything
familiar, including foreground mixtures. Its apparent 100% event detection
is therefore uninformative without its 100% background-attention rate. This
is evidence against that unchanged historical baseline for this task, not
a comparison with a tuned modern sound-event detector.

## Evidence and reproduction

Run from the python directory, using new output directories:

```powershell
uv run chuff-sim-attention --seeds 5 --seed-start 100 --rounds 120 --snr-db 0 6 12 --structured-distractors --output ../eval/results/my-main-run --save-audio
uv run chuff-sim-attention --seeds 5 --seed-start 200 --rounds 120 --snr-db -6 --structured-distractors --output ../eval/results/my-stress-run
uv run python -m unittest discover -s tests -v
uv run python ../eval/tasks/sound_memory_and_voice/grader.py --candidate-dir src/chuff_mpu
```

The review checks pass: eight behavioural tests, the original implementation
grader (1.0), and an installed-console-entry smoke check. The eight tests cover
deterministic rendering and SNR calibration, independent schedules, background
and impact rejection, temporal-pattern learning/habituation, candidate expiry,
consecutive-window nonconfirmation, bounded memory, and invalid input handling.

The repository preserves [main-sweep evidence](results/foreground-novelty/main-sweep/summary.json)
and [masked-sweep evidence](results/foreground-novelty/masked-sweep/summary.json),
along with manifests and compressed per-window observations in those folders.
The main sweep has 5,400 window/mode records; the stress sweep has 1,800.
Source hashes identify the implementation used in a dirty working tree based
on revision 6094aed. The dependency lockfile was subsequently updated only to
declare the already-installed SciPy dependency directly; recorded runtime
versions and the numerical implementation are unchanged. Separate source
hashes are retained rather than claiming the dirty base commit alone
reconstructs the new implementation.

These are initial five-seed descriptive results, not a preregistered test or
proof of general robustness. New recordings, rooms, event families and higher
overlap should be evaluated before choosing final thresholds.

## Next research boundary

The strongest immediate limitation is the -6 dB failure: aggressively ignoring
background also hides real events when they are sufficiently masked. The
next stage needs recorded foreground/background mixtures, broadband events
such as footsteps and speech, longer alarms, multiple concurrent sources,
and room/microphone conditions varied independently from event identity.
An adaptive importance or learning-progress signal should be compared with
recurrence, so a new but important one-off can draw a response without every
random noise becoming a permanent memory. These are recorded as a separate
Beads follow-up rather than counted as solved here.

## Follow-up, 17 October 2026

The [masked-event study](02-masked-foreground-events.md) now reports three
development iterations and a new, paired evaluation of band-local evidence
and quiet-frame noise estimation. It substantially improves the procedural
−6 dB case while retaining the original frontend as a comparison. Long alarms
and more severe masking remain difficult; the follow-up archives those
failures rather than superseding this paper's original measurements.
