# Foreground novelty rather than room memorisation

The older demos ask whether a whole one-second MFCC vector is unfamiliar.
That can memorise the ambience first, then classify a short event mixed into
the same ambience as familiar. This experiment instead asks **what changed
above the background, and does that change recur as a coherent pattern?**

## Run it

From the python directory:

```powershell
uv run chuff-sim-attention --seeds 5 --seed-start 100 --rounds 120 --snr-db 0 6 12 --structured-distractors --output ../eval/results/my-run --save-audio
uv run chuff-sim-attention --seeds 5 --seed-start 200 --snr-db -6 --structured-distractors --output ../eval/results/my-masked-run
uv run python -m unittest discover -s tests -v
```

Output directories must be new, to avoid overwriting prior evidence. For a
fast frontend-only check, add --no-baseline. A direct module invocation also
works: `python -m chuff_mpu.sim.attention_experiment`.

## The scene

Each continuous stream contains coloured broadband noise, 100/200 Hz hum,
slow amplitude drift, and a stronger 340 Hz fan-like background introduced
halfway through. A room impulse response adds early echoes and a sparse
decaying reflection tail. A persistent bandpass filter approximates microphone
colouration. Filter state and room tails carry across window boundaries.

Foreground events are irregularly scheduled, jittered harmonic double chimes
and rising/falling calls with smooth envelopes. Some other windows contain
broadband impacts. With --structured-distractors, some nuisance events are
one-off random three-note sequences. These are acoustically salient but have
no stable recurring identity: they test confirmation separately from detection.
An absence of recurrence is an engineering admission criterion, not a claim
that one-off sounds cannot be important to a person.

SNR is calibrated over the source event's active segment relative to that
segment's background **before** the common room/microphone channel. It is not
full-window SNR, physical sound-pressure level, or an oracle input to the
learner. These remain procedural signals, not realistic recordings of named
animals or appliances. The echo model is not a measured room response.

Schedule, source synthesis, background, and channel have independent seeded
generators. Every comparator receives the exact same mixture, including the
same noise samples and event order. A nominal seed never substitutes for
paired audio. Labels, event flags and true SNR reach the scorer only after
the learner has made its decision.

## What is learned

`ForegroundLearner.observe(clip)` in chuff_mpu/attention.py:

1. Estimates a frequency-specific background from previous windows and the
   current window's lower power quantile. The quantile is corrected for the
   approximate exponential distribution of Gaussian-noise periodogram power.
2. Subtracts that estimate in the power spectrum and detects short periods of
   sufficient excess energy. Duration and spectral-flatness checks reject
   sustained ambience and many broadband impacts.
3. Summarises residual mel-band energy in three ordered temporal parts, so a
   rising and a falling sweep need not collapse into the same average.
4. Holds novel structure in bounded candidate memory. A similar event must
   recur after at least one intervening window without that candidate before
   admission to long-term prototype memory. Consecutive windows alone do not
   confirm a candidate. Unconfirmed candidates expire after 40 windows.
5. Recognises subsequent matches with a low novelty score. Prototypes remain
   fixed after admission to prevent gradual drift into background. Both memory
   stages are bounded at 32 entries; admitted entries use last-recognised
   eviction, and pending entries use last-observed eviction.

Settings live in the validated AttentionConfig dataclass, separately from the
legacy MFCC thresholds. Default match distance is 0.18, calibration lasts
three windows, and admission requires two separated sightings. The simulator
starts events after six windows, but the learner does not inspect that schedule.

Example integration into a capture loop:

```python
from chuff_mpu.attention import ForegroundLearner

listener = ForegroundLearner()
result = listener.observe(microphone_clip)  # mono 16 kHz, one second
if result.status == "candidate":
    pass  # curious, but not yet committed to lasting memory
elif result.status == "learned":
    pass  # a new recurring foreground pattern; trigger a learning display
elif result.status == "recognised":
    pass  # familiar pattern; habituate instead of replaying the background
```

The prototype ID is an unsupervised identity, not an environmental label.
Novelty is distance to the nearest admitted pattern; salience is excess over
estimated ambience. A candidate can have high values without becoming a
lasting memory. This is a perception/admission prototype, not an automatically
trained voice or a change to the existing ChuffAgent's MFCC contract. Its
96-dimensional context features should not be mixed with the legacy 13-D
MFCC repertoire exemplars.

## Comparators and metrics

- **foreground:** background-aware features with recurrence confirmation.
- **foreground_single_sighting:** the same frontend and memory budget, but
  one sighting admits a pattern. This isolates the confirmation trade-off.
- **whole_clip_mfcc:** the original mean-MFCC cosine rule, novelty admission
  threshold 0.7 and recognition threshold 0.2, at the same 32-entry committed
  memory budget. This is an intentionally unchanged historical baseline,
  not a tuned state-of-the-art audio detector. It has no pending-memory stage.

The scorer associates a prototype with the label present **when it was
admitted**. That label never updates its features or influences the learner.
Correct repeat recognition requires an ID admitted on that same event class;
calling everything familiar is not success. The denominator includes events
after the first two ground-truth occurrences of their class, including events
the learner missed. This does not excuse missed detections. Learned-category
counts alone are weaker: admitting a contaminated prototype on a labelled
window can count there but fail subsequent recognition.

Event detection means candidate/learned/recognised status on a true event;
it must be read alongside background-attention rate. The historical baseline
can obtain 100% detection by attending to everything. False admissions count
commits on background-only or nuisance windows, including calibration windows.
`wrong_recognition` means not a correct foreground match; a baseline correctly
recognising its stored room background still fails this foreground task.

First/early novelty averages the first two occurrences of a class; repeat
novelty averages later occurrences. Missing scores are omitted and recorded
as null if there are none. A novelty decline is therefore not a substitute
for detection recall. Metrics are reported per seed; different SNR runs with
the same seed share a schedule and are not independent replicates.

The output includes manifest.json (configurations, dependency versions,
revision, dirty status, source and lock hashes), per-window JSONL,
summary.json, and optional mixture.wav. The WAV has headroom added for a
file preview; the scoring used the original unscaled floating-point mixture.
The CLI does not play audio on hardware.

## Limits and next experiments

The first sweep succeeds for these three short tonal families at 0/+6/+12 dB.
At -6 dB, detection and recognition fail badly. Silence is not a solution to
the signal-from-noise problem: both nuisance rejection and true-event recall
must improve together.

The duration/flatness biases may also reject meaningful broadband sounds,
speech, long alarms and a novel sustained tone. Multiple events within one
window can be merged. A repeating nuisance may still become a memory, and
background changes are deliberately treated conservatively. There is no
semantic importance model or claim of general source separation.

The next evaluation needs recorded backgrounds and foregrounds, overlapping
sources, independently varied rooms/microphones, and a check that novelty
predicts learnable improvement rather than just acoustic unpredictability.
That last distinction follows the motivation of learning-progress approaches
such as [Oudeyer, Kaplan & Hafner (2007)](https://www.pyoudeyer.com/ims.pdf);
this implementation does not implement their intrinsic-motivation algorithm.
Spectral subtraction is a classical noise-estimation approach
([Boll, 1979](https://course.ece.cmu.edu/~ece491/homework/Boll79.pdf)); the
particular detection/admission heuristics here require task-specific validation.
