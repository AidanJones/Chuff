# Selective capture and mood control

`--frontend spectral-gain` is an additional experimental option for modulated
backgrounds. It has its own saved schema, needs three initial background-only
calibration windows, and recalibrates its gain profile on restart. It improves
the procedural modulation cases but loses stationary-background recall; see
[paper 04](../../../papers/04-shared-spectral-gain.md). Use a separate state file.

`--frontend calibrated-gain` makes a fixed raw/compensated choice from the
same initial background calibration. It preserves stationary recall on the
controlled evaluation while improving modulated cases. Its own schema and
selection reset on restart; observation JSON reports the selected mode. It
does not adapt that choice to later environment changes. The original frontend
remains the default.

Exact silence no longer freezes the spectral template at its numerical floor.
Until usable band energy exists, the fixed spectral frontend falls back to
raw power. It fits its template from the first three usable windows; late
calibration resets background estimates/pending confirmations and suppresses
admission on those fitting windows, while retaining admitted prototypes.
Ordinary noisy startup is unchanged. For `calibrated-gain`, an initial raw
choice still stays raw; fitting a later shadow template does not enable gain
or alter that baseline path. This fixes silent-profile recovery, not the
broader problem of adapting a frozen mode to a changed environment.

For the experimental masked-event frontend, add `--frontend band` and use
a separate state file. See [MASKED.md](sim/MASKED.md) for results and limitations.
Original and band frontend states cannot be cross-loaded; the default is
`--frontend original`.

`chuff-selective` connects the foreground listener to one-second microphone
capture or silent WAV replay. It can send the existing mood/energy JSON
protocol to an MCU. This is opt-in: historical MFCC demos and their voice
and dream routines continue to use their original feature representation.

## Try it without hardware

From the python directory, use a mono/stereo 16 kHz WAV:

```powershell
uv run chuff-selective --input-wav path/to/recording.wav --state ../eval/results/selective-state.npz
```

The command prints session metadata, one JSON object per processed window,
and a summary. It does not open a microphone, speaker or serial port in this
mode unless a serial --port is explicitly provided. Replay cannot enable
the learning cue. To generate a new procedural input first:

```powershell
uv run chuff-sim-attention --seeds 1 --seed-start 100 --rounds 60 --snr-db 6 --structured-distractors --output ../eval/results/my-selective-scene --save-audio
uv run chuff-selective --input-wav ../eval/results/my-selective-scene/mixture.wav --state ../eval/results/selective-state.npz
```

Run the second command again to check recognition after restart. Existing
state is validated and loaded before processing. A file from the legacy
EmbeddingStore, or an incompatible/corrupt foreground file, causes an error;
it is not silently discarded or overwritten.

WAV replay accepts signed PCM, unsigned 8-bit PCM, and finite floating-point
audio. Stereo is downmixed. The sample rate must match the loaded model
(16 kHz by default); there is no hidden resampling. An incomplete final
one-second window is discarded, with the sample count reported in the
session header. The current reader loads a WAV into memory, so use bounded
recordings rather than multi-hour files.

## Microphone and optional MCU

```powershell
uv run chuff-selective --mic --state ../eval/results/selective-state.npz
uv run chuff-selective --mic --port COM5 --state ../eval/results/selective-state.npz
```

Use --max-windows 30 for a bounded capture. Ctrl+C stops capture and saves
the admitted memories. The serial connection closes on exit. Omitting
--state keeps the session in memory only; omitting --port keeps its display
messages in stdout only.

The default controller is silent. If desired, --play-learning-cue in mic
mode enables a short fixed, tapered 900 Hz notification only when a new
foreground prototype is admitted. It never plays the recorded input or
routes foreground features into the old call repertoire. All cue samples
pass through the existing numerical limiter and safe_play path. Candidate,
familiar, background, calibration and idle outcomes do not request a cue.
The cue_samples field reports a requested waveform, not proof of successful
hardware playback; the existing playback helper is best-effort.

Playback is blocking. The listener then discards a 250 ms guard capture
before starting the next analysis window. The summary reports discarded
guard time. This reduces immediate self-echo exposure but is not echo
cancellation. Capture is currently a sequence of blocking windows, so sounds
during processing, cue playback or the guard can be missed. Actual capture
latency, gaps and echo behaviour remain to be measured on hardware.

## Display policy

| Listener outcome | Display | Cue eligibility |
|---|---|---|
| Calibrating | contentment, low energy | never |
| Novel candidate | mild_novelty, curiosity | never |
| Newly admitted recurring pattern | new_sound_learned, high energy | optional |
| Familiar pattern | song_recognised, lower energy | never |
| Background or rejected unstructured input | contentment | never |
| Prolonged absence of accepted foreground | bored | never |

These are designed display mappings, not measures of felt emotion. A
background window returns the controller to calm rather than leaving the
last learning display/cue latched. Accepted foreground resets the idle
timer; the default boredom timeout is 30 minutes of analysed windows with
no accepted foreground. The controller does not emit a dream prediction,
because it has not connected a compatible predictive model.

The capture API is also usable directly:

```python
from chuff_mpu.selective import SelectiveController

controller = SelectiveController()  # no hardware side effects
update = controller.process(microphone_clip)  # one mono 16 kHz window
print(update.as_dict())
```

An optional sink implements send_mood(mood, energy), matching McuLink.
The controller returns a cue array only when explicitly enabled; playing
that returned array is the caller's responsibility. The controller itself
opens no devices and plays no samples.

## Saved memory and feature boundaries

ForegroundLearner.save()/load() use a versioned, non-pickle NPZ schema.
Snapshots contain the full frontend configuration, admitted feature vectors,
stable IDs, hit counts, next unused ID and eviction ordering. Writes use a
temporary sibling file and atomic replacement, so a failed write leaves the
previous snapshot intact.

The loader validates schema, configured dimensions, capacity, nonnegative
finite unit vectors, integer IDs and confirmation counts. Default features
are 96-D (three temporal summaries of 32 bands). Legacy voice/dream memories
use 13-D mean MFCCs. The selective controller accepts foreground memory only,
and never passes these vectors to those legacy components.

Pending candidates, background estimates, analysis counters and controller
idle/display state are not persisted. On restart the room estimator calibrates
for three windows again, candidate confirmation begins afresh, and boredom
starts from zero. Admitted patterns and their IDs survive, so recognising an
old pattern after calibration does not cause another learning cue. Patterns
are not time-decayed while the application is stopped.

## Validation and limitations

The test suite covers the full candidate -> admitted -> familiar transition,
reset to calm, changing backgrounds and silence, cue bounds, real serial
JSON formatting through a mocked port, microphone capture through a mocked
device, state round trips, incompatible files, atomic-write failure, WAV
formats and partial windows. No physical audio or MCU test is claimed.

A 120-window replay of the previously saved 0 dB synthetic mixture admitted
four prototypes and recognised 32 windows. Replaying it with restored memory
admitted no additional prototypes and recognised 40 windows. Both runs were
silent and returned to three-window calibration after restart. Four prototypes
for three source classes illustrates fragmentation, not new semantic classes.

This integration leaves the detector's measured -6 dB failure and its
broadband/long-duration biases unchanged. See the
[foreground experiment guide](sim/ATTENTION.md) and
[research results](../../../papers/01-foreground-novelty.md).
