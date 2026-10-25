# Continuous STFT v1

`streaming_stft.ContinuousSTFT` is an independent framing primitive. Existing
clip learners and frozen experiment methods do not use it. It introduces no
new memory, admission rule or learned feature representation.

```python
framer = ContinuousSTFT(n_fft=512, hop=160)
frames = framer.push(mono_samples)
# frames.spectrum: complex128, frequency bins by frames
# frames.start_samples / stop_samples: absolute int64 half-open spans
tail = framer.finish()
```

Frame starts are `0, hop, 2*hop, ...`. The periodic Hann window and division
by its sum match SciPy `stft(..., boundary=None, padded=False,
scaling="spectrum", detrend=False)`. Real mono inputs convert to float64;
NaNs, infinities, complex/object/bool arrays and invalid geometry are rejected
before stream state changes. Spectral overflow is also rejected before state
advances. No centering, start padding or final padding occurs.

A frame becomes available when its last required sample has arrived. For a
push spanning absolute input lengths `old` to `new`, it owns frames satisfying
`old < stop_sample <= new`; the frame may start in an earlier chunk. Its
center is `start_sample + n_fft/2`, but it cannot be emitted until
`stop_sample`. At 16 kHz the default 512-sample frame spans 32 ms. The caller
owns sample-rate validation and conversion of absolute sample spans to time.
Empty chunks emit no frames. Short chunks accumulate until a full frame is
available. Persistent pending state contains fewer than `n_fft` samples.

`finish()` is idempotent, reports input/frame counts and samples **after the
last complete frame**, then clears pending samples and closes the stream.
Buffered overlap already covered by a frame is not counted as uncovered
tail. A stream shorter than one frame has its entire input uncovered. Pushes
after finish fail; `reset()` starts a fresh stream at sample zero. There is
no implicit padding/flush that could invent a tail observation.

Tests compare complex spectra and centers to one-shot SciPy for 512/160,
512/128, odd 511/127 and nonoverlapping 32/32 geometry. They exercise
single-sample chunks, empty/irregular/whole-stream chunks, frame ownership,
invalid-input state preservation, input aliasing, reset and tail accounting.
A unit impulse at sample 15,999 is omitted by separate unpadded one-second
clips, but appears in the continuous frame emitted after the next push.
This establishes framing behavior, not improved event identity or relevance.

Beads ch-6w4.7.3.1 tracks this primitive. Learner integration must explicitly
define decision ownership, floor estimation, smoothing, latency and schema
before using these frames with PCEN/MCRA. Repartitioning frames alone does
not guarantee chunk-invariant learner decisions: the current detector's
within-window statistics and candidate tick timing depend on clip boundaries.
That integration and a separately frozen behavior comparison remain under
ch-6w4.7.3.2 and its parent.
