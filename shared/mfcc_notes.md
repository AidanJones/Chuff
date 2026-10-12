# MFCC Pipeline Notes

Both tracks reduce raw audio to a small per-clip feature vector (MFCCs)
before anything else — novelty scoring, the dream loop, and comms all
operate on these vectors, never on raw PCM.

## Pipeline (conceptual, both tracks)

```
raw PCM (mono, fixed sample rate)
  -> windowed FFT
  -> mel filterbank
  -> log + DCT -> N_MFCC coefficients
  -> (optionally) time-average across the clip -> one fixed-length vector
```

## Track A — ESP32-S3 (MicroPython + emlearn-micropython)

- No numpy/scipy/librosa available on-device — all DSP is hand-rolled or
  via **emlearn-micropython**, a C99 ML library with MicroPython bindings
  providing FFT, IIR filters, Random Forest, and KNN with on-device
  runtime learning.
- emlearn-micropython must be cross-compiled for ESP32-S3 and included as
  a `.mpy` file in the repo (not yet vendored — tracked in `TASKS.md`).
- KNN is the core memory structure: new embeddings are added to PSRAM at
  runtime without reflashing.
- Constants: `SAMPLE_RATE = 8000`, `FFT_SIZE = 256`, `N_MFCC = 13`,
  `CLIP_DURATION_S = 2` (see the esp32 config header once the MicroPython
  audio pipeline lands — the *current* esp32/ code is ESP-IDF C/LVGL, not
  MicroPython; see the top-level README for why, and the note below).

## Track B — Arduino UNO Q (Python 3 on Debian)

- Full `librosa` + `numpy` + `scikit-learn` available (already added to
  `python/pyproject.toml` via `uv add`). No on-device DSP constraints.
- `scikit-learn.neighbors.NearestNeighbors` stands in for emlearn's KNN.
- Constants: `SAMPLE_RATE = 16000`, `N_MFCC = 13`,
  `CLIP_DURATION_S = 1.0` (see `python/src/chuff_mpu/config.py`).

## Current status

Neither track's real MFCC extraction is implemented yet
(`audio.extract_mfcc()` raises `NotImplementedError` on the Python side;
there is no MicroPython audio pipeline on the esp32 track at all right
now — see the note on hardware/software track divergence in the
top-level README). The running MVP stands in for MFCC-based novelty with
plain RMS loudness banding (`python/src/chuff_mpu/mvp_demo.py`) so the
mood-over-serial plumbing could be proven before the ML pipeline exists.
