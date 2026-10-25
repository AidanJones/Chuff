"""Versioned, unpadded continuous framing; independent of clip-based learners."""

from dataclasses import dataclass

import numpy as np
from scipy.signal import get_window


@dataclass(frozen=True)
class SpectralFrames:
    """Complex frequency-by-frame spectra and absolute half-open sample spans."""

    spectrum: np.ndarray
    start_samples: np.ndarray
    stop_samples: np.ndarray


class ContinuousSTFT:
    """Hann/spectrum-scaled real STFT, emitting a frame when its last sample arrives.

    Frame starts are 0, hop, ... regardless of push boundaries. A push owns
    frames whose exclusive stop is after its previous input length and at
    most its new input length. No centering or padding is performed. Inputs
    are mono samples; callers own sample-rate validation and event timing.
    """

    SCHEMA = "chuff.continuous-stft.v1"

    def __init__(self, n_fft=512, hop=160):
        if (isinstance(n_fft, bool) or not isinstance(n_fft, int) or n_fft < 2
                or isinstance(hop, bool) or not isinstance(hop, int) or not 0 < hop <= n_fft):
            raise ValueError("expected integer n_fft >= 2 and 0 < hop <= n_fft")
        self.n_fft, self.hop = n_fft, hop
        self._window = get_window("hann", n_fft, fftbins=True).astype(np.float64)
        self.reset()

    def reset(self):
        self._pending = np.empty(0, dtype=np.float64)
        self._next_start = 0
        self.input_samples = 0
        self.frame_count = 0
        self._last_stop = 0
        self._finished = False

    def push(self, samples):
        if self._finished:
            raise ValueError("stream is finished; reset before pushing again")
        values = np.asarray(samples)
        if (values.ndim != 1 or values.dtype.kind not in "iuf"
                or not np.isfinite(values).all()):
            raise ValueError("expected finite real mono samples")
        values = values.astype(np.float64)
        if not np.isfinite(values).all():
            raise ValueError("sample conversion overflow")
        combined = np.concatenate((self._pending, values))
        count = max(0, 1+(len(combined)-self.n_fft)//self.hop)
        starts = self._next_start+np.arange(count, dtype=np.int64)*self.hop
        stops = starts+self.n_fft
        if count:
            frames = np.lib.stride_tricks.sliding_window_view(combined, self.n_fft)[::self.hop]
            spectrum = np.fft.rfft(frames*self._window, axis=1).T/self._window.sum()
        else:
            spectrum = np.empty((self.n_fft//2+1, 0), dtype=np.complex128)
        # Validate before advancing state, including overflow in the transform.
        if not np.isfinite(spectrum).all():
            raise ValueError("nonfinite spectrum")
        consumed = count*self.hop
        self._pending = combined[consumed:].copy()
        self._next_start += consumed
        self.input_samples += len(values)
        self.frame_count += count
        if count:
            self._last_stop = int(stops[-1])
        return SpectralFrames(spectrum, starts, stops)

    def finish(self):
        """Close without zero padding; return physical tail accounting.

        Buffered overlap may already belong to a completed frame. Only samples
        after the last complete frame's stop count as an uncovered tail.
        """
        result = dict(schema=self.SCHEMA, input_samples=self.input_samples, frames=self.frame_count,
                      uncovered_tail_samples=self.input_samples-self._last_stop)
        self._pending = np.empty(0, dtype=np.float64)
        self._finished = True
        return result
