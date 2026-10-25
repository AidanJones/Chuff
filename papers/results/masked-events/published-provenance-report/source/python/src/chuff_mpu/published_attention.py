"""Published normalization/noise primitives with the project's discovery backend.

These are adapters, not reproductions of keyword spotting or OM-LSA. PCEN
uses librosa0.11 on deliberately scaled magnitudes. MCRA implements the
Cohen/Berdugo2002 recursions, with explicitly chosen three-bin weights.
"""

from dataclasses import dataclass

import librosa
import numpy as np
from scipy.ndimage import convolve1d, uniform_filter1d
from scipy.signal import get_window

from chuff_mpu.band_attention import BandAttentionConfig, BandForegroundLearner


def _matrix(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or 0 in values.shape or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("expected a finite nonnegative frequency-by-time matrix")
    return values


@dataclass(frozen=True)
class PCENConfig(BandAttentionConfig):
    pcen_gain: float = .98
    pcen_bias: float = 2.
    pcen_power: float = .5
    pcen_time_s: float = .4
    pcen_eps: float = 1e-6
    pcen_amplitude_scale: float = float(2**31)

    def __post_init__(self):
        super().__post_init__()
        if not np.isfinite(self.pcen_gain) or not 0 < self.pcen_gain <= 1:
            raise ValueError("pcen_gain must be in (0, 1]")
        for name in ("pcen_bias", "pcen_power", "pcen_time_s", "pcen_eps", "pcen_amplitude_scale"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")


class PCENState:
    def __init__(self, settings):
        self.settings = settings
        self._zi = None
        self._bins = None

    def process(self, magnitude):
        magnitude = _matrix(magnitude)
        if self._bins is not None and len(magnitude) != self._bins:
            raise ValueError("frequency dimension cannot change during a PCEN stream")
        c = self.settings
        result, final = librosa.pcen(magnitude, sr=c.sample_rate, hop_length=c.hop,
            gain=c.pcen_gain, bias=c.pcen_bias, power=c.pcen_power, time_constant=c.pcen_time_s,
            eps=c.pcen_eps, max_size=1, zi=self._zi, return_zf=True)
        if not np.isfinite(result).all():
            raise ValueError("nonfinite PCEN output")
        self._zi, self._bins = final, len(magnitude)
        return result


class PCENForegroundLearner(BandForegroundLearner):
    FEATURE_SCHEMA = "chuff.foreground.pcen-band.v1"
    SETTINGS_CLASS = PCENConfig

    def __init__(self, settings=None):
        super().__init__(settings)
        self._pcen = PCENState(self.settings)
        self._pcen_background = None
        self._pcen_relative = None
        c = self.settings
        mel = lambda f: 2595*np.log10(1+f/700)
        edges = 700*(10**(np.linspace(mel(100), mel(c.sample_rate/2), c.bands+2)/2595)-1)
        self._area = 2/(edges[2:]-edges[:-2])
        self._window_sum = float(get_window("hann", c.n_fft).sum())

    def _pcen_magnitude(self, power):
        # scipy STFT divides by the Hann sum; undo that before the documented
        # amplitude scale and HTK-triangle area normalization.
        return ((self._filters @ np.sqrt(power))*self._area[:, None]
                *self._window_sum*self.settings.pcen_amplitude_scale)

    def _prepare_power(self, power):
        # Streaming normalization also processes the unlabelled warmup.
        pcen = self._pcen.process(self._pcen_magnitude(power))
        c = self.settings
        energy = uniform_filter1d(power.sum(axis=0), c.smoothing_frames, mode="nearest")
        quiet = energy <= np.quantile(energy, .2)
        floor = np.maximum(pcen[:, quiet].mean(axis=1), 1e-12)
        if self._pcen_background is None:
            self._pcen_background = floor.copy()
        background = np.maximum(self._pcen_background, floor)
        self._pcen_background += c.background_rate*(floor-self._pcen_background)
        self._pcen_relative = pcen/background[:, None]
        return power

    def _detection_relative(self, power, bands, noise):
        return self._pcen_relative


@dataclass(frozen=True)
class MCRAConfig(BandAttentionConfig):
    hop: int = 128  # 8ms at16kHz, as in the original MCRA experiment
    mcra_alpha_s: float = .8
    mcra_alpha_d: float = .95
    mcra_alpha_p: float = .2
    mcra_delta: float = 5.
    mcra_min_frames: int = 125
    mcra_eps: float = 1e-12

    def __post_init__(self):
        super().__post_init__()
        for name in ("mcra_alpha_s", "mcra_alpha_d", "mcra_alpha_p"):
            value = getattr(self, name)
            if not np.isfinite(value) or not 0 < value < 1:
                raise ValueError(f"{name} must be in (0, 1)")
        if not np.isfinite(self.mcra_delta) or self.mcra_delta <= 1:
            raise ValueError("mcra_delta must exceed one")
        if type(self.mcra_min_frames) is not int or self.mcra_min_frames < 2:
            raise ValueError("mcra_min_frames must be an integer >=2")
        if not np.isfinite(self.mcra_eps) or self.mcra_eps <= 0:
            raise ValueError("mcra_eps must be finite and positive")


class MCRAState:
    def __init__(self, settings):
        self.settings = settings
        self._noise = self._smooth = self._minimum = self._temporary = self._presence = None
        self.frames = 0

    def process(self, power):
        power = _matrix(power)
        if self._noise is not None and len(power) != len(self._noise):
            raise ValueError("frequency dimension cannot change during an MCRA stream")
        c = self.settings
        smoothed_frequency = convolve1d(power, np.array([.25, .5, .25]), axis=0, mode="nearest")
        output = np.empty_like(power)
        for index in range(power.shape[1]):
            current = power[:, index]
            if self._noise is None:
                self._noise = current.copy()
                self._smooth = smoothed_frequency[:, index].copy()
                self._minimum = self._smooth.copy()
                self._temporary = self._smooth.copy()
                self._presence = np.zeros(len(power))
            else:
                self._smooth = c.mcra_alpha_s*self._smooth+(1-c.mcra_alpha_s)*smoothed_frequency[:, index]
                if self.frames % c.mcra_min_frames == 0:
                    self._minimum = np.minimum(self._temporary, self._smooth)
                    self._temporary = self._smooth.copy()
                else:
                    self._minimum = np.minimum(self._minimum, self._smooth)
                    self._temporary = np.minimum(self._temporary, self._smooth)
                indicator = self._smooth/np.maximum(self._minimum, c.mcra_eps) > c.mcra_delta
                self._presence = c.mcra_alpha_p*self._presence+(1-c.mcra_alpha_p)*indicator
            # Equation4 predicts lambda(l+1) from lambda(l). The current
            # detector receives lambda(l), then this frame updates the state.
            output[:, index] = np.maximum(self._noise, c.mcra_eps)
            alpha = c.mcra_alpha_d+(1-c.mcra_alpha_d)*self._presence
            self._noise = alpha*self._noise+(1-alpha)*current
            self.frames += 1
        return output


class MCRAForegroundLearner(BandForegroundLearner):
    FEATURE_SCHEMA = "chuff.foreground.mcra-band.v1"
    SETTINGS_CLASS = MCRAConfig

    def __init__(self, settings=None):
        super().__init__(settings)
        self._mcra = MCRAState(self.settings)

    def _noise_background(self, power):
        return self._mcra.process(power)
