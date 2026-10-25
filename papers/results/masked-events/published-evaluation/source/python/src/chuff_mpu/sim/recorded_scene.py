"""Full recorded episodes and overlap-added recorded backgrounds, actor-independent."""

from dataclasses import replace
import hashlib
import math
from pathlib import Path
import wave

import numpy as np
from scipy.signal import resample_poly

from chuff_mpu.sim.recorded_catalog import REVISION
from chuff_mpu.sim.varied_scene import VariedScene


FOREGROUNDS = ("dog", "rooster", "crying_baby", "sneezing", "clock_tick")
BACKGROUNDS = ("rain", "sea_waves", "crackling_fire", "helicopter", "chainsaw")


def load_audio(clips, cache, sample_rate=16000):
    """Verify immutable bytes, resample full clips; no event selection by energy."""
    result = {}
    for clip in clips:
        path = Path(cache)/REVISION/clip["filename"]
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != clip["sha256"]:
            raise ValueError(f"recording checksum mismatch: {path}")
        with wave.open(str(path), "rb") as handle:
            if (handle.getnchannels(), handle.getsampwidth(), handle.getframerate(), handle.getnframes()) != (1, 2, 44100, 220500):
                raise ValueError("expected complete five-second mono PCM16 source")
            signal = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2").astype(np.float64)/32768
        divisor = math.gcd(sample_rate, 44100)
        signal = resample_poly(signal, sample_rate//divisor, 44100//divisor)
        if len(signal) != 5*sample_rate or not np.isfinite(signal).all():
            raise ValueError("invalid full-duration resampling")
        result[clip["filename"]] = signal
    return result


class RecordedScene(VariedScene):
    def __init__(self, seed, settings, *, environment_seed, audio, foregrounds, backgrounds,
                 modulation=False, control=False, repetitions=4):
        if (settings.duration_s != 5 or settings.warmup != 2 or settings.rounds != 2+2*len(foregrounds)*repetitions
                or not settings.collect_stems or settings.snr_reference != "post_residual"
                or settings.background_change or settings.nuisance_probability):
            raise ValueError("recorded scene requires complete five-second episodes, two warmup chunks and post-channel components")
        if not foregrounds or not backgrounds or set(foregrounds)&set(backgrounds):
            raise ValueError("separate nonempty foreground/background recording identities required")
        self.audio_bank = audio
        for name in (*foregrounds, *backgrounds):
            signal = np.asarray(audio[name])
            if signal.shape != (5*settings.sample_rate,) or not np.isfinite(signal).all() or np.linalg.norm(signal) == 0:
                raise ValueError("invalid complete recording")
        super().__init__(seed, settings, environment_seed=environment_seed, condition="joint")
        self.control, self.modulation = control, modulation
        self.event_labels = tuple(foregrounds)
        order_rng = np.random.default_rng(np.random.SeedSequence([seed, 83]))
        ordered = [name for _ in range(repetitions) for name in order_rng.permutation(foregrounds)]
        self.program = {2+2*index: str(name) for index, name in enumerate(ordered)}
        n = 5*settings.sample_rate
        total = settings.rounds*n
        self._bed = np.zeros(total)
        rng = np.random.default_rng(np.random.SeedSequence([seed, environment_seed, 140]))
        weights = np.hanning(n+1)[:-1]
        grains = []
        for start in range(-n//2, total, n//2):
            name = str(rng.choice(backgrounds))
            signal = audio[name]
            scale = settings.background_rms/max(float(np.sqrt(np.mean(signal**2))), 1e-12)
            scale *= 10**(float(rng.uniform(-3, 3))/20)
            lo, hi = max(start, 0), min(start+n, total)
            self._bed[lo:hi] += signal[lo-start:hi-start]*weights[lo-start:hi-start]*scale
            grains.append(dict(start_sample=start, recording=name, scale=scale))
        for unused in ("background_pole", "hum_hz", "fan_hz", "halfway_level_multiplier"):
            self.parameters.pop(unused, None)
        self.parameters.update(condition="recorded_room_mic", source_resampling_range=None,
            foreground_recordings=list(foregrounds), background_recordings=list(backgrounds),
            background_grains=grains, modulation=modulation, source_program=self.program,
            background_input_sha256=hashlib.sha256(self._bed.astype("<f8").tobytes()).hexdigest(),
            source_processing="Full 5s resample_poly; 10ms injection-edge fades; one post-channel scale per episode",
            background_processing="Full 5s grains, periodic Hann weights, 50% overlap; independent +/-3dB grain levels")

    def _background(self, n):
        start = self._step*n
        signal = self._bed[start:start+n].copy()
        if self.modulation:
            time = (start+np.arange(n))/self.settings.sample_rate
            signal *= 1+.45*np.sin(2*np.pi*1.3*time+self._hum_phase)
        return signal

    def _event(self, label):
        signal = self.audio_bank[label].copy()
        ramp = int(.01*self.settings.sample_rate)
        signal[:ramp] *= np.linspace(0, 1, ramp)
        signal[-ramp:] *= np.linspace(1, 0, ramp)
        return signal

    def next_frame(self):
        settings, labels = self.settings, self.event_labels
        target = self.program.get(self._step) if not self.control else None
        self.settings = replace(settings, event_probability=float(target is not None))
        if target is not None:
            self.event_labels = (target,)
        try:
            return super().next_frame()
        finally:
            self.settings, self.event_labels = settings, labels
