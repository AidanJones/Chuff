"""Opt-in selective listening: replay WAV or capture mic, optionally update an MCU.

Replay never plays audio. Microphone learning cues require an explicit flag;
they are fixed notifications, not reproduction of the observed waveform.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
import json
from pathlib import Path
import sys

import numpy as np
from scipy.io import wavfile

from chuff_mpu import audio_safety, config
from chuff_mpu.attention import ForegroundLearner
from chuff_mpu.band_attention import BandForegroundLearner
from chuff_mpu.gain_attention import SpectralGainForegroundLearner, CalibratedGainForegroundLearner
from chuff_mpu.selective import SelectiveController


def read_wav(path: str | Path, expected_rate: int) -> np.ndarray:
    """Read PCM/float mono or stereo WAV; reject rate mismatches rather than resample."""
    rate, samples = wavfile.read(path)
    if rate != expected_rate:
        raise ValueError(f"WAV sample rate {rate} does not match foreground memory's {expected_rate} Hz")
    if samples.dtype == np.uint8:
        clip = (samples.astype(np.float64) - 128) / 128
    elif samples.dtype.kind == "i":
        clip = samples.astype(np.float64) / (2 ** (8 * samples.dtype.itemsize - 1))
    elif samples.dtype.kind == "f":
        clip = samples.astype(np.float64)
    else:
        raise ValueError("unsupported WAV sample type")
    if clip.ndim == 2 and clip.shape[1] in (1, 2):
        clip = clip.mean(axis=1)
    if clip.ndim != 1 or not np.isfinite(clip).all():
        raise ValueError("WAV must contain finite mono or stereo samples")
    return clip.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input-wav", type=Path, help="Headless replay; no device or serial port required")
    source.add_argument("--mic", action="store_true", help="Capture one-second windows from the default microphone")
    parser.add_argument("--state", type=Path, help="Load/save versioned admitted foreground memories (.npz)")
    parser.add_argument("--frontend", choices=["original", "band", "spectral-gain", "calibrated-gain"], default="original",
                        help="Experimental band/gain frontends require separate state files and background calibration")
    parser.add_argument("--port", default=None, help="Optional MCU serial port; omitted means JSON-only dry run")
    parser.add_argument("--baud", type=int, default=config.SERIAL_BAUD)
    parser.add_argument("--max-windows", type=int, default=None, help="Stop after this many complete windows")
    parser.add_argument("--play-learning-cue", action="store_true", help="Mic mode only: play a fixed cue on new admission")
    args = parser.parse_args()
    if args.max_windows is not None and args.max_windows < 1:
        parser.error("--max-windows must be positive")
    if args.play_learning_cue and not args.mic:
        parser.error("--play-learning-cue is available only with --mic; WAV replay is silent")
    try:
        listener_class = {"original": ForegroundLearner, "band": BandForegroundLearner,
                          "spectral-gain": SpectralGainForegroundLearner,
                          "calibrated-gain": CalibratedGainForegroundLearner}[args.frontend]
        listener = listener_class.load(args.state) if args.state and args.state.exists() else listener_class()
        rate = listener.settings.sample_rate
        clip = read_wav(args.input_wav, rate) if args.input_wav else None
        if clip is not None and len(clip) < rate:
            parser.error("WAV must contain at least one complete one-second window")
        skipped_samples = 0 if clip is None else len(clip) % rate
        count = 0
        discarded_cue_guard_s = 0.0
        statuses: Counter = Counter()
        interrupted = False
        with ExitStack() as stack:
            sink = None
            if args.port:
                from chuff_mpu.ipc import McuLink

                sink = stack.enter_context(McuLink(args.port, args.baud))
            controller = SelectiveController(listener, sink=sink, learning_cue=args.play_learning_cue)
            print(json.dumps({"type": "session", "mode": "mic" if args.mic else "wav",
                              "frontend": args.frontend, "feature_schema": listener.FEATURE_SCHEMA,
                              "sample_rate": rate, "window_s": 1, "restored_memories": listener.memory_size,
                              "restart_policy": "admitted prototypes retained; ambience/candidates/timers reset",
                              "discarded_tail_samples": skipped_samples}), flush=True)
            try:
                while args.max_windows is None or count < args.max_windows:
                    if clip is not None:
                        start = count * rate
                        if start + rate > len(clip):
                            break
                        window = clip[start:start + rate]
                    else:
                        from chuff_mpu.audio import capture_clip

                        window = capture_clip(1.0, rate)
                    update = controller.process(window)
                    if update.cue is not None:
                        # Blocking playback completes before the next mic capture;
                        # no part of the input window is sent to the speaker.
                        audio_safety.safe_play(update.cue, rate)
                        # Discard a short post-cue capture so a nearby echo is
                        # not immediately admitted as a new external event.
                        # This is a guard interval, not acoustic echo cancellation.
                        capture_clip(controller.settings.cue_guard_s, rate)
                        discarded_cue_guard_s += controller.settings.cue_guard_s
                    diagnostics = ({"gain_compensation_enabled": listener.gain_compensation_enabled}
                                   if isinstance(listener, CalibratedGainForegroundLearner) else {})
                    print(json.dumps({"type": "observation", "window": count,
                                      "memory_size": listener.memory_size, **diagnostics, **update.as_dict()},
                                     allow_nan=False), flush=True)
                    statuses[update.observation.status] += 1
                    count += 1
                    if args.state and update.observation.status == "learned":
                        listener.save(args.state)
            except KeyboardInterrupt:
                interrupted = True
            finally:
                if args.state:
                    listener.save(args.state)
        print(json.dumps({"type": "summary", "windows": count, "statuses": dict(statuses),
                          "memory_size": listener.memory_size, "interrupted": interrupted,
                          "discarded_cue_guard_s": discarded_cue_guard_s}), flush=True)
    except (OSError, ValueError, EOFError) as error:
        parser.exit(1, f"selective listening: {error}\n")


if __name__ == "__main__":
    main()
