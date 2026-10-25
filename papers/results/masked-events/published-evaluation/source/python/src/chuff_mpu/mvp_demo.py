"""Chuff MVP demo: mic loudness -> mood -> serial line to the MCU.

This is the "really simple MVP" for the UNO Q / Arduino track: it doesn't
need the real UNO Q board, Modulino kit, or the full MFCC+KNN pipeline —
just this laptop's mic and *any* Arduino running
arduino/chuff_mcu/chuff_mcu.ino, wired over USB serial.

It stands in for novelty scoring (memory.py / mood.py, not yet
implemented) with plain RMS loudness banding, just so the two tracks
(Python mood source, Arduino mood display) can be proven end-to-end
before the real audio and ML pipeline lands.

Run with:  uv run chuff-mvp-demo --port COM3   (or /dev/ttyACM0, etc.)
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import sounddevice as sd

from chuff_mpu import config
from chuff_mpu.ipc import McuLink


def _rms_energy(block: np.ndarray) -> float:
    """Root-mean-square loudness of one audio block, in [0, ~1]."""
    return float(np.sqrt(np.mean(np.square(block))))


def _energy_to_mood(energy: float) -> tuple[str, float]:
    """Map a raw RMS energy reading to a (mood, normalised energy) pair."""
    if energy < config.MVP_CALM_ENERGY_MAX:
        return "contentment", energy / config.MVP_CALM_ENERGY_MAX * 0.3
    if energy < config.MVP_CURIOUS_ENERGY_MAX:
        return "mild_novelty", 0.3 + 0.4 * (
            (energy - config.MVP_CALM_ENERGY_MAX)
            / (config.MVP_CURIOUS_ENERGY_MAX - config.MVP_CALM_ENERGY_MAX)
        )
    return "strong_novelty", min(1.0, 0.7 + 0.3 * (energy / config.MVP_CURIOUS_ENERGY_MAX))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default=config.SERIAL_PORT, help="Serial port to the MCU")
    parser.add_argument("--baud", type=int, default=config.SERIAL_BAUD)
    parser.add_argument(
        "--sample-rate", type=int, default=config.SAMPLE_RATE, help="Mic sample rate (Hz)"
    )
    parser.add_argument(
        "--interval", type=float, default=config.MVP_POLL_INTERVAL_S, help="Seconds between mood updates"
    )
    args = parser.parse_args()

    block_frames = int(args.sample_rate * args.interval)

    with McuLink(args.port, args.baud) as link:
        print(f"Listening on default mic @ {args.sample_rate} Hz, "
              f"sending moods to {args.port} every {args.interval}s. Ctrl+C to stop.")
        while True:
            block = sd.rec(block_frames, samplerate=args.sample_rate, channels=1, dtype="float32")
            sd.wait()
            energy = _rms_energy(block)
            mood, norm_energy = _energy_to_mood(energy)
            link.send_mood(mood, norm_energy)
            print(f"energy={energy:.4f} -> mood={mood} energy={norm_energy:.2f}")

            event = link.poll_event()
            if event is not None:
                print(f"  <- MCU: {event}")

            time.sleep(0)  # sd.wait() already paced us to ~--interval


if __name__ == "__main__":
    main()
