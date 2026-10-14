#!/usr/bin/env python3
"""Chuff, entirely in software: listen, remember, repeat — and prove the
output guardrails actually hold, before running any real rounds.

No microphone or speaker required (sim/world.py synthesises everything);
pass --play to also attempt real playback of what Chuff "says" through
the default output device (best-effort — silently skipped if there's no
device). Every waveform is safety-limited either way; see
audio_safety.py.

Usage:
    uv run chuff-sim
    uv run chuff-sim --rounds 20 --seed 7 --play
    uv run chuff-sim --log run1.jsonl
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from chuff_mpu import audio_safety, comms, config
from chuff_mpu.sim.agent import ChuffAgent
from chuff_mpu.sim.world import Environment


def _demonstrate_guardrail() -> None:
    """Prove the limiter actually clamps a deliberately pathological signal
    before trusting it with anything a real agent will say."""
    print("=== Guardrail check ===")
    n = int(config.SAMPLE_RATE * 1.0)
    pathological = np.full(n, 50.0, dtype=np.float32)  # 50x full-scale: a bug, not a real tone
    pathological[10] = float("nan")
    pathological[20] = float("inf")
    safe = audio_safety.limit_audio(pathological, config.SAMPLE_RATE)
    peak_in = float(np.max(np.abs(pathological[np.isfinite(pathological)])))
    peak_out = float(np.max(np.abs(safe)))
    nan_in, nan_out = int(np.sum(~np.isfinite(pathological))), int(np.sum(~np.isfinite(safe)))
    print(f"  input:  peak={peak_in:.1f}  non-finite samples={nan_in}")
    print(f"  output: peak={peak_out:.3f}  non-finite samples={nan_out}  "
          f"(ceiling = config.SAFE_PEAK_AMPLITUDE = {config.SAFE_PEAK_AMPLITUDE})")
    assert peak_out <= config.SAFE_PEAK_AMPLITUDE + 1e-6, "guardrail failed to clamp peak amplitude"
    assert nan_out == 0, "guardrail failed to sanitise non-finite samples"
    print("  OK: clamped and sanitised.\n")


def run(rounds: int, seed: int, play: bool, log_path: str | None, noise: float) -> None:
    _demonstrate_guardrail()

    env = Environment(config.SAMPLE_RATE, config.CLIP_DURATION_S, seed=seed, background_noise_amp=noise)
    agent = ChuffAgent(seed=seed + 1)

    print(f"=== {rounds} rounds ===")
    for i in range(1, rounds + 1):
        label, target = env.present()
        mic_in = env.hear(target)
        embedding, novelty, learned = agent.hear(mic_in)

        call_index, freq, call_wave = agent.respond(embedding)
        agent.speak(call_wave, play=play)

        mic_echo = env.hear(call_wave)
        recognised = agent.recognise_own_echo(call_index, mic_echo)
        agent.reinforce_or_decay(call_index, embedding, recognised)

        peak = float(np.max(np.abs(call_wave)))
        print(
            f"[{i:02d}] heard={label:<16} novelty={novelty:.3f} "
            f"{'(learned)' if learned else '          '}  "
            f"-> call#{call_index:<2} {freq:6.1f} Hz (peak={peak:.3f})  "
            f"self-recognised={'yes' if recognised else 'no '}"
        )

        if log_path:
            comms.log_event(
                log_path,
                {
                    "round": i,
                    "heard_label": label,
                    "novelty_score": novelty,
                    "learned": learned,
                    "memory_size": len(agent.memory),
                    "call_index": call_index,
                    "call_freq_hz": freq,
                    "call_peak_amplitude": peak,
                    "self_recognised": recognised,
                },
            )

    print(f"\nFinal memory size: {len(agent.memory)} distinct sounds learned "
          f"(of {len(env.labels)} possible).")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--play", action="store_true", help="Attempt real playback (best-effort)")
    parser.add_argument("--log", default=None, help="Append per-round JSONL to this path")
    parser.add_argument(
        "--noise", type=float, default=0.03, help="Background noise amplitude (0-1ish); try ~0.3-0.5 to see self-recognition fail and calls get decayed/switched"
    )
    args = parser.parse_args()
    run(args.rounds, args.seed, args.play, args.log, args.noise)


if __name__ == "__main__":
    sys.exit(main())
