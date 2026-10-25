#!/usr/bin/env python3
"""CH-53: does a self-directed babbling phase change which call an agent
reaches for on its very first encounter with a real sound?

A fresh `ChuffAgent` is built per trial (no cross-contamination between
conditions); the only difference is whether `agent.babble()` ran first.
For each trial we record the call chosen on the agent's first-ever
`respond()`, and how far that call's frequency sits from the target
sound's own characteristic frequency (its tone frequency, or the
midpoint of a chirp's start/end) — a crude proxy for "did it reach for
something acoustically in the right neighbourhood," since nothing in
this architecture ties `closest_call()`'s untrained tie-break to
frequency at all (see comms.py's docstring on why `Repertoire`
deliberately doesn't sort by frequency).

Usage:
    uv run chuff-sim-babble --seeds 20
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from chuff_mpu import config
from chuff_mpu.sim.agent import ChuffAgent
from chuff_mpu.sim.world import SOUND_LIBRARY, Environment


def _characteristic_freq(label: str) -> float:
    spec = SOUND_LIBRARY[label]
    if spec["kind"] == "tone":
        return float(spec["freq"])
    return float(spec["freq"] + spec["freq_end"]) / 2


def first_encounter_choice(babble: bool, world_seed: int, agent_seed: int, label: str) -> tuple[int, float]:
    env = Environment(config.SAMPLE_RATE, config.CLIP_DURATION_S, seed=world_seed)
    agent = ChuffAgent(seed=agent_seed)
    if babble:
        agent.babble()
    _, target = env.present(label=label)
    mic_in = env.hear(target)
    context_embedding, _, _ = agent.hear(mic_in)
    call_index, freq, _ = agent.respond(context_embedding)
    return call_index, freq


def run(seeds: int, labels: list[str]) -> list[dict]:
    rows = []
    for seed in range(seeds):
        for label in labels:
            call_cold, freq_cold = first_encounter_choice(False, seed, seed + 1, label)
            call_babbled, freq_babbled = first_encounter_choice(True, seed, seed + 1, label)
            target_freq = _characteristic_freq(label)
            rows.append(
                {
                    "seed": seed,
                    "label": label,
                    "differs": call_cold != call_babbled,
                    "err_cold": abs(freq_cold - target_freq),
                    "err_babbled": abs(freq_babbled - target_freq),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=20)
    args = parser.parse_args()

    labels = list(SOUND_LIBRARY)
    rows = run(args.seeds, labels)

    n_differ = sum(r["differs"] for r in rows)
    mean_err_cold = float(np.mean([r["err_cold"] for r in rows]))
    mean_err_babbled = float(np.mean([r["err_babbled"] for r in rows]))

    print(f"=== {args.seeds} seeds x {len(labels)} sounds = {len(rows)} trials ===\n")
    print(f"Round-1 choice differs between cold-start and babbled: {n_differ}/{len(rows)} ({n_differ / len(rows):.0%})")
    print(
        f"Mean |chosen call freq - target characteristic freq|: "
        f"cold-start={mean_err_cold:.0f} Hz, babbled={mean_err_babbled:.0f} Hz"
    )
    if mean_err_babbled < mean_err_cold:
        print(
            f"Babbling moves the first-encounter choice {mean_err_cold - mean_err_babbled:.0f} Hz closer "
            f"to acoustically appropriate, on average."
        )
    elif mean_err_babbled > mean_err_cold:
        print(
            f"Babbling moved the first-encounter choice {mean_err_babbled - mean_err_cold:.0f} Hz FARTHER "
            f"from acoustically appropriate, on average — report this as-is, don't rerun for a better seed."
        )
    else:
        print("No measurable difference.")


if __name__ == "__main__":
    sys.exit(main())
