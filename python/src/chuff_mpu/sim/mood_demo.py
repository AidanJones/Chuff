#!/usr/bin/env python3
"""CH-33/CH-14: mood.py's state machine and dream.py's dream loop, wired
into something that actually exercises them — not just unit tests.

Every other sim/ demo presents a sound every tick, so idle_s never
really accumulates. This one has real gaps: most ticks, nothing
happens. Once idle_s crosses config.IDLE_TIMEOUT_S, the agent
occasionally "dreams" — dream.predict_next() interpolates two stored
embeddings, then the next real sound that comes along is compared
against that prediction via dream.surprise_score(), producing a real
SURPRISE or CONTENTMENT mood transition driven by an actual prediction
error, not a hand-picked test value.

Usage:
    uv run chuff-sim-mood --ticks 400
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from chuff_mpu import audio, config, dream
from chuff_mpu.mood import Mood, MoodState, update_mood
from chuff_mpu.sim.agent import ChuffAgent
from chuff_mpu.sim.world import Environment


def run(ticks: int, seed: int, tick_duration_s: float, sound_probability: float) -> None:
    env = Environment(config.SAMPLE_RATE, config.CLIP_DURATION_S, seed=seed)
    agent = ChuffAgent(seed=seed + 1)
    rng = np.random.default_rng(seed + 2)

    mood_state = MoodState(Mood.CONTENTMENT, energy=0.3)
    idle_s = 0.0
    dream_fired = 0
    mood_counts: dict[str, int] = {}

    print(f"=== {ticks} ticks, {tick_duration_s}s each, p(sound)={sound_probability} ===\n")
    for tick in range(1, ticks + 1):
        novelty: float | None = None
        surprise: float | None = None
        label = "-"

        if rng.random() < sound_probability:
            label, target = env.present()
            mic_in = env.hear(target)
            _, novelty, _ = agent.hear(mic_in)
            idle_s = 0.0
        else:
            idle_s += tick_duration_s
            if idle_s >= config.IDLE_TIMEOUT_S and len(agent.memory) >= 2 and rng.random() < 0.3:
                # Dream: predict, then "wake up" to whatever the world presents next.
                predicted = dream.predict_next(agent.memory)
                wake_label, wake_target = env.present()
                wake_mic = env.hear(wake_target)
                observed = audio.extract_mfcc(wake_mic, config.SAMPLE_RATE, config.N_MFCC)
                surprise = dream.surprise_score(observed, predicted)
                label = f"(dream->{wake_label})"
                dream_fired += 1
                idle_s = 0.0

        mood_state = update_mood(mood_state, novelty, surprise, idle_s)
        mood_counts[mood_state.mood.value] = mood_counts.get(mood_state.mood.value, 0) + 1

        novelty_str = f"{novelty:.3f}" if novelty is not None else "  -  "
        surprise_str = f"{surprise:.3f}" if surprise is not None else "  -  "
        print(
            f"[{tick:03d}] heard={label:<22} novelty={novelty_str} surprise={surprise_str} "
            f"idle={idle_s:6.0f}s -> mood={mood_state.mood.value:<16} energy={mood_state.energy:.2f}"
        )

    print(f"\nDream loop fired {dream_fired} time(s) over {ticks} ticks.")
    print("Mood state frequency:")
    for mood, count in sorted(mood_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {mood:<18} {count:4d} ticks ({count / ticks:.0%})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticks", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--tick-duration", type=float, default=10.0, help="Simulated seconds per tick (default 10s/tick)"
    )
    parser.add_argument("--sound-probability", type=float, default=0.15)
    args = parser.parse_args()
    run(args.ticks, args.seed, args.tick_duration, args.sound_probability)


if __name__ == "__main__":
    sys.exit(main())
