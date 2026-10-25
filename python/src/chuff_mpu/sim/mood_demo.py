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
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import sys

import numpy as np

from chuff_mpu import audio, config, dream
from chuff_mpu.mood import Mood, MoodState, update_mood
from chuff_mpu.sim.agent import ChuffAgent
from chuff_mpu.sim.world import Environment


def run(ticks: int, seed: int, tick_duration_s: float, sound_probability: float,
        *, ack_ticks: tuple[int, ...] = ()) -> dict:
    if ticks < 1 or seed < 0 or not np.isfinite(tick_duration_s) or tick_duration_s <= 0:
        raise ValueError("positive ticks/duration and nonnegative seed required")
    if not np.isfinite(sound_probability) or not 0 <= sound_probability <= 1:
        raise ValueError("sound probability must be in [0, 1]")
    if any(type(tick) is not int or not 1 <= tick <= ticks for tick in ack_ticks):
        raise ValueError("ack ticks must lie within the one-based run")
    env = Environment(config.SAMPLE_RATE, config.CLIP_DURATION_S, seed=seed)
    agent = ChuffAgent(seed=seed + 1)
    rng = np.random.default_rng(seed + 2)
    dream_rng = np.random.default_rng(seed + 3)

    mood_state = MoodState(Mood.CONTENTMENT, energy=0.3)
    idle_s = 0.0
    dream_fired = 0
    mood_counts: dict[str, int] = {}
    rows = []

    print(f"=== {ticks} ticks, {tick_duration_s}s each, p(sound)={sound_probability} ===\n")
    for tick in range(1, ticks + 1):
        novelty: float | None = None
        surprise: float | None = None
        label = "-"
        learned = False
        predicted = None

        if rng.random() < sound_probability:
            label, target = env.present()
            mic_in = env.hear(target)
            _, novelty, learned = agent.hear(mic_in)
            idle_s = 0.0
        else:
            idle_s += tick_duration_s
            if idle_s >= config.IDLE_TIMEOUT_S and len(agent.memory) >= 2 and rng.random() < 0.3:
                # Dream: predict, then "wake up" to whatever the world presents next.
                predicted = dream.predict_next(agent.memory, rng=dream_rng)
                wake_label, wake_target = env.present()
                wake_mic = env.hear(wake_target)
                observed = audio.extract_mfcc(wake_mic, config.SAMPLE_RATE, config.N_MFCC)
                surprise = dream.surprise_score(observed, predicted)
                label = f"(dream->{wake_label})"
                dream_fired += 1
                idle_s = 0.0

        # Explicit fixture protocol inputs, not an acoustic ACK detector.
        ack = tick in ack_ticks
        mood_state = update_mood(mood_state, novelty, surprise, idle_s, ack_received=ack, learned=learned)
        mood_counts[mood_state.mood.value] = mood_counts.get(mood_state.mood.value, 0) + 1
        rows.append(dict(tick=tick, label=label, novelty=novelty, surprise=surprise,
                         idle_s=idle_s, learned=bool(learned), ack_received=ack,
                         prediction=None if predicted is None else predicted.tolist(),
                         mood=mood_state.mood.value, energy=mood_state.energy))

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
    return dict(ticks=ticks, seed=seed, tick_duration_s=tick_duration_s,
                sound_probability=sound_probability, ack_ticks=list(ack_ticks),
                rng_seeds=dict(environment=seed, agent=seed+1, schedule=seed+2, dream=seed+3),
                dream_events=dream_fired, mood_counts=mood_counts, observations=rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticks", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--tick-duration", type=float, default=10.0, help="Simulated seconds per tick (default 10s/tick)"
    )
    parser.add_argument("--sound-probability", type=float, default=0.15)
    parser.add_argument("--ack-ticks", type=int, nargs="*", default=[],
                        help="One-based synthetic protocol ACK input ticks; not acoustic detection")
    parser.add_argument("--output", type=Path, help="New directory for exact sources and raw trajectory JSON")
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        parser.error("output directory must not already exist")
    result = run(args.ticks, args.seed, args.tick_duration, args.sound_probability,
                 ack_ticks=tuple(args.ack_ticks))
    if args.output is not None:
        args.output.mkdir(parents=True, exist_ok=False)
        root = Path(__file__).resolve().parents[4]
        sources = {}
        for name in ("sim/mood_demo.py", "sim/agent.py", "sim/world.py", "sim/tones.py",
                     "memory.py", "dream.py", "mood.py", "audio.py", "audio_safety.py", "comms.py", "config.py"):
            relative = "python/src/chuff_mpu/"+name
            data = (root/relative).read_bytes()
            target = args.output/"source"/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            sources[relative] = hashlib.sha256(data).hexdigest()
        for relative in ("python/pyproject.toml", "python/uv.lock"):
            data = (root/relative).read_bytes()
            target = args.output/"source"/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            sources[relative] = hashlib.sha256(data).hexdigest()
        manifest = dict(source_sha256=sources, python=platform.python_version(),
                        dependencies={name: version(name) for name in ("numpy", "scipy", "librosa", "scikit-learn")})
        (args.output/"manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        (args.output/"trajectory.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
