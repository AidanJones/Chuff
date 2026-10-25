#!/usr/bin/env python3
"""CH-60: can mutation+selection search over call_dsl's tiny DSL actually
find a program that reproduces a target sound?

Mirrors clebran-arc-agi's fitness-driven search loop (see
shared/research_backlog.md's "Related work: clebran-arc-agi" section) at
Chuff's scale: elitism + mutation + occasional crossover, no fancier
machinery needed for a first pass. Fitness is cosine similarity in MFCC
space between a candidate program's synthesised waveform and a target
sound — the same metric/convention memory.py and comms.py already use,
so this plugs into the rest of the codebase rather than inventing a
parallel one.

This is a genuine *structural* search test, not just parameter-fitting:
a chirp target (frequency sweep) can only be matched well by a program
that picks the chirp generator, not the tone generator, so the headline
number to watch is whether evolution finds the right generator kind, not
just whether the loss goes down.

Usage:
    uv run chuff-sim-dsl --target kettle_whistle   # a pure tone
    uv run chuff-sim-dsl --target dog_bark         # a chirp - structural test
    uv run chuff-sim-dsl --target dog_bark --generations 40 --population 50
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from chuff_mpu import audio, config
from chuff_mpu.sim import call_dsl as dsl
from chuff_mpu.sim.world import SOUND_LIBRARY, Environment


def fitness(program: dsl.CallProgram, target_embedding: np.ndarray, sample_rate: int, n_mfcc: int) -> float:
    """Cosine similarity (higher is better) between a candidate and the target."""
    wave = dsl.synthesize(program, sample_rate)
    embedding = audio.extract_mfcc(wave, sample_rate, n_mfcc)
    denom = (np.linalg.norm(embedding) * np.linalg.norm(target_embedding)) or 1e-9
    return float(np.dot(embedding, target_embedding) / denom)


def evolve(
    target_embedding: np.ndarray,
    sample_rate: int,
    n_mfcc: int,
    population_size: int,
    generations: int,
    elite_frac: float,
    crossover_rate: float,
    seed: int,
) -> tuple[dsl.CallProgram, list[float]]:
    rng = np.random.default_rng(seed)
    population = [dsl.random_program(rng) for _ in range(population_size)]
    n_elite = max(1, int(population_size * elite_frac))
    best_per_gen: list[float] = []
    scored: list[tuple[float, dsl.CallProgram]] = []

    for _ in range(generations):
        scored = sorted(
            ((fitness(p, target_embedding, sample_rate, n_mfcc), p) for p in population),
            key=lambda t: -t[0],
        )
        best_per_gen.append(scored[0][0])
        elites = [p for _, p in scored[:n_elite]]

        new_population = list(elites)
        while len(new_population) < population_size:
            parent = elites[int(rng.integers(0, len(elites)))]
            if len(elites) > 1 and rng.random() < crossover_rate:
                other = elites[int(rng.integers(0, len(elites)))]
                child = dsl.crossover(parent, other, rng)
            else:
                child = dsl.mutate(parent, rng)
            new_population.append(child)
        population = new_population

    return scored[0][1], best_per_gen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=list(SOUND_LIBRARY), default="kettle_whistle")
    parser.add_argument("--population", type=int, default=30)
    parser.add_argument("--generations", type=int, default=20)
    parser.add_argument("--elite-frac", type=float, default=0.2)
    parser.add_argument("--crossover-rate", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    env = Environment(config.SAMPLE_RATE, config.CLIP_DURATION_S, seed=args.seed)
    label, target_wave = env.present(label=args.target, jitter_hz=0.0)
    target_embedding = audio.extract_mfcc(target_wave, config.SAMPLE_RATE, config.N_MFCC)
    spec = SOUND_LIBRARY[args.target]

    print(f"=== target: {label} ({spec}) ===")
    print(f"population={args.population}, generations={args.generations}\n")

    best, history = evolve(
        target_embedding,
        config.SAMPLE_RATE,
        config.N_MFCC,
        args.population,
        args.generations,
        args.elite_frac,
        args.crossover_rate,
        args.seed,
    )

    for gen, score in enumerate(history):
        marker = " <-- best so far" if gen == 0 or score > max(history[:gen]) else ""
        print(f"gen {gen:3d}: best fitness = {score:.4f}{marker}")

    print(f"\nBest program found: {best.pretty()}")
    print(f"Final fitness: {history[-1]:.4f} (1.0 = perfect match)")
    print(
        f"Generator kind: {best.ops[0][0]} "
        f"({'matches' if best.ops[0][0] == spec['kind'] else 'does NOT match'} the target's own generator kind)"
    )


if __name__ == "__main__":
    sys.exit(main())
