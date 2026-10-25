"""A small, composable call-construction DSL for Chuff's repertoire.

Inspired by clebran-arc-agi (C:\\Users\\aidan\\repos\\clebran-arc-agi) — a
separate research project that evolves typed linear-IR programs over a
grid DSL and mines recurring subroutines into a reusable library
(DreamCoder/Stitch-style). See shared/research_backlog.md's "Related
work: clebran-arc-agi" section for the full comparison and what is/isn't
transferable; this module is the first concrete piece (CH-60).

Where comms.Repertoire represents each call as a single frequency (pick
one of N fixed tones), a CallProgram represents a call as a short
*program*: one generator op (a tone or chirp) optionally followed by a
few modifier ops (amplitude modulation, an attack/decay envelope). This
gives calls actual compositional structure — mutation and crossover
operate on program structure, not just a scalar parameter — which is
what CH-51 (learnable call duration/bandwidth) was gesturing at without
yet having a representation to make it concrete.

Deliberately much smaller than clebran's genome: one generator, 0-3
modifiers, four op types. The point is to prove the representation and
the search loop work at all (see dsl_evolution_demo.py), not to match
clebran's scale.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from chuff_mpu import audio_safety

Op = tuple  # ("tone", freq, duration) | ("chirp", f0, f1, duration) | ("am", rate, depth) | ("envelope", attack_frac, decay_frac)

MAX_MODIFIERS = 3
FREQ_RANGE = (200.0, 4000.0)
DURATION_RANGE = (0.2, 1.0)
AM_RATE_RANGE = (1.0, 40.0)
AM_DEPTH_RANGE = (0.1, 0.9)
ENV_FRAC_RANGE = (0.05, 0.4)

_GENERATORS = ("tone", "chirp")
_MODIFIERS = ("am", "envelope")


@dataclass(frozen=True)
class CallProgram:
    """One generator op followed by 0..MAX_MODIFIERS modifier ops."""

    ops: tuple[Op, ...]

    def __post_init__(self) -> None:
        if not self.ops or self.ops[0][0] not in _GENERATORS:
            raise ValueError("a CallProgram must start with a generator op (tone/chirp)")
        if any(op[0] not in _MODIFIERS for op in self.ops[1:]):
            raise ValueError("every op after the first must be a modifier op (am/envelope)")
        if len(self.ops) - 1 > MAX_MODIFIERS:
            raise ValueError(f"too many modifiers (max {MAX_MODIFIERS})")

    def pretty(self) -> str:
        return " -> ".join(f"{op[0]}({', '.join(f'{v:.1f}' for v in op[1:])})" for op in self.ops)


def _clip(value: float, bounds: tuple[float, float]) -> float:
    return float(np.clip(value, bounds[0], bounds[1]))


def random_op(kind: str, rng: np.random.Generator) -> Op:
    if kind == "tone":
        return ("tone", float(rng.uniform(*FREQ_RANGE)), float(rng.uniform(*DURATION_RANGE)))
    if kind == "chirp":
        f0, f1 = rng.uniform(*FREQ_RANGE, size=2)
        return ("chirp", float(f0), float(f1), float(rng.uniform(*DURATION_RANGE)))
    if kind == "am":
        return ("am", float(rng.uniform(*AM_RATE_RANGE)), float(rng.uniform(*AM_DEPTH_RANGE)))
    if kind == "envelope":
        a, d = rng.uniform(*ENV_FRAC_RANGE, size=2)
        return ("envelope", float(a), float(d))
    raise ValueError(kind)


def random_program(rng: np.random.Generator, n_modifiers: int | None = None) -> CallProgram:
    generator = random_op(rng.choice(_GENERATORS), rng)
    k = rng.integers(0, MAX_MODIFIERS + 1) if n_modifiers is None else n_modifiers
    modifiers = tuple(random_op(rng.choice(_MODIFIERS), rng) for _ in range(k))
    return CallProgram((generator, *modifiers))


def synthesize(program: CallProgram, sample_rate: int) -> np.ndarray:
    """Render a CallProgram to a waveform. Always safety-limited before return."""
    gen = program.ops[0]
    if gen[0] == "tone":
        _, freq, duration = gen
        n = int(duration * sample_rate)
        t = np.linspace(0, duration, n, endpoint=False)
        signal = 0.6 * np.sin(2 * np.pi * freq * t)
    else:  # chirp
        _, f0, f1, duration = gen
        n = int(duration * sample_rate)
        t = np.linspace(0, duration, n, endpoint=False)
        freq_t = np.linspace(f0, f1, n)
        phase = 2 * np.pi * np.cumsum(freq_t) / sample_rate
        signal = 0.6 * np.sin(phase)

    t_real = np.arange(len(signal)) / sample_rate
    for op in program.ops[1:]:
        if op[0] == "am":
            _, rate, depth = op
            signal = signal * (1 - depth + depth * np.sin(2 * np.pi * rate * t_real))
        elif op[0] == "envelope":
            _, attack_frac, decay_frac = op
            n = len(signal)
            env = np.ones(n)
            a_n, d_n = int(attack_frac * n), int(decay_frac * n)
            if a_n > 0:
                env[:a_n] = np.linspace(0, 1, a_n)
            if d_n > 0:
                env[n - d_n :] = np.linspace(1, 0, d_n)
            signal = signal * env

    return audio_safety.limit_audio(signal.astype(np.float32), sample_rate)


# ---- Mutation / crossover, mirroring clebran's genome.py vocabulary at Chuff's scale ----


def mutate_const(program: CallProgram, rng: np.random.Generator, scale: float = 0.15) -> CallProgram:
    """Perturb one numeric parameter of a random op by a small relative delta."""
    idx = int(rng.integers(0, len(program.ops)))
    op = program.ops[idx]
    kind = op[0]
    bounds_by_kind = {
        "tone": [FREQ_RANGE, DURATION_RANGE],
        "chirp": [FREQ_RANGE, FREQ_RANGE, DURATION_RANGE],
        "am": [AM_RATE_RANGE, AM_DEPTH_RANGE],
        "envelope": [ENV_FRAC_RANGE, ENV_FRAC_RANGE],
    }[kind]
    values = list(op[1:])
    pos = int(rng.integers(0, len(values)))
    span = bounds_by_kind[pos][1] - bounds_by_kind[pos][0]
    values[pos] = _clip(values[pos] + rng.normal(scale=scale * span), bounds_by_kind[pos])
    new_ops = list(program.ops)
    new_ops[idx] = (kind, *values)
    return CallProgram(tuple(new_ops))


def mutate_swap_modifier(program: CallProgram, rng: np.random.Generator) -> CallProgram:
    """Replace one modifier op with a freshly-random op of a (possibly different) modifier kind."""
    if len(program.ops) == 1:
        return program
    idx = int(rng.integers(1, len(program.ops)))
    new_ops = list(program.ops)
    new_ops[idx] = random_op(rng.choice(_MODIFIERS), rng)
    return CallProgram(tuple(new_ops))


def mutate_grow(program: CallProgram, rng: np.random.Generator) -> CallProgram:
    """Append a new random modifier, if under the length cap."""
    if len(program.ops) - 1 >= MAX_MODIFIERS:
        return program
    return CallProgram((*program.ops, random_op(rng.choice(_MODIFIERS), rng)))


def mutate_delete(program: CallProgram, rng: np.random.Generator) -> CallProgram:
    """Remove a random modifier, if there is one to remove."""
    if len(program.ops) == 1:
        return program
    idx = int(rng.integers(1, len(program.ops)))
    new_ops = program.ops[:idx] + program.ops[idx + 1 :]
    return CallProgram(new_ops)


def mutate(program: CallProgram, rng: np.random.Generator) -> CallProgram:
    """Pick one mutation operator at random and apply it."""
    choice = rng.choice(["const", "swap", "grow", "delete"], p=[0.5, 0.2, 0.15, 0.15])
    return {
        "const": mutate_const,
        "swap": mutate_swap_modifier,
        "grow": mutate_grow,
        "delete": mutate_delete,
    }[str(choice)](program, rng)


def crossover(a: CallProgram, b: CallProgram, rng: np.random.Generator) -> CallProgram:
    """Splice: take `a`'s generator, then as many of `b`'s modifiers as fit.

    Mirrors clebran's fragment extraction/injection — a modifier
    sub-sequence discovered useful in one program gets transplanted onto
    another's generator — without needing a hole-filling mechanism,
    since every modifier here is a closed, self-contained op (no
    internal references to fix up).
    """
    keep = b.ops[1:][:MAX_MODIFIERS]
    return CallProgram((a.ops[0], *keep))
