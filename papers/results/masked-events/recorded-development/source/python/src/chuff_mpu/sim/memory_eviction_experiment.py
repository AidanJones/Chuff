#!/usr/bin/env python3
"""CH-52: does a recency-weighted eviction policy retain a rare-but-
recurring sound better than plain FIFO, under capacity pressure from
many one-off sounds competing for the same slots?

See shared/research_backlog.md Theme 2.2. Uses real synthesised audio
(tones, real MFCC extraction) through the same `memory.EmbeddingStore`
the rest of this project uses — just with `eviction="fifo"` vs
`eviction="recency"` compared side by side on an identical sequence.

Scenario: a small-capacity store (capacity well below the number of
distinct sounds it'll be asked to deal with) is shown one "rare" sound
at the very start, then a long run of never-repeated filler sounds —
except every `--revisit-every` rounds, the rare sound is presented
again (recognised-or-not is checked, but it's not force-re-added; a
real agent wouldn't re-learn something it already knows). At the end,
is the rare sound still recognised?

Usage:
    uv run chuff-sim-memory
    uv run chuff-sim-memory --rounds 60 --capacity 4 --revisit-every 5
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from chuff_mpu import audio, config, memory
from chuff_mpu.sim import tones

RARE_FREQ_HZ = 1500.0


def _embed(freq_hz: float, rng: np.random.Generator | None = None) -> np.ndarray:
    clip = tones.sine_tone(freq_hz, config.CLIP_DURATION_S, config.SAMPLE_RATE)
    if rng is not None:
        clip = tones.mix(clip, tones.white_noise(config.CLIP_DURATION_S, config.SAMPLE_RATE, 0.02, rng))
    return audio.extract_mfcc(clip, config.SAMPLE_RATE, config.N_MFCC)


def _well_separated_filler_freqs(count: int, rng: np.random.Generator) -> list[float]:
    """`count` filler frequencies, spaced far enough apart that they never
    accidentally "recognise" each other — isolates the eviction-policy
    effect from noise caused by lucky/unlucky filler-filler collisions."""
    grid = np.linspace(200, 4000, max(count * 4, 50))
    rng.shuffle(grid)
    return list(grid[:count])


def run_one(eviction: str, rounds: int, capacity: int, revisit_every: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    store = memory.EmbeddingStore(max_embeddings=capacity, eviction=eviction)

    rare_embedding = _embed(RARE_FREQ_HZ, rng)
    store.add(rare_embedding)  # round 0: learn the rare sound

    filler_rounds = [i for i in range(1, rounds + 1) if not (revisit_every and i % revisit_every == 0)]
    filler_freqs = iter(_well_separated_filler_freqs(len(filler_rounds), rng))

    recognised_on_revisit: list[bool] = []
    for i in range(1, rounds + 1):
        if revisit_every and i % revisit_every == 0:
            score = store.novelty_score(rare_embedding)
            recognised_on_revisit.append(score < config.RECOGNITION_THRESHOLD)
            # A real agent wouldn't re-learn a sound it already recognises.
            if score >= config.NOVELTY_THRESHOLD_HIGH:
                store.add(rare_embedding)
        else:
            filler_embedding = _embed(next(filler_freqs), rng)
            score = store.novelty_score(filler_embedding)
            if score >= config.NOVELTY_THRESHOLD_HIGH:
                store.add(filler_embedding)

    final_score = store.novelty_score(rare_embedding)
    return {
        "final_novelty_of_rare": final_score,
        "still_recognised_at_end": final_score < config.RECOGNITION_THRESHOLD,
        "recognised_on_revisit_fraction": (
            sum(recognised_on_revisit) / len(recognised_on_revisit) if recognised_on_revisit else float("nan")
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--capacity", type=int, default=3)
    parser.add_argument("--revisit-every", type=int, default=10)
    parser.add_argument("--seeds", type=int, default=10, help="Average over this many seeds (0..N-1)")
    args = parser.parse_args()

    print(
        f"=== capacity={args.capacity}, {args.rounds} rounds, rare sound revisited "
        f"every {args.revisit_every} rounds, averaged over {args.seeds} seeds ===\n"
    )
    for eviction in ("fifo", "recency"):
        rows = [run_one(eviction, args.rounds, args.capacity, args.revisit_every, seed) for seed in range(args.seeds)]
        still_recognised_rate = sum(r["still_recognised_at_end"] for r in rows) / len(rows)
        avg_revisit_fraction = sum(r["recognised_on_revisit_fraction"] for r in rows) / len(rows)
        avg_final_novelty = sum(r["final_novelty_of_rare"] for r in rows) / len(rows)
        print(
            f"[{eviction:>7}] still_recognised_at_end: {still_recognised_rate:.0%} of seeds  |  "
            f"recognised-on-revisit (avg across run): {avg_revisit_fraction:.0%}  |  "
            f"avg final novelty of rare sound: {avg_final_novelty:.3f}"
        )


if __name__ == "__main__":
    sys.exit(main())
