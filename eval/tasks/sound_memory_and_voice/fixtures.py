"""Deterministic synthetic data for the sound_memory_and_voice task.

Real MFCC vectors aren't needed to grade the *algorithms* in memory.py and
comms.py (nearest-neighbour novelty scoring, call-to-context weighting) —
only a stable, repeatable notion of "distinct sound classes with small
within-class jitter" is needed. These fixtures stand in for real audio so
the grader runs in milliseconds with no mic/speaker involved, and so every
run (agent's or reference's) is scored against the exact same inputs.

13-dim vectors are used throughout to match N_MFCC in config.py.
"""

from __future__ import annotations

import numpy as np

N_DIM = 13
N_CLASSES = 6
JITTER_STD = 0.05  # small Gaussian noise = "the same real-world sound, recorded again"


def class_centroids(seed: int = 1000) -> np.ndarray:
    """Return N_CLASSES well-separated unit vectors in N_DIM-space.

    Each represents one distinct "sound" Chuff might hear repeatedly
    (e.g. doorbell, dog bark, kettle whistle, ...).
    """
    rng = np.random.default_rng(seed)
    centroids = rng.normal(size=(N_CLASSES, N_DIM))
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)
    return centroids


def jittered_sample(centroid: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One noisy "recording" of a sound class — same class, not bit-identical."""
    sample = centroid + rng.normal(scale=JITTER_STD, size=centroid.shape)
    return sample.astype(np.float32)


def memory_test_sequence(seed: int = 42) -> list[tuple[np.ndarray, int | None]]:
    """A fixed sequence of (embedding, true_class_or_None) observations.

    `true_class_or_None` is the index into class_centroids() the embedding
    was drawn from, or None if it's a one-off outlier never repeated
    (always "genuinely novel"). The grader uses this ground truth to check
    the candidate's novelty_score() calls the right way each time; the
    candidate under test never sees this label.
    """
    rng = np.random.default_rng(seed)
    centroids = class_centroids()
    sequence: list[tuple[np.ndarray, int | None]] = []

    # Each class heard 3 times (first = "new", next two = "repeats"),
    # interleaved in a fixed shuffled order so recognition can't rely on
    # the classes arriving in neat blocks.
    entries: list[tuple[int, np.ndarray]] = []
    for cls in range(N_CLASSES):
        for _ in range(3):
            entries.append((cls, jittered_sample(centroids[cls], rng)))
    rng.shuffle(entries)
    for cls, emb in entries:
        sequence.append((emb, cls))

    # A handful of genuine one-off outliers, far from every centroid.
    for _ in range(4):
        outlier = rng.normal(loc=0, scale=1, size=N_DIM).astype(np.float32)
        outlier *= 5.0 / (np.linalg.norm(outlier) + 1e-6)  # push well outside the unit-norm clusters
        sequence.append((outlier, None))

    rng.shuffle(sequence)
    return sequence


def comms_test_contexts(seed: int = 7) -> tuple[np.ndarray, np.ndarray, np.random.Generator]:
    """Two distinct, stable "context" embeddings for the comms rubric, plus an RNG."""
    rng = np.random.default_rng(seed)
    centroids = class_centroids(seed=seed)
    ctx1 = jittered_sample(centroids[0], rng)
    ctx2 = jittered_sample(centroids[1], rng)
    return ctx1, ctx2, rng
