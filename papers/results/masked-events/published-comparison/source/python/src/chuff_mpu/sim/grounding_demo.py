#!/usr/bin/env python3
"""CH-54: do two independently-learning Chuff units converge on a
*shared* per-sound signaling convention, or just two independently-
stable (but uncoordinated) ones?

two_unit_demo.py is one-directional: Unit A speaks, Unit B only ever
listens and acknowledges. That can show A's repertoire stabilising, but
it can't answer the grounding question (shared/research_backlog.md
Theme 3.3) at all, because B never develops a repertoire of its own to
compare A's against.

This version is symmetric: both A and B are full ChuffAgents (their own
memory + repertoire, independently seeded), both independently hear
every presented sound (through their own noisy mic — two microphones in
two positions, not one shared signal), and both speak *and* listen every
round — each one's call judged by the other's independently-learned
"do I recognise this" model, each reinforcing/decaying its own choice
based on whether its own call got acknowledged. shared/comms_protocol.md's
loop, run in both directions at once.

After the run, for each sound label: has each unit settled on one
preferred call for it (purity of its last-N choices), and — the actual
grounding question — do *different* labels end up mapped to *different*
(A's call, B's call) pairs, or does everything collapse onto the same
pair regardless of what was heard (no real discrimination at all)?

Usage:
    uv run chuff-sim-grounding --rounds 200
    uv run chuff-sim-grounding --rounds 200 --cutoff 900
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict

import numpy as np

from chuff_mpu import comms, config
from chuff_mpu.sim import tones
from chuff_mpu.sim.agent import ChuffAgent
from chuff_mpu.sim.two_unit_demo import Listener, _detect_tone
from chuff_mpu.sim.world import Environment


def _speak_and_ack(
    speaker: ChuffAgent,
    listener_of_speaker: Listener,
    context_embedding: np.ndarray,
    env: Environment,
    attenuation: float,
    cutoff_hz: float | None,
    ack_duration_s: float,
) -> tuple[int, float, bool]:
    """`speaker` picks+plays a call for `context_embedding`; the receiving
    unit's `listener_of_speaker` (its independently-learned model of THIS
    speaker's calls) decides whether to acknowledge; `speaker` hears back
    (or doesn't) over the same lossy channel. Returns (call_index, freq,
    heard_ack)."""
    call_index, freq, wave = speaker.respond(context_embedding)
    heard_by_listener = env.transmit(wave, attenuation=attenuation, cutoff_hz=cutoff_hz)
    recognised = listener_of_speaker.hear_call(heard_by_listener)

    heard_ack = False
    if recognised:
        ack_wave = tones.sine_tone(config.ACK_FREQ_HZ, ack_duration_s, config.SAMPLE_RATE, amplitude=0.6)
        speaker_hears_ack = env.transmit(ack_wave, attenuation=attenuation, cutoff_hz=cutoff_hz)
        heard_ack = _detect_tone(speaker_hears_ack, config.SAMPLE_RATE, config.ACK_FREQ_HZ)
    return call_index, freq, heard_ack


def run(
    rounds: int,
    seed: int,
    attenuation: float,
    cutoff_hz: float | None,
    log_path: str | None,
    relevance_threshold: float | None = None,
) -> dict:
    env = Environment(config.SAMPLE_RATE, config.CLIP_DURATION_S, seed=seed)
    agent_a = ChuffAgent(seed=seed + 1)
    agent_b = ChuffAgent(seed=seed + 2)
    if relevance_threshold is not None:
        # ch-63: override both units' relevance gate for this run, without
        # touching ChuffAgent's constructor (this is sim-only sweeping, not
        # part of the graded Repertoire contract).
        agent_a.voice._relevance_threshold = relevance_threshold
        agent_b.voice._relevance_threshold = relevance_threshold
    listener_for_a = Listener(config.MAX_EMBEDDINGS)  # B's model of A's calls
    listener_for_b = Listener(config.MAX_EMBEDDINGS)  # A's model of B's calls
    ack_duration_s = config.ACK_DURATION_MS / 1000

    history_a: dict[str, list[int]] = defaultdict(list)
    history_b: dict[str, list[int]] = defaultdict(list)

    for i in range(1, rounds + 1):
        label, target = env.present()

        mic_a = env.hear(target)
        ctx_a, _, _ = agent_a.hear(mic_a)
        mic_b = env.hear(target)  # independently noisy: a different microphone position
        ctx_b, _, _ = agent_b.hear(mic_b)

        call_idx_a, freq_a, heard_ack_a = _speak_and_ack(
            agent_a, listener_for_a, ctx_a, env, attenuation, cutoff_hz, ack_duration_s
        )
        agent_a.reinforce_or_decay(call_idx_a, ctx_a, heard_ack_a)
        history_a[label].append(call_idx_a)

        call_idx_b, freq_b, heard_ack_b = _speak_and_ack(
            agent_b, listener_for_b, ctx_b, env, attenuation, cutoff_hz, ack_duration_s
        )
        agent_b.reinforce_or_decay(call_idx_b, ctx_b, heard_ack_b)
        history_b[label].append(call_idx_b)

        if log_path:
            comms.log_event(
                log_path,
                {
                    "round": i,
                    "label": label,
                    "a_call": call_idx_a,
                    "a_ack": heard_ack_a,
                    "b_call": call_idx_b,
                    "b_ack": heard_ack_b,
                },
            )

    return {"history_a": history_a, "history_b": history_b}


def _converged_choice(history: list[int], tail_frac: float = 0.3) -> tuple[int, float]:
    """Majority call index over the last `tail_frac` of a label's
    occurrences, and its purity (fraction of the tail that agrees)."""
    tail = history[-max(1, int(len(history) * tail_frac)) :]
    counts = Counter(tail)
    call, n = counts.most_common(1)[0]
    return call, n / len(tail)


def summarize(result: dict, labels: list[str]) -> dict:
    """Per-label (call, purity) for both units, plus the distinct-pairs count."""
    history_a, history_b = result["history_a"], result["history_b"]
    pairs: dict[str, tuple[int, int]] = {}
    purities: list[float] = []
    for label in labels:
        if label not in history_a or label not in history_b:
            continue
        call_a, purity_a = _converged_choice(history_a[label])
        call_b, purity_b = _converged_choice(history_b[label])
        pairs[label] = (call_a, call_b)
        purities.extend([purity_a, purity_b])
    return {
        "pairs": pairs,
        "distinct_pairs": len(set(pairs.values())),
        "n_labels": len(pairs),
        "mean_purity": float(np.mean(purities)) if purities else 0.0,
    }


def report(result: dict, labels: list[str]) -> None:
    history_a, history_b = result["history_a"], result["history_b"]
    print(f"{'label':<16} {'A call (purity)':<18} {'B call (purity)':<18}")
    for label in labels:
        if label not in history_a or label not in history_b:
            print(f"{label:<16} (not enough occurrences this run)")
            continue
        call_a, purity_a = _converged_choice(history_a[label])
        call_b, purity_b = _converged_choice(history_b[label])
        print(f"{label:<16} #{call_a:<2} ({purity_a:.0%}){'':<9} #{call_b:<2} ({purity_b:.0%})")

    summary = summarize(result, labels)
    distinct_pairs, n_labels = summary["distinct_pairs"], summary["n_labels"]
    print(
        f"\n{distinct_pairs} distinct (A-call, B-call) pairs across {n_labels} labels "
        f"({'fully discriminating' if distinct_pairs == n_labels else 'some labels share a pair - not fully grounded'})."
    )
    print(
        "This is the grounding question (Theme 3.3): a distinct pair per label means each "
        "sound got its own mutually-recognised convention, not just 'anything gets acked'."
    )


def sweep_relevance_threshold(rounds: int, seeds: int, thresholds: list[float]) -> None:
    """CH-63: does a tighter relevance gate get closer to one-pair-per-label
    grounding, and at what cost to per-label consistency?"""
    from chuff_mpu.sim.world import SOUND_LIBRARY

    labels = list(SOUND_LIBRARY)
    print(f"=== relevance-threshold sweep: {seeds} seeds x {rounds} rounds each ===\n")
    for threshold in thresholds:
        distinct_counts, purities = [], []
        for seed in range(seeds):
            result = run(rounds, seed, config.CHANNEL_ATTENUATION, None, None, relevance_threshold=threshold)
            summary = summarize(result, labels)
            distinct_counts.append(summary["distinct_pairs"])
            purities.append(summary["mean_purity"])
        print(
            f"threshold={threshold:.2f}  avg distinct pairs: {np.mean(distinct_counts):.1f}/6  "
            f"(range {min(distinct_counts)}-{max(distinct_counts)})  avg purity: {np.mean(purities):.0%}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--attenuation", type=float, default=config.CHANNEL_ATTENUATION)
    parser.add_argument("--cutoff", type=float, default=None)
    parser.add_argument("--log", default=None, help="Append per-round JSONL to this path")
    parser.add_argument(
        "--relevance-threshold", type=float, default=None, help="Override config.RELEVANCE_SIM_THRESHOLD for this run"
    )
    parser.add_argument(
        "--sweep-thresholds",
        type=int,
        default=0,
        help="Instead of one run, sweep relevance thresholds [0.3, 0.5, 0.7, 0.9] "
        "over this many seeds each (ch-63) and print a summary.",
    )
    args = parser.parse_args()

    if args.sweep_thresholds:
        sweep_relevance_threshold(args.rounds, args.sweep_thresholds, [0.3, 0.5, 0.7, 0.9])
        return

    from chuff_mpu.sim.world import SOUND_LIBRARY

    print(f"=== {args.rounds} rounds, symmetric A<->B, seed={args.seed} ===\n")
    result = run(args.rounds, args.seed, args.attenuation, args.cutoff, args.log, args.relevance_threshold)
    report(result, list(SOUND_LIBRARY))


if __name__ == "__main__":
    sys.exit(main())
