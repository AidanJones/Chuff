#!/usr/bin/env python3
"""Two independent Chuff units, one simulated acoustic channel, no shared
data structure — a software stand-in for shared/comms_protocol.md's
real two-unit experiment (TASKS.md CH-23, blocked on a second physical
unit) and a direct test of two open questions in
shared/research_backlog.md: does repertoire convergence actually need
contingent acknowledgement (Theme 1.3, "social gating"), and if it has
it, how does it unfold round to round (Theme 3.2)?

Unit A hears a real-world sound, picks the closest call in its own
repertoire, and "says" it. Unit B's microphone picks up that call,
attenuated by distance and mixed with fresh room noise — B has no
access to A's internals, only to that noisy acoustic signal. If B
recognises the call (has heard something close enough to it before), B
plays a fixed acknowledgement tone; A listens for that tone the same
lossy way. A reinforces the call it used if it heard the ack, decays it
otherwise — exactly shared/comms_protocol.md's loop, nothing more.

--condition passive removes B's ability to ever acknowledge, as a
control: the social-gating hypothesis predicts A's repertoire should
never stabilise under passive exposure alone, only under contingent
acknowledgement.

Usage:
    uv run chuff-sim-2unit --rounds 60 --condition social
    uv run chuff-sim-2unit --rounds 60 --condition passive
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from chuff_mpu import audio, comms, config, memory
from chuff_mpu.sim import tones
from chuff_mpu.sim.agent import ChuffAgent
from chuff_mpu.sim.world import Environment


def _detect_tone(clip: np.ndarray, sample_rate: int, target_freq: float, tolerance_hz: float = 50.0, snr_factor: float = 3.0) -> bool:
    """Did `clip` contain a tone at ~target_freq, above the noise floor?

    Deliberately simple FFT-peak detection, not MFCC/KNN — the
    acknowledgement is a single fixed, universally-known tone by design
    (shared/comms_protocol.md), unlike the evolving repertoire calls, so
    it doesn't need the same machinery used to recognise those.
    """
    spectrum = np.abs(np.fft.rfft(clip))
    freqs = np.fft.rfftfreq(len(clip), 1 / sample_rate)
    peak_idx = int(np.argmax(spectrum))
    # Cast off numpy scalar types explicitly: left uncast, these compare to
    # native Python bool/float|int fine at runtime but produce numpy.bool_,
    # not bool — which json.dumps rejects. Found by actually exercising
    # --log for the first time (ch-22), not by inspection.
    peak_freq, peak_mag = float(freqs[peak_idx]), float(spectrum[peak_idx])
    noise_floor = float(np.median(spectrum)) + 1e-9
    return bool(abs(peak_freq - target_freq) < tolerance_hz and peak_mag > noise_floor * snr_factor)


class Listener:
    """Unit B's "have I heard a call like this before" memory.

    Separate from a ChuffAgent's own `memory.EmbeddingStore` of
    *environmental* sounds — "do I recognise a call from another unit"
    and "have I heard this environmental sound before" are different
    questions, so they get different stores even though both reuse the
    same EmbeddingStore implementation.
    """

    def __init__(self, max_embeddings: int) -> None:
        self.store = memory.EmbeddingStore(max_embeddings)

    def hear_call(self, clip: np.ndarray) -> bool:
        embedding = audio.extract_mfcc(clip, config.SAMPLE_RATE, config.N_MFCC)
        recognised = self.store.novelty_score(embedding) < config.RECOGNITION_THRESHOLD
        if not recognised:
            self.store.add(embedding)  # learn it now, so it IS recognised next time
        return recognised


def run(
    rounds: int,
    seed: int,
    condition: str,
    attenuation: float,
    cutoff_hz: float | None,
    log_path: str | None,
    verbose: bool = True,
) -> dict:
    env = Environment(config.SAMPLE_RATE, config.CLIP_DURATION_S, seed=seed)
    unit_a = ChuffAgent(seed=seed + 1)
    unit_b = Listener(config.MAX_EMBEDDINGS)
    ack_duration_s = config.ACK_DURATION_MS / 1000

    ack_history: list[bool] = []
    reinforced_freqs: list[float] = []  # calls that got an ack this run — Theme 1.2's AAH test
    cutoff_label = "none" if cutoff_hz is None else f"{cutoff_hz:.0f} Hz"
    if verbose:
        print(f"=== {rounds} rounds, condition={condition}, attenuation={attenuation}, channel cutoff={cutoff_label} ===")
    for i in range(1, rounds + 1):
        label, target = env.present()
        mic_in = env.hear(target)
        context_embedding, novelty, learned = unit_a.hear(mic_in)
        call_index, freq, call_wave = unit_a.respond(context_embedding)

        # What B's mic actually picks up, 2-3m away: optional frequency-
        # dependent loss, attenuation, then fresh room noise.
        b_mic_in = env.transmit(call_wave, attenuation=attenuation, cutoff_hz=cutoff_hz)

        if condition == "passive":
            recognised_by_b = False  # B never acknowledges, no matter what it hears
        else:
            recognised_by_b = unit_b.hear_call(b_mic_in)

        heard_ack = False
        if recognised_by_b:
            ack_wave = tones.sine_tone(config.ACK_FREQ_HZ, ack_duration_s, config.SAMPLE_RATE, amplitude=0.6)
            a_hears_ack = env.transmit(ack_wave, attenuation=attenuation, cutoff_hz=cutoff_hz)
            heard_ack = _detect_tone(a_hears_ack, config.SAMPLE_RATE, config.ACK_FREQ_HZ)

        unit_a.reinforce_or_decay(call_index, context_embedding, heard_ack)
        ack_history.append(heard_ack)
        if heard_ack:
            reinforced_freqs.append(freq)

        if verbose:
            print(
                f"[{i:03d}] A heard={label:<16} -> call#{call_index:<2} {freq:6.1f} Hz  "
                f"B recognised={'yes' if recognised_by_b else 'no ':<3} ack heard by A={'yes' if heard_ack else 'no'}"
            )
        if log_path:
            comms.log_event(
                log_path,
                {
                    "round": i,
                    "condition": condition,
                    "cutoff_hz": cutoff_hz,
                    "heard_label": label,
                    "call_index": call_index,
                    "call_freq_hz": freq,
                    "b_recognised": recognised_by_b,
                    "a_heard_ack": heard_ack,
                    "b_memory_size": len(unit_b.store),
                },
            )

    n = len(ack_history)
    first_half = ack_history[: n // 2] or [False]
    second_half = ack_history[n // 2 :] or [False]
    stats = {
        "ack_rate_first_half": sum(first_half) / len(first_half),
        "ack_rate_second_half": sum(second_half) / len(second_half),
        "mean_reinforced_freq_hz": float(np.mean(reinforced_freqs)) if reinforced_freqs else None,
        "n_reinforced": len(reinforced_freqs),
    }
    if verbose:
        print(
            f"\nAck rate - first half: {stats['ack_rate_first_half']:.2f}  "
            f"second half: {stats['ack_rate_second_half']:.2f}  (condition={condition})"
        )
        if reinforced_freqs:
            print(
                f"Reinforced-call frequencies - mean: {np.mean(reinforced_freqs):.0f} Hz, "
                f"min: {np.min(reinforced_freqs):.0f} Hz, max: {np.max(reinforced_freqs):.0f} Hz "
                f"(channel cutoff={cutoff_label}; the Acoustic Adaptation Hypothesis predicts "
                f"a tight cutoff should push this toward lower frequencies)"
            )
        else:
            print("No acks heard this run - nothing was reinforced.")
    return stats


def sweep(rounds: int, seeds: int, condition: str, attenuation: float) -> None:
    """Average Theme 1.2's AAH comparison (no cutoff vs. a tight cutoff)
    over many seeds — a single seed is too noisy to trust (verified while
    building this: the first seed tried showed literally no effect,
    because of the index/frequency confound fixed in comms.py; a later
    single seed showed one, but still wasn't enough to be confident the
    direction was real rather than lucky)."""
    print(f"=== AAH sweep: {seeds} seeds, {rounds} rounds each, condition={condition} ===\n")
    for cutoff_hz in (None, 900.0):
        means, acks = [], []
        for seed in range(seeds):
            stats = run(rounds, seed, condition, attenuation, cutoff_hz, None, verbose=False)
            if stats["mean_reinforced_freq_hz"] is not None:
                means.append(stats["mean_reinforced_freq_hz"])
            acks.append(stats["ack_rate_second_half"])
        label = "none" if cutoff_hz is None else f"{cutoff_hz:.0f} Hz"
        print(
            f"cutoff={label:>6}  avg reinforced-call mean freq: {np.mean(means):.0f} Hz "
            f"(range {np.min(means):.0f}-{np.max(means):.0f} across {len(means)} seeds)  "
            f"avg second-half ack rate: {np.mean(acks):.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--condition", choices=["social", "passive"], default="social")
    parser.add_argument("--attenuation", type=float, default=config.CHANNEL_ATTENUATION)
    parser.add_argument(
        "--cutoff",
        type=float,
        default=None,
        help="Channel low-pass cutoff in Hz (e.g. 900) to test the Acoustic Adaptation "
        "Hypothesis; default None = no filtering, matching the original behaviour.",
    )
    parser.add_argument("--log", default=None, help="Append per-round JSONL to this path")
    parser.add_argument(
        "--sweep-seeds",
        type=int,
        default=0,
        help="Instead of one verbose run, average the no-cutoff-vs-900Hz-cutoff AAH "
        "comparison (Theme 1.2) over this many seeds and print a summary.",
    )
    args = parser.parse_args()
    if args.sweep_seeds:
        sweep(args.rounds, args.sweep_seeds, args.condition, args.attenuation)
    else:
        run(args.rounds, args.seed, args.condition, args.attenuation, args.cutoff, args.log)


if __name__ == "__main__":
    sys.exit(main())
