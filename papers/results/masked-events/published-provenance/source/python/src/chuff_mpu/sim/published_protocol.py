"""Finite, matched frontend grids and explicit non-silent development selection."""

from dataclasses import asdict
from itertools import product
import argparse
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np

from chuff_mpu.band_attention import BandAttentionConfig, BandForegroundLearner
from chuff_mpu.published_attention import (PCENConfig, PCENForegroundLearner,
                                         MCRAConfig, MCRAForegroundLearner)
from chuff_mpu.sim.recorded_catalog import audit_reservation, digest


FAMILIES = {
    "quiet160": (BandAttentionConfig, BandForegroundLearner, 160),
    "quiet128": (BandAttentionConfig, BandForegroundLearner, 128),
    "pcen160": (PCENConfig, PCENForegroundLearner, 160),
    "mcra128": (MCRAConfig, MCRAForegroundLearner, 128),
}


def candidates():
    """Twelve explicit settings per family; no outcomes affect this grid."""
    result = {}
    for family, (config, _, hop) in FAMILIES.items():
        for feature, local, broad in product((0, 1), (1.5, 2.5, 3.5), (1.15, 1.5)):
            name = f"{family}-f{feature}-l{local:g}-b{broad:g}"
            result[name] = dict(family=family, settings=asdict(config(hop=hop,
                feature_mode=feature, local_threshold=local, broad_threshold=broad)))
    return result


def actor(specification):
    config, learner, _ = FAMILIES[specification["family"]]
    return learner(config(**specification["settings"]))


def protocol(reservation_sha256):
    package = Path(__file__).resolve().parents[1]
    sources = ("attention.py", "band_attention.py", "published_attention.py", "gain_attention.py",
               "adaptive_gain.py", "sim/recorded_scene.py", "sim/recorded_experiment.py",
               "sim/recorded_metrics.py", "sim/published_protocol.py")
    return dict(schema="chuff.recorded.published-protocol.v1",
        reservation_sha256=reservation_sha256, candidates=candidates(),
        method_source_sha256={name: digest((package/name).read_bytes()) for name in sources},
        dependencies={name: version(name) for name in ("numpy", "scipy", "librosa")},
        development_seeds=[2800, 2801], evaluation_seeds=list(range(2900, 2910)),
        positive_suites=[f"{condition}/{snr}/positive" for condition in ("recorded", "modulated")
                         for snr in ("0", "-6")],
        control_suites=[f"{condition}/0/control" for condition in ("recorded", "modulated")],
        scene=dict(duration_s=5, rounds=42, warmup=2, nuisance_probability=0,
                   background_change=False, collect_stems=True, snr_reference="post_residual"),
        primary="strict_episode_exclusion",
        selection=dict(min_detection_each_positive_suite=.8,
            min_mean_first_episode_novel_attention=.6, min_mean_strict_episode_exclusion=.2,
            feasible_order=["min_mean_control_background_admissions", "max_mean_strict_episode_exclusion",
                            "min_mean_wrong_repeat_identity", "lexical_config_id"],
            infeasible="No selected configuration; report diagnostic highest-strict configuration, then lowest background admissions, wrong identity, lexical ID",
            groups="One overall and one per feature_mode per family; no extra tuning budget"),
        evaluation="Untuned defaults plus feasible selections (or explicitly infeasible diagnostic choices), deduplicated; adaptive_gain untuned anchor",
        runtime="Warm imports/JIT using separate actors; perf_counter_ns around actual observe only; exclude branch copy/probe, physical rendering and warmup clips; record each duration",
        scope="Full-recording injection/replay, not semantic event recognition; bootstrap conditional on five fixed foreground recordings",
        boundary="Each observe uses its own unpadded STFT; PCEN/MCRA spectral state persists but missing boundary frames are not reconstructed",
        tuning="Published primitive smoother defaults fixed; same 12 detection/feature settings per family; quiet128 is the MCRA hop-matched control")


def select(summaries, plan):
    """Reject incomplete grids; constraints are never relaxed after outcomes."""
    expected_grid = plan["candidates"]
    expected_runs = {(suite, seed) for suite in plan["positive_suites"]+plan["control_suites"]
                     for seed in plan["development_seeds"]}
    actual_runs = [(row["suite"], row["seed"]) for row in summaries]
    if len(actual_runs) != len(set(actual_runs)) or set(actual_runs) != expected_runs:
        raise ValueError("development suite/seed matrix differs from frozen protocol")
    if any(set(row["episodes"]) != set(expected_grid) | {"adaptive_gain"} for row in summaries):
        raise ValueError("development candidate matrix differs from frozen protocol")
    positives = [row for row in summaries if row["suite"] in plan["positive_suites"]]
    controls = [row for row in summaries if row["suite"] in plan["control_suites"]]
    scored = {}
    c = plan["selection"]
    for name, spec in expected_grid.items():
        mean = lambda rows, metric: float(np.mean([row["episodes"][name]["rates"][metric] for row in rows]))
        detection = {suite: mean([row for row in positives if row["suite"] == suite], "episode_detection")
                     for suite in plan["positive_suites"]}
        strict = mean(positives, "strict_episode_exclusion")
        novel = mean(positives, "first_episode_novel_attention")
        wrong = mean(positives, "wrong_repeat_identity")
        background = float(np.mean([row["episodes"][name]["counts"]["background_admissions"] for row in controls]))
        if not np.isfinite([*detection.values(), strict, novel, wrong, background]).all():
            raise ValueError("nonfinite development score")
        scored[name] = dict(family=spec["family"], feature_mode=spec["settings"]["feature_mode"],
            detection_by_suite=detection, strict=strict, first_novel=novel, wrong=wrong,
            control_admissions=background, feasible=(min(detection.values()) >= c["min_detection_each_positive_suite"]
                and strict >= c["min_mean_strict_episode_exclusion"]
                and novel >= c["min_mean_first_episode_novel_attention"]))
    selected = {}
    for family in FAMILIES:
        for feature in (None, 0, 1):
            members = [name for name, row in scored.items() if row["family"] == family
                       and (feature is None or row["feature_mode"] == feature)]
            feasible = [name for name in members if scored[name]["feasible"]]
            rank = lambda name: (scored[name]["control_admissions"], -scored[name]["strict"], scored[name]["wrong"], name)
            diagnostic = min(members, key=lambda name: (-scored[name]["strict"], scored[name]["control_admissions"], scored[name]["wrong"], name))
            key = family if feature is None else f"{family}/feature{feature}"
            selected[key] = dict(selected=min(feasible, key=rank) if feasible else None,
                                 diagnostic=diagnostic, feasible_count=len(feasible), candidates=len(members))
    evaluation_ids = {f"{family}-f0-l3.5-b1.5" for family in FAMILIES}
    evaluation_ids.update(item["selected"] or item["diagnostic"] for item in selected.values())
    return dict(schema="chuff.recorded.published-selection.v1", scores=scored, selections=selected,
                evaluation_ids=sorted(evaluation_ids))


def audit_protocol(path, reservation):
    audit_reservation(reservation)
    plan = json.loads(path.read_text())
    expected = protocol(digest((reservation/"reservation.json").read_bytes()))
    if plan != expected:
        raise ValueError("protocol, method sources or dependencies differ from the frozen declaration")
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "audit", "select"))
    parser.add_argument("--reservation", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.action == "freeze":
        audit_reservation(args.reservation)
        plan = protocol(digest((args.reservation/"reservation.json").read_bytes()))
        with args.protocol.open("x", encoding="utf-8") as stream:
            json.dump(plan, stream, indent=2, allow_nan=False)
    else:
        plan = audit_protocol(args.protocol, args.reservation)
        if args.action == "select":
            if args.summary is None or args.output is None:
                parser.error("select requires --summary and --output")
            result = select(json.loads(args.summary.read_text()), plan)
            result.update(protocol_sha256=digest(args.protocol.read_bytes()),
                          summary_sha256=digest(args.summary.read_bytes()))
            with args.output.open("x", encoding="utf-8") as stream:
                json.dump(result, stream, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
