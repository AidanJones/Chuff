"""Offline audit of frozen frontend grids, raw replay evidence and selection."""

import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path

import numpy as np

from chuff_mpu.sim.published_experiment import runtime
from chuff_mpu.sim.published_protocol import select
from chuff_mpu.sim.recorded_catalog import digest
from chuff_mpu.sim.recorded_metrics import aggregate, episode_metrics


def audit(folder):
    manifest = json.loads((folder/"manifest.json").read_text())
    assert manifest["schema"] == "chuff.recorded.published-experiment.v1"
    for name, expected in manifest["source_sha256"].items():
        assert digest((folder/"source"/name).read_bytes()) == expected, name
    for group in ("input_sha256", "output_sha256", "examples"):
        for name, expected in manifest[group].items():
            assert digest((folder/name).read_bytes()) == expected, name
    plan = json.loads((folder/"input/protocol.json").read_text())
    assert manifest["dependencies"] == plan["dependencies"]
    for name, expected in plan["method_source_sha256"].items():
        assert manifest["source_sha256"]["python/src/chuff_mpu/"+name] == expected
    assert digest((folder/"input/reservation.json").read_bytes()) == plan["reservation_sha256"]
    reservation = json.loads((folder/"input/reservation.json").read_text())
    catalog = json.loads((folder/"input/catalog.json").read_text())
    phase = manifest["phase"]
    assert catalog["partitions"] == [phase] and catalog["reservation_sha256"] == plan["reservation_sha256"]
    reserved = [row for row in reservation["selected_rows"] if row["split"] == phase]
    assert [(r["filename"], r["src_file"], int(r["fold"])) for r in reserved] == [
        (r["filename"], r["source_id"], r["fold"]) for r in catalog["clips"]]
    assert not {r["source_id"] for r in catalog["clips"]} & set(reservation["selection_plan"]["excluded_sources"])
    allowed = {r["filename"] for r in catalog["clips"]}
    assert set(manifest["foregrounds"]) | set(manifest["backgrounds"]) == allowed
    if phase == "development":
        expected_modes = set(plan["candidates"]) | {"adaptive_gain"}
    else:
        selection = select(json.loads((folder/"input/development-summary.json").read_text()), plan)
        assert selection == json.loads((folder/"input/development-selection.json").read_text())
        expected_modes = set(selection["evaluation_ids"]) | {"adaptive_gain"}
    assert set(manifest["settings"]) == expected_modes
    for mode in expected_modes-{"adaptive_gain"}:
        assert manifest["settings"][mode] == plan["candidates"][mode]["settings"]
    seeds = plan[phase+"_seeds"]
    suites = plan["positive_suites"]+plan["control_suites"]
    assert manifest["seed_start"] == seeds[0] and manifest["seeds"] == len(seeds)
    assert manifest["suites"] == suites
    with gzip.open(folder/"observations.jsonl.gz", "rt", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    indexed = {(r["suite"], r["seed"], r["step"], r["mode"]): r for r in rows}
    expected = {(suite, seed, step, mode) for suite in suites for seed in seeds
                for step in range(210) for mode in expected_modes}
    assert len(rows) == len(indexed) and set(indexed) == expected
    grouped = defaultdict(list)
    for row in rows:
        assert type(row["observe_ns"]) is int and row["observe_ns"] > 0
        grouped[row["suite"], row["seed"]].append(row)
    summaries = json.loads((folder/"summary.json").read_text())
    by_run = {(row["suite"], row["seed"]): row for row in summaries}
    assert len(by_run) == len(summaries) and set(by_run) == set(grouped)
    for (suite, seed), summary in by_run.items():
        selected = grouped[suite, seed]
        assert summary["episodes"] == episode_metrics(selected, details=True, exclusion=True)
        assert summary["runtime"] == runtime(selected, manifest["settings"])
        assert summary["max_component_error"] < 1e-5
        condition, snr, kind = suite.split("/")
        room_rng = np.random.default_rng(np.random.SeedSequence(200000+seed).spawn(4)[0])
        room_length = float(room_rng.uniform(.03, .5))
        assert summary["scene"] == {**scene_settings(plan), "snr_db": float(snr), "room_decay_s": room_length}
        assert summary["environment"]["environment_seed"] == 200000+seed
        program = {int(key): value for key, value in summary["environment"]["source_program"].items()}
        occurrences = Counter()
        episodes = {}
        for chunk in range(42):
            label = program.get(chunk) if kind == "positive" else None
            if label is not None:
                occurrences[label] += 1
                episodes[chunk] = label
            for offset in range(5):
                reference = indexed[suite, seed, chunk*5+offset, "adaptive_gain"]
                for mode in expected_modes:
                    row = indexed[suite, seed, chunk*5+offset, mode]
                    assert row["chunk"] == chunk and row["recording_id"] == label
                    assert row["episode_id"] == (chunk if label is not None else None)
                    assert row["occurrence"] == (occurrences[label] if label is not None else None)
                    assert row["episode_offset"] == (offset if label is not None else None)
                    assert ("without_episode_probe" in row) == (label is not None)
                    if label is not None:
                        assert abs(row["episode_post_snr_db"]-float(snr)) < 1e-8
                        if row["window_post_snr_db"] is not None:
                            ratio = 20*np.log10(row["current_episode_window_rms"]/row["residual_window_rms"])
                            assert abs(ratio-row["window_post_snr_db"]) < 1e-10
                    for field in ("current_episode_window_rms", "residual_window_rms", "window_post_snr_db"):
                        assert row[field] == reference[field]
        assert {r["episode_id"]: r["recording_id"] for r in summary["physical_episodes"]} == episodes
        if kind == "positive":
            assert set(occurrences) == set(manifest["foregrounds"]) and set(occurrences.values()) == {4}
        for item in summary["physical_episodes"]:
            assert item["stop_step"]-item["start_step"] == 5
            ratio = 20*np.log10(item["source_rms"]/item["residual_rms"])
            assert abs(ratio-item["post_snr_db"]) < 2e-5
        matched = by_run[condition+"/0/control", seed]
        assert summary["background_output_sha256"] == matched["background_output_sha256"]
        reference = by_run["recorded/0/control", seed]["environment"]
        for key in ("source_program", "background_input_sha256", "microphone_low_hz",
                    "microphone_high_hz", "gain", "room_response_length_s"):
            assert summary["environment"][key] == reference[key], key
    for name in manifest["examples"]:
        with np.load(folder/name, allow_pickle=False) as sample:
            assert len(sample["mixture"]) == 80000
            np.testing.assert_allclose(sample["mixture"], sample["background"]+sample["all_sources"], atol=1e-7)
            np.testing.assert_allclose(sample["mixture"], sample["without_episode"]+sample["current_episode"], atol=1e-7)
            info = json.loads(str(sample["metadata"]))
            ratio = 20*np.log10(np.linalg.norm(sample["current_episode"].astype(float))/np.linalg.norm(sample["without_episode"].astype(float)))
            assert abs(ratio-info["post_snr_db"]) < 2e-4
    assert json.loads((folder/"analysis.json").read_text()) == aggregate(summaries)
    if phase == "development":
        assert json.loads((folder/"selection.json").read_text()) == select(summaries, plan)
    print(f"{len(rows)} decisions: source/method/input/output hashes, complete frozen matrix, physical episodes/SNR, paired controls, timings, metrics and selection verified")


def scene_settings(plan):
    from dataclasses import asdict
    from chuff_mpu.sim.scene import SceneConfig
    return asdict(SceneConfig(**plan["scene"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    audit(args.folder)


if __name__ == "__main__":
    main()
