"""Audit full-duration recorded experiment structure, hashes and episode scores."""

import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from chuff_mpu.sim.recorded_metrics import aggregate, episode_metrics


def audit(folder):
    manifest = json.loads((folder/"manifest.json").read_text())
    for name, digest in manifest["source_sha256"].items():
        assert hashlib.sha256((folder/"source"/name).read_bytes()).hexdigest() == digest, name
    for group in ("input_sha256", "output_sha256", "examples"):
        for name, digest in manifest[group].items():
            assert hashlib.sha256((folder/name).read_bytes()).hexdigest() == digest, name
    catalog = json.loads((folder/"input/catalog.json").read_text())
    groups = {split: {r["source_id"] for r in catalog["clips"] if r["split"] == split}
              for split in ("development", "evaluation")}
    assert not groups["development"] & groups["evaluation"]
    allowed = {r["filename"] for r in catalog["clips"] if r["split"] == manifest["partition"]}
    assert set(manifest["foregrounds"]) <= allowed and set(manifest["backgrounds"]) <= allowed
    with gzip.open(folder/"observations.jsonl.gz", "rt", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    modes = tuple(manifest["settings"])
    seeds = range(manifest["seed_start"], manifest["seed_start"]+manifest["seeds"])
    suites = set(manifest["suites"])
    assert suites == {f"{condition}/{snr}/{kind}" for condition in ("recorded", "modulated")
        for snr in ("0", "-6") for kind in ("positive", "control")}
    indexed = {(r["suite"], r["seed"], r["step"], r["mode"]): r for r in rows}
    expected = {(suite, seed, step, mode) for suite in suites for seed in seeds
                for step in range(210) for mode in modes}
    assert len(indexed) == len(rows) and set(indexed) == expected
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["suite"], row["seed"]].append(row)
    summaries = json.loads((folder/"summary.json").read_text())
    by_run = {(r["suite"], r["seed"]): r for r in summaries}
    assert len(by_run) == len(summaries) and set(by_run) == set(grouped)
    for summary in summaries:
        suite, seed = summary["suite"], summary["seed"]
        selected = grouped[suite, seed]
        assert summary["episodes"] == episode_metrics(selected)
        assert summary["max_component_error"] < 1e-5
        program = {int(key): value for key, value in summary["environment"]["source_program"].items()}
        occurrences = defaultdict(int)
        actual_episodes = {}
        for chunk in range(42):
            label = program.get(chunk) if suite.endswith("/positive") else None
            if label is not None:
                occurrences[label] += 1
                actual_episodes[chunk] = label
            for offset in range(5):
                reference = indexed[suite, seed, chunk*5+offset, modes[0]]
                for mode in modes:
                    row = indexed[suite, seed, chunk*5+offset, mode]
                    assert row["recording_id"] == label
                    assert row["episode_id"] == (chunk if label is not None else None)
                    assert row["occurrence"] == (occurrences[label] if label is not None else None)
                    if label is not None:
                        assert abs(row["episode_post_snr_db"]-summary["scene"]["snr_db"]) < 1e-8
                        assert row["episode_offset"] == offset and "without_episode_probe" in row
                    if row["window_post_snr_db"] is not None:
                        actual = 20*np.log10(row["current_episode_window_rms"]/row["residual_window_rms"])
                        assert abs(actual-row["window_post_snr_db"]) < 1e-10
                    for field in ("recording_id", "episode_id", "occurrence", "window_post_snr_db"):
                        assert row[field] == reference[field]
        if suite.endswith("/positive"):
            assert set(occurrences) == set(manifest["foregrounds"]) and set(occurrences.values()) == {4}
        assert {r["episode_id"]: r["recording_id"] for r in summary["physical_episodes"]} == actual_episodes
        for item in summary["physical_episodes"]:
            assert item["stop_step"]-item["start_step"] == 5
            assert abs(20*np.log10(item["source_rms"]/item["residual_rms"])-item["post_snr_db"]) < 2e-5
        condition, snr, kind = suite.split("/")
        matched = by_run[condition+"/"+snr+"/"+("control" if kind == "positive" else "positive"), seed]
        for field in ("background_output_sha256",):
            assert summary[field] == matched[field]
        assert summary["environment"]["source_program"] == matched["environment"]["source_program"]
    for name in manifest["examples"]:
        with np.load(folder/name, allow_pickle=False) as sample:
            assert len(sample["mixture"]) == 80000
            np.testing.assert_allclose(sample["mixture"], sample["background"]+sample["all_sources"], atol=1e-7)
            np.testing.assert_allclose(sample["mixture"], sample["without_episode"]+sample["current_episode"], atol=1e-7)
            info = json.loads(str(sample["metadata"]))
            ratio = 20*np.log10(np.linalg.norm(sample["current_episode"].astype(float))/np.linalg.norm(sample["without_episode"].astype(float)))
            assert abs(ratio-info["post_snr_db"]) < 2e-4
    assert json.loads((folder/"analysis.json").read_text()) == aggregate(summaries)
    print(f"{len(rows)} decisions: complete matrix, source/input/example hashes, SNR, episode repeats, background pairing and metrics verified")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    audit(args.folder)


if __name__ == "__main__":
    main()
