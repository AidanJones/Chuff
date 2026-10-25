"""Audit linear-component examples, calibration, transform pairing and probes."""

import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from chuff_mpu.sim.audit_masked import audit
from chuff_mpu.sim.channel_metrics import intervention_metrics, paired_estimates, condition_contrasts
from chuff_mpu.sim.localization import score_regions
from chuff_mpu.sim.memory_diagnostics import diagnose


def audit_channel(folder):
    summaries = audit(folder)
    manifest = json.loads((folder/"manifest.json").read_text())
    expected = {(suite, seed) for suite in manifest["suites"]
                for seed in range(manifest["seed_start"], manifest["seed_start"]+manifest["seeds"])}
    assert len(summaries) == len(expected)
    assert {(s["suite"], s["seed"]) for s in summaries} == expected
    groups, schedules, transform_parameters = defaultdict(list), {}, {}
    with gzip.open(folder/"observations.jsonl.gz", "rt", encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            groups[row["suite"], row["seed"]].append(row)
            if row["suite"].startswith("background/"):
                assert row["truth"] is None and not row["nuisance"]
                continue
            key = row["seed"], row["step"]
            assert schedules.setdefault(key, row["truth"]) == row["truth"]
            if row["truth"] is None:
                continue
            assert "without_current_probe" in row
            assert row["event_stop_s"]-row["event_start_s"] <= .94+1/16000
            measured = row["active_snr_db_before_channel"] if row["suite"].endswith("/input") else row["post_channel_residual_snr_db"]
            assert abs(measured-row["snr_db"]) < 1e-8
            parameters = row["source_parameters"]
            transform = row["suite"].split("/")[0]
            parameter_key = transform, row["seed"], row["step"]
            assert transform_parameters.setdefault(parameter_key, parameters) == parameters
    for (transform, seed, step), parameters in transform_parameters.items():
        if transform == "pitch":
            assert parameters["applied_duration_factor"] == 1
            assert parameters["pitch_ratio"] == transform_parameters["pitch_duration", seed, step]["pitch_ratio"]
        if transform == "duration":
            assert parameters["pitch_ratio"] == 1
            assert parameters["realized_duration_factor"] == transform_parameters["pitch_duration", seed, step]["realized_duration_factor"]
    for summary in summaries:
        rows = groups.pop((summary["suite"], summary["seed"]))
        assert diagnose(rows) == summary["diagnostics"]
        assert score_regions(rows) == summary["localization"]
        assert intervention_metrics(rows) == summary["interventions"]
        events = [r for r in rows if r["mode"] == "original" and r["truth"] is not None]
        assert len(events) == len(summary["snr_events"])
        for row, event in zip(events, summary["snr_events"]):
            assert row["step"] == event["step"] and row["truth"] == event["label"]
            assert row["active_snr_db_before_channel"] == event["before"]
            assert row["post_channel_background_snr_db"] == event["post_background"]
            assert row["post_channel_residual_snr_db"] == event["post_residual"]
        assert np.isfinite(summary["max_component_recombination_error"])
    assert not groups
    for name, digest in manifest["examples"].items():
        path = folder/name.replace("\\", "/")
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
        with np.load(path, allow_pickle=False) as data:
            mixture = data["mixture"]
            np.testing.assert_allclose(mixture, data["background"]+data["all_sources"], atol=1e-7, rtol=3e-6)
            np.testing.assert_allclose(mixture, data["current_source"]+data["without_current"], atol=1e-7, rtol=3e-6)
            info = json.loads(str(data["metadata"].item()))
            section = slice(round(info["event_start_s"]*16000), round(info["event_stop_s"]*16000))
            rms = lambda array: np.sqrt(np.mean(np.asarray(array, dtype=np.float64)**2))
            snr = 20*np.log10(rms(data["current_source"][section])/rms(data["without_current"][section]))
            assert abs(snr-info["post_residual"]) < 2e-4
    estimates = paired_estimates(summaries)
    assert estimates == json.loads((folder/"analysis.json").read_text())
    print("SNR targets, transform independence, linear examples, timing and identical-state probe metrics verified")
    return estimates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--contrasts-output", type=Path, help="New derived analysis JSON plus exact analysis-source copies")
    args = parser.parse_args()
    if args.contrasts_output is not None and args.contrasts_output.exists():
        parser.error("contrasts output must be new")
    audit_channel(args.folder)
    if args.contrasts_output is not None:
        source = args.contrasts_output.parent/(args.contrasts_output.stem+"-source")
        source.mkdir(exist_ok=False)
        hashes = {}
        for name in ("channel_metrics.py", "audit_channel.py"):
            content = Path(__file__).with_name(name).read_bytes()
            (source/name).write_bytes(content)
            hashes[name] = hashlib.sha256(content).hexdigest()
        summaries = json.loads((args.folder/"summary.json").read_text())
        result = dict(input_summary_sha256=hashlib.sha256((args.folder/"summary.json").read_bytes()).hexdigest(),
            analysis_source_sha256=hashes, source_directory=source.name, contrasts=condition_contrasts(summaries))
        args.contrasts_output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
