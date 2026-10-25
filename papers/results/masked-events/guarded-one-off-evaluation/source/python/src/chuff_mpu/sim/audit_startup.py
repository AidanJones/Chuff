"""Audit the archived silent-prefix regression's stored decisions and counts."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def audit_startup(folder):
    manifest = json.loads((folder/"manifest.json").read_text())
    for name, digest in manifest["source_sha256"].items():
        assert hashlib.sha256((folder/"source"/name).read_bytes()).hexdigest() == digest
    observations = json.loads((folder/"observations.json").read_text())
    summaries = json.loads((folder/"summary.json").read_text())
    modes = ("old_clean", "old_silent", "new_clean", "new_silent")
    seeds = range(manifest["seed_start"], manifest["seed_start"]+manifest["seeds"])
    indexed = {(r["seed"], r["step"], r["mode"]): r for r in observations}
    expected = {(seed, step, mode) for seed in seeds for step in range(120) for mode in modes}
    assert len(indexed) == len(observations) and set(indexed) == expected
    assert len(summaries) == len(seeds)
    assert {r["seed"] for r in summaries} == set(seeds)
    for summary in summaries:
        for mode in modes:
            rows = [indexed[summary["seed"], step, mode] for step in range(120)]
            count = Counter()
            for row in rows:
                count["events"] += row["truth"] is not None
                count["detected_events"] += row["truth"] is not None and row["status"] in ("candidate", "learned", "recognised")
                count["false_admissions"] += row["truth"] is None and row["status"] == "learned"
            for key in ("events", "detected_events", "false_admissions"):
                assert count[key] == summary["modes"][mode][key]
        for step in range(120):
            for left, right in (("new_clean", "old_clean"), ("new_silent", "new_clean")):
                a, b = indexed[summary["seed"], step, left], indexed[summary["seed"], step, right]
                assert (a["truth"], a["status"], a["prototype_id"], a["novelty"]) == (b["truth"], b["status"], b["prototype_id"], b["novelty"])
    print(f"{len(observations)} decisions: complete matrix, source hashes, detection counts and stored trajectory equivalence verified")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    audit_startup(args.folder)


if __name__ == "__main__":
    main()
