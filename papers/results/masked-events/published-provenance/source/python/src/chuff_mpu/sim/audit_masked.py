"""Audit saved decisions and produce label-level seed statistics for the paper."""

from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from chuff_mpu.sim.masked_experiment import aggregate, class_metrics


def audit(folder: Path):
    manifest = json.loads((folder / "manifest.json").read_text())
    summaries = json.loads((folder / "summary.json").read_text())
    for name, digest in manifest["source_sha256"].items():
        assert hashlib.sha256((folder / "source" / name).read_bytes()).hexdigest() == digest, name
    groups = defaultdict(list)
    with gzip.open(folder / "observations.jsonl.gz", "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            groups[(row["suite"], row["snr_db"], row["seed"])].append(row)
    totals = 0
    for summary in summaries:
        rows = groups.pop((summary["suite"], summary["snr_db"], summary["seed"]))
        totals += len(rows)
        assert class_metrics(rows) == summary["by_class"]
        for mode, metrics in summary["metrics"].items():
            observations = [r for r in rows if r["mode"] == mode]
            assert [r["step"] for r in observations] == list(range(manifest["rounds"]))
            assert sum(r["status"] == "learned" and r["truth"] is None for r in observations) == metrics["false_admissions"]
            classes = class_metrics(observations)[mode]
            repeats = sum(v.get("repeat_events", 0) for v in classes.values())
            correct = sum(v.get("correct_repeats", 0) for v in classes.values())
            assert repeats == metrics["counts"]["repeat_windows"]
            assert correct == metrics["counts"]["correct_repeat_recognition"]
            assert metrics["repeat_recognition_rate"] == (correct / repeats if repeats else None)
    assert not groups
    assert aggregate(summaries) == json.loads((folder / "aggregate.json").read_text())
    print(f"{folder.name}: {totals} decisions, source hashes, class counts, paired metrics and intervals verified")
    return summaries


def by_class(summaries):
    result = []
    for suite, snr in sorted({(s["suite"], s["snr_db"]) for s in summaries if s["suite"] != "background"}):
        runs = [s for s in summaries if (s["suite"], s["snr_db"]) == (suite, snr)]
        labels = sorted({label for s in runs for label in s["by_class"]["original"]})
        for label in labels:
            row = {"suite": suite, "snr_db": snr, "label": label, "modes": {}}
            for mode in ["original", "quiet_band"]:
                counts = [s["by_class"][mode].get(label, {}) for s in runs]
                values = [c.get("correct_repeats", 0) / c["repeat_events"] for c in counts if c.get("repeat_events", 0)]
                detection = [c.get("detected", 0) / c["events"] for c in counts if c.get("events", 0)]
                ci = None
                if values:
                    values = np.array(values)
                    indices = np.random.default_rng(731).integers(0, len(values), size=(10000, len(values)))
                    ci = np.quantile(values[indices].mean(axis=1), [0.025, 0.975]).tolist()
                row["modes"][mode] = {"repeat_recognition": float(np.mean(values)) if len(values) else None,
                                      "bootstrap_95": ci, "eligible_seeds": len(values),
                                      "detection": float(np.mean(detection)),
                                      "repeat_events": sum(c.get("repeat_events", 0) for c in counts)}
            result.append(row)
    return result


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folders", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summaries = []
    for folder in args.folders:
        summaries.extend(audit(folder))
    args.output.write_text(json.dumps(by_class(summaries), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
