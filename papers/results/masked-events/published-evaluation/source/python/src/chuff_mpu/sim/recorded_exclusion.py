"""Secondary stricter episode exclusion, retaining the prespecified window metric."""

import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path

from chuff_mpu.sim.recorded_metrics import aggregate, episode_metrics


def derive(folder):
    with gzip.open(folder/"observations.jsonl.gz", "rt", encoding="utf-8") as stream:
        grouped = defaultdict(list)
        for line in stream:
            row = json.loads(line)
            grouped[row["suite"], row["seed"]].append(row)
    summaries = [dict(suite=suite, seed=seed, episodes=episode_metrics(rows, details=True, exclusion=True))
                 for (suite, seed), rows in sorted(grouped.items())]
    return summaries, aggregate(summaries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    summary, analysis = derive(args.folder)
    inputs = {name: hashlib.sha256((args.folder/name).read_bytes()).hexdigest()
              for name in ("manifest.json", "observations.jsonl.gz")}
    if args.audit:
        manifest = json.loads((args.output/"manifest.json").read_text())
        assert inputs == manifest["input_sha256"]
        for name, digest in manifest["source_sha256"].items():
            assert hashlib.sha256((args.output/"source"/name).read_bytes()).hexdigest() == digest
        for name, digest in manifest["output_sha256"].items():
            assert hashlib.sha256((args.output/name).read_bytes()).hexdigest() == digest
        assert json.loads((args.output/"summary.json").read_text()) == summary
        assert json.loads((args.output/"analysis.json").read_text()) == analysis
        print(f"{len(summary)} stream summaries: strict episode exclusion, original endpoint, hashes and seed estimates verified")
        return
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = dict(input_sha256=inputs, source_sha256={}, output_sha256={},
        scope="Secondary diagnostic developed after development evidence; no replacement of prespecified primary; no learner tuning",
        definition="Eligible episode has prior-prototype correct recognition and NO current-recording recognition in any of its per-window source-deleted probes",
        caveat="Probe priors retain actual earlier learner exposure; different window probes are independent discarded branches, not one all-episode counterfactual trajectory")
    for name in ("recorded_metrics.py", "recorded_exclusion.py"):
        content = Path(__file__).with_name(name).read_bytes()
        destination = args.output/"source"/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
        manifest["source_sha256"][name] = hashlib.sha256(content).hexdigest()
    for name, value in (("summary.json", summary), ("analysis.json", analysis)):
        (args.output/name).write_text(json.dumps(value, indent=2), encoding="utf-8")
        manifest["output_sha256"][name] = hashlib.sha256((args.output/name).read_bytes()).hexdigest()
    (args.output/"manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Derived stricter source exclusion for {len(summary)} streams")


if __name__ == "__main__":
    main()
