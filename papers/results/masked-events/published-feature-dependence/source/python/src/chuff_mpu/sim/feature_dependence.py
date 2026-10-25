"""Full observed-recording selected-feature probe replay; no policy changes."""

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import gzip
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np

from chuff_mpu.sim.audit_provenance import audit as audit_provenance
from chuff_mpu.sim.feature_probes import FeatureProbeEvaluator, InterventionScene, match_dependence
from chuff_mpu.sim.memory_provenance import instrument
from chuff_mpu.sim.provenance_replay import context
from chuff_mpu.sim.published_experiment import actors
from chuff_mpu.sim.recorded_catalog import audit as audit_catalog, digest
from chuff_mpu.sim.recorded_experiment import evaluate
from chuff_mpu.sim.recorded_scene import RecordedScene, load_audio
from chuff_mpu.sim.scene import SceneConfig


KINDS = ("actual_outside_radius", "probe_no_feature", "probe_matches_stored", "probe_feature_outside_radius")
ROLES = ("actual", "stored", "without_current", "bed")


def summarize_records(records, modes):
    result = {}
    for mode in modes:
        stages = {}
        for stage in ("all", "birth", "confirmation", "adjacent", "admission"):
            counts = Counter({key: 0 for key in ("windows", "actual_zero", "stored_zero", "bed_noop", "current_noop",
                "bed_exact_feature", "bed_match_disjoint_region", "bed_match_changed_region")})
            counts.update({role+"/"+kind: 0 for role in ("bed", "without_current") for kind in KINDS})
            for row in records:
                if row["mode"] != mode:
                    continue
                included = dict(all=True, birth=row["is_birth"], confirmation=row["is_confirmation"],
                    adjacent=not row["is_birth"] and not row["is_confirmation"], admission=row["is_admission"])[stage]
                if not included:
                    continue
                packets = row["observations"]
                counts["windows"] += 1
                counts["actual_zero"] += packets["actual"]["feature_norm"] == 0
                counts["stored_zero"] += row["stored_feature_norm"] == 0
                counts["bed_noop"] += row["waveform_sha256"]["bed"] == row["waveform_sha256"]["actual"]
                counts["current_noop"] += row["waveform_sha256"]["without_current"] == row["waveform_sha256"]["actual"]
                counts["bed_exact_feature"] += packets["bed"]["exact_actual_feature"]
                for role in ("bed", "without_current"):
                    counts[role+"/"+match_dependence(row, role)] += 1
                if match_dependence(row) == "probe_matches_stored":
                    counts["bed_match_disjoint_region"] += packets["bed"]["region_iou"] == 0
                    counts["bed_match_changed_region"] += any(packets["bed"][key] != packets["actual"][key]
                        for key in ("region_start_s", "region_stop_s"))
            stages[stage] = dict(counts)
        result[mode] = stages
    return result


def admission_pairs(records, admissions):
    grouped = defaultdict(list)
    for record in records:
        grouped[record["mode"], record["candidate_id"]].append(record)
    result = []
    actual_admissions = {(row["mode"], row["candidate_id"]) for row in records if row["is_admission"]}
    expected_admissions = {(row["mode"], row["candidate_id"]) for row in admissions}
    if actual_admissions != expected_admissions or len(expected_admissions) != len(admissions):
        raise ValueError("admission feature coverage differs from provenance")
    for admission in admissions:
        selected = grouped[admission["mode"], admission["candidate_id"]]
        births = [row for row in selected if row["is_birth"]]
        admitted = [row for row in selected if row["is_admission"]]
        if len(births) != 1 or len(admitted) != 1:
            raise ValueError("admission does not have one captured birth and admission")
        birth, final = births[0], admitted[0]
        if (birth["step"] != admission["birth_step"] or final["step"] != admission["admission_step"]
                or final["prototype_id"] != admission["prototype_id"]
                or birth["observations"]["actual"]["feature_sha256"] != admission["birth_feature_sha256"]
                or final["stored_feature_sha256"] != admission["stored_feature_sha256"]
                or final["observations"]["actual"]["feature_sha256"] != admission["admission_observed_feature_sha256"]):
            raise ValueError("admission features differ from archived candidate history")
        first, last = match_dependence(birth), match_dependence(final)
        if "actual_outside_radius" in (first, last):
            joint = "unmatchable_actual"
        elif first == last == "probe_matches_stored":
            joint = "both_reproducible"
        elif first == "probe_matches_stored":
            joint = "birth_only_reproducible"
        elif last == "probe_matches_stored":
            joint = "admission_only_reproducible"
        else:
            joint = "neither_reproducible"
        result.append(dict(admission, birth_dependence=first, admission_dependence=last, joint=joint,
            birth_vector_file=birth["vector_file"], birth_vector_index=birth["vector_index"],
            admission_vector_file=final["vector_file"], admission_vector_index=final["vector_index"]))
    return result


def replay(original, provenance, catalog_folder, cache, output):
    audit_provenance(provenance)
    audit_catalog(catalog_folder, cache)
    old = json.loads((original/"manifest.json").read_text())
    prov = json.loads((provenance/"manifest.json").read_text())
    if digest((original/"manifest.json").read_bytes()) != prov["input_sha256"]["input/original-manifest.json"]:
        raise ValueError("original evaluation differs from provenance input")
    package = Path(__file__).resolve().parents[1]
    root = package.parents[2]
    for name, expected in old["source_sha256"].items():
        if digest((root/name).read_bytes()) != expected:
            raise ValueError("original source changed: "+name)
    tracer = "python/src/chuff_mpu/sim/memory_provenance.py"
    if digest((root/tracer).read_bytes()) != prov["source_sha256"][tracer]:
        raise ValueError("provenance tracer changed")
    plan = json.loads((original/"input/protocol.json").read_text())
    dependencies = {name: version(name) for name in plan["dependencies"]}
    if dependencies != plan["dependencies"]:
        raise ValueError("original dependency versions changed")
    catalog = json.loads((catalog_folder/"catalog.json").read_text())
    if digest((catalog_folder/"catalog.json").read_bytes()) != old["input_sha256"]["input/catalog.json"]:
        raise ValueError("replay catalog differs")
    names = [name for name in old["settings"] if name != "adaptive_gain"]
    if {name: asdict(model.settings) for name, model in actors(plan, names).items()} != old["settings"]:
        raise ValueError("replay actor settings differ")
    audio = load_audio(catalog["clips"], cache)
    with gzip.open(original/"observations.jsonl.gz", "rt", encoding="utf-8") as stream:
        gold_rows = [json.loads(line) for line in stream]
    rows_by_run = defaultdict(list)
    for row in gold_rows:
        rows_by_run[row["suite"], row["seed"]].append(row)
    trace_by_run = defaultdict(list)
    with gzip.open(provenance/"trace.jsonl.gz", "rt", encoding="utf-8") as stream:
        for line in stream:
            event = json.loads(line)
            trace_by_run[event["suite"], event["seed"]].append({key: value for key, value in event.items()
                if key not in ("seed", "suite", "window_context")})
    admissions_by_run = defaultdict(list)
    for row in json.loads((provenance/"admissions.json").read_text()):
        admissions_by_run[row["suite"], row["seed"]].append(row)
    original_summary = {(row["suite"], row["seed"]): row for row in json.loads((original/"summary.json").read_text())}
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(schema="chuff.recorded.feature-dependence.v1", phase="observed-data-diagnostic",
        scope="Selected-feature intervention under actual prior exposure; not fixed-patch causality, semantic relevance or a new policy",
        source_sha256={}, input_sha256={}, output_sha256={}, vectors={}, settings=old["settings"], dependencies=dependencies,
        verified_actual_decisions=0, feature_probe_windows=0, source_removal_probes=0)
    for source in (*package.rglob("*.py"), root/"python/uv.lock"):
        name = source.relative_to(root).as_posix()
        content = source.read_bytes()
        destination = output/"source"/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
        manifest["source_sha256"][name] = digest(content)
    inputs = {"original-manifest.json": original/"manifest.json", "observations.jsonl.gz": original/"observations.jsonl.gz",
        "original-summary.json": original/"summary.json", "protocol.json": original/"input/protocol.json",
        "catalog.json": original/"input/catalog.json", "LICENSE": original/"input/LICENSE",
        "provenance-manifest.json": provenance/"manifest.json", "trace.jsonl.gz": provenance/"trace.jsonl.gz",
        "admissions.json": provenance/"admissions.json"}
    for name, source in inputs.items():
        content = source.read_bytes()
        destination = output/"input"/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
        manifest["input_sha256"]["input/"+name] = digest(content)
    summaries, pairs = [], []
    with gzip.open(output/"probes.jsonl.gz", "wt", encoding="utf-8") as stream:
        for suite in plan["positive_suites"]+plan["control_suites"]:
            condition, snr, kind = suite.split("/")
            for seed in plan["evaluation_seeds"]:
                key = suite, seed
                expected_events = trace_by_run[key]
                slots = {mode: [event["tick"]-1 for event in expected_events
                    if event["mode"] == mode and event["kind"] == "candidate_observed"] for mode in old["settings"]}
                evaluators = {mode: FeatureProbeEvaluator(instrument(model), slots[mode])
                              for mode, model in actors(plan, names).items()}
                scene = RecordedScene(seed, SceneConfig(**plan["scene"], snr_db=float(snr)), environment_seed=200000+seed,
                    audio=audio, foregrounds=tuple(old["foregrounds"]), backgrounds=tuple(old["backgrounds"]),
                    modulation=condition == "modulated", control=kind == "control")
                rows, physical = evaluate(InterventionScene(scene, evaluators), evaluators)
                gold = {(row["mode"], row["step"]): row for row in rows_by_run[key]}
                for row in rows:
                    row.update(seed=seed, suite=suite)
                    if row != {name: value for name, value in gold[row["mode"], row["step"]].items() if name != "observe_ns"}:
                        raise ValueError(f"actual/probe trajectory changed: {suite}/{seed}/{row['mode']}/{row['step']}")
                if len(rows) != len(gold) or any(value != original_summary[key][name] for name, value in physical.items()):
                    raise ValueError("original decision matrix or physical replay changed")
                actual_events = [dict(event, mode=mode) for mode, evaluator in evaluators.items()
                                 for event in evaluator.model.provenance_events]
                if actual_events != expected_events:
                    raise ValueError("original provenance history changed")
                vector_name = f"vectors/{suite.replace('/', '--')}/{seed}.npz"
                vector_path = output/vector_name
                vector_path.parent.mkdir(parents=True, exist_ok=True)
                arrays, records = {}, []
                for mode, evaluator in evaluators.items():
                    arrays.update({mode+"__"+role: values for role, values in evaluator.finish().items()})
                    for record in evaluator.records:
                        reference = gold[mode, record["step"]]
                        if "without_episode_probe" in reference:
                            probe = record["observations"]["without_current"]
                            if {name: probe[name] for name in ("status", "prototype_id")} != reference["without_episode_probe"]:
                                raise ValueError("feature source-removal probe differs from original")
                        record.update(mode=mode, suite=suite, seed=seed, vector_file=vector_name, window_context=context(reference))
                        records.append(record)
                        stream.write(json.dumps(record, allow_nan=False)+"\n")
                with vector_path.open("xb") as handle:
                    np.savez_compressed(handle, **arrays)
                manifest["vectors"][vector_name] = digest(vector_path.read_bytes())
                pairs.extend(admission_pairs(records, admissions_by_run[key]))
                summaries.append(dict(suite=suite, seed=seed, verified_decisions=len(rows), probe_windows=len(records),
                    counts=summarize_records(records, old["settings"]), **physical))
                manifest["verified_actual_decisions"] += len(rows)
                manifest["source_removal_probes"] += sum("without_episode_probe" in row for row in rows)
                manifest["feature_probe_windows"] += len(records)
                print(f"{suite} seed={seed}: {len(rows)} decisions identical; {len(records)} candidate feature windows", flush=True)
    for name, content in (("summary.json", summaries), ("admission-pairs.json", pairs)):
        (output/name).write_text(json.dumps(content, indent=2, allow_nan=False), encoding="utf-8")
    manifest["output_sha256"] = {name: digest((output/name).read_bytes()) for name in ("probes.jsonl.gz", "summary.json", "admission-pairs.json")}
    (output/"manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("original", "provenance", "catalog", "cache", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    replay(args.original, args.provenance, args.catalog, args.cache, args.output)


if __name__ == "__main__":
    main()
