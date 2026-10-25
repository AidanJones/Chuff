"""Observed-data replay and birth/admission label diagnostics; no learner retuning."""

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import gzip
import json
from pathlib import Path
import re

import numpy as np

from chuff_mpu.sim.audit_published import audit as audit_original
from chuff_mpu.sim.memory_provenance import instrument
from chuff_mpu.sim.provenance_metrics import provenance_episode_metrics
from chuff_mpu.sim.published_experiment import actors
from chuff_mpu.sim.recorded_catalog import audit as audit_catalog, digest
from chuff_mpu.sim.recorded_experiment import evaluate
from chuff_mpu.sim.recorded_metrics import aggregate, episode_metrics
from chuff_mpu.sim.recorded_scene import RecordedScene, load_audio
from chuff_mpu.sim.scene import SceneConfig


TRANSITIONS = ("same_recording", "foreground_to_background", "background_to_foreground",
               "different_recording", "background_to_background")


def context(row):
    return {key: row[key] for key in ("recording_id", "episode_id", "occurrence",
        "previous_source_tail_s", "current_episode_window_rms")}


def join_trace(rows, events):
    """Validate lifecycle/coverage, then join truth outside the waveform-only actor."""
    indexed = {(row["mode"], row["step"]): row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError("duplicate observation keys")
    origins, admissions, joined = {}, [], []
    observed, recognised, admitted, last_ticks = set(), set(), set(), {}
    counts = {mode: Counter({key: 0 for key in (*TRANSITIONS, "admissions", "birth_with_room_tail",
        "background_birth_with_room_tail", "bed_only_birth", "stored_hash_changed_since_birth",
        "stored_differs_from_admission_observation")}) for mode, _ in indexed}
    for event in events:
        mode, tick, kind = event["mode"], event["tick"], event["kind"]
        if tick < last_ticks.get(mode, 0) or (mode, tick-1) not in indexed:
            raise ValueError("invalid trace tick/order")
        last_ticks[mode] = tick
        row = indexed[mode, tick-1]
        key = mode, event["candidate_id"]
        if kind == "memory_observation_unresolved":
            raise ValueError("unresolved memory observation cannot support a provenance audit")
        for name in ("stored_feature_sha256", "observed_feature_sha256", "previous_feature_sha256"):
            if name in event and not re.fullmatch(r"[0-9a-f]{64}", event[name]):
                raise ValueError("invalid feature hash")
        if kind == "candidate_created":
            if key in origins or event["birth_tick"] != tick:
                raise ValueError("duplicate candidate or invalid birth tick")
            origins[key] = dict(birth_tick=tick, birth_hash=event["stored_feature_sha256"],
                stored_hash=event["stored_feature_sha256"], hits=0, previous_tick=None,
                alive=True, prototype_id=None)
        elif key not in origins or not origins[key]["alive"]:
            raise ValueError("trace refers to unknown/retired candidate")
        origin = origins[key]
        if event.get("birth_tick", origin["birth_tick"]) != origin["birth_tick"]:
            raise ValueError("candidate birth changed")
        if kind == "stored_vector_changed":
            if event["previous_feature_sha256"] != origin["stored_hash"]:
                raise ValueError("vector-change predecessor mismatch")
            origin["stored_hash"] = event["stored_feature_sha256"]
        elif event.get("stored_feature_sha256") != origin["stored_hash"]:
            raise ValueError("stored feature differs without an explicit vector-change event")
        if kind == "candidate_observed":
            if origin["prototype_id"] is not None:
                raise ValueError("admitted prototype reported as pending")
            previous = origin["previous_tick"]
            increment = previous is None or tick-previous >= 2
            if (event["hits_before"] != origin["hits"] or event["previous_observation_tick"] != previous
                    or event["hits_after"] != origin["hits"]+increment
                    or row["status"] not in ("candidate", "learned") or (mode, tick) in observed):
                raise ValueError("candidate observation/confirmation mismatch")
            origin.update(hits=event["hits_after"], previous_tick=tick)
            observed.add((mode, tick))
        elif kind == "prototype_admitted":
            identifier = event["prototype_id"]
            if (origin["prototype_id"] is not None or origin["previous_tick"] != tick
                    or event["hits"] != origin["hits"] or row["status"] != "learned"
                    or row["prototype_id"] != identifier or (mode, identifier) in admitted):
                raise ValueError("admission differs from memory/observation evidence")
            origin["prototype_id"] = identifier
            admitted.add((mode, identifier))
            birth_row = indexed[mode, origin["birth_tick"]-1]
            birth, admission = birth_row["recording_id"], row["recording_id"]
            transition = ("background_to_background" if birth is None and admission is None else
                "background_to_foreground" if birth is None else
                "foreground_to_background" if admission is None else
                "same_recording" if birth == admission else "different_recording")
            item = dict(mode=mode, candidate_id=event["candidate_id"], prototype_id=identifier,
                birth_step=origin["birth_tick"]-1, admission_step=tick-1,
                birth_context=context(birth_row), admission_context=context(row), transition=transition,
                birth_feature_sha256=origin["birth_hash"], stored_feature_sha256=origin["stored_hash"],
                admission_observed_feature_sha256=event["observed_feature_sha256"])
            admissions.append(item)
            tally = counts[mode]
            tally.update(admissions=1, **{transition: 1})
            tally["birth_with_room_tail"] += birth_row["previous_source_tail_s"] > 0
            tally["background_birth_with_room_tail"] += birth is None and birth_row["previous_source_tail_s"] > 0
            tally["bed_only_birth"] += birth is None and birth_row["previous_source_tail_s"] <= 0
            tally["stored_hash_changed_since_birth"] += origin["birth_hash"] != origin["stored_hash"]
            tally["stored_differs_from_admission_observation"] += origin["stored_hash"] != event["observed_feature_sha256"]
        elif kind == "prototype_recognised":
            if (origin["prototype_id"] != event["prototype_id"] or row["status"] != "recognised"
                    or row["prototype_id"] != event["prototype_id"] or (mode, tick) in recognised):
                raise ValueError("recognition differs from live prototype/observation")
            recognised.add((mode, tick))
        elif kind in ("candidate_retired", "prototype_retired"):
            if (kind == "prototype_retired") != (origin["prototype_id"] is not None):
                raise ValueError("retirement kind mismatch")
            origin["alive"] = False
        elif kind not in ("candidate_created", "stored_vector_changed"):
            raise ValueError("unknown trace event")
        joined.append(dict(event, window_context=context(row)))
    expected_observed = {(r["mode"], r["step"]+1) for r in rows if r["status"] in ("candidate", "learned")}
    expected_recognised = {(r["mode"], r["step"]+1) for r in rows if r["status"] == "recognised"}
    expected_admitted = {(r["mode"], r["prototype_id"]) for r in rows if r["status"] == "learned"}
    if observed != expected_observed or recognised != expected_recognised or admitted != expected_admitted:
        raise ValueError("incomplete trace coverage")
    return joined, admissions, {mode: dict(value) for mode, value in counts.items()}


def score(rows, admissions):
    admission_labels = {(r["mode"], r["prototype_id"]): r["admission_context"]["recording_id"] for r in admissions}
    birth_labels = {(r["mode"], r["prototype_id"]): r["birth_context"]["recording_id"] for r in admissions}
    original = episode_metrics(rows, details=True, exclusion=True)
    if provenance_episode_metrics(rows, labels=admission_labels, details=True, exclusion=True) != original:
        raise ValueError("explicit-label scorer does not reproduce original admission scores")
    return dict(admission=original,
                birth=provenance_episode_metrics(rows, labels=birth_labels, details=True, exclusion=True))


def analysis(summaries):
    return {label: aggregate([dict(seed=row["seed"], suite=row["suite"], episodes=row["scores"][label])
                             for row in summaries]) for label in ("admission", "birth")}


def replay(original, catalog_folder, cache, output):
    audit_original(original)
    audit_catalog(catalog_folder, cache)
    original_manifest = json.loads((original/"manifest.json").read_text())
    if original_manifest["phase"] != "evaluation":
        raise ValueError("this diagnostic requires the frozen published evaluation")
    plan = json.loads((original/"input/protocol.json").read_text())
    catalog = json.loads((catalog_folder/"catalog.json").read_text())
    if digest((catalog_folder/"catalog.json").read_bytes()) != original_manifest["input_sha256"]["input/catalog.json"]:
        raise ValueError("replay input catalog differs from original")
    package = Path(__file__).resolve().parents[1]
    root = package.parents[2]
    for name, expected in plan["method_source_sha256"].items():
        if digest((package/name).read_bytes()) != expected:
            raise ValueError("original method/scorer source changed: "+name)
    if {key: asdict(value.settings) for key, value in actors(plan,
            [name for name in original_manifest["settings"] if name != "adaptive_gain"]).items()} != original_manifest["settings"]:
        raise ValueError("replay settings mismatch")
    audio = load_audio(catalog["clips"], cache)
    with gzip.open(original/"observations.jsonl.gz", "rt", encoding="utf-8") as stream:
        gold_rows = [json.loads(line) for line in stream]
    grouped = defaultdict(list)
    for row in gold_rows:
        grouped[row["suite"], row["seed"]].append(row)
    gold_summary = {(r["suite"], r["seed"]): r for r in json.loads((original/"summary.json").read_text())}
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(schema="chuff.recorded.provenance-replay.v1", phase="observed-data-diagnostic",
        scope="Birth/admission window labels, not causal feature attribution; original scores retained; no retuning",
        source_sha256={}, input_sha256={}, output_sha256={}, verified_actual_decisions=0,
        original_phase=original_manifest["phase"], settings=original_manifest["settings"])
    for source in (*package.rglob("*.py"), root/"python/uv.lock"):
        name = source.relative_to(root).as_posix()
        destination = output/"source"/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
        manifest["source_sha256"][name] = digest(source.read_bytes())
    inputs = {"original-manifest.json": original/"manifest.json", "summary.json": original/"summary.json",
        "observations.jsonl.gz": original/"observations.jsonl.gz", "protocol.json": original/"input/protocol.json",
        "catalog.json": original/"input/catalog.json", "LICENSE": original/"input/LICENSE"}
    for name, source in inputs.items():
        content = source.read_bytes()
        destination = output/"input"/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
        manifest["input_sha256"]["input/"+name] = digest(content)
    summaries, all_admissions = [], []
    names = [name for name in original_manifest["settings"] if name != "adaptive_gain"]
    with gzip.open(output/"trace.jsonl.gz", "wt", encoding="utf-8") as stream:
        for suite in plan["positive_suites"]+plan["control_suites"]:
            condition, snr, kind = suite.split("/")
            for seed in plan["evaluation_seeds"]:
                models = {name: instrument(model) for name, model in actors(plan, names).items()}
                scene = RecordedScene(seed, SceneConfig(**plan["scene"], snr_db=float(snr)),
                    environment_seed=200000+seed, audio=audio, foregrounds=tuple(original_manifest["foregrounds"]),
                    backgrounds=tuple(original_manifest["backgrounds"]), modulation=condition == "modulated", control=kind == "control")
                rows, physical = evaluate(scene, models)
                gold = {(r["mode"], r["step"]): r for r in grouped[suite, seed]}
                for row in rows:
                    row.update(seed=seed, suite=suite)
                    expected = {key: value for key, value in gold[row["mode"], row["step"]].items() if key != "observe_ns"}
                    if row != expected:
                        raise ValueError(f"trajectory differs: {suite}/{seed}/{row['mode']}/{row['step']}")
                if len(rows) != len(gold):
                    raise ValueError("replay matrix mismatch")
                for key, value in physical.items():
                    if value != gold_summary[suite, seed][key]:
                        raise ValueError("physical replay differs: "+key)
                events = [dict(event, mode=name) for name, model in models.items() for event in model.provenance_events]
                joined, admissions, transitions = join_trace(rows, events)
                scores = score(rows, admissions)
                if scores["admission"] != gold_summary[suite, seed]["episodes"]:
                    raise ValueError("historical scores differ")
                for event in joined:
                    stream.write(json.dumps(dict(event, seed=seed, suite=suite), allow_nan=False)+"\n")
                all_admissions.extend(dict(item, seed=seed, suite=suite) for item in admissions)
                summaries.append(dict(seed=seed, suite=suite, scores=scores, transitions=transitions,
                    trace_events=len(joined), verified_decisions=len(rows), **physical))
                manifest["verified_actual_decisions"] += len(rows)
                print(f"{suite} seed={seed}: {len(rows)} decisions identical, {len(admissions)} admissions traced", flush=True)
    outputs = dict(summary=summaries, admissions=all_admissions, analysis=analysis(summaries))
    for name, data in outputs.items():
        (output/(name+".json")).write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
    manifest["output_sha256"] = {name: digest((output/name).read_bytes())
        for name in ("trace.jsonl.gz", *(name+".json" for name in outputs))}
    (output/"manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("original", "catalog", "cache", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    replay(args.original, args.catalog, args.cache, args.output)


if __name__ == "__main__":
    main()
