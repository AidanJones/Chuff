"""Outer evaluator for feature interventions; stems never enter the learner.

The legacy evaluator can still copy this wrapper for its ordinary source-
removal probe: that copy is a bare learner with the same prior state. Extra
feature probes run only at all pre-recorded candidate-observation slots and
are checked against the actual waveform-only trace, without selecting labels.
"""

from copy import deepcopy
import hashlib
from pathlib import Path
import re

import numpy as np

from chuff_mpu.attention import Observation, _distance
from chuff_mpu.sim.memory_provenance import feature_hash


def waveform_hash(audio):
    return hashlib.sha256(np.asarray(audio, dtype="<f4").tobytes()).hexdigest()


def feature_packet(observation, stored, actual, radius):
    vector = observation.embedding
    packet = dict(status=observation.status, prototype_id=observation.prototype_id,
        salience=observation.salience, novelty=observation.novelty,
        region_start_s=observation.region_start_s, region_stop_s=observation.region_stop_s,
        feature_sha256=None, feature_norm=None, distance_to_stored=None, distance_to_actual=None,
        matches_stored=False, exact_actual_feature=False, region_iou=None)
    if vector is not None:
        vector = np.asarray(vector, dtype=np.float32)
        if vector.shape != stored.shape or not np.isfinite(vector).all():
            raise ValueError("invalid probe feature shape or values")
        packet.update(feature_sha256=feature_hash(vector), feature_norm=float(np.linalg.norm(vector.astype(float))),
            distance_to_stored=_distance(vector, stored), distance_to_actual=_distance(vector, actual))
        packet["matches_stored"] = packet["distance_to_stored"] < radius
        packet["exact_actual_feature"] = packet["feature_sha256"] == feature_hash(actual)
    return packet


def region_iou(first, second):
    if any(packet["region_start_s"] is None or packet["region_stop_s"] is None for packet in (first, second)):
        return None
    start = max(first["region_start_s"], second["region_start_s"])
    stop = min(first["region_stop_s"], second["region_stop_s"])
    union = max(first["region_stop_s"], second["region_stop_s"])-min(first["region_start_s"], second["region_start_s"])
    return max(0., stop-start)/union if union > 0 else None


def match_dependence(record, intervention="bed"):
    """Selected-feature comparison, not acoustic identity or semantic causality."""
    if intervention not in ("bed", "without_current"):
        raise ValueError("unknown feature intervention")
    if not record["observations"]["actual"]["matches_stored"]:
        return "actual_outside_radius"
    probe = record["observations"][intervention]
    if probe["feature_sha256"] is None:
        return "probe_no_feature"
    return "probe_matches_stored" if probe["matches_stored"] else "probe_feature_outside_radius"


def validate_records(records, arrays):
    """Recompute vector hashes, norms, distances, region comparisons and no-ops."""
    if set(arrays) != {"actual", "stored", "without_current", "bed"}:
        raise ValueError("unexpected feature arrays")
    shape = arrays["actual"].shape
    if len(shape) != 2 or shape[0] != len(records) or shape[1] < 1:
        raise ValueError("invalid feature array shape")
    if any(array.shape != shape or array.dtype != np.float32 or not np.isfinite(array).all() for array in arrays.values()):
        raise ValueError("inconsistent feature array shape, dtype or values")
    previous_step = -1
    for index, record in enumerate(records):
        if record["vector_index"] != index or record["step"] <= previous_step:
            raise ValueError("feature row index/order mismatch")
        previous_step = record["step"]
        actual, stored = arrays["actual"][index], arrays["stored"][index]
        if (feature_hash(stored) != record["stored_feature_sha256"]
                or float(np.linalg.norm(stored.astype(float))) != record["stored_feature_norm"]
                or not 0 < record["match_distance"] < 2):
            raise ValueError("invalid stored feature or matching radius")
        packets = {}
        for name in ("actual", "without_current", "bed"):
            old = record["observations"][name]
            vector = arrays[name][index] if old["feature_sha256"] is not None else None
            if vector is None and np.any(arrays[name][index]):
                raise ValueError("absent probe feature has nonzero placeholder")
            if (old["status"] in ("candidate", "learned", "recognised")) != (vector is not None):
                raise ValueError("feature availability differs from detector status")
            observation = Observation(old["status"], old["salience"], old["novelty"], old["prototype_id"],
                                      vector, old["region_start_s"], old["region_stop_s"])
            packets[name] = feature_packet(observation, stored, actual, record["match_distance"])
        for name in packets:
            packets[name]["region_iou"] = region_iou(packets["actual"], packets[name])
        if packets != record["observations"] or packets["actual"]["feature_sha256"] is None:
            raise ValueError("feature packet differs from archived vectors")
        hashes = record["waveform_sha256"]
        if set(hashes) != set(packets) or any(not re.fullmatch(r"[0-9a-f]{64}", value) for value in hashes.values()):
            raise ValueError("invalid waveform hashes")
        for name, metric in (("without_current", "removed_current_rms"), ("bed", "removed_foreground_rms")):
            if not np.isfinite(record[metric]) or record[metric] < 0:
                raise ValueError("invalid removed-component RMS")
            if hashes[name] == hashes["actual"] and (packets[name] != packets["actual"] or record[metric] != 0):
                raise ValueError("identical-audio control differs from actual")
    return True


class FeatureProbeEvaluator:
    """One-second observer facade around a fresh provenance-instrumented model."""
    def __init__(self, model, expected_steps):
        if (not hasattr(model, "_provenance_history") or model._tick
                or model.memory_size or model.candidate_count or model._provenance_history):
            raise ValueError("a fresh provenance-instrumented learner is required")
        expected_steps = tuple(expected_steps)
        if (any(type(step) is not int or step < 0 for step in expected_steps)
                or len(set(expected_steps)) != len(expected_steps)):
            raise ValueError("nonnegative integer probe steps required")
        self.model = model
        self.expected_steps = set(expected_steps)
        self.visited = set()
        self.step = 0
        self.records = []
        self.vectors = {key: [] for key in ("actual", "stored", "without_current", "bed")}
        self._without_current = self._bed = None
        self._offset = 0

    @property
    def settings(self):
        return self.model.settings

    @property
    def memory_size(self):
        return self.model.memory_size

    def __deepcopy__(self, memo):
        # The legacy evaluator's probe gets no intervention arrays or collector.
        return deepcopy(self.model, memo)

    def set_interventions(self, without_current, bed):
        without_current, bed = np.asarray(without_current), np.asarray(bed)
        if (without_current.ndim != 1 or without_current.shape != bed.shape
                or not len(bed) or len(bed) % self.settings.sample_rate
                or not np.isfinite(without_current).all() or not np.isfinite(bed).all()):
            raise ValueError("finite equal-length whole-second mono interventions required")
        if self._bed is not None and self._offset != len(self._bed):
            raise ValueError("previous intervention frame was not fully consumed")
        self._without_current, self._bed, self._offset = without_current, bed, 0

    def observe(self, audio):
        n = self.settings.sample_rate
        if self._bed is None or self._offset+n > len(self._bed) or len(audio) != n:
            raise ValueError("set intervention frame before each one-second observation")
        expected = self.step in self.expected_steps
        before = deepcopy(self.model) if expected else None
        history = len(self.model._provenance_history)
        observation = self.model.observe(audio)
        new_events = self.model._provenance_history[history:]
        observed = [event.row() for event in new_events if event.kind == "candidate_observed"]
        if len(observed) != int(expected):
            raise ValueError("actual candidate-observation slots differ from archived schedule")
        if expected:
            event = observed[0]
            stored = next((pattern.embedding.copy() for pattern, origin in self.model._provenance_objects.items()
                           if origin.candidate_id == event["candidate_id"]), None)
            if stored is None or feature_hash(stored) != event["stored_feature_sha256"]:
                raise ValueError("stored candidate does not match trace")
            actual = np.asarray(observation.embedding, dtype=np.float32).copy()
            if feature_hash(actual) != event["observed_feature_sha256"]:
                raise ValueError("current feature does not match trace")
            section = slice(self._offset, self._offset+n)
            without_current, bed = self._without_current[section], self._bed[section]
            probes = {"actual": observation,
                      "without_current": deepcopy(before).observe(without_current),
                      "bed": deepcopy(before).observe(bed)}
            packets = {name: feature_packet(probe, stored, actual, self.settings.match_distance)
                       for name, probe in probes.items()}
            for name in packets:
                packets[name]["region_iou"] = region_iou(packets["actual"], packets[name])
            hashes = {"actual": waveform_hash(audio), "without_current": waveform_hash(without_current), "bed": waveform_hash(bed)}
            for name in ("without_current", "bed"):
                if hashes[name] == hashes["actual"] and packets[name] != packets["actual"]:
                    raise ValueError("identical-audio probe did not reproduce the actual observation")
            record = dict(step=self.step, vector_index=len(self.records), candidate_id=event["candidate_id"],
                birth_tick=event["birth_tick"], prototype_id=event["prototype_id"],
                hits_before=event["hits_before"], hits_after=event["hits_after"],
                previous_observation_tick=event["previous_observation_tick"],
                is_birth=event["hits_before"] == 0, is_confirmation=event["hits_before"] > 0 and event["hits_after"] > event["hits_before"],
                is_admission=observation.status == "learned", stored_feature_sha256=feature_hash(stored),
                stored_feature_norm=float(np.linalg.norm(stored.astype(float))), match_distance=self.settings.match_distance,
                waveform_sha256=hashes,
                removed_current_rms=float(np.sqrt(np.mean((np.asarray(audio, dtype=float)-without_current)**2))),
                removed_foreground_rms=float(np.sqrt(np.mean((np.asarray(audio, dtype=float)-bed)**2))),
                observations=packets)
            self.records.append(record)
            self.vectors["stored"].append(stored.astype(np.float32))
            for name, probe in probes.items():
                self.vectors[name].append(np.zeros_like(actual) if probe.embedding is None else probe.embedding.astype(np.float32).copy())
            self.visited.add(self.step)
        self.step += 1
        self._offset += n
        return observation

    def finish(self, path: Path | None = None):
        if self.visited != self.expected_steps or self._bed is None or self._offset != len(self._bed):
            raise ValueError("incomplete intervention/probe schedule")
        arrays = {name: np.stack(values) if values else np.empty((0, self.settings.bands*3), dtype=np.float32)
                  for name, values in self.vectors.items()}
        validate_records(self.records, arrays)
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                np.savez_compressed(stream, **arrays)
        return arrays


class InterventionScene:
    def __init__(self, scene, evaluators):
        self.scene, self.evaluators = scene, evaluators
        self.settings = scene.settings

    def __iter__(self):
        for frame in self.scene:
            for evaluator in self.evaluators.values():
                evaluator.set_interventions(frame.without_current_source_audio, frame.background_audio)
            yield frame
