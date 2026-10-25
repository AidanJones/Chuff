"""Waveform-only evaluator instrumentation; no truth enters learner methods.

This observes the existing private memory objects without changing selection,
confirmation, admission or stored vectors. Labels belong in a later evaluator
join on tick, never in these events. Tick 1 corresponds to replay step 0.
"""

from copy import deepcopy
from dataclasses import dataclass
import hashlib

import numpy as np


def feature_hash(vector):
    """Canonical little-endian float32 feature bytes, not an acoustic source ID."""
    return hashlib.sha256(np.asarray(vector, dtype="<f4").tobytes()).hexdigest()


@dataclass(frozen=True)
class _Event:
    tick: int
    kind: str
    candidate_id: int
    prototype_id: int | None
    fields: tuple

    def row(self):
        return dict(tick=self.tick, kind=self.kind, candidate_id=self.candidate_id,
                    prototype_id=self.prototype_id, **dict(self.fields))


@dataclass
class _Origin:
    candidate_id: int
    birth_tick: int
    stored_hash: str
    prototype_id: int | None


class _TraceMemory:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._provenance_history = ()
        self._provenance_objects = {}
        self._provenance_next_id = 0

    @property
    def provenance_events(self):
        # Expose fresh JSON-ready dictionaries; immutable internal events stay private.
        return [event.row() for event in self._provenance_history]

    @classmethod
    def load(cls, path):
        raise ValueError("saved memory does not contain candidate birth provenance; replay from a fresh actor")

    def __deepcopy__(self, memo):
        clone = type(self).__new__(type(self))
        memo[id(self)] = clone
        for key, value in self.__dict__.items():
            # Event fields are immutable scalar tuples. Branches share their past
            # and append a new tuple; they never append to the actual actor's log.
            setattr(clone, key, value if key == "_provenance_history" else deepcopy(value, memo))
        return clone

    def _provenance_emit(self, kind, origin, **fields):
        event = _Event(self._tick, kind, origin.candidate_id, origin.prototype_id,
                       tuple(sorted(fields.items())))
        self._provenance_history += (event,)

    def _provenance_sync(self):
        live = self._candidates+self._patterns
        live_set = set(live)
        for pattern, origin in list(self._provenance_objects.items()):
            if pattern not in live_set:
                self._provenance_emit("prototype_retired" if origin.prototype_id is not None else "candidate_retired",
                    origin, stored_feature_sha256=origin.stored_hash, birth_tick=origin.birth_tick)
                del self._provenance_objects[pattern]
        for pattern in live:
            current_hash = feature_hash(pattern.embedding)
            identifier = pattern.identifier if pattern.identifier >= 0 else None
            if pattern not in self._provenance_objects:
                origin = _Origin(self._provenance_next_id, self._tick, current_hash, identifier)
                self._provenance_next_id += 1
                self._provenance_objects[pattern] = origin
                self._provenance_emit("candidate_created", origin,
                    birth_tick=origin.birth_tick, stored_feature_sha256=current_hash)
            else:
                origin = self._provenance_objects[pattern]
                origin.prototype_id = identifier
                if current_hash != origin.stored_hash:
                    self._provenance_emit("stored_vector_changed", origin,
                        previous_feature_sha256=origin.stored_hash, stored_feature_sha256=current_hash,
                        birth_tick=origin.birth_tick)
                    origin.stored_hash = current_hash

    def observe(self, clip):
        observation = super().observe(clip)
        # Also catches expiration, calibration resets and mode switches that do
        # not reach _remember. No reason/source attribution is inferred here.
        self._provenance_sync()
        return observation

    def _remember(self, embedding, salience):
        self._provenance_sync()
        before = {pattern: (pattern.hits, pattern.last_seen) for pattern in self._candidates}
        observation = super()._remember(embedding, salience)
        self._provenance_sync()
        if observation.status in ("learned", "recognised"):
            touched = [pattern for pattern in self._patterns if pattern.identifier == observation.prototype_id]
        elif observation.status == "candidate":
            touched = [pattern for pattern in self._candidates if pattern.last_seen == self._tick]
        else:
            touched = []
        if len(touched) != 1:
            # Private custom memory implementations can return a candidate with
            # no memory object. Record the limitation, never guess an identity.
            self._provenance_history += (_Event(self._tick, "memory_observation_unresolved", -1,
                observation.prototype_id, (("status", observation.status), ("touched_objects", len(touched)))),)
            return observation
        pattern = touched[0]
        origin = self._provenance_objects[pattern]
        fields = dict(birth_tick=origin.birth_tick, stored_feature_sha256=feature_hash(pattern.embedding),
                      observed_feature_sha256=feature_hash(embedding))
        if observation.status == "recognised":
            self._provenance_emit("prototype_recognised", origin, **fields)
        else:
            prior_hits, prior_tick = before.get(pattern, (0, None))
            self._provenance_emit("candidate_observed", origin, hits_before=prior_hits,
                hits_after=pattern.hits, previous_observation_tick=prior_tick, **fields)
            if observation.status == "learned":
                self._provenance_emit("prototype_admitted", origin, hits=pattern.hits, **fields)
        return observation


_CLASSES = {}


def instrument(fresh_actor):
    """Construct an instrumented fresh actor with the same class/settings.

    Refuse existing state: historical candidate birth cannot be reconstructed
    from a saved prototype. This API is for evaluator replays, not live use.
    """
    if (isinstance(fresh_actor, _TraceMemory) or fresh_actor._tick
            or fresh_actor.memory_size or fresh_actor.candidate_count):
        raise ValueError("provenance instrumentation requires a fresh, uninstrumented actor")
    base = type(fresh_actor)
    if base not in _CLASSES:
        _CLASSES[base] = type("Provenance"+base.__name__, (_TraceMemory, base), {})
    return _CLASSES[base](fresh_actor.settings)
