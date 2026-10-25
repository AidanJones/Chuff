# Candidate and stored-vector provenance

`memory_provenance.py` instruments fresh learner instances for evaluator
replays. It observes existing memory objects around their original methods;
it does not change detection, matching, confirmations, admission policy or
feature vectors. No recording labels, episode IDs, clean stems or truth
metadata enter the actor. Beads ch-6w4.10.7 tracks the broader scoring audit.

```python
from chuff_mpu.band_attention import BandForegroundLearner
from chuff_mpu.sim.memory_provenance import instrument

actor = instrument(BandForegroundLearner())
for clip in clips:
    observation = actor.observe(clip)
events = actor.provenance_events
```

Tick 1 corresponds to evaluator step 0. The evaluator can join tick and
birth tick to its own recording/episode context **after** observing audio.
Candidate IDs are deterministic, monotonic and local to one actor/stream;
they are distinct from prototype IDs and are never inferred from feature
hashes or Python object addresses. Different candidates can have identical
feature bytes.

Events record candidate creation, observations with before/after hit counts
and previous observation tick, prototype admission, recognition, retirement
and changes to a stored vector. Birth tick survives confirmation and
admission. Retirement means the object left live memory; the tracer does
not guess whether expiration, capacity, calibration or a mode switch caused
it. Candidate observations are not automatically counted as confirmations:
adjacent windows can leave the core hit count unchanged.

Stored and current-observation feature hashes are separate fields. Hashes
use canonical little-endian float32 bytes. The core may admit a candidate's
old stored vector while returning a current-window embedding; the two hashes
can therefore differ at admission. A hash identifies feature bytes, not the
physical source responsible for them. An explicitly recorded vector change
also prevents silently treating later feature contents as the original
birth vector.

Internal event history is an immutable tuple of frozen scalar events.
`deepcopy` preserves the learner's ordinary state and its candidate-origin
mapping while sharing the immutable history prefix. A probe appends its own
new history, leaving the actual actor's log and memory unchanged. Exported
event dictionaries are fresh copies. Branch histories have their own future
and must not be merged into the actual stream's trace. History grows with
the replay; this is evaluator instrumentation, not a bounded live logger.

Instrumentation rejects actors with prior ticks or memory and rejects
already instrumented actors. Loading a saved memory through the traced class
is also rejected: existing persistence does not contain candidate birth
history, so a historical origin cannot be reconstructed honestly.

Tests cover stored-versus-observed vectors, confirmation gaps, immediate
admission, feature ownership, capacity retirement, TTL expiration, pending
resets, stored-vector changes, invalid clips and copied-branch isolation.
They compare exact actual decisions and candidate/prototype contents over
65 windows for original, quiet-band, fixed spectral-gain, calibrated gain,
adaptive gain, guarded gain, PCEN and MCRA families, with source-removal
branch comparisons. All existing learner source files remain unchanged.

The trace alone does not establish acoustic causality or revise reported
accuracy. A birth-window recording label still does not prove that its
foreground caused the stored embedding. The completed observed-recording
replay in `provenance_replay.py` matches the original 126,000 decisions and
40,000 probes, then joins labels outside actors. `audit_provenance.py`
validates lifecycle coverage, contexts and both scoring views;
`provenance_report.py` derives paired-seed label sensitivity. The diagnostic
scorer in `provenance_metrics.py` is separate from the unchanged historical
scorer and must reproduce it given admission labels in every stream.

Paper 09 and Beads ch-6w4.10.7.2 document the completed audit. Historical
scores and frozen archives remain available alongside birth-label
diagnostics. Candidate feature dependence at birth/confirmation is a
different question, tracked in ch-6w4.10.8; window labels alone do not answer it.
