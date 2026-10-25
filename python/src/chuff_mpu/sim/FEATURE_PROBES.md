# Candidate selected-feature interventions

`feature_probes.py` is an outer evaluator around a fresh provenance-traced
learner. It measures whether the feature selected from a counterfactual
window still matches the candidate's stored vector. It does not change
learning, matching, admission or historical scores. Beads ch-6w4.10.8 tracks
the investigation; ch-6w4.10.8.1 supplies this probe layer and ch-6w4.10.8.2
tracks the full observed-recording replay and interpretation.

```python
from pathlib import Path
from chuff_mpu.band_attention import BandForegroundLearner
from chuff_mpu.sim.memory_provenance import instrument
from chuff_mpu.sim.feature_probes import FeatureProbeEvaluator, InterventionScene
from chuff_mpu.sim.recorded_experiment import evaluate

observer = FeatureProbeEvaluator(instrument(BandForegroundLearner()), expected_steps)
observers = {"quiet": observer}
rows, physical = evaluate(InterventionScene(scene, observers), observers)
arrays = observer.finish(Path("new-feature-vectors.npz"))
records = observer.records
```

The scene supplies full component frames; observations remain one second.
`expected_steps` must include **every** archived candidate-observation slot
for that actor/stream, including candidates never admitted. It must not be
restricted by recording labels or successful admissions. The schedule only
avoids needless evaluator snapshots; actual waveform-only tracing must
reproduce it exactly or the run fails.

## State and intervention boundaries

The outer evaluator receives two waveform interventions: current-episode
removal (`without_current_source_audio`) and the recorded bed with **all**
injected foreground contributions removed (`background_audio`). The latter
also removes older foreground tails. The learner receives only the actual
mixture through its original `observe(audio)` method. No intervention array,
label, clean stem or probe result enters actual learner state.

At a scheduled window, a copy of the learner state is retained before actual
observation. After that observation, two independent copies of the retained
state process the intervention audio. Their updates and provenance futures
are discarded. Actual prior exposure remains in every branch; these probes
do not recreate a wholly foreground-deleted learning history.

The legacy recorded evaluator still makes its ordinary source-removal
branch. Copying this facade deliberately returns a bare learner copy,
without intervention arrays or the collector. The facade is an evaluator
adapter, not a general-purpose copyable/persistable learner.

## Measurements and interpretation

Each record includes candidate/birth identity, hit counts and stage flags,
actual and probe status/regions, exact feature hashes, norms, cosine
distances to the stored and actual feature, and region intersection-over-
union. Stored and current vectors remain distinct. The matching threshold
is the existing learner radius; no diagnostic-specific tuning is introduced.
Waveform hashes and removed-component RMS describe the delivered float32
interventions. Identical input hashes must reproduce the entire actual
observation packet exactly from the copied prior state.

`match_dependence(record, intervention)` returns one of four descriptions:

- `actual_outside_radius`: the actual selected feature does not match its
  stored vector, so it cannot support a source-required match claim.
- `probe_no_feature`: the actual match exists but the intervention selects
  no feature.
- `probe_matches_stored`: the intervention still selects a feature within
  the stored vector's ordinary matching radius.
- `probe_feature_outside_radius`: a feature is selected, but it does not
  match the stored vector.

An exact zero feature is distinct from an absent feature. Zero vectors can
have identical bytes while cosine distance is one, so an identical zero
probe must never be interpreted as positive foreground evidence. Hash
equality and within-radius matching remain separate fields.

A probe can move to a different selected region and still match the stored
vector; region overlap is reported separately. This is an intervention on
the full selected-feature pipeline, including its noise-estimator update,
not a fixed-patch attribution or proof of acoustic source identity. The bed
can itself contain semantically meaningful sounds. Clean simulation stems
provide evaluator evidence and are not a deployable admission oracle.

## Feature archives and checks

`finish()` returns four float32 matrices: `actual`, `stored`, `without_current`
and `bed`. `vector_index` links each record to its row. Absent probe features
use zero placeholders with null feature hashes; a detected zero feature has
a non-null hash and zero norm. `finish(path)` writes a compressed NPZ and
refuses an existing file.

`validate_records(records, arrays)` independently recomputes feature hashes,
norms, distances, stored matches, exact feature equality, selected-region
overlap and identical-audio controls from the archived vectors. It rejects
shape/dtype/value inconsistencies, changed measurements and nonzero absent
placeholders. The full replay auditor must additionally check original
decision, provenance and candidate-slot coverage; this helper does not
re-render audio or establish waveform-hash provenance by itself.

Tests preserve exact legacy observations, physical metadata and provenance
histories across eight frontend families on synthetic full-duration recorded
scenes and no-target controls. They also exercise source removal, older-tail
removal, region changes, absent/zero features, schedule errors, copied-branch
isolation, archive tampering and overwrite protection. These tests validate
the instrument, not real-recording foreground dependence. Empirical results
remain the separate observed replay task.
