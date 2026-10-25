# Observed candidate-provenance replay

This archive diagnoses the existing published-frontend evaluation. It is
observed-data analysis, not a new held-out test or a new admission policy.
See [paper 09](../../../09-candidate-birth-and-admission-labels.md).

The replay checked all 126,000 actual output rows and 40,000 source-removal
decisions against the original archive, omitting only the recorded runtime
field. Physical episode metadata, component error and background hashes also
matched. The frozen classifier, scene and historical scorer bytes were
checked before replay. No original learner or scorer file was changed.

`trace.jsonl.gz` contains 121,767 lifecycle events with evaluator-joined
window context. `admissions.json` describes all 5,940 admissions, including
birth/admission contexts, tail metadata and stored/current feature hashes.
`summary.json` retains both admission-label and birth-label metrics per
stream and recording, with transition counts. `analysis.json` preserves
suite-specific estimates for both views. The explicit-label scorer must
exactly reproduce original admission scores in every stream.

From `python/`, audit offline:

```powershell
uv run python -m chuff_mpu.sim.audit_provenance ../papers/results/masked-events/published-provenance
```

This audit checks source/input/output hashes, original raw-data links,
complete matrix coverage, lifecycle IDs/hits/hashes, contextual joins,
admissions and both label-score derivations. It does not render waveforms;
waveform trajectory equality was asserted during the replay itself.
To repeat the waveform replay into a **new** output directory:

```powershell
uv run python -m chuff_mpu.sim.provenance_replay --original ../papers/results/masked-events/published-evaluation --catalog ../papers/data/esc10-published-evaluation --cache ../data/cache/esc10 --output ../papers/results/masked-events/published-provenance-reproduction
```

Fetch the pinned evaluation catalog first if the ignored audio cache is
absent. The source snapshot and dependency lock preserve replay code. Exact
method hashes include line endings; use the archived source tree when a
fresh checkout changes those bytes. No new audio examples are exported.
Input recording provenance and attribution remain in `input/catalog.json`
and `input/LICENSE`; the data license is separate from the code license.

Window labels do not identify the acoustic cause of a feature. Background
births may have earlier source tails, while a recording-labelled window can
still yield a background-driven feature. Different exact feature hashes do
not imply a different perceptual source. The original admission scores are
retained rather than replaced. Pure-control admissions remain no-target
births/admissions with no prior foreground tails under either mapping.
