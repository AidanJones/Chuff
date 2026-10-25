# Observed candidate-feature intervention replay

This archive retains every candidate observation, not just candidates later
admitted. It verifies all 126,000 original actual decisions and 40,000 legacy
source-removal checks, then saves 28,813 feature-probe windows and 5,940
birth/admission pairs. Sixty NPZ bundles contain four float32 matrices per
configuration. JSON `vector_file` and `vector_index` fields identify rows.

The manifest hashes executed source, copied inputs, output packets and vector
bundles separately. Core learner methods, protocol, dependency versions,
catalog and full original provenance history were checked during replay.
`observe_ns` is deliberately excluded from decision equality. Original
timings remain in the copied input; extra probe cost is not a deployment
latency measurement.

From the repository's `python/` directory with the locked environment:

```powershell
./.venv/Scripts/python.exe -m chuff_mpu.sim.audit_feature_dependence ../papers/results/masked-events/published-feature-dependence
```

The auditor independently recomputes vector hashes/norms/distances, regions,
no-op packets, stage counts, lifecycle coverage and admission pair links.
It checks saved physical metadata against original evidence; it does not
independently rerender the source waveforms. Replay performed that check.

To rerender into a **new** output directory, with the verified evaluation
recordings already cached:

```powershell
$env:OPENBLAS_NUM_THREADS='1'
./.venv/Scripts/python.exe -m chuff_mpu.sim.feature_dependence --original ../papers/results/masked-events/published-evaluation --provenance ../papers/results/masked-events/published-provenance --catalog ../papers/data/esc10-published-evaluation --cache ../data/cache/esc10 --output ../papers/results/masked-events/NEW-feature-replay
```

Frozen source hashes must match, including line endings. The executed-source
snapshot predates the standalone auditor/report and contains package Python
files plus `uv.lock`, but omits `pyproject.toml`. For an isolated scratch
reproduction, copy the original evaluation's hashed
`source/python/pyproject.toml` into the scratch source tree before running;
do not mutate this archive. The companion report source snapshot includes
the standalone auditor, report builder and package metadata. Dependency
versions must match `input/protocol.json`.

The learner receives only mixtures. Intervention copies retain earlier
actual exposure and discard probe updates. Bed removal includes older source
tails. Feature reselection can move regions; a bed match is not fixed-patch
attribution. Detected zero vectors and absent features remain distinct.
The diagnostic neither supplies a deployable clean-stem oracle nor measures
semantic relevance. See paper 10 and the companion report for interpretation.
