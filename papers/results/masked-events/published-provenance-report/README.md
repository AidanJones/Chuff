# Birth/admission label sensitivity

`report.json` and `report.md` describe the provenance replay without changing
learner decisions, parameters or historical scores. Input hashes link the
copied summaries and protocol to the original provenance manifest. Source
snapshots and the dependency lock preserve the derivation.

Each seed's four positive suites are averaged before 10,000 paired
seed-bootstrap draws. Birth-minus-admission differences use the same draws.
Intervals condition on five fixed foreground recordings; replays and
correlated configuration variants are not independent sources. This is
exploratory observed-data label sensitivity, not corrected accuracy or a
new held-out model result. Per-recording counts and pure-control transition
totals remain explicit.

From `python/`, audit this derivation:

```powershell
uv run python -m chuff_mpu.sim.provenance_report audit ../papers/results/masked-events/published-provenance-report
```

Recreate into a new output folder:

```powershell
uv run python -m chuff_mpu.sim.provenance_report build ../papers/results/masked-events/published-provenance-report-reproduction --replay ../papers/results/masked-events/published-provenance
```

The build first audits the full provenance replay. The report audit verifies
its hash links and recomputes aggregate scores and paired differences; it
does not repeat waveform rendering. [Paper 09](../../../09-candidate-birth-and-admission-labels.md)
explains window-label ambiguity, concentration in individual recordings and
the separate need for acoustic feature-dependence probes.
