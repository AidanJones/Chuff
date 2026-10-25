# Selected-feature dependence report

This observed-data diagnostic is derived from the separately audited replay.
`report.json` preserves all ten correlated configurations, context/stage
counts, per-seed ratios, admission joint categories and transitions, plus
per-recording birth/admission counts. `report.md` is a compact overview;
paper 10 also reports the changed-audio subset.

Recording windows, gaps with tails, gaps without tails and pure controls
partition observations. `recording_changed_audio` overlaps recording windows
and excludes exact float32 waveform no-ops; it is not an annotated event
mask or an active-energy SNR. Birth, confirmation and adjacent partition
candidate observations; admission overlaps confirmation.

Ratios require an actual-to-stored within-radius match. Counts pool windows;
estimates aggregate suites within seed and then equally weight seeds. Ten
thousand seed-bootstrap draws are conditional on five fixed foreground
recordings. Undefined seed denominators produce null means and intervals
instead of silently dropping a seed. Per-recording counts are descriptive,
not independent category samples. No learned policy is selected here.

From the repository's `python/` directory:

```powershell
./.venv/Scripts/python.exe -m chuff_mpu.sim.feature_dependence_report audit ../papers/results/masked-events/published-feature-dependence-report
```

This recomputes the complete report and checks source/input/output hashes
and links to the copied replay manifest. Run the replay auditor separately
to verify the underlying vectors and original lifecycle coverage.

Rebuild to a new directory:

```powershell
./.venv/Scripts/python.exe -m chuff_mpu.sim.feature_dependence_report build ../papers/results/masked-events/NEW-feature-report --replay ../papers/results/masked-events/published-feature-dependence
```

The source snapshot includes all package Python files, `pyproject.toml` and
`uv.lock`; the report is reproducible without audio rerendering. Actual prior
learner exposure is retained in the probes. Selected regions can change and
background-role recordings may themselves contain meaningful sounds.
