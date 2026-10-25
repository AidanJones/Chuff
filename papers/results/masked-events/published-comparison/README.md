# Descriptive held-out comparison

`report.json` and `report.md` summarize the audited frozen held-out experiment
without selecting or retuning any method. `input/` preserves the evaluation
summary/manifest, source catalog, frozen protocol and development selection
and summaries. Source/input/output hashes link the derivation to the original
evaluation. The source snapshot includes the complete package and dependency
lock at report creation; only the report derivation runs here.

Each seed's four positive suites are averaged before 10,000 seed-bootstrap
draws. Controls and warmed inference similarly average their matched suites
within seed. Paired contrasts share the same draws across methods. Five
fixed foreground recordings remain five distinct inputs, with recording/run
pairs and replay counts reported separately. Intervals condition on these
recordings and are not corpus-level confidence intervals. Post-run contrasts
are exploratory and not adjusted for multiple comparisons.

From `python/`, audit this derivation offline:

```powershell
uv run python -m chuff_mpu.sim.published_report audit ../papers/results/masked-events/published-comparison
```

To recreate into a **new** output folder:

```powershell
uv run python -m chuff_mpu.sim.published_report build ../papers/results/masked-events/published-comparison-reproduction --evaluation ../papers/results/masked-events/published-evaluation
```

The build first audits the complete underlying raw experiment. The report
audit verifies its input links and recomputes the aggregation and contrasts;
it does not duplicate the full raw-data audit. See [paper 08](../../../08-published-frontends-and-recorded-novelty.md)
for interpretation, per-recording failures and the distinction between
admission labels and candidate birth provenance. Selected relative/raw PCEN
pipelines use different detection thresholds, so their contrast does not
isolate stored-feature normalization.
