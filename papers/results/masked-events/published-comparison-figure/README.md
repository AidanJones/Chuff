# Held-out comparison figure

`comparison.png` and `comparison.svg` visualize the audited paired-seed
summary from `../published-comparison/report.json`. Five representative
pipelines are shown; the complete ten-configuration table remains in that
report. The MCRA point is diagnostic because it did not meet the frozen
development retention requirement. Neither axes nor timing establish
semantic relevance, corpus-level generalization or optimized algorithm cost.

Bars are marginal 95% seed-bootstrap intervals conditional on five fixed
foreground recordings. The runtime panel is mean warmed actual-observe
milliseconds per one-second input on the shared desktop, excluding initial
calibration, probes, copying and rendering.

The figure was visually checked. `manifest.json` pins the plotting source,
input JSON, outputs and matplotlib version. `source/` and `input/` preserve
the exact bytes. No audio is included. From `python/`, recreate into a new
folder with an isolated plotting environment:

```powershell
uv run --no-project --python 3.11 --with matplotlib==3.10.7 python src/chuff_mpu/sim/plot_published.py ../papers/results/masked-events/published-comparison/report.json ../papers/results/masked-events/published-comparison-figure-reproduction
```

The original experiment environment does not require matplotlib. SVG metadata
and renderer differences may change regenerated file bytes; the archived
hashes identify the actual reviewed outputs.
