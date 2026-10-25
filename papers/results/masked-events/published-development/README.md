# Frozen frontend development comparison

This is parameter-selection evidence for [paper 08](../../../08-published-frontends-and-recorded-novelty.md),
not held-out evaluation. It contains 48 finite candidate settings plus one
untuned adaptive-gain anchor, six suites, two seeds and 210 one-second
observations per stream: 123,480 actual decisions plus 39,200 episode-removal
probe decisions. Controls are unique per ambience condition/seed.

`input/protocol.json` was frozen before the fold-2 development audio was
downloaded. `selection.json` applies its declared constraints exactly.
Quiet-band and PCEN have feasible configurations; MCRA has none and its
evaluation choices are explicitly diagnostic. Every grid point and both
feature modes are retained in `summary.json`, `analysis.json` and raw rows.
The protocol's fixed defaults and selected/diagnostic IDs determine the
subsequent evaluation list without retuning.

From `python/`, audit without audio downloads:

```powershell
uv run python -m chuff_mpu.sim.audit_published ../papers/results/masked-events/published-development
```

The audit checks exact source/input/output/example bytes, the complete
suite/seed/step/actor matrix, method settings, source exclusion, physical
episodes/SNR and matched controls, raw-derived timing/identity metrics and
the exact selection. `source/` retains the package and dependency lock at
execution. The frozen protocol enforces exact method bytes; for reproducing
after a checkout that changes source line endings, use this archived source
tree rather than altering the protocol. Archive files preserve raw bytes.

Timing rows cover actual `observe` calls on this shared desktop, excluding
probe calls, copying and rendering. Imports/JIT were warmed on disposable
actors, and runtime summaries exclude initial calibration windows. Host and
thread settings are in `manifest.json`; this is not isolated hardware or
energy profiling.

`examples/` contains four full five-second component examples, one per
positive suite, with recombination and post-channel SNR checks. These derived
recordings retain upstream ESC-10 CC BY 3.0 and individual attribution in
`input/LICENSE` and `input/catalog.json`; they do not inherit the code's MIT
license. Five fixed foreground recordings, source pauses, episode-injection
labels and conditional seed resampling limit interpretation. Source removal
retains earlier actual learner exposure and does not recreate a wholly
episode-deleted learning history.
