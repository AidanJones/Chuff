# Frozen held-out frontend comparison

This archive evaluates the settings selected before fold-4 audio was
downloaded. [Paper 08](../../../08-published-frontends-and-recorded-novelty.md)
documents the method, development selection and limitations.

The fixed matrix comprises nine configurations plus an untuned adaptive-gain
anchor, six unique suites, ten seeds (2900–2909) and 210 one-second
observations per stream: 126,000 actual decisions and 40,000 current-episode
source-removal probe decisions. Twenty no-target controls are unique by
ambience condition/seed; they are not duplicated at nominal target SNRs.

The exact input protocol, development summaries and selection are in
`input/`. No evaluation outcomes change the settings or feasibility labels.
MCRA choices remain diagnostic because no development configuration met the
declared foreground-retention requirement. The original defaults and
feature-specific pipeline choices remain in the evaluation.

From `python/`, audit offline:

```powershell
uv run python -m chuff_mpu.sim.audit_published ../papers/results/masked-events/published-evaluation
```

The audit checks hashes, frozen settings and selection provenance, the full
suite/seed/window/actor matrix, full episode duration and post-channel SNR,
component examples and paired backgrounds, source separation, timings and
raw-derived identity metrics. `source/` retains execution-time code and the
dependency lock. Method byte hashes include line endings; use the preserved
source tree if a checkout changes root-source bytes.

Source-deletion branches retain actual prior learner exposure and are
discarded after each probe. They do not reconstruct a wholly episode-deleted
learning history. Identity uses admission-window labels, which do not prove
candidate birth provenance. Wrong and correct recognition can occur in
different windows of one episode, so their rates are not complements.

There are five fixed foreground recordings and five fixed background
recordings. Mixer seeds and repeated exposure do not increase the number of
independent source recordings. Whole-episode SNR can conceal large variation
within recordings. These are full-recording injection/replay results, not
natural event-boundary, semantic relevance or field-robustness validation.

Per-window warmed timings concern this Python implementation on a shared
desktop, excluding probes/copying/rendering and initial calibration windows.
They are not isolated algorithmic cost or hardware power measurements.
Derived full-duration component examples retain ESC-10 CC BY 3.0 and the
individual creator/source attribution in `input/LICENSE` and `input/catalog.json`.
