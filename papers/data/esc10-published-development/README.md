# Development-only recorded inputs

This v2 catalog materializes the ten development clips from the
[source reservation](../esc10-published-reservation/README.md). The
[comparison protocol](../../results/masked-events/published-protocol.json)
was frozen before these recordings were downloaded. No evaluation waveform
is included or required by this catalog's audit.

`catalog.json` pins each complete five-second PCM WAV's hash, format,
immutable upstream URL and creator/source attribution. The reservation,
observed-source exclusion catalogs and exact builder are archived alongside
it. Audio resides in the ignored cache. Upstream ESC-10 CC BY 3.0 and
individual attribution records in `upstream/LICENSE` apply to the data.

From `python/`:

```powershell
uv run python -m chuff_mpu.sim.recorded_catalog fetch --catalog ../papers/data/esc10-published-development --cache ../data/cache/esc10
uv run python -m chuff_mpu.sim.recorded_catalog audit --catalog ../papers/data/esc10-published-development --cache ../data/cache/esc10
```

`fetch` restores missing files against pinned hashes. Existing conflicting
bytes fail validation. The development split supports parameter selection;
its scores are not held-out results. See [paper 08](../../08-published-frontends-and-recorded-novelty.md)
for scope, protocol and interpretation.
