# Held-out recorded comparison inputs

This catalog materializes only the ten fold-4 evaluation recordings from
the [source reservation](../esc10-published-reservation/README.md). The
development settings and selection were committed in `5cfe07c` before
these waveforms were downloaded. Original source IDs are disjoint from both
the development split and all observed fold-1/fold-5 pilot inputs.

`catalog.json` records each complete five-second WAV's immutable source URL,
SHA-256, format and individual creator/source attribution. The exact builder,
reservation, exclusions and upstream metadata/license are archived here.
Audio is in the ignored cache; existing conflicting bytes are rejected.
ESC-10 CC BY 3.0 and the attribution in `upstream/LICENSE` apply to the data.

From `python/`:

```powershell
uv run python -m chuff_mpu.sim.recorded_catalog fetch --catalog ../papers/data/esc10-published-evaluation --cache ../data/cache/esc10
uv run python -m chuff_mpu.sim.recorded_catalog audit --catalog ../papers/data/esc10-published-evaluation --cache ../data/cache/esc10
```

This split contains only one recording per category: five foreground and
five background files. Source separation does not establish independence of
creators, equipment or sessions, clean event stems or semantic relevance.
See [paper 08](../../08-published-frontends-and-recorded-novelty.md) for the
fixed comparison protocol, results and limits.
