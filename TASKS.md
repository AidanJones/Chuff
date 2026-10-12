# Chuff Task Backlog

Tracked in [beads](https://github.com/steveyegge/beads) (`bd`), not here —
`bd.exe` was blocked on this machine by Windows Smart App Control; once
unblocked, the full backlog below was migrated into real `bd` issues
(`ch-1` through `ch-54`).

```
bd ready              # what's actionable right now, dependency-resolved
bd list               # everything
bd show ch-13         # one issue's full detail, labels, dependencies
```

No `bd` installed? `.beads/issues.jsonl` is a plain-text export of every
issue (title, description, status, labels, dependencies) — readable
without the tool, re-exportable with `bd export > .beads/issues.jsonl`
after any change. The live Dolt database (`.beads/embeddeddolt/`) is
gitignored; that export is the portable, human-readable copy.

## Current snapshot (37 issues, as of the last `bd export`)

Ready to work (open, no unmet dependencies): `ch-11`, `ch-12`, `ch-15`,
`ch-16`, `ch-17`, `ch-18`, `ch-22`, `ch-33`, `ch-36`, `ch-41`, `ch-51`,
`ch-53`, `ch-54`.

Blocked: `ch-23b`, `ch-30` (both on physical hardware not yet in hand),
`ch-34`, `ch-35` (both depend on `ch-30`).

Done: everything else — the full esp32/arduino/python/eval scaffold,
the real `memory.py`/`comms.py`/`audio.py`/`audio_safety.py`
implementations, both `sim/` demos, and the research-backlog follow-ups
in `shared/research_backlog.md` (Acoustic Adaptation Hypothesis test,
memory eviction policy comparison).

See `shared/research_backlog.md` for the deeper write-up behind the
research-track issues (`ch-50` through `ch-54`), and each track's own
README for narrower context.
