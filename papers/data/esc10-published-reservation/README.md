# Reserved inputs for the published-frontend comparison

This metadata-only reservation supports Beads ch-6w4.7.2. It selects ten
development sources from ESC-10 fold 2 and ten evaluation sources from fold 4
using the lexically first eligible filename per category. Every original
Freesound `source_id` in the earlier fold-1/fold-5 pilot is excluded, including
alternative excerpts of those sources. The selected 20 original source IDs
are distinct from each other and from all 20 observed pilot source IDs.

`reservation.json` records the exact rows, fold rule and exclusions. Hashed
copies of the prior catalog, pinned upstream CSV, license, README and builder
make selection reproducible without reading audio. The reservation operation
does not download, decode or characterize any waveform. Audio hashes are
established when a partition is subsequently materialized from the immutable
upstream revision; this metadata reservation alone is not an audio audit.

From `python/`, verify the reservation offline:

```powershell
uv run python -m chuff_mpu.sim.recorded_catalog audit-reservation --catalog ../papers/data/esc10-published-reservation
```

Materialize development separately, preserving the held-out partition:

```powershell
uv run python -m chuff_mpu.sim.recorded_catalog prepare --catalog ../papers/data/esc10-published-development --reservation ../papers/data/esc10-published-reservation --partition development --cache ../data/cache/esc10
```

After the comparison protocol, development selection and learner settings
have been frozen, materialize evaluation into a separate new catalog:

```powershell
uv run python -m chuff_mpu.sim.recorded_catalog prepare --catalog ../papers/data/esc10-published-evaluation --reservation ../papers/data/esc10-published-reservation --partition evaluation --cache ../data/cache/esc10
```

`prepare` refuses an existing destination and pins each downloaded file's
bytes, attribution and full five-second WAV format. Subsequent `fetch` and
`audit` operations consume those hashes; they do not reselect recordings.
The older v1 pilot catalog remains readable by the current auditor.

The source recordings retain the upstream ESC-10 CC BY 3.0 license and
individual attribution records in `upstream/LICENSE`; the code license does
not replace these data licenses. Source-ID separation does not establish
independence of creators, locations or recording equipment. There are only
five foreground recordings and five background recordings per partition.
Neither selection nor source separation establishes foreground recognition
accuracy, natural event boundaries or generalization to a wider corpus.

Dataset: Karol J. Piczak (2015), *ESC: Dataset for Environmental Sound
Classification*, ACM Multimedia, [doi:10.1145/2733373.2806390](https://doi.org/10.1145/2733373.2806390).
