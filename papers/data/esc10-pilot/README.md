# Pinned recorded-audio input catalog

This catalog supplies inputs for Beads ch-6w4.1. It contains no accuracy
results; [paper 07](../../07-recorded-replay-and-source-exclusion.md) reports
the subsequent full-duration replay pilot. `catalog.json` selects one original source per category and split:
ten development clips from fold 1 and ten evaluation clips from fold 5.
The builder excludes any original Freesound `src_file` already used in the
other split, including another excerpt/take. It also requires ESC-10
membership and preserves each selected clip's upstream attribution.

The immutable upstream revision is
`33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6`. The repository's
[license](https://github.com/karolpiczak/ESC-50/blob/33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6/LICENSE)
identifies ESC-10 as CC BY 3.0 and supplies individual source attributions.
Those exact records, the CSV metadata and the upstream README are retained
in `upstream/`, with hashes. Each catalog entry preserves its creator,
original title/URL, original license tag, immutable download URL, SHA-256,
byte count and checked WAV format. Data licenses remain separate from the
project's code license; retain the attribution information when using clips.

Audio is cached under ignored `data/cache/esc10/<revision>/` rather than
committed with the project. Twenty cached files total approximately 8.8 MB,
each five seconds, mono, 44.1 kHz, 16-bit PCM. The offline audit verifies all
metadata, attribution and audio hashes, formats and original-source splits.
The frozen builder is in `source/recorded_catalog.py`.

From `python/`:

```powershell
uv run python -m chuff_mpu.sim.recorded_catalog fetch --catalog ../papers/data/esc10-pilot --cache ../data/cache/esc10
uv run python -m chuff_mpu.sim.recorded_catalog audit --catalog ../papers/data/esc10-pilot --cache ../data/cache/esc10
```

`fetch` restores missing files only after checking the catalog hash. Existing
different bytes fail audit rather than being silently replaced. `prepare`
creates a new catalog folder and downloads the pinned upstream selection;
it does not overwrite this catalog.

This small catalog covers dog, rooster, crying baby, sneezing and clock
ticks, plus rain, sea waves, fire, helicopter and chainsaw recordings.
The original clips have category labels, not precise event onset/offset
annotations. They are not clean isolated stems, single identifiable animals,
or a balanced representation of deployment conditions. Distinct source IDs
do not prove distinct recording sessions, locations, equipment or creators.
Do not treat repeats of one file as independent natural recurrences, or
category labels as physical source identities.

The remaining benchmark work is tracked in Beads: ingestion into continuous
calibrated mixtures, explicit onset/patch scoring scope, speech, footsteps,
long alarms, multiple recordings per class and independent overlapping
sources. This catalog does not fulfill those broader acceptance criteria.
Evaluation waveforms have been downloaded and format/hash checked, without
selecting methods from their acoustic outcomes. Dataset inclusion does not
mean the existing foreground learner has been validated on recordings.

Dataset citation: Karol J. Piczak (2015), *ESC: Dataset for Environmental
Sound Classification*, ACM Multimedia, pp. 1015–1018.
[DOI](https://doi.org/10.1145/2733373.2806390).
