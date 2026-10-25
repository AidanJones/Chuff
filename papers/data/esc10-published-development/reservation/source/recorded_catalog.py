"""Prepare/fetch/audit a pinned, attributed ESC-10 input catalog, not accuracy results."""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import re
from urllib.request import urlopen
import wave


REVISION = "33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6"
REPOSITORY = "https://github.com/karolpiczak/ESC-50"
BASE = f"https://raw.githubusercontent.com/karolpiczak/ESC-50/{REVISION}/"
CATEGORIES = ("dog", "rooster", "rain", "sea_waves", "crackling_fire", "crying_baby",
              "sneezing", "clock_tick", "helicopter", "chainsaw")
UPSTREAM = {"upstream/esc50.csv": "meta/esc50.csv", "upstream/LICENSE": "LICENSE",
            "upstream/README.md": "README.md"}


def digest(content):
    return hashlib.sha256(content).hexdigest()


def download(url):
    with urlopen(url, timeout=30) as response:
        content = response.read(5_000_001)
    if len(content) > 5_000_000:
        raise ValueError("catalog download exceeds size limit")
    return content


def filename(value):
    if not re.fullmatch(r"\d+-\d+-[A-Z]-\d+\.wav", value):
        raise ValueError("unexpected ESC filename")
    return value


def select_rows(text, categories=CATEGORIES, *, development_fold=1, evaluation_fold=5,
                excluded_sources=()):
    if (type(development_fold) is not int or type(evaluation_fold) is not int
            or development_fold not in range(1, 6) or evaluation_fold not in range(1, 6)
            or development_fold == evaluation_fold):
        raise ValueError("distinct integer development/evaluation folds in [1, 5] required")
    if not categories or len(set(categories)) != len(categories) or not set(categories) <= set(CATEGORIES):
        raise ValueError("nonempty unique ESC-10 categories required")
    excluded_sources = tuple(excluded_sources)
    if any(not isinstance(source, str) or not source.isdigit() for source in excluded_sources):
        raise ValueError("excluded original source IDs must be numeric strings")
    rows = list(csv.DictReader(io.StringIO(text)))
    selected, used_sources = [], set(excluded_sources)
    for split, fold in (("development", str(development_fold)), ("evaluation", str(evaluation_fold))):
        for category in categories:
            choices = sorted((r for r in rows if r["fold"] == fold and r["category"] == category
                and r["esc10"] == "True" and r["src_file"] not in used_sources), key=lambda r: r["filename"])
            if not choices:
                raise ValueError(f"no source-disjoint ESC-10 input for {split}/{category}")
            row = dict(choices[0], split=split)
            filename(row["filename"])
            used_sources.add(row["src_file"])
            selected.append(row)
    return selected


def reserve(folder, metadata_from, exclude_catalogs, *, development_fold=2, evaluation_fold=4):
    """Pin a metadata-only selection. Never read, fetch or characterize audio."""
    if not exclude_catalogs:
        raise ValueError("at least one observed catalog must be excluded")
    upstream = {name: (metadata_from/name).read_bytes() for name in UPSTREAM}
    old_metadata = json.loads((metadata_from/"catalog.json").read_text())
    if old_metadata["revision"] != REVISION:
        raise ValueError("metadata revision differs from pinned upstream")
    for name, content in upstream.items():
        if digest(content) != old_metadata["upstream_sha256"][name]:
            raise ValueError("metadata checksum mismatch")
    exclusions, source_ids = {}, set()
    for index, path in enumerate(exclude_catalogs):
        content = (path/"catalog.json").read_bytes()
        catalog = json.loads(content)
        if catalog["revision"] != REVISION or catalog["repository"] != REPOSITORY:
            raise ValueError("excluded catalog must use the pinned ESC revision")
        source_ids.update(clip["source_id"] for clip in catalog["clips"])
        exclusions[f"excluded/catalog-{index:02d}.json"] = content
    plan = dict(development_fold=development_fold, evaluation_fold=evaluation_fold,
                excluded_sources=sorted(source_ids))
    rows = select_rows(upstream["upstream/esc50.csv"].decode("utf-8"), **plan)
    attribution = attributions(upstream["upstream/LICENSE"].decode("utf-8"))
    for row in rows:
        key = "-".join(row["filename"][:-4].split("-")[:-1])
        if key not in attribution or attribution[key]["original_license_tag"] not in ("CC0", "CC-BY"):
            raise ValueError("missing or unsupported selected-source attribution")
    files = {**upstream, **exclusions, "source/recorded_catalog.py": Path(__file__).read_bytes()}
    manifest = dict(schema="chuff.recorded.esc10-reservation.v1", revision=REVISION,
        repository=REPOSITORY, selection_plan=plan, selected_rows=rows,
        scope="Metadata-only reservation; no audio downloaded, decoded, listened to or evaluated by this operation",
        file_sha256={name: digest(content) for name, content in files.items()})
    folder.mkdir(parents=True, exist_ok=False)
    for name, content in files.items():
        destination = folder/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
    (folder/"reservation.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    audit_reservation(folder)


def audit_reservation(folder):
    manifest = json.loads((folder/"reservation.json").read_text())
    if (manifest["schema"] != "chuff.recorded.esc10-reservation.v1"
            or manifest["revision"] != REVISION or manifest["repository"] != REPOSITORY):
        raise ValueError("unsupported reservation")
    hashes = manifest["file_sha256"]
    required = set(UPSTREAM) | {"source/recorded_catalog.py"}
    excluded_paths = [name for name in hashes if re.fullmatch(r"excluded/catalog-\d{2}\.json", name)]
    if not excluded_paths or set(hashes) != required | set(excluded_paths):
        raise ValueError("unexpected or missing reservation files")
    for name, expected in hashes.items():
        if digest((folder/name).read_bytes()) != expected:
            raise ValueError(f"reservation checksum mismatch: {name}")
    source_ids = set()
    for name in excluded_paths:
        catalog = json.loads((folder/name).read_text())
        if catalog["revision"] != REVISION or catalog["repository"] != REPOSITORY:
            raise ValueError("excluded catalog revision/repository mismatch")
        for upstream_name in UPSTREAM:
            if catalog["upstream_sha256"][upstream_name] != hashes[upstream_name]:
                raise ValueError("excluded catalog metadata differs from reservation")
        source_ids.update(clip["source_id"] for clip in catalog["clips"])
    plan = manifest["selection_plan"]
    if plan["excluded_sources"] != sorted(source_ids):
        raise ValueError("excluded sources differ from archived observed catalogs")
    rows = select_rows((folder/"upstream/esc50.csv").read_text(encoding="utf-8"), **plan)
    if rows != manifest["selected_rows"]:
        raise ValueError("reserved selection differs from deterministic metadata rule")
    print(f"{len(rows)} metadata-only reserved clips; {len(source_ids)} observed original sources excluded")
    return manifest


def attributions(text):
    pattern = re.compile(r"^- \[([^\]]+)\.ogg\]: clip derived from (.+) \((https?://[^)]+)\) by (.+) \[([^\]]+)\]$", re.M)
    return {key: dict(original_title=title, original_url=url, creator=creator,
                     original_license_tag=license_tag, attribution=match.group(0))
            for match in pattern.finditer(text)
            for key, title, url, creator, license_tag in [match.groups()]}


def audio_info(content):
    with wave.open(io.BytesIO(content), "rb") as handle:
        info = dict(sample_rate=handle.getframerate(), channels=handle.getnchannels(),
                    frames=handle.getnframes(), sample_width=handle.getsampwidth(),
                    compression=handle.getcomptype())
    if info != dict(sample_rate=44100, channels=1, frames=220500, sample_width=2, compression="NONE"):
        raise ValueError(f"unexpected upstream audio format: {info}")
    return info


def prepare(folder, cache, *, reservation=None, partitions=("development", "evaluation")):
    if (not partitions or len(set(partitions)) != len(partitions)
            or not set(partitions) <= {"development", "evaluation"}):
        raise ValueError("unique nonempty development/evaluation partitions required")
    if reservation is None and set(partitions) != {"development", "evaluation"}:
        raise ValueError("partition-only preparation requires a reservation")
    reserved = audit_reservation(reservation) if reservation is not None else None
    folder.mkdir(parents=True, exist_ok=False)
    hashes, upstream = {}, {}
    for name, remote in UPSTREAM.items():
        content = (reservation/name).read_bytes() if reserved is not None else download(BASE+remote)
        destination = folder/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
        hashes[name] = digest(content)
        upstream[name] = content.decode("utf-8")
    attribution = attributions(upstream["upstream/LICENSE"])
    plan = reserved["selection_plan"] if reserved is not None else {}
    selected = select_rows(upstream["upstream/esc50.csv"], **plan)
    selected = [row for row in selected if row["split"] in partitions]
    manifest = dict(schema="chuff.recorded.esc10-catalog.v1", repository=REPOSITORY, revision=REVISION,
        dataset_license="CC-BY-3.0 (ESC-10 subset); code license does not replace data licenses",
        selection="Lexically first source-disjoint file per ESC-10 category, development fold1/evaluation fold5",
        source_group="Original Freesound src_file, not excerpt filename; no cross-split source reuse",
        scope="Recorded input preparation only. No learner evaluation or onset annotations. Twenty 5-second clips.",
        citation="Karol J. Piczak (2015), ESC: Dataset for Environmental Sound Classification, ACM Multimedia, doi:10.1145/2733373.2806390",
        upstream_sha256=hashes, clips=[])
    if reserved is not None:
        manifest.update(schema="chuff.recorded.esc10-catalog.v2", selection_plan=plan,
            partitions=list(partitions),
            scope=f"Recorded input preparation only; no learner evaluation. {len(selected)} full five-second clips.",
            selection="Lexically first eligible source-disjoint file per category and declared fold, excluding archived observed sources")
        # Copy only the audited reservation, never incidental files in its folder.
        for name in ("reservation.json", *reserved["file_sha256"]):
            destination = folder/"reservation"/name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes((reservation/name).read_bytes())
        manifest["reservation_sha256"] = digest((folder/"reservation/reservation.json").read_bytes())
    for row in selected:
        name = row["filename"]
        key = "-".join(name[:-4].split("-")[:-1])
        details = attribution.get(key)
        if details is None or details["original_license_tag"] not in ("CC0", "CC-BY"):
            raise ValueError(f"missing/unexpected per-clip attribution for {name}")
        content = download(BASE+"audio/"+name)
        destination = cache/REVISION/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Do not overwrite an existing different cache entry silently.
        if destination.exists() and destination.read_bytes() != content:
            raise ValueError(f"cache conflict for {name}")
        destination.write_bytes(content)
        manifest["clips"].append(dict(filename=name, category=row["category"], split=row["split"],
            fold=int(row["fold"]), source_id=row["src_file"], take=row["take"],
            url=BASE+"audio/"+name, sha256=digest(content), bytes=len(content),
            audio=audio_info(content), **details))
        print(f"cached {row['split']}/{row['category']}: {name}", flush=True)
    source = Path(__file__).read_bytes()
    destination = folder/"source/recorded_catalog.py"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(source)
    manifest["builder_sha256"] = digest(source)
    (folder/"catalog.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    audit(folder, cache)


def audit(folder, cache, *, fetch=False):
    manifest = json.loads((folder/"catalog.json").read_text())
    assert manifest["schema"] in ("chuff.recorded.esc10-catalog.v1", "chuff.recorded.esc10-catalog.v2") and manifest["revision"] == REVISION
    plan = {}
    if manifest["schema"] == "chuff.recorded.esc10-catalog.v2":
        reserved = audit_reservation(folder/"reservation")
        assert digest((folder/"reservation/reservation.json").read_bytes()) == manifest["reservation_sha256"]
        assert manifest["selection_plan"] == reserved["selection_plan"]
        plan = reserved["selection_plan"]
        for name in UPSTREAM:
            assert manifest["upstream_sha256"][name] == reserved["file_sha256"][name]
    assert digest((folder/"source/recorded_catalog.py").read_bytes()) == manifest["builder_sha256"]
    for name in UPSTREAM:
        assert digest((folder/name).read_bytes()) == manifest["upstream_sha256"][name]
    rows = select_rows((folder/"upstream/esc50.csv").read_text(encoding="utf-8"), **plan)
    if manifest["schema"] == "chuff.recorded.esc10-catalog.v2":
        partitions = manifest["partitions"]
        assert partitions and len(set(partitions)) == len(partitions)
        assert set(partitions) <= {"development", "evaluation"}
        rows = [row for row in rows if row["split"] in partitions]
    details = attributions((folder/"upstream/LICENSE").read_text(encoding="utf-8"))
    assert [r["filename"] for r in rows] == [r["filename"] for r in manifest["clips"]]
    groups = {split: {r["source_id"] for r in manifest["clips"] if r["split"] == split}
              for split in ("development", "evaluation")}
    assert not groups["development"] & groups["evaluation"]
    for row, clip in zip(rows, manifest["clips"]):
        name = filename(clip["filename"])
        assert clip["category"] == row["category"] and clip["source_id"] == row["src_file"]
        assert clip["split"] == row["split"] and clip["fold"] == int(row["fold"])
        assert clip["url"] == BASE+"audio/"+name
        key = "-".join(name[:-4].split("-")[:-1])
        for field, value in details[key].items():
            assert clip[field] == value
        destination = cache/REVISION/name
        if fetch and not destination.exists():
            content = download(clip["url"])
            assert digest(content) == clip["sha256"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
        content = destination.read_bytes()
        assert digest(content) == clip["sha256"] and len(content) == clip["bytes"]
        assert audio_info(content) == clip["audio"]
    print(f"{len(manifest['clips'])} clips: pinned metadata/attribution/audio hashes, format and disjoint original sources verified")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "fetch", "audit", "reserve", "audit-reservation"))
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--metadata-from", type=Path)
    parser.add_argument("--exclude-catalog", type=Path, action="append", default=[])
    parser.add_argument("--development-fold", type=int, default=2)
    parser.add_argument("--evaluation-fold", type=int, default=4)
    parser.add_argument("--reservation", type=Path)
    parser.add_argument("--partition", choices=("development", "evaluation"), action="append")
    args = parser.parse_args()
    if args.action == "reserve":
        if args.metadata_from is None or not args.exclude_catalog:
            parser.error("reserve requires --metadata-from and --exclude-catalog")
        reserve(args.catalog, args.metadata_from, args.exclude_catalog,
                development_fold=args.development_fold, evaluation_fold=args.evaluation_fold)
    elif args.action == "audit-reservation":
        audit_reservation(args.catalog)
    else:
        if args.cache is None:
            parser.error("prepare/fetch/audit require --cache")
        if args.action == "prepare":
            prepare(args.catalog, args.cache, reservation=args.reservation,
                    partitions=args.partition or ("development", "evaluation"))
        else:
            audit(args.catalog, args.cache, fetch=args.action == "fetch")


if __name__ == "__main__":
    main()
