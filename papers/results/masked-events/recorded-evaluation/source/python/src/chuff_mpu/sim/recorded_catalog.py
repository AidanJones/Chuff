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


def select_rows(text, categories=CATEGORIES):
    rows = list(csv.DictReader(io.StringIO(text)))
    selected, used_sources = [], set()
    for split, fold in (("development", "1"), ("evaluation", "5")):
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


def prepare(folder, cache):
    folder.mkdir(parents=True, exist_ok=False)
    hashes, upstream = {}, {}
    for name, remote in UPSTREAM.items():
        content = download(BASE+remote)
        destination = folder/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
        hashes[name] = digest(content)
        upstream[name] = content.decode("utf-8")
    attribution = attributions(upstream["upstream/LICENSE"])
    selected = select_rows(upstream["upstream/esc50.csv"])
    manifest = dict(schema="chuff.recorded.esc10-catalog.v1", repository=REPOSITORY, revision=REVISION,
        dataset_license="CC-BY-3.0 (ESC-10 subset); code license does not replace data licenses",
        selection="Lexically first source-disjoint file per ESC-10 category, development fold1/evaluation fold5",
        source_group="Original Freesound src_file, not excerpt filename; no cross-split source reuse",
        scope="Recorded input preparation only. No learner evaluation or onset annotations. Twenty 5-second clips.",
        citation="Karol J. Piczak (2015), ESC: Dataset for Environmental Sound Classification, ACM Multimedia, doi:10.1145/2733373.2806390",
        upstream_sha256=hashes, clips=[])
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
    assert manifest["schema"] == "chuff.recorded.esc10-catalog.v1" and manifest["revision"] == REVISION
    assert digest((folder/"source/recorded_catalog.py").read_bytes()) == manifest["builder_sha256"]
    for name in UPSTREAM:
        assert digest((folder/name).read_bytes()) == manifest["upstream_sha256"][name]
    rows = select_rows((folder/"upstream/esc50.csv").read_text(encoding="utf-8"))
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
    parser.add_argument("action", choices=("prepare", "fetch", "audit"))
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.catalog, args.cache)
    else:
        audit(args.catalog, args.cache, fetch=args.action == "fetch")


if __name__ == "__main__":
    main()
