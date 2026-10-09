#!/usr/bin/env python3
"""Validate, group, split, and optionally materialize the YOLO plate dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
PLATE_RE = re.compile(r"^(?P<plate>[A-Z]{4}\d{2}|[A-Z]{2}\d{4})(?:\(\d+\))?$")


class UnionFind:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


@dataclass(frozen=True)
class Item:
    stem: str
    image: Path
    label: Path
    sha256: str
    dhash: int
    inferred_text: str


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def difference_hash(path: Path) -> int:
    with Image.open(path) as image:
        pixels = list(image.convert("L").resize((17, 16)).getdata())
    value = 0
    for y in range(16):
        for x in range(16):
            if pixels[y * 17 + x] > pixels[y * 17 + x + 1]:
                value |= 1 << (y * 16 + x)
    return value


def inferred_plate(stem: str) -> str:
    match = PLATE_RE.fullmatch(stem.upper())
    return match.group("plate") if match else ""


def validate_label(path: Path) -> int:
    count = 0
    for line_number, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"{path}:{line_number}: expected five YOLO fields")
        class_id = int(fields[0])
        coords = [float(value) for value in fields[1:]]
        if class_id != 0:
            raise ValueError(f"{path}:{line_number}: only class 0 is supported")
        if any(value < 0 or value > 1 for value in coords):
            raise ValueError(f"{path}:{line_number}: coordinates must be normalized")
        if coords[2] <= 0 or coords[3] <= 0:
            raise ValueError(f"{path}:{line_number}: width and height must be positive")
        count += 1
    if count == 0:
        raise ValueError(f"{path}: label contains no boxes")
    return count


def load_transcriptions(path: Path) -> dict[str, tuple[str, str, str]]:
    if not path.exists():
        return {}
    result: dict[str, tuple[str, str, str]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            text = row.get("plate_text", "").strip().upper().replace("-", "").replace(" ", "")
            group = row.get("group_id", "").strip()
            verified = row.get("verified", "no").strip().lower()
            if text and not PLATE_RE.fullmatch(text):
                raise ValueError(f"Invalid plate transcription for {row['stem']}: {text}")
            if verified not in {"yes", "no"}:
                raise ValueError(f"Invalid verified value for {row['stem']}: {verified}")
            result[row["stem"]] = (text, group, verified)
    return result


def write_transcription_template(path: Path, items: list[Item], overrides: dict[str, tuple[str, str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["stem", "plate_text", "group_id", "verified"])
        writer.writeheader()
        for item in items:
            text, group, verified = overrides.get(item.stem, (item.inferred_text, "", "no"))
            writer.writerow(
                {
                    "stem": item.stem,
                    "plate_text": text or item.inferred_text,
                    "group_id": group,
                    "verified": verified if text else "no",
                }
            )


def assign_splits(group_sizes: dict[str, int], seed: int) -> dict[str, str]:
    rng = random.Random(seed)
    groups = list(group_sizes)
    rng.shuffle(groups)
    groups.sort(key=lambda group: group_sizes[group], reverse=True)
    total = sum(group_sizes.values())
    targets = {"train": total * 0.70, "val": total * 0.15, "test": total * 0.15}
    counts = {name: 0 for name in targets}
    result: dict[str, str] = {}
    for group in groups:
        split = max(targets, key=lambda name: targets[name] - counts[name])
        result[group] = split
        counts[split] += group_sizes[group]
    return result


def scan(dataset_root: Path, overrides: dict[str, tuple[str, str, str]]) -> tuple[list[Item], int]:
    images = sorted(path for path in dataset_root.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    labels = {path.stem: path for path in dataset_root.glob("*.txt")}
    image_stems = {path.stem for path in images}
    if missing := sorted(image_stems - labels.keys()):
        raise ValueError(f"Images without labels: {missing}")
    if orphaned := sorted(labels.keys() - image_stems):
        raise ValueError(f"Labels without images: {orphaned}")

    box_count = 0
    items: list[Item] = []
    for image in images:
        box_count += validate_label(labels[image.stem])
        override_text = overrides.get(image.stem, ("", "", "no"))[0]
        items.append(
            Item(
                stem=image.stem,
                image=image,
                label=labels[image.stem],
                sha256=sha256(image),
                dhash=difference_hash(image),
                inferred_text=override_text or inferred_plate(image.stem),
            )
        )
    return items, box_count


def build_manifest(
    dataset_root: Path,
    manifest_path: Path,
    transcriptions_path: Path,
    seed: int,
) -> list[dict[str, str]]:
    overrides = load_transcriptions(transcriptions_path)
    items, box_count = scan(dataset_root, overrides)
    union = UnionFind(item.stem for item in items)

    by_sha: dict[str, list[Item]] = defaultdict(list)
    by_plate: dict[str, list[Item]] = defaultdict(list)
    for item in items:
        by_sha[item.sha256].append(item)
        if item.inferred_text:
            by_plate[item.inferred_text].append(item)
    for group in [*by_sha.values(), *by_plate.values()]:
        for other in group[1:]:
            union.union(group[0].stem, other.stem)

    for index, left in enumerate(items):
        for right in items[index + 1 :]:
            if (left.dhash ^ right.dhash).bit_count() <= 8:
                union.union(left.stem, right.stem)

    for item in items:
        custom_group = overrides.get(item.stem, ("", "", "no"))[1]
        if not custom_group:
            continue
        for other in items:
            if overrides.get(other.stem, ("", "", "no"))[1] == custom_group:
                union.union(item.stem, other.stem)

    components: dict[str, list[Item]] = defaultdict(list)
    for item in items:
        components[union.find(item.stem)].append(item)

    duplicate_of: dict[str, str] = {}
    for exact_group in by_sha.values():
        if len(exact_group) < 2:
            continue
        canonical = min(exact_group, key=lambda item: (not bool(item.inferred_text), item.stem))
        for item in exact_group:
            if item.stem != canonical.stem:
                duplicate_of[item.stem] = canonical.stem

    group_ids: dict[str, str] = {}
    for component in components.values():
        texts = sorted({item.inferred_text for item in component if item.inferred_text})
        if len(texts) > 1:
            raise ValueError(f"Conflicting transcriptions in group: {texts}")
        fallback = min(item.sha256[:12] for item in component)
        group_id = texts[0] if texts else f"unknown-{fallback}"
        for item in component:
            group_ids[item.stem] = group_id

    group_sizes: dict[str, int] = defaultdict(int)
    for item in items:
        if item.stem not in duplicate_of:
            group_sizes[group_ids[item.stem]] += 1
    split_by_group = assign_splits(dict(group_sizes), seed)

    rows: list[dict[str, str]] = []
    for item in items:
        rows.append(
            {
                "image": item.image.name,
                "label": item.label.name,
                "plate_text": item.inferred_text,
                "group_id": group_ids[item.stem],
                "split": split_by_group[group_ids[item.stem]],
                "include": "no" if item.stem in duplicate_of else "yes",
                "duplicate_of": duplicate_of.get(item.stem, ""),
            }
        )

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_transcription_template(transcriptions_path, items, overrides)

    included = sum(row["include"] == "yes" for row in rows)
    unknown = sum(not row["plate_text"] and row["include"] == "yes" for row in rows)
    print(f"Validated {len(items)} images and {box_count} boxes")
    print(f"Manifest contains {included} included images; removed {len(items) - included} exact duplicates")
    print(f"Manual OCR transcriptions remaining: {unknown}")
    return rows


def materialize(dataset_root: Path, destination: Path, rows: list[dict[str, str]]) -> None:
    for split in ("train", "val", "test"):
        (destination / "images" / split).mkdir(parents=True, exist_ok=True)
        (destination / "labels" / split).mkdir(parents=True, exist_ok=True)
    for row in rows:
        if row["include"] != "yes":
            continue
        split = row["split"]
        shutil.copy2(dataset_root / row["image"], destination / "images" / split / row["image"])
        shutil.copy2(dataset_root / row["label"], destination / "labels" / split / row["label"])
    (destination / "data.yaml").write_text(
        "\n".join(
            [
                f"path: {destination.resolve().as_posix()}",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                "names:",
                "  0: license_plate",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"Materialized dataset at {destination.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("training/manifest.csv"))
    parser.add_argument("--transcriptions", type=Path, default=Path("training/transcriptions.csv"))
    parser.add_argument("--materialize", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.dataset_root.resolve()
    rows = build_manifest(root, args.manifest, args.transcriptions, args.seed)
    if args.materialize:
        materialize(root, args.materialize, rows)


if __name__ == "__main__":
    main()
