#!/usr/bin/env python3
"""Audit CA-1M frame archives with explicit, reproducible rules."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tarfile
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ca1m_audit_core import (
    AuditRules,
    classify_instances,
    sharpness_variance_of_laplacian,
)


CAPTURE_PATTERN = re.compile(r"ca1m-val-(\d+)\.tar$")
INSTANCE_SUFFIX = "/instances.json"
OUTPUT_FIELDS = (
    "archive",
    "capture_id",
    "frame_index",
    "temporal_bin",
    "total",
    "geometric",
    "oracle",
    "natural",
    "natural_interior",
    "sharpness",
    "width",
    "height",
    "image_member",
    "instance_member",
)


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Audit one or more local CA-1M TAR archives."
    )
    parser.add_argument(
        "archives",
        nargs="+",
        type=Path,
        help="CA-1M TAR archives, processed in the supplied order.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=project_root,
        help="Project root. Defaults to the parent of scripts/.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination JSONL path, absolute or relative to project root.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        help=(
            "Metadata JSON path. Defaults beside the output as "
            "<output-stem>_metadata.json."
        ),
    )
    parser.add_argument(
        "--reference-audit",
        type=Path,
        help="Optional historical JSONL audit to compare by frame identity.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing output and metadata files.",
    )
    parser.add_argument(
        "--limit-frames",
        type=int,
        help="Optional maximum number of frames per archive for a smoke test.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print progress after this many total frames; zero disables it.",
    )
    parser.add_argument(
        "--max-mismatches",
        type=int,
        default=10,
        help="Maximum comparison mismatches to print.",
    )
    return parser.parse_args()


def resolve_path(project_root: Path, path: Path) -> Path:
    path = path.expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def display_path(project_root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(project_root))
    except ValueError:
        return str(path)


def capture_id_from_archive(path: Path) -> str:
    match = CAPTURE_PATTERN.search(path.name)
    if match is None:
        raise ValueError(
            f"Archive name does not match ca1m-val-<capture>.tar: {path.name}"
        )
    return match.group(1)


def read_member_bytes(archive: tarfile.TarFile, name: str) -> bytes:
    handle = archive.extractfile(name)
    if handle is None:
        raise FileNotFoundError(f"Member not found in {archive.name}: {name}")
    with handle:
        return handle.read()


def frame_members(archive: tarfile.TarFile) -> list[str]:
    members = [
        member.name
        for member in archive.getmembers()
        if member.isfile() and member.name.endswith(INSTANCE_SUFFIX)
    ]
    return sorted(members)


def temporal_bin(frame_index: int, number_of_frames: int) -> int:
    if number_of_frames <= 0:
        raise ValueError("number_of_frames must be positive")
    return min(5, (frame_index * 5) // number_of_frames + 1)


def audit_archive(
    archive_path: Path,
    project_root: Path,
    rules: AuditRules,
    limit_frames: int | None,
    progress_every: int,
    processed_before: int,
) -> list[dict[str, Any]]:
    capture_id = capture_id_from_archive(archive_path)
    rows: list[dict[str, Any]] = []
    with tarfile.open(archive_path, "r") as archive:
        instance_members = frame_members(archive)
        number_of_frames = len(instance_members)
        selected_members = (
            instance_members
            if limit_frames is None
            else instance_members[:limit_frames]
        )
        for frame_index, instance_member in enumerate(selected_members):
            frame_directory = instance_member[: -len(INSTANCE_SUFFIX)]
            image_member = f"{frame_directory}/image.png"
            instances = json.loads(
                read_member_bytes(archive, instance_member).decode("utf-8")
            )
            if not isinstance(instances, list):
                raise TypeError(f"Expected a list in {instance_member}")
            width, height, sharpness = sharpness_variance_of_laplacian(
                read_member_bytes(archive, image_member)
            )
            groups = classify_instances(instances, width, height, rules)
            row = {
                "archive": display_path(project_root, archive_path),
                "capture_id": capture_id,
                "frame_index": frame_index,
                "temporal_bin": temporal_bin(frame_index, number_of_frames),
                "total": len(instances),
                "geometric": len(groups["geometric"]),
                "oracle": len(groups["oracle"]),
                "natural": len(groups["natural"]),
                "natural_interior": len(groups["natural_interior"]),
                "sharpness": sharpness,
                "width": width,
                "height": height,
                "image_member": image_member,
                "instance_member": instance_member,
            }
            rows.append(row)
            processed = processed_before + len(rows)
            if progress_every and processed % progress_every == 0:
                print(f"  processed {processed} frames")
    return rows


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def frame_identity(row: dict[str, Any]) -> tuple[str, str]:
    return str(row["capture_id"]), str(row["instance_member"])


def compare_reference(
    generated: list[dict[str, Any]],
    reference_path: Path,
    require_complete: bool,
    max_mismatches: int,
) -> tuple[int, list[str]]:
    reference_rows = load_jsonl(reference_path)
    reference = {frame_identity(row): row for row in reference_rows}
    mismatches: list[str] = []
    mismatch_count = 0
    for row in generated:
        identity = frame_identity(row)
        expected = reference.get(identity)
        if expected is None:
            mismatch_count += 1
            if len(mismatches) < max_mismatches:
                mismatches.append(f"missing reference frame: {identity}")
            continue
        differences = [
            field
            for field in OUTPUT_FIELDS
            if row.get(field) != expected.get(field)
        ]
        if differences:
            mismatch_count += 1
            if len(mismatches) < max_mismatches:
                details = ", ".join(
                    f"{field}={expected.get(field)!r}->{row.get(field)!r}"
                    for field in differences
                )
                mismatches.append(f"{identity}: {details}")
    if require_complete:
        generated_ids = {frame_identity(row) for row in generated}
        missing_generated = set(reference) - generated_ids
        mismatch_count += len(missing_generated)
        for identity in sorted(missing_generated):
            if len(mismatches) >= max_mismatches:
                break
            mismatches.append(f"missing generated frame: {identity}")
    return mismatch_count, mismatches


def main() -> int:
    args = parse_args()
    project_root = args.project_root.expanduser().resolve()
    archives = [resolve_path(project_root, path) for path in args.archives]
    output_path = resolve_path(project_root, args.output)
    metadata_path = (
        resolve_path(project_root, args.metadata)
        if args.metadata
        else output_path.with_name(f"{output_path.stem}_metadata.json")
    )
    reference_path = (
        resolve_path(project_root, args.reference_audit)
        if args.reference_audit
        else None
    )
    if args.limit_frames is not None and args.limit_frames <= 0:
        raise ValueError("--limit-frames must be positive")
    if args.progress_every < 0 or args.max_mismatches < 0:
        raise ValueError("Progress and mismatch limits must be nonnegative")
    for archive in archives:
        if not archive.is_file():
            raise FileNotFoundError(f"Archive not found: {archive}")
    for destination in (output_path, metadata_path):
        if destination.exists() and not args.overwrite:
            raise FileExistsError(
                f"Destination exists; pass --overwrite to replace it: {destination}"
            )

    rules = AuditRules()
    rows: list[dict[str, Any]] = []
    capture_counts: Counter[str] = Counter()
    print(f"Archives: {len(archives)}")
    for archive in archives:
        capture_id = capture_id_from_archive(archive)
        print(f"Auditing {archive.name} ...")
        archive_rows = audit_archive(
            archive_path=archive,
            project_root=project_root,
            rules=rules,
            limit_frames=args.limit_frames,
            progress_every=args.progress_every,
            processed_before=len(rows),
        )
        rows.extend(archive_rows)
        capture_counts[capture_id] += len(archive_rows)

    output_sha256 = write_jsonl_atomic(output_path, rows)
    metadata = {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(project_root),
        "archives": [display_path(project_root, path) for path in archives],
        "number_of_archives": len(archives),
        "number_of_frames": len(rows),
        "frames_per_capture": dict(sorted(capture_counts.items())),
        "limited_frames_per_archive": args.limit_frames,
        "rules": rules.to_dict(),
        "sharpness": {
            "color_decode": "Pillow RGB",
            "grayscale": "cv2.COLOR_RGB2GRAY",
            "operator": "cv2.Laplacian(gray, cv2.CV_64F).var()",
        },
        "output": display_path(project_root, output_path),
        "output_sha256": output_sha256,
    }

    comparison_mismatches = 0
    if reference_path is not None:
        comparison_mismatches, samples = compare_reference(
            generated=rows,
            reference_path=reference_path,
            require_complete=args.limit_frames is None,
            max_mismatches=args.max_mismatches,
        )
        metadata["reference_comparison"] = {
            "reference": display_path(project_root, reference_path),
            "mismatched_frames": comparison_mismatches,
            "sample_mismatches": samples,
        }
        print(f"Reference mismatches: {comparison_mismatches}")
        for sample in samples:
            print(f"  {sample}")

    write_json_atomic(metadata_path, metadata)
    print()
    print(f"Frames: {len(rows)}")
    print(f"Output: {output_path}")
    print(f"Metadata: {metadata_path}")
    print(f"Output SHA-256: {output_sha256}")
    return 1 if comparison_mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
