#!/usr/bin/env python3
"""Evaluate 3D pair feasibility for ranked CA-1M scene candidates."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import statistics
import tarfile
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ca1m_audit_core import (
    AuditRules,
    classify_instances,
    interior_valid,
    normalized_category,
    valid_box,
)


VISIBILITY_THRESHOLDS = (0.8, 0.9, 1.0)


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Evaluate 3D object-pair feasibility in a scene shortlist."
    )
    parser.add_argument("shortlist", type=Path, help="Ranked shortlist JSONL.")
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
        help="Destination pair-feasibility JSONL.",
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        help="Protocol JSON path. Defaults beside the output.",
    )
    parser.add_argument(
        "--minimum-distance-m",
        type=float,
        default=0.25,
        help="Minimum 3D center-to-center distance for an eligible pair.",
    )
    parser.add_argument(
        "--minimum-pairs",
        type=int,
        default=12,
        help="Minimum eligible relations required for basic feasibility.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing output and protocol files.",
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


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"Expected an object at {path}:{line_number}")
            rows.append(row)
    return rows


def read_member_json(archive: tarfile.TarFile, name: str) -> list[dict[str, Any]]:
    handle = archive.extractfile(name)
    if handle is None:
        raise FileNotFoundError(f"Member not found in {archive.name}: {name}")
    with handle:
        data = json.loads(handle.read().decode("utf-8"))
    if not isinstance(data, list) or not all(isinstance(row, dict) for row in data):
        raise TypeError(f"Expected a list of objects in {name}")
    return data


def visible_fraction(
    box: tuple[float, float, float, float], width: int, height: int
) -> float:
    area = (box[2] - box[0]) * (box[3] - box[1])
    if area <= 0:
        return 0.0
    clipped_width = max(0.0, min(box[2], width) - max(box[0], 0.0))
    clipped_height = max(0.0, min(box[3], height) - max(box[1], 0.0))
    return min(1.0, max(0.0, clipped_width * clipped_height / area))


def euclidean_distance(a: list[Any], b: list[Any]) -> float:
    distance = math.sqrt(
        sum((float(a[index]) - float(b[index])) ** 2 for index in range(3))
    )
    if not math.isfinite(distance):
        raise ValueError("Non-finite 3D distance")
    return distance


def object_record(
    instance: dict[str, Any],
    instance_index: int,
    width: int,
    height: int,
    rules: AuditRules,
) -> dict[str, Any]:
    box = valid_box(instance, rules.box_field)
    if box is None:
        raise ValueError(f"Natural instance without valid box: {instance_index}")
    position = instance.get("position")
    if not isinstance(position, list) or len(position) != 3:
        raise ValueError(f"Natural instance without 3D position: {instance_index}")
    return {
        "instance_index": instance_index,
        "instance_id": str(instance.get("id", "")),
        "category": normalized_category(instance),
        "position": [float(value) for value in position],
        "box_2d_rend": list(box),
        "visible_fraction": visible_fraction(box, width, height),
        "interior": interior_valid(instance, width, height, rules),
    }


def build_pairs(objects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for first, second in itertools.combinations(objects, 2):
        distance = euclidean_distance(first["position"], second["position"])
        pairs.append(
            {
                "first_instance_index": first["instance_index"],
                "second_instance_index": second["instance_index"],
                "first_instance_id": first["instance_id"],
                "second_instance_id": second["instance_id"],
                "first_category": first["category"],
                "second_category": second["category"],
                "distance_m": distance,
                "minimum_visible_fraction": min(
                    first["visible_fraction"], second["visible_fraction"]
                ),
                "both_interior": first["interior"] and second["interior"],
            }
        )
    return sorted(
        pairs,
        key=lambda pair: (
            pair["distance_m"],
            pair["first_instance_index"],
            pair["second_instance_index"],
        ),
    )


def distance_summary(pairs: list[dict[str, Any]]) -> dict[str, float | None]:
    distances = [pair["distance_m"] for pair in pairs]
    if not distances:
        return {"minimum_m": None, "median_m": None, "maximum_m": None}
    return {
        "minimum_m": min(distances),
        "median_m": statistics.median(distances),
        "maximum_m": max(distances),
    }


def validate_audit_counts(
    candidate: dict[str, Any], groups: dict[str, list[dict[str, Any]]]
) -> None:
    comparisons = {
        "oracle": len(groups["oracle"]),
        "natural": len(groups["natural"]),
        "natural_interior": len(groups["natural_interior"]),
    }
    mismatches = {
        field: (candidate.get(field), observed)
        for field, observed in comparisons.items()
        if candidate.get(field) != observed
    }
    if mismatches:
        raise ValueError(
            f"Audit-count mismatch for {candidate.get('capture_id')}/"
            f"{candidate.get('frame_index')}: {mismatches}"
        )


def evaluate_candidate(
    candidate: dict[str, Any],
    archive: tarfile.TarFile,
    rules: AuditRules,
    minimum_distance_m: float,
    minimum_pairs: int,
) -> dict[str, Any]:
    instances = read_member_json(archive, str(candidate["instance_member"]))
    width = int(candidate["width"])
    height = int(candidate["height"])
    groups = classify_instances(instances, width, height, rules)
    validate_audit_counts(candidate, groups)
    index_by_identity = {
        id(instance): index
        for index, instance in enumerate(instances, start=1)
    }
    objects = [
        object_record(
            instance,
            index_by_identity[id(instance)],
            width,
            height,
            rules,
        )
        for instance in groups["natural"]
    ]
    all_pairs = build_pairs(objects)
    eligible_pairs = [
        pair for pair in all_pairs if pair["distance_m"] >= minimum_distance_m
    ]
    visibility_counts = {
        f"at_least_{int(threshold * 100)}_percent": sum(
            pair["minimum_visible_fraction"] >= threshold
            for pair in eligible_pairs
        )
        for threshold in VISIBILITY_THRESHOLDS
    }
    result = dict(candidate)
    result.update(
        {
            "natural_objects": objects,
            "number_of_all_natural_pairs": len(all_pairs),
            "number_of_eligible_pairs": len(eligible_pairs),
            "number_of_interior_eligible_pairs": sum(
                pair["both_interior"] for pair in eligible_pairs
            ),
            "eligible_pairs_by_visibility": visibility_counts,
            "eligible_distance_summary": distance_summary(eligible_pairs),
            "basic_pair_feasible": len(eligible_pairs) >= minimum_pairs,
            "eligible_pairs": eligible_pairs,
        }
    )
    return result


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
    return sha256_file(path)


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


def main() -> int:
    args = parse_args()
    project_root = args.project_root.expanduser().resolve()
    shortlist_path = resolve_path(project_root, args.shortlist)
    output_path = resolve_path(project_root, args.output)
    protocol_path = (
        resolve_path(project_root, args.protocol)
        if args.protocol
        else output_path.with_name(f"{output_path.stem}_protocol.json")
    )
    if not shortlist_path.is_file():
        raise FileNotFoundError(f"Shortlist not found: {shortlist_path}")
    if args.minimum_distance_m < 0 or not math.isfinite(args.minimum_distance_m):
        raise ValueError("--minimum-distance-m must be finite and nonnegative")
    if args.minimum_pairs <= 0:
        raise ValueError("--minimum-pairs must be positive")
    for destination in (output_path, protocol_path):
        if destination.exists() and not args.overwrite:
            raise FileExistsError(
                f"Destination exists; pass --overwrite to replace it: {destination}"
            )

    candidates = load_jsonl(shortlist_path)
    required = (
        "archive",
        "capture_id",
        "frame_index",
        "candidate_rank",
        "instance_member",
        "width",
        "height",
        "oracle",
        "natural",
        "natural_interior",
    )
    for candidate in candidates:
        missing = [field for field in required if field not in candidate]
        if missing:
            raise KeyError(f"Missing candidate fields: {', '.join(missing)}")

    grouped: dict[Path, list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        archive_path = resolve_path(project_root, Path(str(candidate["archive"])))
        grouped[archive_path].append(candidate)
    for archive_path in grouped:
        if not archive_path.is_file():
            raise FileNotFoundError(f"Archive not found: {archive_path}")

    rules = AuditRules()
    results: list[dict[str, Any]] = []
    for archive_path, archive_candidates in sorted(
        grouped.items(), key=lambda item: str(item[0])
    ):
        print(f"Reading {archive_path.name} ...")
        with tarfile.open(archive_path, "r") as archive:
            for candidate in sorted(
                archive_candidates,
                key=lambda row: (
                    str(row["capture_id"]),
                    int(row["candidate_rank"]),
                ),
            ):
                results.append(
                    evaluate_candidate(
                        candidate=candidate,
                        archive=archive,
                        rules=rules,
                        minimum_distance_m=args.minimum_distance_m,
                        minimum_pairs=args.minimum_pairs,
                    )
                )
    results.sort(
        key=lambda row: (str(row["capture_id"]), int(row["candidate_rank"]))
    )

    output_sha256 = write_jsonl_atomic(output_path, results)
    selected_results = [
        row for row in results if bool(row.get("selected_by_frame_rule"))
    ]
    visibility_selected = {
        key: sum(
            row["eligible_pairs_by_visibility"][key] >= args.minimum_pairs
            for row in selected_results
        )
        for key in (
            "at_least_80_percent",
            "at_least_90_percent",
            "at_least_100_percent",
        )
    }
    protocol = {
        "schema_version": "1.0",
        "status": "development_pair_feasibility",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "shortlist": display_path(project_root, shortlist_path),
        "shortlist_sha256": sha256_file(shortlist_path),
        "minimum_distance_m": args.minimum_distance_m,
        "minimum_pairs": args.minimum_pairs,
        "visibility_thresholds_reported": list(VISIBILITY_THRESHOLDS),
        "number_of_candidates": len(results),
        "number_basic_pair_feasible": sum(
            row["basic_pair_feasible"] for row in results
        ),
        "number_of_frame_rule_selections": len(selected_results),
        "frame_rule_selections_basic_pair_feasible": sum(
            row["basic_pair_feasible"] for row in selected_results
        ),
        "frame_rule_selections_with_minimum_pairs_by_visibility": (
            visibility_selected
        ),
        "rules": rules.to_dict(),
        "output": display_path(project_root, output_path),
        "output_sha256": output_sha256,
        "next_required_stage": (
            "choose a visibility rule, then construct balanced 12-pair manifests"
        ),
    }
    write_json_atomic(protocol_path, protocol)

    print("Pair-feasibility analysis")
    print(f"Candidates: {len(results)}")
    print(
        "Basic feasible: "
        f"{sum(row['basic_pair_feasible'] for row in results)}/{len(results)}"
    )
    print()
    for row in results:
        visibility = row["eligible_pairs_by_visibility"]
        print(
            f"{row['capture_id']} rank={row['candidate_rank']} "
            f"frame={row['frame_index']} natural={row['natural']} "
            f"pairs={row['number_of_eligible_pairs']} "
            f"interior_pairs={row['number_of_interior_eligible_pairs']} "
            f"vis80={visibility['at_least_80_percent']} "
            f"vis90={visibility['at_least_90_percent']} "
            f"vis100={visibility['at_least_100_percent']} "
            f"feasible={row['basic_pair_feasible']}"
        )
    print()
    print(f"Output: {output_path}")
    print(f"Protocol: {protocol_path}")
    print(f"Output SHA-256: {output_sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
