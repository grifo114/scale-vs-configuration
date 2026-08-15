#!/usr/bin/env python3
"""Rank CA-1M scene candidates without using model outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Rank audited CA-1M frames by explicit selection rules."
    )
    parser.add_argument(
        "audits",
        nargs="+",
        type=Path,
        help="One or more frame-audit JSONL files.",
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
        help="Destination JSONL for the ranked shortlist.",
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        help="Protocol JSON path. Defaults beside the output.",
    )
    parser.add_argument("--minimum-oracle", type=int, default=8)
    parser.add_argument("--minimum-natural", type=int, default=6)
    parser.add_argument("--minimum-natural-interior", type=int, default=3)
    parser.add_argument(
        "--sharpness-quantile",
        type=float,
        default=0.75,
        help="Per-capture sharpness quantile applied after eligibility filters.",
    )
    parser.add_argument(
        "--candidates-per-capture",
        type=int,
        default=5,
        help="Maximum number of temporal-bin winners retained per capture.",
    )
    parser.add_argument(
        "--exclude-capture",
        action="append",
        default=[],
        help="Capture ID to exclude. Repeat for multiple captures.",
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


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def linear_quantile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("Cannot compute a quantile of an empty list")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def finite_float(value: Any, field: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Non-finite {field}: {value!r}")
    return number


def finite_int(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"Invalid {field}: {value!r}")
    number = int(value)
    if number != float(value):
        raise ValueError(f"Non-integral {field}: {value!r}")
    return number


def validated_row(row: dict[str, Any]) -> dict[str, Any]:
    required = (
        "capture_id",
        "frame_index",
        "temporal_bin",
        "oracle",
        "natural",
        "natural_interior",
        "sharpness",
        "image_member",
        "instance_member",
    )
    missing = [field for field in required if field not in row]
    if missing:
        raise KeyError(f"Missing audit fields: {', '.join(missing)}")
    checked = dict(row)
    checked["capture_id"] = str(row["capture_id"])
    for field in (
        "frame_index",
        "temporal_bin",
        "oracle",
        "natural",
        "natural_interior",
    ):
        checked[field] = finite_int(row[field], field)
    checked["sharpness"] = finite_float(row["sharpness"], "sharpness")
    if not 1 <= checked["temporal_bin"] <= 5:
        raise ValueError(f"Invalid temporal_bin: {checked['temporal_bin']}")
    return checked


def rank_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        -row["natural_interior"],
        -row["natural"],
        -row["oracle"],
        -row["sharpness"],
        row["frame_index"],
        str(row["image_member"]),
    )


def eligible(
    row: dict[str, Any],
    minimum_oracle: int,
    minimum_natural: int,
    minimum_natural_interior: int,
) -> bool:
    return (
        row["oracle"] >= minimum_oracle
        and row["natural"] >= minimum_natural
        and row["natural_interior"] >= minimum_natural_interior
    )


def rank_capture(
    rows: list[dict[str, Any]],
    minimum_oracle: int,
    minimum_natural: int,
    minimum_natural_interior: int,
    sharpness_quantile: float,
    candidates_per_capture: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    eligible_rows = [
        row
        for row in rows
        if eligible(
            row,
            minimum_oracle,
            minimum_natural,
            minimum_natural_interior,
        )
    ]
    if not eligible_rows:
        return [], {
            "audited_frames": len(rows),
            "eligible_frames": 0,
            "sharpness_cutoff": None,
            "high_sharpness_frames": 0,
            "temporal_bins_represented": [],
            "shortlist_size": 0,
        }
    cutoff = linear_quantile(
        [row["sharpness"] for row in eligible_rows], sharpness_quantile
    )
    high_sharpness = [
        row for row in eligible_rows if row["sharpness"] >= cutoff
    ]
    by_bin: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in high_sharpness:
        by_bin[row["temporal_bin"]].append(row)
    bin_winners = [
        sorted(bin_rows, key=rank_key)[0]
        for _, bin_rows in sorted(by_bin.items())
    ]
    shortlist = sorted(bin_winners, key=rank_key)[:candidates_per_capture]
    ranked: list[dict[str, Any]] = []
    for candidate_rank, row in enumerate(shortlist, start=1):
        candidate = dict(row)
        candidate.update(
            {
                "candidate_rank": candidate_rank,
                "selected_by_frame_rule": candidate_rank == 1,
                "capture_sharpness_cutoff": cutoff,
                "capture_eligible_frames": len(eligible_rows),
                "capture_high_sharpness_frames": len(high_sharpness),
            }
        )
        ranked.append(candidate)
    summary = {
        "audited_frames": len(rows),
        "eligible_frames": len(eligible_rows),
        "sharpness_cutoff": cutoff,
        "high_sharpness_frames": len(high_sharpness),
        "temporal_bins_represented": sorted(by_bin),
        "shortlist_size": len(ranked),
    }
    return ranked, summary


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
    audit_paths = [resolve_path(project_root, path) for path in args.audits]
    output_path = resolve_path(project_root, args.output)
    protocol_path = (
        resolve_path(project_root, args.protocol)
        if args.protocol
        else output_path.with_name(f"{output_path.stem}_protocol.json")
    )
    if args.minimum_oracle < 0 or args.minimum_natural < 0:
        raise ValueError("Minimum counts must be nonnegative")
    if args.minimum_natural_interior < 0:
        raise ValueError("Minimum counts must be nonnegative")
    if not 0 <= args.sharpness_quantile <= 1:
        raise ValueError("--sharpness-quantile must be between zero and one")
    if not 1 <= args.candidates_per_capture <= 5:
        raise ValueError("--candidates-per-capture must be between one and five")
    for path in audit_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Audit not found: {path}")
    for destination in (output_path, protocol_path):
        if destination.exists() and not args.overwrite:
            raise FileExistsError(
                f"Destination exists; pass --overwrite to replace it: {destination}"
            )

    excluded = {str(value) for value in args.exclude_capture}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    identities: set[tuple[str, str]] = set()
    total_rows = 0
    for path in audit_paths:
        for raw_row in load_jsonl(path):
            row = validated_row(raw_row)
            total_rows += 1
            capture_id = row["capture_id"]
            if capture_id in excluded:
                continue
            identity = (capture_id, str(row["instance_member"]))
            if identity in identities:
                raise ValueError(f"Duplicate audited frame: {identity}")
            identities.add(identity)
            grouped[capture_id].append(row)

    ranked_rows: list[dict[str, Any]] = []
    capture_summaries: dict[str, dict[str, Any]] = {}
    rejected_captures: list[str] = []
    for capture_id, rows in sorted(grouped.items()):
        ranked, summary = rank_capture(
            rows=rows,
            minimum_oracle=args.minimum_oracle,
            minimum_natural=args.minimum_natural,
            minimum_natural_interior=args.minimum_natural_interior,
            sharpness_quantile=args.sharpness_quantile,
            candidates_per_capture=args.candidates_per_capture,
        )
        capture_summaries[capture_id] = summary
        if not ranked:
            rejected_captures.append(capture_id)
        ranked_rows.extend(ranked)

    output_sha256 = write_jsonl_atomic(output_path, ranked_rows)
    protocol = {
        "schema_version": "1.0",
        "status": "development_candidate_ranking",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": [
            {
                "path": display_path(project_root, path),
                "sha256": sha256_file(path),
            }
            for path in audit_paths
        ],
        "input_rows": total_rows,
        "excluded_captures": sorted(excluded),
        "selection_rule": {
            "minimum_oracle": args.minimum_oracle,
            "minimum_natural": args.minimum_natural,
            "minimum_natural_interior": args.minimum_natural_interior,
            "sharpness_quantile": args.sharpness_quantile,
            "quantile_interpolation": "linear, position=(n-1)*q",
            "sharpness_scope": "eligible frames within each capture",
            "ranking_priority": [
                "natural_interior descending",
                "natural descending",
                "oracle descending",
                "sharpness descending",
                "frame_index ascending",
                "image_member ascending",
            ],
            "temporal_diversity": "retain the best frame per temporal bin",
            "candidates_per_capture": args.candidates_per_capture,
            "uses_model_outputs": False,
        },
        "captures_considered": len(grouped),
        "captures_with_shortlist": len(grouped) - len(rejected_captures),
        "rejected_captures": rejected_captures,
        "shortlist_rows": len(ranked_rows),
        "capture_summaries": capture_summaries,
        "output": display_path(project_root, output_path),
        "output_sha256": output_sha256,
        "next_required_stage": "3D pair-feasibility validation",
    }
    write_json_atomic(protocol_path, protocol)

    print("Scene-candidate ranking")
    print(f"Input rows: {total_rows}")
    print(f"Captures considered: {len(grouped)}")
    print(f"Captures with shortlist: {len(grouped) - len(rejected_captures)}")
    print(f"Rejected captures: {len(rejected_captures)}")
    print(f"Shortlist rows: {len(ranked_rows)}")
    print()
    for capture_id in sorted(grouped):
        candidates = [
            row for row in ranked_rows if row["capture_id"] == capture_id
        ]
        if not candidates:
            print(f"{capture_id}: rejected")
            continue
        chosen = candidates[0]
        print(
            f"{capture_id}: frame={chosen['frame_index']} "
            f"bin={chosen['temporal_bin']} "
            f"natural={chosen['natural']} "
            f"interior={chosen['natural_interior']} "
            f"oracle={chosen['oracle']} "
            f"sharpness={chosen['sharpness']:.1f}"
        )
    print()
    print(f"Output: {output_path}")
    print(f"Protocol: {protocol_path}")
    print(f"Output SHA-256: {output_sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
