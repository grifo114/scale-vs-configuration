#!/usr/bin/env python3
"""Select balanced 3D relation pairs from CA-1M scene candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STRATUM_NAMES = ("short", "medium", "long")


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Select 12 balanced relation pairs per chosen scene."
    )
    parser.add_argument("feasibility", type=Path, help="Pair-feasibility JSONL.")
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
        help="Destination selected-pair manifest JSONL.",
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        help="Protocol JSON path. Defaults beside the output.",
    )
    parser.add_argument("--minimum-distance-m", type=float, default=0.25)
    parser.add_argument("--minimum-visible-fraction", type=float, default=1.0)
    parser.add_argument("--maximum-box-iou", type=float, default=0.50)
    parser.add_argument(
        "--maximum-intersection-over-smaller", type=float, default=0.80
    )
    parser.add_argument(
        "--minimum-box-area-fraction", type=float, default=0.00375
    )
    parser.add_argument("--minimum-box-side-px", type=float, default=30.0)
    parser.add_argument("--minimum-border-margin-px", type=float, default=20.0)
    parser.add_argument("--pairs-per-scene", type=int, default=12)
    parser.add_argument("--distance-strata", type=int, default=3)
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


def partition_evenly(items: list[Any], groups: int) -> list[list[Any]]:
    if groups <= 0:
        raise ValueError("groups must be positive")
    if len(items) < groups:
        raise ValueError(f"Cannot split {len(items)} items into {groups} nonempty groups")
    base, remainder = divmod(len(items), groups)
    partitions: list[list[Any]] = []
    start = 0
    for index in range(groups):
        size = base + (1 if index < remainder else 0)
        partitions.append(items[start : start + size])
        start += size
    return partitions


def pair_quality_key(pair: dict[str, Any]) -> tuple[Any, ...]:
    return (
        -float(pair["minimum_box_area_px2"]),
        -float(pair["box_center_distance_px"]),
        float(pair["box_iou"]),
        float(pair["intersection_over_smaller_box"]),
        not bool(pair["both_interior"]),
        int(pair["first_instance_index"]),
        int(pair["second_instance_index"]),
    )


def hard_pair_filter(
    pair: dict[str, Any],
    minimum_distance_m: float,
    minimum_visible_fraction: float,
    maximum_box_iou: float,
    maximum_intersection_over_smaller: float,
    minimum_box_area_fraction: float,
    minimum_box_side_px: float,
    minimum_border_margin_px: float,
) -> bool:
    return (
        float(pair["distance_m"]) >= minimum_distance_m
        and float(pair["minimum_visible_fraction"])
        >= minimum_visible_fraction
        and float(pair["box_iou"]) <= maximum_box_iou
        and float(pair["intersection_over_smaller_box"])
        <= maximum_intersection_over_smaller
        and float(pair["minimum_box_area_fraction"])
        >= minimum_box_area_fraction
        and float(pair["minimum_box_side_px"]) >= minimum_box_side_px
        and float(pair["minimum_border_margin_px"])
        >= minimum_border_margin_px
    )


def object_legibility(
    obj: dict[str, Any], width: int, height: int
) -> dict[str, float]:
    box = [float(value) for value in obj["box_2d_rend"]]
    box_width = box[2] - box[0]
    box_height = box[3] - box[1]
    if box_width <= 0 or box_height <= 0:
        raise ValueError(f"Invalid object box: {box}")
    return {
        "box_area_fraction": (box_width * box_height) / (width * height),
        "box_minimum_side_px": min(box_width, box_height),
        "box_border_margin_px": min(
            box[0], box[1], width - box[2], height - box[3]
        ),
    }


def hard_eligible_pairs(
    scene: dict[str, Any],
    minimum_distance_m: float,
    minimum_visible_fraction: float,
    maximum_box_iou: float,
    maximum_intersection_over_smaller: float,
    minimum_box_area_fraction: float,
    minimum_box_side_px: float,
    minimum_border_margin_px: float,
) -> list[dict[str, Any]]:
    width = int(scene["width"])
    height = int(scene["height"])
    objects = {
        int(obj["instance_index"]): object_legibility(obj, width, height)
        for obj in scene["natural_objects"]
    }
    enriched_pairs: list[dict[str, Any]] = []
    for source_pair in scene["eligible_pairs"]:
        pair = dict(source_pair)
        first = objects[int(pair["first_instance_index"])]
        second = objects[int(pair["second_instance_index"])]
        pair.update(
            {
                "minimum_box_area_fraction": min(
                    first["box_area_fraction"],
                    second["box_area_fraction"],
                ),
                "minimum_box_side_px": min(
                    first["box_minimum_side_px"],
                    second["box_minimum_side_px"],
                ),
                "minimum_border_margin_px": min(
                    first["box_border_margin_px"],
                    second["box_border_margin_px"],
                ),
            }
        )
        if hard_pair_filter(
            pair,
            minimum_distance_m,
            minimum_visible_fraction,
            maximum_box_iou,
            maximum_intersection_over_smaller,
            minimum_box_area_fraction,
            minimum_box_side_px,
            minimum_border_margin_px,
        ):
            enriched_pairs.append(pair)
    return sorted(
        enriched_pairs,
        key=lambda pair: (
            float(pair["distance_m"]),
            int(pair["first_instance_index"]),
            int(pair["second_instance_index"]),
        ),
    )


def select_scene_pairs(
    scene: dict[str, Any],
    minimum_distance_m: float,
    minimum_visible_fraction: float,
    maximum_box_iou: float,
    maximum_intersection_over_smaller: float,
    minimum_box_area_fraction: float,
    minimum_box_side_px: float,
    minimum_border_margin_px: float,
    pairs_per_scene: int,
    distance_strata: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    hard_eligible = hard_eligible_pairs(
        scene,
        minimum_distance_m,
        minimum_visible_fraction,
        maximum_box_iou,
        maximum_intersection_over_smaller,
        minimum_box_area_fraction,
        minimum_box_side_px,
        minimum_border_margin_px,
    )
    if len(hard_eligible) < pairs_per_scene:
        raise ValueError(
            f"Capture {scene['capture_id']} has {len(hard_eligible)} eligible pairs; "
            f"{pairs_per_scene} required"
        )
    interior_eligible = [
        pair for pair in hard_eligible if bool(pair["both_interior"])
    ]
    if len(interior_eligible) >= pairs_per_scene:
        selection_pool = interior_eligible
        interior_policy = "interior_only"
    else:
        selection_pool = hard_eligible
        interior_policy = "full_visible_fallback"
    pairs_per_stratum = pairs_per_scene // distance_strata
    distance_partitions = partition_evenly(selection_pool, distance_strata)
    selected: list[dict[str, Any]] = []
    stratum_summaries: dict[str, dict[str, Any]] = {}
    for stratum_index, (stratum_name, stratum_pairs) in enumerate(
        zip(STRATUM_NAMES, distance_partitions), start=1
    ):
        subranges = partition_evenly(stratum_pairs, pairs_per_stratum)
        stratum_selected = [
            min(subrange, key=pair_quality_key) for subrange in subranges
        ]
        stratum_selected.sort(
            key=lambda pair: (
                float(pair["distance_m"]),
                int(pair["first_instance_index"]),
                int(pair["second_instance_index"]),
            )
        )
        for within_stratum_index, pair in enumerate(stratum_selected, start=1):
            chosen = dict(pair)
            chosen["distance_stratum"] = stratum_name
            chosen["distance_stratum_index"] = stratum_index
            chosen["within_stratum_index"] = within_stratum_index
            selected.append(chosen)
        stratum_distances = [float(pair["distance_m"]) for pair in stratum_pairs]
        selected_distances = [
            float(pair["distance_m"]) for pair in stratum_selected
        ]
        stratum_summaries[stratum_name] = {
            "candidate_pairs": len(stratum_pairs),
            "candidate_minimum_m": min(stratum_distances),
            "candidate_maximum_m": max(stratum_distances),
            "selected_minimum_m": min(selected_distances),
            "selected_maximum_m": max(selected_distances),
        }

    objects = {
        int(obj["instance_index"]): obj for obj in scene["natural_objects"]
    }
    manifest_rows: list[dict[str, Any]] = []
    object_usage: Counter[int] = Counter()
    for pair_index, pair in enumerate(selected, start=1):
        first_index = int(pair["first_instance_index"])
        second_index = int(pair["second_instance_index"])
        object_usage[first_index] += 1
        object_usage[second_index] += 1
        fold = "A" if pair_index % 2 == 1 else "B"
        manifest_rows.append(
            {
                "scene_id": str(scene["capture_id"]),
                "capture_id": str(scene["capture_id"]),
                "scene_candidate_rank": int(scene["candidate_rank"]),
                "frame_index": int(scene["frame_index"]),
                "temporal_bin": int(scene["temporal_bin"]),
                "archive": str(scene["archive"]),
                "image_member": str(scene["image_member"]),
                "instance_member": str(scene["instance_member"]),
                "pair_id": f"{scene['capture_id']}-P{pair_index:02d}",
                "pair_index": pair_index,
                "fold": fold,
                "distance_stratum": pair["distance_stratum"],
                "distance_stratum_index": pair["distance_stratum_index"],
                "within_stratum_index": pair["within_stratum_index"],
                "ground_truth_m": float(pair["distance_m"]),
                "object_a": objects[first_index],
                "object_b": objects[second_index],
                "pair_quality": {
                    "both_interior": bool(pair["both_interior"]),
                    "minimum_visible_fraction": float(
                        pair["minimum_visible_fraction"]
                    ),
                    "box_iou": float(pair["box_iou"]),
                    "intersection_over_smaller_box": float(
                        pair["intersection_over_smaller_box"]
                    ),
                    "minimum_box_area_px2": float(pair["minimum_box_area_px2"]),
                    "minimum_box_area_fraction": float(
                        pair["minimum_box_area_fraction"]
                    ),
                    "minimum_box_side_px": float(
                        pair["minimum_box_side_px"]
                    ),
                    "minimum_border_margin_px": float(
                        pair["minimum_border_margin_px"]
                    ),
                    "box_center_distance_px": float(
                        pair["box_center_distance_px"]
                    ),
                },
            }
        )
    fold_strata = Counter(
        (row["fold"], row["distance_stratum"]) for row in manifest_rows
    )
    expected_per_fold_stratum = pairs_per_stratum // 2
    for fold in ("A", "B"):
        for stratum_name in STRATUM_NAMES:
            observed = fold_strata[(fold, stratum_name)]
            if observed != expected_per_fold_stratum:
                raise AssertionError(
                    f"Unexpected {fold}/{stratum_name} count: {observed}"
                )
    distances = [row["ground_truth_m"] for row in manifest_rows]
    scene_summary = {
        "candidate_rank": int(scene["candidate_rank"]),
        "frame_index": int(scene["frame_index"]),
        "hard_filter_pairs": len(hard_eligible),
        "interior_filter_pairs": len(interior_eligible),
        "selection_pool_pairs": len(selection_pool),
        "interior_policy": interior_policy,
        "selected_pairs": len(manifest_rows),
        "selected_minimum_m": min(distances),
        "selected_maximum_m": max(distances),
        "selected_interior_pairs": sum(
            row["pair_quality"]["both_interior"] for row in manifest_rows
        ),
        "unique_objects_used": len(object_usage),
        "maximum_object_uses": max(object_usage.values()),
        "object_usage": {
            str(index): count for index, count in sorted(object_usage.items())
        },
        "distance_strata": stratum_summaries,
    }
    return manifest_rows, scene_summary


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
    feasibility_path = resolve_path(project_root, args.feasibility)
    output_path = resolve_path(project_root, args.output)
    protocol_path = (
        resolve_path(project_root, args.protocol)
        if args.protocol
        else output_path.with_name(f"{output_path.stem}_protocol.json")
    )
    if not feasibility_path.is_file():
        raise FileNotFoundError(f"Feasibility file not found: {feasibility_path}")
    finite_thresholds = (
        args.minimum_distance_m,
        args.minimum_visible_fraction,
        args.maximum_box_iou,
        args.maximum_intersection_over_smaller,
        args.minimum_box_area_fraction,
        args.minimum_box_side_px,
        args.minimum_border_margin_px,
    )
    if not all(math.isfinite(value) for value in finite_thresholds):
        raise ValueError("Pair thresholds must be finite")
    if args.minimum_distance_m < 0:
        raise ValueError("--minimum-distance-m must be nonnegative")
    if not 0 <= args.minimum_visible_fraction <= 1:
        raise ValueError("--minimum-visible-fraction must be between zero and one")
    if not 0 <= args.maximum_box_iou <= 1:
        raise ValueError("--maximum-box-iou must be between zero and one")
    if not 0 <= args.maximum_intersection_over_smaller <= 1:
        raise ValueError(
            "--maximum-intersection-over-smaller must be between zero and one"
        )
    if not 0 <= args.minimum_box_area_fraction <= 1:
        raise ValueError(
            "--minimum-box-area-fraction must be between zero and one"
        )
    if args.minimum_box_side_px < 0:
        raise ValueError("--minimum-box-side-px must be nonnegative")
    if args.minimum_border_margin_px < 0:
        raise ValueError("--minimum-border-margin-px must be nonnegative")
    if args.distance_strata != len(STRATUM_NAMES):
        raise ValueError(f"--distance-strata must be {len(STRATUM_NAMES)}")
    if args.pairs_per_scene <= 0 or args.pairs_per_scene % args.distance_strata:
        raise ValueError("--pairs-per-scene must be positive and divisible by strata")
    pairs_per_stratum = args.pairs_per_scene // args.distance_strata
    if pairs_per_stratum % 2:
        raise ValueError("Pairs per distance stratum must be even for folds A/B")
    for destination in (output_path, protocol_path):
        if destination.exists() and not args.overwrite:
            raise FileExistsError(
                f"Destination exists; pass --overwrite to replace it: {destination}"
            )

    rows = load_jsonl(feasibility_path)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["capture_id"]), []).append(row)
    selected_scenes: list[dict[str, Any]] = []
    rejected_captures: list[str] = []
    scene_selection_summaries: dict[str, dict[str, Any]] = {}
    for capture_id, candidates in sorted(grouped.items()):
        evaluated: list[tuple[int, dict[str, Any]]] = []
        for candidate in candidates:
            pair_count = len(
                hard_eligible_pairs(
                    candidate,
                    args.minimum_distance_m,
                    args.minimum_visible_fraction,
                    args.maximum_box_iou,
                    args.maximum_intersection_over_smaller,
                    args.minimum_box_area_fraction,
                    args.minimum_box_side_px,
                    args.minimum_border_margin_px,
                )
            )
            evaluated.append((pair_count, candidate))
        evaluated.sort(
            key=lambda item: (
                -item[0],
                int(item[1]["candidate_rank"]),
                int(item[1]["frame_index"]),
            )
        )
        best_count, best_candidate = evaluated[0]
        scene_selection_summaries[capture_id] = {
            "candidate_frames": len(candidates),
            "maximum_eligible_pairs": best_count,
            "selected": best_count >= args.pairs_per_scene,
            "selected_candidate_rank": (
                int(best_candidate["candidate_rank"])
                if best_count >= args.pairs_per_scene
                else None
            ),
            "selected_frame_index": (
                int(best_candidate["frame_index"])
                if best_count >= args.pairs_per_scene
                else None
            ),
        }
        if best_count >= args.pairs_per_scene:
            selected_scenes.append(best_candidate)
        else:
            rejected_captures.append(capture_id)
    if not selected_scenes:
        raise ValueError("No capture has enough legible relation pairs")

    manifest: list[dict[str, Any]] = []
    scene_summaries: dict[str, dict[str, Any]] = {}
    for scene in sorted(selected_scenes, key=lambda row: str(row["capture_id"])):
        scene_rows, scene_summary = select_scene_pairs(
            scene=scene,
            minimum_distance_m=args.minimum_distance_m,
            minimum_visible_fraction=args.minimum_visible_fraction,
            maximum_box_iou=args.maximum_box_iou,
            maximum_intersection_over_smaller=(
                args.maximum_intersection_over_smaller
            ),
            minimum_box_area_fraction=args.minimum_box_area_fraction,
            minimum_box_side_px=args.minimum_box_side_px,
            minimum_border_margin_px=args.minimum_border_margin_px,
            pairs_per_scene=args.pairs_per_scene,
            distance_strata=args.distance_strata,
        )
        manifest.extend(scene_rows)
        scene_summaries[str(scene["capture_id"])] = scene_summary

    output_sha256 = write_jsonl_atomic(output_path, manifest)
    protocol = {
        "schema_version": "1.0",
        "status": "development_balanced_pair_selection",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": display_path(project_root, feasibility_path),
        "input_sha256": sha256_file(feasibility_path),
        "scene_rule": (
            "choose the candidate frame with the largest hard-filtered pair "
            "pool; break ties by candidate_rank then frame_index; reject a "
            "capture when its best frame has fewer than pairs_per_scene pairs"
        ),
        "hard_pair_filters": {
            "minimum_distance_m": args.minimum_distance_m,
            "minimum_visible_fraction": args.minimum_visible_fraction,
            "maximum_box_iou": args.maximum_box_iou,
            "maximum_intersection_over_smaller_box": (
                args.maximum_intersection_over_smaller
            ),
            "minimum_box_area_fraction": args.minimum_box_area_fraction,
            "minimum_box_side_px": args.minimum_box_side_px,
            "minimum_border_margin_px": args.minimum_border_margin_px,
        },
        "selection": {
            "pairs_per_scene": args.pairs_per_scene,
            "distance_strata": list(STRATUM_NAMES),
            "pairs_per_stratum": pairs_per_stratum,
            "distance_partition": (
                "sort eligible pairs by 3D distance and split contiguously "
                "into three groups with sizes differing by at most one"
            ),
            "within_stratum_partition": (
                "split each distance stratum contiguously into four "
                "subranges and select one pair per subrange"
            ),
            "interior_pool_policy": (
                "use only pairs with both boxes at least 5 px from the image "
                "border when at least 12 such pairs exist; otherwise use the "
                "full hard-filtered pool"
            ),
            "within_subrange_priority": [
                "minimum 2D box area descending",
                "2D box-center distance descending",
                "box IoU ascending",
                "intersection over smaller box ascending",
                "both objects interior first",
                "instance indices ascending",
            ],
            "fold_assignment": "odd pair indices=A; even pair indices=B",
        },
        "number_of_scenes": len(selected_scenes),
        "number_of_rejected_captures": len(rejected_captures),
        "rejected_captures": rejected_captures,
        "scene_selection_summaries": scene_selection_summaries,
        "number_of_pairs": len(manifest),
        "scene_summaries": scene_summaries,
        "output": display_path(project_root, output_path),
        "output_sha256": output_sha256,
        "next_required_stage": "render and visually audit red/blue pair stimuli",
    }
    write_json_atomic(protocol_path, protocol)

    print("Balanced pair selection")
    print(f"Scenes: {len(selected_scenes)}")
    print(f"Rejected captures: {len(rejected_captures)}")
    if rejected_captures:
        print(f"Rejected IDs: {', '.join(rejected_captures)}")
    print(f"Pairs: {len(manifest)}")
    print()
    for scene_id, summary in scene_summaries.items():
        print(
            f"{scene_id}: rank={summary['candidate_rank']} "
            f"frame={summary['frame_index']} "
            f"eligible={summary['hard_filter_pairs']} "
            f"interior_pool={summary['interior_filter_pairs']} "
            f"policy={summary['interior_policy']} "
            f"selected={summary['selected_pairs']} "
            f"distance={summary['selected_minimum_m']:.3f}-"
            f"{summary['selected_maximum_m']:.3f} m "
            f"interior={summary['selected_interior_pairs']} "
            f"objects={summary['unique_objects_used']} "
            f"max_uses={summary['maximum_object_uses']}"
        )
    print()
    print(f"Output: {output_path}")
    print(f"Protocol: {protocol_path}")
    print(f"Output SHA-256: {output_sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
