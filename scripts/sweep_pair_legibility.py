#!/usr/bin/env python3
"""Sweep deterministic object-legibility thresholds over scene candidates."""

from __future__ import annotations

import argparse
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_AREA_FRACTIONS = (0.00125, 0.00250, 0.00375, 0.00500)
DEFAULT_MINIMUM_SIDES_PX = (20.0, 30.0, 40.0, 50.0)
DEFAULT_BORDER_MARGINS_PX = (5.0, 10.0, 15.0, 20.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Count pair candidates under global object-legibility thresholds."
        )
    )
    parser.add_argument("feasibility", type=Path, help="Pair-feasibility JSONL.")
    parser.add_argument("--minimum-distance-m", type=float, default=0.25)
    parser.add_argument("--minimum-visible-fraction", type=float, default=1.0)
    parser.add_argument("--maximum-box-iou", type=float, default=0.50)
    parser.add_argument(
        "--maximum-intersection-over-smaller", type=float, default=0.80
    )
    parser.add_argument("--minimum-pairs", type=int, default=12)
    parser.add_argument("--top-configurations", type=int, default=12)
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON report. The console summary is always printed.",
    )
    return parser.parse_args()


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


def hard_pair_filter(pair: dict[str, Any], args: argparse.Namespace) -> bool:
    return (
        float(pair["distance_m"]) >= args.minimum_distance_m
        and float(pair["minimum_visible_fraction"])
        >= args.minimum_visible_fraction
        and float(pair["box_iou"]) <= args.maximum_box_iou
        and float(pair["intersection_over_smaller_box"])
        <= args.maximum_intersection_over_smaller
    )


def object_legibility(
    obj: dict[str, Any], width: int, height: int
) -> dict[str, float]:
    box = [float(value) for value in obj["box_2d_rend"]]
    box_width = box[2] - box[0]
    box_height = box[3] - box[1]
    if box_width <= 0 or box_height <= 0:
        raise ValueError(f"Invalid object box: {box}")
    margin = min(box[0], box[1], width - box[2], height - box[3])
    return {
        "area_fraction": (box_width * box_height) / (width * height),
        "minimum_side_px": min(box_width, box_height),
        "border_margin_px": margin,
    }


def enrich_pairs(
    row: dict[str, Any], args: argparse.Namespace
) -> list[dict[str, Any]]:
    width = int(row["width"])
    height = int(row["height"])
    objects = {
        int(obj["instance_index"]): object_legibility(obj, width, height)
        for obj in row["natural_objects"]
    }
    enriched: list[dict[str, Any]] = []
    for pair in row["eligible_pairs"]:
        if not hard_pair_filter(pair, args):
            continue
        first = objects[int(pair["first_instance_index"])]
        second = objects[int(pair["second_instance_index"])]
        enriched.append(
            {
                "minimum_area_fraction": min(
                    first["area_fraction"], second["area_fraction"]
                ),
                "minimum_side_px": min(
                    first["minimum_side_px"], second["minimum_side_px"]
                ),
                "minimum_border_margin_px": min(
                    first["border_margin_px"], second["border_margin_px"]
                ),
            }
        )
    return enriched


def count_pairs(
    pairs: list[dict[str, Any]],
    minimum_area_fraction: float,
    minimum_side_px: float,
    minimum_border_margin_px: float,
) -> int:
    return sum(
        pair["minimum_area_fraction"] >= minimum_area_fraction
        and pair["minimum_side_px"] >= minimum_side_px
        and pair["minimum_border_margin_px"] >= minimum_border_margin_px
        for pair in pairs
    )


def main() -> int:
    args = parse_args()
    if args.minimum_pairs <= 0:
        raise ValueError("--minimum-pairs must be positive")
    thresholds = (
        args.minimum_distance_m,
        args.minimum_visible_fraction,
        args.maximum_box_iou,
        args.maximum_intersection_over_smaller,
    )
    if not all(math.isfinite(value) for value in thresholds):
        raise ValueError("Pair thresholds must be finite")
    rows = load_jsonl(args.feasibility.expanduser().resolve())
    if not rows:
        raise ValueError("No candidate rows found")

    candidates: list[dict[str, Any]] = []
    by_capture: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        candidate = {
            "capture_id": str(row["capture_id"]),
            "candidate_rank": int(row["candidate_rank"]),
            "frame_index": int(row["frame_index"]),
            "selected_by_frame_rule": bool(row.get("selected_by_frame_rule")),
            "pairs": enrich_pairs(row, args),
        }
        candidates.append(candidate)
        by_capture[candidate["capture_id"]].append(candidate)

    configurations: list[dict[str, Any]] = []
    for area, side, margin in itertools.product(
        DEFAULT_AREA_FRACTIONS,
        DEFAULT_MINIMUM_SIDES_PX,
        DEFAULT_BORDER_MARGINS_PX,
    ):
        capture_results: dict[str, dict[str, Any]] = {}
        for capture_id, capture_candidates in sorted(by_capture.items()):
            evaluated = []
            for candidate in capture_candidates:
                count = count_pairs(candidate["pairs"], area, side, margin)
                evaluated.append(
                    {
                        "candidate_rank": candidate["candidate_rank"],
                        "frame_index": candidate["frame_index"],
                        "selected_by_frame_rule": candidate[
                            "selected_by_frame_rule"
                        ],
                        "pair_count": count,
                    }
                )
            evaluated.sort(
                key=lambda item: (
                    -item["pair_count"],
                    item["candidate_rank"],
                    item["frame_index"],
                )
            )
            capture_results[capture_id] = {
                "feasible": evaluated[0]["pair_count"] >= args.minimum_pairs,
                "best_candidate": evaluated[0],
                "candidates": evaluated,
            }
        feasible_captures = sum(
            result["feasible"] for result in capture_results.values()
        )
        configurations.append(
            {
                "minimum_area_fraction": area,
                "minimum_side_px": side,
                "minimum_border_margin_px": margin,
                "feasible_captures": feasible_captures,
                "minimum_best_pair_count": min(
                    result["best_candidate"]["pair_count"]
                    for result in capture_results.values()
                ),
                "captures": capture_results,
            }
        )

    number_of_captures = len(by_capture)
    full_coverage = [
        config
        for config in configurations
        if config["feasible_captures"] == number_of_captures
    ]
    full_coverage.sort(
        key=lambda config: (
            -config["minimum_border_margin_px"],
            -config["minimum_side_px"],
            -config["minimum_area_fraction"],
            -config["minimum_best_pair_count"],
        )
    )

    selected_counts = []
    for candidate in sorted(
        (item for item in candidates if item["selected_by_frame_rule"]),
        key=lambda item: item["capture_id"],
    ):
        selected_counts.append(
            {
                "capture_id": candidate["capture_id"],
                "candidate_rank": candidate["candidate_rank"],
                "frame_index": candidate["frame_index"],
                "baseline_pairs": len(candidate["pairs"]),
                "moderate_pairs": count_pairs(
                    candidate["pairs"], 0.0025, 30.0, 10.0
                ),
                "strict_pairs": count_pairs(
                    candidate["pairs"], 0.0050, 40.0, 20.0
                ),
            }
        )

    report = {
        "input": str(args.feasibility),
        "number_of_captures": number_of_captures,
        "number_of_candidates": len(candidates),
        "minimum_pairs": args.minimum_pairs,
        "hard_pair_filters": {
            "minimum_distance_m": args.minimum_distance_m,
            "minimum_visible_fraction": args.minimum_visible_fraction,
            "maximum_box_iou": args.maximum_box_iou,
            "maximum_intersection_over_smaller": (
                args.maximum_intersection_over_smaller
            ),
        },
        "selected_scene_diagnostics": selected_counts,
        "full_coverage_configurations": full_coverage,
        "all_configurations": configurations,
    }
    if args.output:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    print("Pair-legibility sweep")
    print(f"Captures: {number_of_captures}")
    print(f"Candidates: {len(candidates)}")
    print(f"Configurations: {len(configurations)}")
    print(f"Full-coverage configurations: {len(full_coverage)}")
    print()
    print("Current selected scenes:")
    for item in selected_counts:
        print(
            f"{item['capture_id']} rank={item['candidate_rank']} "
            f"frame={item['frame_index']} baseline={item['baseline_pairs']} "
            f"moderate={item['moderate_pairs']} strict={item['strict_pairs']}"
        )
    print()
    print("Strictest full-coverage configurations:")
    for index, config in enumerate(
        full_coverage[: args.top_configurations], start=1
    ):
        choices = ", ".join(
            f"{capture_id}:r{result['best_candidate']['candidate_rank']}="
            f"{result['best_candidate']['pair_count']}"
            for capture_id, result in config["captures"].items()
        )
        print(
            f"{index:02d}. area>={config['minimum_area_fraction']:.5f} "
            f"side>={config['minimum_side_px']:.0f}px "
            f"margin>={config['minimum_border_margin_px']:.0f}px "
            f"min_support={config['minimum_best_pair_count']} | {choices}"
        )
    if args.output:
        print()
        print(f"Report: {args.output.expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
