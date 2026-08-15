#!/usr/bin/env python3
"""Apply the frozen confirmatory scene and pair acceptance protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from select_balanced_pairs import (
    hard_eligible_pairs,
    load_jsonl,
    select_scene_pairs,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Select or reject captures using the frozen confirmatory protocol."
    )
    parser.add_argument("feasibility", type=Path, help="Pair-feasibility JSONL.")
    parser.add_argument(
        "--confirmatory-protocol",
        type=Path,
        default=Path("configs/confirmatory_capture_protocol.json"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=project_root)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_path(project_root: Path, path: Path) -> Path:
    path = path.expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def display_path(project_root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(project_root))
    except ValueError:
        return str(path)


def read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return data


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
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def frozen_rules(config: dict[str, Any]) -> dict[str, Any]:
    pair = config["pair_filters"]
    scene = config["scene_acceptance"]
    if pair.get("both_objects_interior") is not True:
        raise ValueError("Confirmatory protocol must require both objects interior")
    if int(scene["minimum_eligible_pairs"]) != int(scene["selected_pairs"]):
        raise ValueError("Minimum and selected pair counts must match")
    if int(scene["pairs_per_stratum"]) * len(scene["distance_strata"]) != int(
        scene["selected_pairs"]
    ):
        raise ValueError("Distance-stratum counts do not match selected_pairs")
    return {
        "minimum_distance_m": float(pair["minimum_distance_m"]),
        "minimum_visible_fraction": float(pair["minimum_visible_fraction"]),
        "maximum_box_iou": float(pair["maximum_pair_iou"]),
        "maximum_intersection_over_smaller": float(
            pair["maximum_intersection_over_smaller_box"]
        ),
        "minimum_box_area_fraction": float(pair["minimum_box_area_fraction"]),
        "minimum_box_side_px": float(pair["minimum_box_side_px"]),
        "minimum_border_margin_px": float(pair["minimum_border_margin_px"]),
        "pairs_per_scene": int(scene["selected_pairs"]),
        "distance_strata": len(scene["distance_strata"]),
        "minimum_unique_objects": int(scene["minimum_unique_objects"]),
        "maximum_uses_per_object": int(scene["maximum_uses_per_object"]),
        "minimum_distance_ratio": float(scene["minimum_distance_ratio"]),
    }


def strict_pairs(scene: dict[str, Any], rules: dict[str, Any]) -> list[dict[str, Any]]:
    pairs = hard_eligible_pairs(
        scene,
        rules["minimum_distance_m"],
        rules["minimum_visible_fraction"],
        rules["maximum_box_iou"],
        rules["maximum_intersection_over_smaller"],
        rules["minimum_box_area_fraction"],
        rules["minimum_box_side_px"],
        rules["minimum_border_margin_px"],
    )
    return [pair for pair in pairs if bool(pair["both_interior"])]


def rejection_reasons(summary: dict[str, Any], rules: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if int(summary["unique_objects_used"]) < rules["minimum_unique_objects"]:
        reasons.append("minimum_unique_objects")
    if int(summary["maximum_object_uses"]) > rules["maximum_uses_per_object"]:
        reasons.append("maximum_uses_per_object")
    minimum = float(summary["selected_minimum_m"])
    maximum = float(summary["selected_maximum_m"])
    ratio = maximum / minimum
    if not math.isfinite(ratio) or ratio < rules["minimum_distance_ratio"]:
        reasons.append("minimum_distance_ratio")
    if int(summary["selected_interior_pairs"]) != rules["pairs_per_scene"]:
        reasons.append("both_objects_interior")
    return reasons


def main() -> int:
    args = parse_args()
    project_root = args.project_root.expanduser().resolve()
    feasibility_path = resolve_path(project_root, args.feasibility)
    config_path = resolve_path(project_root, args.confirmatory_protocol)
    output_path = resolve_path(project_root, args.output)
    protocol_path = resolve_path(project_root, args.protocol)
    for source in (feasibility_path, config_path):
        if not source.is_file():
            raise FileNotFoundError(source)
    for destination in (output_path, protocol_path):
        if destination.exists() and not args.overwrite:
            raise FileExistsError(
                f"Destination exists; pass --overwrite to replace it: {destination}"
            )

    config = read_json(config_path)
    rules = frozen_rules(config)
    rows = load_jsonl(feasibility_path)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["capture_id"]), []).append(row)
    if not grouped:
        raise ValueError("Pair-feasibility input is empty")

    manifest: list[dict[str, Any]] = []
    decisions: dict[str, dict[str, Any]] = {}
    for capture_id, candidates in sorted(grouped.items()):
        evaluated = [
            (len(strict_pairs(candidate, rules)), candidate) for candidate in candidates
        ]
        evaluated.sort(
            key=lambda item: (
                -item[0],
                int(item[1]["candidate_rank"]),
                int(item[1]["frame_index"]),
            )
        )
        strict_count, best = evaluated[0]
        decision: dict[str, Any] = {
            "capture_id": capture_id,
            "candidate_frames": len(candidates),
            "best_candidate_rank": int(best["candidate_rank"]),
            "best_frame_index": int(best["frame_index"]),
            "best_strict_pair_count": strict_count,
            "accepted": False,
            "rejection_reasons": [],
        }
        if strict_count < rules["pairs_per_scene"]:
            decision["rejection_reasons"] = ["minimum_eligible_pairs"]
            decisions[capture_id] = decision
            continue

        selected, summary = select_scene_pairs(
            scene=best,
            minimum_distance_m=rules["minimum_distance_m"],
            minimum_visible_fraction=rules["minimum_visible_fraction"],
            maximum_box_iou=rules["maximum_box_iou"],
            maximum_intersection_over_smaller=rules[
                "maximum_intersection_over_smaller"
            ],
            minimum_box_area_fraction=rules["minimum_box_area_fraction"],
            minimum_box_side_px=rules["minimum_box_side_px"],
            minimum_border_margin_px=rules["minimum_border_margin_px"],
            pairs_per_scene=rules["pairs_per_scene"],
            distance_strata=rules["distance_strata"],
        )
        reasons = rejection_reasons(summary, rules)
        decision["selection_summary"] = summary
        decision["rejection_reasons"] = reasons
        decision["accepted"] = not reasons
        if not reasons:
            manifest.extend(selected)
        decisions[capture_id] = decision

    output_sha256 = write_jsonl_atomic(output_path, manifest)
    accepted = [capture for capture, item in decisions.items() if item["accepted"]]
    rejected = [capture for capture, item in decisions.items() if not item["accepted"]]
    report = {
        "schema_version": "1.0",
        "status": "confirmatory_pair_selection",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "confirmatory_protocol": display_path(project_root, config_path),
        "confirmatory_protocol_sha256": sha256_file(config_path),
        "input": display_path(project_root, feasibility_path),
        "input_sha256": sha256_file(feasibility_path),
        "rules": rules,
        "capture_decisions": decisions,
        "accepted_captures": accepted,
        "rejected_captures": rejected,
        "number_of_pairs": len(manifest),
        "output": display_path(project_root, output_path),
        "output_sha256": output_sha256,
    }
    write_json_atomic(protocol_path, report)

    print("Confirmatory pair selection")
    print(f"Captures: {len(grouped)}")
    print(f"Accepted: {len(accepted)}")
    print(f"Rejected: {len(rejected)}")
    print(f"Pairs: {len(manifest)}")
    print()
    for capture_id, decision in decisions.items():
        status = "ACCEPTED" if decision["accepted"] else "REJECTED"
        reasons = ",".join(decision["rejection_reasons"]) or "none"
        print(
            f"{capture_id}: {status} rank={decision['best_candidate_rank']} "
            f"frame={decision['best_frame_index']} "
            f"strict_pairs={decision['best_strict_pair_count']} "
            f"reasons={reasons}"
        )
    print()
    print(f"Output: {output_path}")
    print(f"Protocol: {protocol_path}")
    print(f"Output SHA-256: {output_sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
