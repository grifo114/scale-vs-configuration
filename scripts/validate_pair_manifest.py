#!/usr/bin/env python3
"""Validate the frozen pair-selection manifest before model inference."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import pathlib
import sys
from typing import Any


EXPECTED_STRATA = ("short", "medium", "long")
EXPECTED_FOLDS = ("A", "B")
EPSILON = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate balance, legibility, diversity, and distance coverage."
    )
    parser.add_argument("manifest", type=pathlib.Path)
    parser.add_argument("--protocol", type=pathlib.Path)
    parser.add_argument("--pairs-per-scene", type=int, default=12)
    parser.add_argument("--minimum-distance-m", type=float, default=0.25)
    parser.add_argument("--minimum-distance-ratio", type=float, default=2.5)
    parser.add_argument("--minimum-unique-objects", type=int, default=6)
    parser.add_argument("--maximum-object-uses", type=int, default=5)
    parser.add_argument("--minimum-visible-fraction", type=float, default=1.0)
    parser.add_argument("--maximum-pair-iou", type=float, default=0.5)
    parser.add_argument("--maximum-intersection-over-smaller", type=float, default=0.8)
    parser.add_argument("--minimum-box-area-fraction", type=float, default=0.00375)
    parser.add_argument("--minimum-box-side-px", type=float, default=30.0)
    parser.add_argument("--minimum-border-margin-px", type=float, default=20.0)
    return parser.parse_args()


def load_jsonl(path: pathlib.Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"line {line_number}: invalid JSON: {error}") from error
            if not isinstance(row, dict):
                raise ValueError(f"line {line_number}: expected a JSON object")
            row["_line_number"] = line_number
            rows.append(row)
    return rows


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def object_key(row: dict[str, Any], side: str) -> str:
    obj = row.get(f"object_{side}")
    if not isinstance(obj, dict):
        return f"missing-{side}"
    for key in ("id", "instance_id", "instance_index", "index"):
        if key in obj:
            return f"{key}:{obj[key]}"
    return json.dumps(obj, sort_keys=True, ensure_ascii=False)


def metric(row: dict[str, Any], *names: str) -> Any:
    quality = row.get("pair_quality")
    if isinstance(quality, dict):
        for name in names:
            if name in quality:
                return quality[name]
    for name in names:
        if name in row:
            return row[name]
    return None


def report_failure(failures: list[str], scene_id: str, message: str) -> None:
    failures.append(f"{scene_id}: {message}")


def validate_quality(
    row: dict[str, Any],
    scene_id: str,
    args: argparse.Namespace,
    failures: list[str],
) -> None:
    pair_id = str(row.get("pair_id", f"line-{row['_line_number']}"))

    def minimum(names: tuple[str, ...], threshold: float, label: str) -> None:
        value = metric(row, *names)
        if not finite_number(value):
            report_failure(failures, scene_id, f"{pair_id}: missing {label}")
        elif float(value) + EPSILON < threshold:
            report_failure(
                failures,
                scene_id,
                f"{pair_id}: {label}={float(value):.6g} < {threshold:.6g}",
            )

    def maximum(names: tuple[str, ...], threshold: float, label: str) -> None:
        value = metric(row, *names)
        if not finite_number(value):
            report_failure(failures, scene_id, f"{pair_id}: missing {label}")
        elif float(value) - EPSILON > threshold:
            report_failure(
                failures,
                scene_id,
                f"{pair_id}: {label}={float(value):.6g} > {threshold:.6g}",
            )

    minimum(("minimum_visible_fraction", "visible_fraction"), args.minimum_visible_fraction, "visible fraction")
    maximum(("box_iou", "pair_iou", "iou"), args.maximum_pair_iou, "pair IoU")
    maximum(
        (
            "intersection_over_smaller_box",
            "intersection_over_smaller",
            "pair_intersection_over_smaller",
            "ios",
        ),
        args.maximum_intersection_over_smaller,
        "intersection over smaller",
    )
    minimum(
        ("minimum_box_area_fraction", "min_box_area_fraction"),
        args.minimum_box_area_fraction,
        "minimum box area fraction",
    )
    minimum(
        ("minimum_box_side_px", "min_box_side_px"),
        args.minimum_box_side_px,
        "minimum box side",
    )
    minimum(
        ("minimum_border_margin_px", "min_border_margin_px"),
        args.minimum_border_margin_px,
        "minimum border margin",
    )

    interior = metric(row, "both_interior", "interior_pair")
    if interior is not True:
        report_failure(failures, scene_id, f"{pair_id}: both objects are not interior")


def validate_scene(
    scene_id: str,
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
    failures: list[str],
) -> dict[str, Any]:
    rows = sorted(rows, key=lambda row: int(row.get("pair_index", 10**9)))
    if len(rows) != args.pairs_per_scene:
        report_failure(
            failures,
            scene_id,
            f"pairs={len(rows)}; expected {args.pairs_per_scene}",
        )

    pair_ids = [str(row.get("pair_id")) for row in rows]
    duplicates = [key for key, count in collections.Counter(pair_ids).items() if count > 1]
    if duplicates:
        report_failure(failures, scene_id, f"duplicate pair IDs: {', '.join(duplicates)}")

    expected_indices = list(range(1, len(rows) + 1))
    actual_indices = [row.get("pair_index") for row in rows]
    if actual_indices != expected_indices:
        report_failure(failures, scene_id, f"pair indices are not contiguous: {actual_indices}")

    distances: list[float] = []
    identities: list[tuple[str, str]] = []
    object_uses: collections.Counter[str] = collections.Counter()
    strata: collections.Counter[str] = collections.Counter()
    folds: collections.Counter[str] = collections.Counter()
    stratum_folds: collections.Counter[tuple[str, str]] = collections.Counter()

    for row in rows:
        pair_id = str(row.get("pair_id", f"line-{row['_line_number']}"))
        distance = row.get("ground_truth_m")
        if not finite_number(distance) or float(distance) <= 0:
            report_failure(failures, scene_id, f"{pair_id}: invalid ground truth")
        else:
            distance = float(distance)
            distances.append(distance)
            if distance + EPSILON < args.minimum_distance_m:
                report_failure(
                    failures,
                    scene_id,
                    f"{pair_id}: distance={distance:.6g} < {args.minimum_distance_m:.6g}",
                )

        pair_index = row.get("pair_index")
        expected_fold = "A" if isinstance(pair_index, int) and pair_index % 2 == 1 else "B"
        fold = str(row.get("fold"))
        if fold not in EXPECTED_FOLDS:
            report_failure(failures, scene_id, f"{pair_id}: invalid fold={fold!r}")
        elif fold != expected_fold:
            report_failure(failures, scene_id, f"{pair_id}: fold={fold}; expected {expected_fold}")
        folds[fold] += 1

        expected_stratum = None
        if isinstance(pair_index, int) and 1 <= pair_index <= args.pairs_per_scene:
            stratum_size = args.pairs_per_scene // len(EXPECTED_STRATA)
            expected_stratum = EXPECTED_STRATA[min((pair_index - 1) // stratum_size, 2)]
        stratum = str(row.get("distance_stratum", row.get("stratum")))
        if stratum not in EXPECTED_STRATA:
            report_failure(failures, scene_id, f"{pair_id}: invalid stratum={stratum!r}")
        elif expected_stratum is not None and stratum != expected_stratum:
            report_failure(
                failures,
                scene_id,
                f"{pair_id}: stratum={stratum}; expected {expected_stratum}",
            )
        strata[stratum] += 1
        stratum_folds[(stratum, fold)] += 1

        left = object_key(row, "a")
        right = object_key(row, "b")
        if left == right:
            report_failure(failures, scene_id, f"{pair_id}: identical object on both sides")
        identities.append(tuple(sorted((left, right))))
        object_uses.update((left, right))
        validate_quality(row, scene_id, args, failures)

    duplicate_relations = [key for key, count in collections.Counter(identities).items() if count > 1]
    if duplicate_relations:
        report_failure(failures, scene_id, f"duplicate unordered object pairs={len(duplicate_relations)}")

    if distances != sorted(distances):
        report_failure(failures, scene_id, "ground-truth distances are not nondecreasing")

    if distances:
        distance_ratio = max(distances) / min(distances)
        if distance_ratio + EPSILON < args.minimum_distance_ratio:
            report_failure(
                failures,
                scene_id,
                f"distance ratio={distance_ratio:.3f} < {args.minimum_distance_ratio:.3f}",
            )
    else:
        distance_ratio = math.nan

    expected_per_stratum = args.pairs_per_scene // len(EXPECTED_STRATA)
    expected_per_fold = args.pairs_per_scene // len(EXPECTED_FOLDS)
    expected_per_stratum_fold = expected_per_stratum // len(EXPECTED_FOLDS)
    for stratum in EXPECTED_STRATA:
        if strata[stratum] != expected_per_stratum:
            report_failure(failures, scene_id, f"{stratum} pairs={strata[stratum]}; expected {expected_per_stratum}")
        for fold in EXPECTED_FOLDS:
            if stratum_folds[(stratum, fold)] != expected_per_stratum_fold:
                report_failure(
                    failures,
                    scene_id,
                    f"{stratum}/fold {fold}={stratum_folds[(stratum, fold)]}; expected {expected_per_stratum_fold}",
                )
    for fold in EXPECTED_FOLDS:
        if folds[fold] != expected_per_fold:
            report_failure(failures, scene_id, f"fold {fold}={folds[fold]}; expected {expected_per_fold}")

    if len(object_uses) < args.minimum_unique_objects:
        report_failure(
            failures,
            scene_id,
            f"unique objects={len(object_uses)} < {args.minimum_unique_objects}",
        )
    max_uses = max(object_uses.values(), default=0)
    if max_uses > args.maximum_object_uses:
        report_failure(
            failures,
            scene_id,
            f"maximum object uses={max_uses} > {args.maximum_object_uses}",
        )

    return {
        "pairs": len(rows),
        "distance_min_m": min(distances, default=math.nan),
        "distance_max_m": max(distances, default=math.nan),
        "distance_ratio": distance_ratio,
        "unique_objects": len(object_uses),
        "maximum_object_uses": max_uses,
    }


def validate_protocol(
    protocol_path: pathlib.Path,
    manifest_path: pathlib.Path,
    scene_count: int,
    pair_count: int,
    failures: list[str],
) -> None:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    expected_hash = sha256(manifest_path)
    for key in ("output_sha256", "manifest_sha256"):
        if key in protocol and protocol[key] != expected_hash:
            failures.append(f"protocol: {key} does not match manifest SHA-256")
    for key in ("number_of_scenes", "selected_scenes"):
        if key in protocol and isinstance(protocol[key], int) and protocol[key] != scene_count:
            failures.append(f"protocol: {key}={protocol[key]}; observed {scene_count}")
    for key in ("number_of_pairs", "selected_pairs"):
        if key in protocol and isinstance(protocol[key], int) and protocol[key] != pair_count:
            failures.append(f"protocol: {key}={protocol[key]}; observed {pair_count}")


def main() -> int:
    args = parse_args()
    failures: list[str] = []
    try:
        rows = load_jsonl(args.manifest)
    except (OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    grouped: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        scene_id = str(row.get("scene_id", row.get("capture_id", "missing-scene")))
        grouped[scene_id].append(row)

    summaries = {
        scene_id: validate_scene(scene_id, scene_rows, args, failures)
        for scene_id, scene_rows in sorted(grouped.items())
    }

    if args.protocol is not None:
        try:
            validate_protocol(args.protocol, args.manifest, len(grouped), len(rows), failures)
        except (OSError, json.JSONDecodeError) as error:
            failures.append(f"protocol: could not read: {error}")

    print("Pair-manifest validation")
    print(f"Manifest: {args.manifest.resolve()}")
    print(f"SHA-256: {sha256(args.manifest)}")
    print(f"Scenes: {len(grouped)}")
    print(f"Pairs: {len(rows)}")
    print()
    for scene_id, summary in summaries.items():
        print(
            f"{scene_id}: pairs={summary['pairs']} "
            f"distance={summary['distance_min_m']:.3f}-{summary['distance_max_m']:.3f} m "
            f"ratio={summary['distance_ratio']:.3f} "
            f"objects={summary['unique_objects']} "
            f"max_uses={summary['maximum_object_uses']}"
        )

    if failures:
        print()
        print(f"Result: FAILED ({len(failures)} issue(s))")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print()
    print("Result: PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
