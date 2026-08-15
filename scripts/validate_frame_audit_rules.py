#!/usr/bin/env python3
"""Reconstruct and compare the rules behind a historical CA-1M frame audit.

The script is diagnostic. It reads the ignored pilot audit and the local CA-1M
TAR files, then evaluates several explicit rule combinations. It does not
select scenes and does not modify the dataset or the historical audit.
"""

from __future__ import annotations

import argparse
import json
import math
import tarfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


DEFAULT_GENERIC_CATEGORIES = (
    "baseboard",
    "ceiling",
    "floor",
    "object",
    "wall",
)
DEFAULT_MARGINS = (0.025, 0.05, 0.075, 0.10, 0.125, 0.15)
DEFAULT_PIXEL_MARGINS = (10.0, 20.0, 25.0, 30.0, 40.0, 50.0)
BOX_FIELDS = ("box_2d_rend", "box_2d_proj")
DUPLICATE_SCOPES = ("all", "geometric", "oracle")
INTERIOR_MODES = ("box", "center")
INTERIOR_PIPELINES = ("subset", "recompute")


@dataclass(frozen=True)
class RuleConfig:
    box_field: str
    duplicate_scope: str
    caption_required: bool
    interior_mode: str
    interior_pipeline: str
    margin_kind: str
    margin: float

    @property
    def label(self) -> str:
        caption = "required" if self.caption_required else "ignored"
        return (
            f"box={self.box_field}, duplicates={self.duplicate_scope}, "
            f"caption={caption}, interior={self.interior_mode}, "
            f"pipeline={self.interior_pipeline}, "
            f"margin={self.margin:g}-{self.margin_kind}"
        )


@dataclass
class MatchCounts:
    total: int = 0
    geometric: int = 0
    oracle: int = 0
    natural: int = 0
    interior: int = 0
    all_fields: int = 0


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Compare candidate CA-1M audit rules against the historical "
            "frame_audit.jsonl artifact."
        )
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=project_root,
        help="Project root. Defaults to the parent of scripts/.",
    )
    parser.add_argument(
        "--audit",
        type=Path,
        default=Path("outputs/pilot/capture_candidates/frame_audit.jsonl"),
        help="Audit JSONL path, absolute or relative to the project root.",
    )
    parser.add_argument(
        "--generic-category",
        action="append",
        dest="generic_categories",
        help=(
            "Generic category to exclude. Repeat the option to replace the "
            "default category set."
        ),
    )
    parser.add_argument(
        "--margin",
        action="append",
        type=float,
        dest="margins",
        help=(
            "Candidate interior margin as a fraction in [0, 0.5). Repeat "
            "the option to replace the default margin set."
        ),
    )
    parser.add_argument(
        "--pixel-margin",
        action="append",
        type=float,
        dest="pixel_margins",
        help=(
            "Candidate fixed interior margin in pixels. Repeat the option "
            "to replace the default pixel-margin set."
        ),
    )
    parser.add_argument(
        "--top",
        type=int,
        default=12,
        help="Number of highest-scoring configurations to print.",
    )
    parser.add_argument(
        "--max-mismatches",
        type=int,
        default=8,
        help="Maximum best-configuration mismatches to print.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional number of audit rows to process for a quick test.",
    )
    return parser.parse_args()


def resolve_path(project_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else project_root / path


def load_audit(path: Path, limit: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            row["_audit_line"] = line_number
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    if not rows:
        raise ValueError(f"No audit rows found in {path}")
    return rows


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def finite_vector(value: Any, length: int) -> bool:
    return (
        isinstance(value, list)
        and len(value) == length
        and all(finite_number(item) for item in value)
    )


def geometric_valid(instance: dict[str, Any]) -> bool:
    position = instance.get("position")
    scale = instance.get("scale")
    corners = instance.get("corners")
    if not finite_vector(position, 3) or not finite_vector(scale, 3):
        return False
    if not all(float(value) > 0 for value in scale):
        return False
    if not isinstance(corners, list) or len(corners) != 8:
        return False
    if not all(finite_vector(corner, 3) for corner in corners):
        return False
    return True


def normalized_category(instance: dict[str, Any]) -> str:
    return str(instance.get("category", "")).strip().casefold()


def valid_box(instance: dict[str, Any], box_field: str) -> list[float] | None:
    box = instance.get(box_field)
    if not finite_vector(box, 4):
        return None
    return [float(value) for value in box]


def oracle_valid(
    instance: dict[str, Any],
    box_field: str,
    width: int,
    height: int,
) -> bool:
    if not geometric_valid(instance):
        return False
    box = valid_box(instance, box_field)
    if box is None or width <= 0 or height <= 0:
        return False
    box_width = box[2] - box[0]
    box_height = box[3] - box[1]
    if box_width < 20 or box_height < 20:
        return False
    area = box_width * box_height
    coverage = area / (width * height)
    return area >= 1000 and coverage <= 0.5


def category_counts(
    instances: list[dict[str, Any]],
    geometric: list[dict[str, Any]],
    oracle: list[dict[str, Any]],
    scope: str,
) -> Counter[str]:
    scoped = {
        "all": instances,
        "geometric": geometric,
        "oracle": oracle,
    }[scope]
    return Counter(normalized_category(instance) for instance in scoped)


def natural_instances(
    instances: list[dict[str, Any]],
    geometric: list[dict[str, Any]],
    oracle: list[dict[str, Any]],
    duplicate_scope: str,
    caption_required: bool,
    generic_categories: frozenset[str],
) -> list[dict[str, Any]]:
    counts = category_counts(instances, geometric, oracle, duplicate_scope)
    selected: list[dict[str, Any]] = []
    for instance in oracle:
        category = normalized_category(instance)
        if not category or category in generic_categories:
            continue
        if counts[category] != 1:
            continue
        if caption_required and not str(instance.get("caption", "")).strip():
            continue
        selected.append(instance)
    return selected


def interior_valid(
    instance: dict[str, Any],
    box_field: str,
    width: int,
    height: int,
    mode: str,
    margin_kind: str,
    margin: float,
) -> bool:
    box = valid_box(instance, box_field)
    if box is None:
        return False
    if margin_kind == "fraction":
        x_margin = margin * width
        y_margin = margin * height
    elif margin_kind == "pixels":
        x_margin = margin
        y_margin = margin
    else:
        raise ValueError(f"Unknown margin kind: {margin_kind}")
    x_low = x_margin
    y_low = y_margin
    x_high = width - x_margin
    y_high = height - y_margin
    if mode == "box":
        return (
            box[0] >= x_low
            and box[1] >= y_low
            and box[2] <= x_high
            and box[3] <= y_high
        )
    center_x = (box[0] + box[2]) / 2.0
    center_y = (box[1] + box[3]) / 2.0
    return x_low <= center_x <= x_high and y_low <= center_y <= y_high


def candidate_configs(
    margins: Iterable[float],
    pixel_margins: Iterable[float],
) -> list[RuleConfig]:
    margin_specs = [
        *(('fraction', margin) for margin in margins),
        *(('pixels', margin) for margin in pixel_margins),
    ]
    return [
        RuleConfig(
            box_field=box_field,
            duplicate_scope=duplicate_scope,
            caption_required=caption_required,
            interior_mode=interior_mode,
            interior_pipeline=interior_pipeline,
            margin_kind=margin_kind,
            margin=margin,
        )
        for box_field in BOX_FIELDS
        for duplicate_scope in DUPLICATE_SCOPES
        for caption_required in (False, True)
        for interior_mode in INTERIOR_MODES
        for interior_pipeline in INTERIOR_PIPELINES
        for margin_kind, margin in margin_specs
    ]


def load_instances(
    archive: tarfile.TarFile,
    member_name: str,
) -> list[dict[str, Any]]:
    handle = archive.extractfile(member_name)
    if handle is None:
        raise FileNotFoundError(
            f"Could not extract {member_name} from {archive.name}"
        )
    with handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise TypeError(f"Expected a list in {member_name}")
    return data


def print_configuration(
    rank: int,
    config: RuleConfig,
    counts: MatchCounts,
    row_count: int,
) -> None:
    print(f"{rank:02d}. {config.label}")
    print(
        "    matches: "
        f"all={counts.all_fields}/{row_count}, "
        f"total={counts.total}/{row_count}, "
        f"geometric={counts.geometric}/{row_count}, "
        f"oracle={counts.oracle}/{row_count}, "
        f"natural={counts.natural}/{row_count}, "
        f"interior={counts.interior}/{row_count}"
    )


def main() -> int:
    args = parse_args()
    project_root = args.project_root.expanduser().resolve()
    audit_path = resolve_path(project_root, args.audit).resolve()
    generic_categories = frozenset(
        category.strip().casefold()
        for category in (
            args.generic_categories or DEFAULT_GENERIC_CATEGORIES
        )
        if category.strip()
    )
    margins = tuple(args.margins or DEFAULT_MARGINS)
    pixel_margins = tuple(args.pixel_margins or DEFAULT_PIXEL_MARGINS)
    if any(not 0 <= margin < 0.5 for margin in margins):
        raise ValueError("Every margin must be in [0, 0.5).")
    if any(margin < 0 for margin in pixel_margins):
        raise ValueError("Every pixel margin must be nonnegative.")
    if args.top <= 0 or args.max_mismatches < 0:
        raise ValueError("--top must be positive and --max-mismatches nonnegative.")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive.")

    rows = load_audit(audit_path, args.limit)
    configs = candidate_configs(margins, pixel_margins)
    match_counts = {config: MatchCounts() for config in configs}
    mismatch_samples: dict[
        RuleConfig, list[tuple[dict[str, Any], tuple[int, ...]]]
    ] = {
        config: [] for config in configs
    }

    rows_by_archive: dict[Path, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        archive_path = resolve_path(project_root, Path(row["archive"])).resolve()
        rows_by_archive[archive_path].append(row)

    processed = 0
    for archive_path in sorted(rows_by_archive):
        if not archive_path.is_file():
            raise FileNotFoundError(f"Archive not found: {archive_path}")
        print(f"Reading {archive_path.name} ...")
        with tarfile.open(archive_path, "r") as archive:
            for row in rows_by_archive[archive_path]:
                instances = load_instances(archive, row["instance_member"])
                geometric = [
                    instance for instance in instances if geometric_valid(instance)
                ]
                total_prediction = len(instances)
                geometric_prediction = len(geometric)
                width = int(row["width"])
                height = int(row["height"])

                oracle_by_box: dict[str, list[dict[str, Any]]] = {}
                natural_by_key: dict[
                    tuple[str, str, bool], list[dict[str, Any]]
                ] = {}
                for box_field in BOX_FIELDS:
                    oracle = [
                        instance
                        for instance in instances
                        if oracle_valid(instance, box_field, width, height)
                    ]
                    oracle_by_box[box_field] = oracle
                    for duplicate_scope in DUPLICATE_SCOPES:
                        for caption_required in (False, True):
                            key = (box_field, duplicate_scope, caption_required)
                            natural_by_key[key] = natural_instances(
                                instances=instances,
                                geometric=geometric,
                                oracle=oracle,
                                duplicate_scope=duplicate_scope,
                                caption_required=caption_required,
                                generic_categories=generic_categories,
                            )

                expected = (
                    int(row["total"]),
                    int(row["geometric"]),
                    int(row["oracle"]),
                    int(row["natural"]),
                    int(row["natural_interior"]),
                )
                for config in configs:
                    oracle = oracle_by_box[config.box_field]
                    natural = natural_by_key[
                        (
                            config.box_field,
                            config.duplicate_scope,
                            config.caption_required,
                        )
                    ]
                    interior_oracle = [
                        instance
                        for instance in oracle
                        if interior_valid(
                            instance=instance,
                            box_field=config.box_field,
                            width=width,
                            height=height,
                            mode=config.interior_mode,
                            margin_kind=config.margin_kind,
                            margin=config.margin,
                        )
                    ]
                    if config.interior_pipeline == "subset":
                        natural_ids = {id(instance) for instance in natural}
                        interior = [
                            instance
                            for instance in interior_oracle
                            if id(instance) in natural_ids
                        ]
                    else:
                        interior_counts = Counter(
                            normalized_category(instance)
                            for instance in interior_oracle
                        )
                        interior = []
                        for instance in interior_oracle:
                            category = normalized_category(instance)
                            if not category or category in generic_categories:
                                continue
                            if interior_counts[category] != 1:
                                continue
                            if (
                                config.caption_required
                                and not str(instance.get("caption", "")).strip()
                            ):
                                continue
                            interior.append(instance)
                    predicted = (
                        total_prediction,
                        geometric_prediction,
                        len(oracle),
                        len(natural),
                        len(interior),
                    )
                    counts = match_counts[config]
                    comparisons = [
                        predicted[index] == expected[index] for index in range(5)
                    ]
                    counts.total += int(comparisons[0])
                    counts.geometric += int(comparisons[1])
                    counts.oracle += int(comparisons[2])
                    counts.natural += int(comparisons[3])
                    counts.interior += int(comparisons[4])
                    counts.all_fields += int(all(comparisons))
                    if (
                        not all(comparisons)
                        and len(mismatch_samples[config]) < args.max_mismatches
                    ):
                        mismatch_samples[config].append((row, predicted))

                processed += 1
                if processed % 250 == 0:
                    print(f"  processed {processed}/{len(rows)} frames")

    ranked = sorted(
        configs,
        key=lambda config: (
            -match_counts[config].all_fields,
            -match_counts[config].interior,
            -match_counts[config].natural,
            -match_counts[config].oracle,
            config.label,
        ),
    )

    print()
    print("Audit rule reconstruction")
    print(f"Rows: {len(rows)}")
    print(f"Generic categories: {', '.join(sorted(generic_categories))}")
    print(f"Configurations: {len(configs)}")
    print()
    print("Highest-scoring configurations:")
    for rank, config in enumerate(ranked[: args.top], start=1):
        print_configuration(rank, config, match_counts[config], len(rows))

    best = ranked[0]
    mismatches = len(rows) - match_counts[best].all_fields
    print()
    print(f"Best: {best.label}")
    print(f"Mismatched rows: {mismatches}/{len(rows)}")
    if mismatches and args.max_mismatches:
        print("First mismatches (expected -> predicted):")
        for row, predicted in mismatch_samples[best]:
            expected = (
                int(row["total"]),
                int(row["geometric"]),
                int(row["oracle"]),
                int(row["natural"]),
                int(row["natural_interior"]),
            )
            print(
                f"  line={row['_audit_line']}, "
                f"capture={row['capture_id']}, frame={row['frame_index']}: "
                f"{expected} -> {predicted}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
