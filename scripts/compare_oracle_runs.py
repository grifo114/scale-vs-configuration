#!/usr/bin/env python3
"""Compare a reproduced G-oracle run with the frozen v0.1 checkpoint."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any


DEFAULT_REFERENCE = Path("checkpoints/v0.1/results.jsonl")
DEFAULT_CANDIDATE = Path(
    "outputs/phase1/reproduction_qwen3vl4b_oracle/results.jsonl"
)
DEFAULT_REFERENCE_METADATA = Path("checkpoints/v0.1/metadata.json")
DEFAULT_CANDIDATE_METADATA = Path(
    "outputs/phase1/reproduction_qwen3vl4b_oracle/metadata.json"
)

ROW_RUNTIME_FIELDS = {
    "run_index",
    "condition",
    "model_id",
    "model_revision",
    "temperature",
    "max_tokens",
    "generation_seed",
    "stimulus_sha256",
    "raw_response",
    "predicted_distance_m",
    "elapsed_seconds",
    "absolute_error_m",
    "relative_error",
    "prediction_to_gt_ratio",
    "log_error",
}

ROW_STABLE_RUNTIME_FIELDS = (
    "run_index",
    "condition",
    "model_id",
    "model_revision",
    "temperature",
    "max_tokens",
    "generation_seed",
    "stimulus_sha256",
)

ROW_NUMERIC_OUTPUT_FIELDS = (
    "predicted_distance_m",
    "absolute_error_m",
    "relative_error",
    "prediction_to_gt_ratio",
    "log_error",
)

METADATA_STABLE_FIELDS = (
    "condition",
    "model_id",
    "model_revision",
    "mlx_version",
    "mlx_vlm_version",
    "transformers_version",
    "python_version",
    "architecture",
    "temperature",
    "max_tokens",
    "generation_seed",
    "order_seed",
    "manifest_sha256",
    "number_of_scenes",
    "number_of_queries",
    "scene_ids",
    "valid_responses",
)


class ComparisonError(RuntimeError):
    """Raised when an input file cannot be compared."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare a candidate G-oracle run with the frozen checkpoint."
        )
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Project root (default: current directory).",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=DEFAULT_REFERENCE,
        help="Reference JSONL results.",
    )
    parser.add_argument(
        "--candidate",
        type=Path,
        default=DEFAULT_CANDIDATE,
        help="Candidate JSONL results.",
    )
    parser.add_argument(
        "--reference-metadata",
        type=Path,
        default=DEFAULT_REFERENCE_METADATA,
        help="Reference metadata JSON.",
    )
    parser.add_argument(
        "--candidate-metadata",
        type=Path,
        default=DEFAULT_CANDIDATE_METADATA,
        help="Candidate metadata JSON.",
    )
    parser.add_argument(
        "--abs-tol",
        type=float,
        default=0.0,
        help="Absolute tolerance for numeric outputs (default: exact).",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Optional JSON report path.",
    )
    args = parser.parse_args()
    if not math.isfinite(args.abs_tol) or args.abs_tol < 0:
        parser.error("--abs-tol must be finite and non-negative")
    return args


def resolve_under(root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ComparisonError(f"File not found: {path}") from error
    except json.JSONDecodeError as error:
        raise ComparisonError(f"Invalid JSON in {path}: {error}") from error
    if not isinstance(value, dict):
        raise ComparisonError(f"Expected a JSON object in {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as error:
        raise ComparisonError(f"File not found: {path}") from error

    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ComparisonError(
                f"Invalid JSON at {path}:{line_number}: {error}"
            ) from error
        if not isinstance(value, dict):
            raise ComparisonError(
                f"Expected a JSON object at {path}:{line_number}"
            )
        rows.append(value)
    return rows


def numeric_equal(left: Any, right: Any, tolerance: float) -> bool:
    if left is None or right is None:
        return left is right
    if isinstance(left, bool) or isinstance(right, bool):
        return left == right
    if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
        return False
    return math.isclose(
        float(left),
        float(right),
        rel_tol=0.0,
        abs_tol=tolerance,
    )


def append_difference(
    differences: list[dict[str, Any]],
    category: str,
    field: str,
    reference: Any,
    candidate: Any,
    pair_id: str | None = None,
) -> None:
    item = {
        "category": category,
        "field": field,
        "reference": reference,
        "candidate": candidate,
    }
    if pair_id is not None:
        item["pair_id"] = pair_id
    differences.append(item)


def compare_rows(
    reference: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    tolerance: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    differences: list[dict[str, Any]] = []
    counters = {
        "row_count_differences": int(len(reference) != len(candidate)),
        "order_differences": 0,
        "input_differences": 0,
        "stable_runtime_differences": 0,
        "raw_response_differences": 0,
        "numeric_output_differences": 0,
    }
    maximum_prediction_difference = 0.0

    if len(reference) != len(candidate):
        append_difference(
            differences,
            "row_count",
            "number_of_rows",
            len(reference),
            len(candidate),
        )

    for index, (ref_row, candidate_row) in enumerate(
        zip(reference, candidate), start=1
    ):
        ref_pair_id = str(ref_row.get("pair_id", ""))
        candidate_pair_id = str(candidate_row.get("pair_id", ""))
        if ref_pair_id != candidate_pair_id:
            counters["order_differences"] += 1
            append_difference(
                differences,
                "order",
                f"run_index_{index}",
                ref_pair_id,
                candidate_pair_id,
            )
            continue

        ref_input = {
            key: value
            for key, value in ref_row.items()
            if key not in ROW_RUNTIME_FIELDS
        }
        candidate_input = {
            key: value
            for key, value in candidate_row.items()
            if key not in ROW_RUNTIME_FIELDS
        }
        if ref_input != candidate_input:
            counters["input_differences"] += 1
            append_difference(
                differences,
                "input",
                "manifest_fields",
                ref_input,
                candidate_input,
                ref_pair_id,
            )

        for field in ROW_STABLE_RUNTIME_FIELDS:
            if ref_row.get(field) != candidate_row.get(field):
                counters["stable_runtime_differences"] += 1
                append_difference(
                    differences,
                    "stable_runtime",
                    field,
                    ref_row.get(field),
                    candidate_row.get(field),
                    ref_pair_id,
                )

        if ref_row.get("raw_response") != candidate_row.get("raw_response"):
            counters["raw_response_differences"] += 1
            append_difference(
                differences,
                "raw_response",
                "raw_response",
                ref_row.get("raw_response"),
                candidate_row.get("raw_response"),
                ref_pair_id,
            )

        for field in ROW_NUMERIC_OUTPUT_FIELDS:
            ref_value = ref_row.get(field)
            candidate_value = candidate_row.get(field)
            if not numeric_equal(ref_value, candidate_value, tolerance):
                counters["numeric_output_differences"] += 1
                append_difference(
                    differences,
                    "numeric_output",
                    field,
                    ref_value,
                    candidate_value,
                    ref_pair_id,
                )
            if field == "predicted_distance_m" and all(
                isinstance(value, (int, float)) and not isinstance(value, bool)
                for value in (ref_value, candidate_value)
            ):
                maximum_prediction_difference = max(
                    maximum_prediction_difference,
                    abs(float(ref_value) - float(candidate_value)),
                )

    counters["maximum_absolute_prediction_difference_m"] = (
        maximum_prediction_difference
    )
    return counters, differences


def compare_metadata(
    reference: dict[str, Any],
    candidate: dict[str, Any],
) -> tuple[int, list[dict[str, Any]]]:
    differences: list[dict[str, Any]] = []
    for field in METADATA_STABLE_FIELDS:
        if reference.get(field) != candidate.get(field):
            append_difference(
                differences,
                "metadata",
                field,
                reference.get(field),
                candidate.get(field),
            )
    return len(differences), differences


def timing_summary(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    values = [
        float(row["elapsed_seconds"])
        for row in rows
        if isinstance(row.get("elapsed_seconds"), (int, float))
        and not isinstance(row.get("elapsed_seconds"), bool)
    ]
    if not values:
        return {"mean_seconds": None, "median_seconds": None}
    return {
        "mean_seconds": statistics.fmean(values),
        "median_seconds": statistics.median(values),
    }


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    reference_path = resolve_under(project_root, args.reference)
    candidate_path = resolve_under(project_root, args.candidate)
    reference_metadata_path = resolve_under(
        project_root, args.reference_metadata
    )
    candidate_metadata_path = resolve_under(
        project_root, args.candidate_metadata
    )

    reference_rows = read_jsonl(reference_path)
    candidate_rows = read_jsonl(candidate_path)
    reference_metadata = read_json(reference_metadata_path)
    candidate_metadata = read_json(candidate_metadata_path)

    row_summary, row_differences = compare_rows(
        reference_rows,
        candidate_rows,
        args.abs_tol,
    )
    metadata_count, metadata_differences = compare_metadata(
        reference_metadata,
        candidate_metadata,
    )
    differences = row_differences + metadata_differences
    passed = not differences

    report = {
        "status": "identical" if passed else "different",
        "absolute_tolerance": args.abs_tol,
        "reference_results": str(reference_path),
        "candidate_results": str(candidate_path),
        "reference_metadata": str(reference_metadata_path),
        "candidate_metadata": str(candidate_metadata_path),
        "reference_rows": len(reference_rows),
        "candidate_rows": len(candidate_rows),
        **row_summary,
        "metadata_differences": metadata_count,
        "reference_timing": timing_summary(reference_rows),
        "candidate_timing": timing_summary(candidate_rows),
        "differences": differences,
    }

    if args.report is not None:
        report_path = resolve_under(project_root, args.report)
        write_report(report_path, report)
        report["report_path"] = str(report_path)

    print("Oracle run comparison")
    print(f"Reference rows: {len(reference_rows)}")
    print(f"Candidate rows: {len(candidate_rows)}")
    print(f"Order differences: {row_summary['order_differences']}")
    print(f"Input differences: {row_summary['input_differences']}")
    print(
        "Stable runtime differences: "
        f"{row_summary['stable_runtime_differences']}"
    )
    print(
        "Raw response differences: "
        f"{row_summary['raw_response_differences']}"
    )
    print(
        "Numeric output differences: "
        f"{row_summary['numeric_output_differences']}"
    )
    print(f"Metadata differences: {metadata_count}")
    print(
        "Maximum absolute prediction difference: "
        f"{row_summary['maximum_absolute_prediction_difference_m']:.12g} m"
    )
    print(
        "Reference median time: "
        f"{report['reference_timing']['median_seconds']:.3f} s"
    )
    print(
        "Candidate median time: "
        f"{report['candidate_timing']['median_seconds']:.3f} s"
    )
    if args.report is not None:
        print(f"Report: {report['report_path']}")

    if passed:
        print("Result: IDENTICAL for every compared deterministic field.")
        return 0

    print(f"Result: DIFFERENT ({len(differences)} difference(s)).")
    for item in differences[:10]:
        location = f" [{item['pair_id']}]" if "pair_id" in item else ""
        print(
            f"- {item['category']}{location} {item['field']}: "
            f"reference={item['reference']!r}, candidate={item['candidate']!r}"
        )
    if len(differences) > 10:
        print(f"- ... {len(differences) - 10} additional difference(s)")
    return 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ComparisonError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
