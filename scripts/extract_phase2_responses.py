#!/usr/bin/env python3
"""Derive analysis-ready distances without mutating Phase 2 raw results."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any


SCRIPT_VERSION = "0.1.0"
DEFAULT_PROTOCOL = Path("configs/phase2_response_extraction_protocol_v1.json")
FENCED_JSON_PATTERN = re.compile(
    r"\A```json[ \t]*\r?\n(?P<payload>.*?)\r?\n```[ \t]*\Z",
    flags=re.DOTALL,
)


class ExtractionError(RuntimeError):
    """Raised when extraction inputs or outputs violate the frozen protocol."""


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Apply the frozen Phase 2 response-extraction protocol."
    )
    parser.add_argument("--project-root", type=Path, default=root)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def resolve_under(root: Path, value: Path) -> Path:
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ExtractionError(f"Expected a JSON object: {path}")
    return value


def validate_protocol(value: dict[str, Any]) -> None:
    if value.get("artifact_name") != "phase2_response_extraction_protocol_v1":
        raise ExtractionError("Unexpected response-extraction protocol")
    if value.get("status") != (
        "predeclared_after_technical_canary_before_full_inference"
    ):
        raise ExtractionError("Unexpected response-extraction protocol status")
    if value.get("full_confirmatory_outputs_observed") is not False:
        raise ExtractionError("The protocol is not pre-full-inference")
    if value.get("prompt_changed") is not False:
        raise ExtractionError("The extraction amendment changed the prompt")
    if value.get("models_changed") is not False:
        raise ExtractionError("The extraction amendment changed the models")
    if value.get("query_manifest_changed") is not False:
        raise ExtractionError("The extraction amendment changed the manifest")
    policy = value.get("two_layer_policy")
    if not isinstance(policy, dict):
        raise ExtractionError("The two-layer extraction policy is absent")
    accepted = policy.get("analysis_extraction", {}).get(
        "accepted_forms_in_precedence_order"
    )
    if accepted != ["strict_json", "single_json_markdown_fence"]:
        raise ExtractionError("Unexpected accepted response forms")
    disallowed = set(policy.get("disallowed_operations", []))
    required_disallowed = {
        "prompt repair",
        "model-specific parser",
        "response retry",
        "prose number extraction",
        "multiple-candidate selection",
        "mutation of raw response rows",
    }
    if disallowed != required_disallowed:
        raise ExtractionError("Unexpected disallowed-operation set")


def validate_distance_object(value: Any) -> tuple[float | None, str | None]:
    if not isinstance(value, dict):
        return None, "top-level JSON value is not an object"
    if set(value) != {"distance_m"}:
        return None, "expected exactly one key named distance_m"
    distance = value["distance_m"]
    if isinstance(distance, bool) or not isinstance(distance, (int, float)):
        return None, "distance_m is not a JSON number"
    numeric = float(distance)
    if not math.isfinite(numeric) or numeric <= 0:
        return None, "distance_m is not positive and finite"
    return numeric, None


def invalid_result(
    *,
    status: str,
    source_format: str,
    error: str,
) -> dict[str, Any]:
    return {
        "strict_json_valid": False,
        "analysis_response_status": status,
        "analysis_extraction_method": None,
        "analysis_source_format": source_format,
        "analysis_distance_m": None,
        "analysis_extraction_error": error,
    }


def extract_distance_response(raw_response: str) -> dict[str, Any]:
    """Apply strict JSON first, then one exact json-labelled Markdown fence."""

    text = raw_response.strip()
    try:
        strict_value = json.loads(text)
    except json.JSONDecodeError as strict_error:
        fence = FENCED_JSON_PATTERN.fullmatch(text)
        if fence is None:
            return invalid_result(
                status="invalid_json",
                source_format="unrecognized",
                error=(
                    "strict JSON parse failed and response is not one accepted "
                    f"json Markdown fence: {strict_error}"
                ),
            )
        payload = fence.group("payload")
        try:
            fenced_value = json.loads(payload)
        except json.JSONDecodeError as fenced_error:
            return invalid_result(
                status="invalid_json",
                source_format="single_json_markdown_fence",
                error=f"fenced JSON parse failed: {fenced_error}",
            )
        numeric, schema_error = validate_distance_object(fenced_value)
        if schema_error is not None:
            return invalid_result(
                status="invalid_schema",
                source_format="single_json_markdown_fence",
                error=schema_error,
            )
        return {
            "strict_json_valid": False,
            "analysis_response_status": "valid",
            "analysis_extraction_method": "single_json_markdown_fence",
            "analysis_source_format": "single_json_markdown_fence",
            "analysis_distance_m": numeric,
            "analysis_extraction_error": None,
        }

    numeric, schema_error = validate_distance_object(strict_value)
    if schema_error is not None:
        return invalid_result(
            status="invalid_schema",
            source_format="strict_json",
            error=schema_error,
        )
    return {
        "strict_json_valid": True,
        "analysis_response_status": "valid",
        "analysis_extraction_method": "strict_json",
        "analysis_source_format": "strict_json",
        "analysis_distance_m": numeric,
        "analysis_extraction_error": None,
    }


def read_source_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ExtractionError(f"Line {line_number} is not a JSON object")
        if not isinstance(value.get("raw_response"), str):
            raise ExtractionError(f"Line {line_number} lacks raw_response text")
        if not isinstance(value.get("sequence_index"), int):
            raise ExtractionError(f"Line {line_number} lacks sequence_index")
        if not isinstance(value.get("pair_id"), str):
            raise ExtractionError(f"Line {line_number} lacks pair_id")
        rows.append(value)
    if not rows:
        raise ExtractionError("The input JSONL contains no result rows")
    if [row["sequence_index"] for row in rows] != list(range(1, len(rows) + 1)):
        raise ExtractionError("Input rows are not an exact sequence prefix")
    if len({row["pair_id"] for row in rows}) != len(rows):
        raise ExtractionError("Input pair IDs are not unique")
    return rows


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def derive_rows(
    source_rows: list[dict[str, Any]],
    *,
    source_sha256: str,
    protocol_sha256: str,
    extractor_sha256: str,
) -> list[dict[str, Any]]:
    derived: list[dict[str, Any]] = []
    for source in source_rows:
        raw_response = source["raw_response"]
        extraction = extract_distance_response(raw_response)
        derived.append(
            {
                "schema_version": "1.0",
                "source_results_sha256": source_sha256,
                "source_sequence_index": source["sequence_index"],
                "pair_id": source["pair_id"],
                "model_key": source.get("model_key"),
                "model_id": source.get("model_id"),
                "model_revision": source.get("model_revision"),
                "raw_response_sha256": sha256_text(raw_response),
                "original_response_status": source.get("response_status"),
                "original_parse_error": source.get("parse_error"),
                "response_extraction_protocol_sha256": protocol_sha256,
                "response_extractor_sha256": extractor_sha256,
                **extraction,
            }
        )
    return derived


def main() -> int:
    args = parse_args()
    root = args.project_root.resolve()
    protocol_path = resolve_under(root, args.protocol)
    input_path = resolve_under(root, args.input)
    output_path = resolve_under(root, args.output)
    report_path = resolve_under(root, args.report)

    if output_path == input_path or report_path == input_path:
        raise ExtractionError("Derived outputs may not overwrite raw results")
    if output_path == report_path:
        raise ExtractionError("Derived JSONL and report paths must differ")
    if output_path.exists() or report_path.exists():
        raise ExtractionError("Refusing to overwrite an existing derived artifact")

    protocol = read_json(protocol_path)
    validate_protocol(protocol)
    source_rows = read_source_rows(input_path)
    source_sha256 = sha256_file(input_path)
    protocol_sha256 = sha256_file(protocol_path)
    extractor_sha256 = sha256_file(Path(__file__))
    derived = derive_rows(
        source_rows,
        source_sha256=source_sha256,
        protocol_sha256=protocol_sha256,
        extractor_sha256=extractor_sha256,
    )
    write_jsonl_atomic(output_path, derived)

    status_counts = Counter(row["analysis_response_status"] for row in derived)
    method_counts = Counter(
        row["analysis_extraction_method"] or "none" for row in derived
    )
    strict_valid = sum(row["strict_json_valid"] is True for row in derived)
    report = {
        "artifact_name": "phase2_response_extraction_report",
        "schema_version": "1.0",
        "extractor_version": SCRIPT_VERSION,
        "input_path": str(input_path),
        "input_sha256": source_sha256,
        "input_rows": len(source_rows),
        "output_path": str(output_path),
        "output_sha256": sha256_file(output_path),
        "protocol_path": str(protocol_path),
        "protocol_sha256": protocol_sha256,
        "extractor_sha256": extractor_sha256,
        "strict_json_valid": strict_valid,
        "analysis_response_status_counts": dict(sorted(status_counts.items())),
        "analysis_extraction_method_counts": dict(sorted(method_counts.items())),
        "raw_results_mutated": False,
        "ground_truth_compared": False,
        "scientific_metrics_computed": False,
    }
    write_json_atomic(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
