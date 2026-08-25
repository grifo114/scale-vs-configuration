#!/usr/bin/env python3
"""Validate the pre-full Phase 2 response-extraction amendment."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any


SCRIPT_VERSION = "0.1.0"


class ValidationError(RuntimeError):
    """Raised when a response-extraction invariant is absent."""


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Validate Phase 2 response extraction v1."
    )
    parser.add_argument("--project-root", type=Path, default=root)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("configs/phase2_response_extraction_protocol_v1.json"),
    )
    parser.add_argument(
        "--evidence",
        type=Path,
        default=Path(
            "configs/phase2_response_extraction_canary_evidence_v1.json"
        ),
    )
    parser.add_argument(
        "--extractor",
        type=Path,
        default=Path("scripts/extract_phase2_responses.py"),
    )
    parser.add_argument("--canary-results", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def resolve_under(root: Path, value: Path) -> Path:
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def load_extractor(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(
        "phase2_response_extractor_under_test", path
    )
    require(spec is not None and spec.loader is not None, "Cannot load extractor")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_protocol(value: dict[str, Any], extractor: Any) -> None:
    extractor.validate_protocol(value)
    require(value["response_extraction_rule_changed"] is True, "Rule change absent")
    require(value["scientific_question_changed"] is False, "Question changed")
    require(value["inference_inputs_changed"] is False, "Inputs changed")
    basis = value["technical_canary_basis"]
    require(basis["attempt_id"] == "20260825T155631Z_9b93e13c", "Attempt mismatch")
    require(basis["ground_truth_compared"] is False, "Ground truth was compared")
    require(basis["scientific_metrics_computed"] is False, "Metrics were computed")
    require(basis["full_inference_executed"] is False, "Full inference was executed")
    require(
        basis["results_sha256"]
        == "ad29e6e1ac86e956955e7eb850923e8a9a67b7b964316b36ae0449ef9897092a",
        "Canary results hash mismatch",
    )


def validate_evidence(value: dict[str, Any]) -> None:
    require(
        value["artifact_name"]
        == "phase2_response_extraction_canary_evidence_v1",
        "Unexpected evidence artifact",
    )
    require(value["validation_scope"] == "response_format_only", "Scope mismatch")
    require(value["ground_truth_compared"] is False, "Ground truth was compared")
    require(value["scientific_metrics_computed"] is False, "Metrics were computed")
    require(value["full_confirmatory_outputs_observed"] is False, "Full outputs seen")
    require(value["raw_response_values_recorded"] is False, "Raw values recorded")
    require(
        value["results_sha256"]
        == "ad29e6e1ac86e956955e7eb850923e8a9a67b7b964316b36ae0449ef9897092a",
        "Evidence source hash mismatch",
    )
    counts = value["counts"]
    require(counts["total_rows"] == 12, "Evidence row count mismatch")
    require(counts["strict_json_valid"] == 0, "Strict-valid count mismatch")
    require(
        counts["analysis_extractable_single_json_markdown_fence"] == 9,
        "Fenced-JSON count mismatch",
    )
    require(counts["unrecognized_prose"] == 3, "Prose count mismatch")
    rows = value["rows"]
    require(len(rows) == 12, "Evidence does not contain twelve rows")
    require(
        [row["sequence_index"] for row in rows] == list(range(1, 13)),
        "Evidence sequence is not 1..12",
    )
    require(len({row["pair_id"] for row in rows}) == 12, "Pair IDs repeat")
    classes = Counter(row["analysis_format_class"] for row in rows)
    require(
        classes == Counter(
            {"single_json_markdown_fence": 9, "unrecognized_prose": 3}
        ),
        "Evidence format classes mismatch",
    )
    require(
        all(row["original_response_status"] == "invalid_json" for row in rows),
        "Original response status mismatch",
    )
    serialized = json.dumps(value, ensure_ascii=False, sort_keys=True)
    require("raw_response\"" not in serialized, "Raw response field leaked")
    require("ground_truth_m" not in serialized, "Ground truth field leaked")
    require("analysis_distance_m" not in serialized, "Predicted distance leaked")


def synthetic_cases() -> list[tuple[str, str, dict[str, Any]]]:
    return [
        (
            "strict_integer",
            '{"distance_m": 1}',
            {"status": "valid", "method": "strict_json", "strict": True},
        ),
        (
            "strict_float_whitespace",
            '  {"distance_m": 1.25}\n',
            {"status": "valid", "method": "strict_json", "strict": True},
        ),
        (
            "fenced_json",
            '```json\n{"distance_m": 1.25}\n```',
            {
                "status": "valid",
                "method": "single_json_markdown_fence",
                "strict": False,
            },
        ),
        (
            "fenced_json_crlf",
            '```json\r\n{"distance_m": 1.25}\r\n```',
            {
                "status": "valid",
                "method": "single_json_markdown_fence",
                "strict": False,
            },
        ),
        (
            "prose_before_fence",
            'Answer:\n```json\n{"distance_m": 1.25}\n```',
            {"status": "invalid_json", "method": None, "strict": False},
        ),
        (
            "prose_after_fence",
            '```json\n{"distance_m": 1.25}\n```\nDone.',
            {"status": "invalid_json", "method": None, "strict": False},
        ),
        (
            "unlabelled_fence",
            '```\n{"distance_m": 1.25}\n```',
            {"status": "invalid_json", "method": None, "strict": False},
        ),
        (
            "uppercase_fence",
            '```JSON\n{"distance_m": 1.25}\n```',
            {"status": "invalid_json", "method": None, "strict": False},
        ),
        (
            "multiple_fences",
            '```json\n{"distance_m": 1}\n```\n```json\n{"distance_m": 2}\n```',
            {"status": "invalid_json", "method": None, "strict": False},
        ),
        (
            "prose_number",
            "The distance is 1.25 meters.",
            {"status": "invalid_json", "method": None, "strict": False},
        ),
        (
            "malformed_fenced_json",
            '```json\n{"distance_m":}\n```',
            {"status": "invalid_json", "method": None, "strict": False},
        ),
        (
            "strict_extra_key",
            '{"distance_m": 1, "unit": "m"}',
            {"status": "invalid_schema", "method": None, "strict": False},
        ),
        (
            "fenced_extra_key",
            '```json\n{"distance_m": 1, "unit": "m"}\n```',
            {"status": "invalid_schema", "method": None, "strict": False},
        ),
        (
            "boolean",
            '{"distance_m": true}',
            {"status": "invalid_schema", "method": None, "strict": False},
        ),
        (
            "string",
            '{"distance_m": "1.25"}',
            {"status": "invalid_schema", "method": None, "strict": False},
        ),
        (
            "zero",
            '{"distance_m": 0}',
            {"status": "invalid_schema", "method": None, "strict": False},
        ),
        (
            "negative",
            '{"distance_m": -1}',
            {"status": "invalid_schema", "method": None, "strict": False},
        ),
        (
            "array",
            '[{"distance_m": 1}]',
            {"status": "invalid_schema", "method": None, "strict": False},
        ),
        (
            "nan",
            '{"distance_m": NaN}',
            {"status": "invalid_schema", "method": None, "strict": False},
        ),
        (
            "infinity",
            '{"distance_m": Infinity}',
            {"status": "invalid_schema", "method": None, "strict": False},
        ),
    ]


def run_synthetic_tests(extractor: Any) -> list[str]:
    passed: list[str] = []
    for name, raw_response, expected in synthetic_cases():
        result = extractor.extract_distance_response(raw_response)
        require(
            result["analysis_response_status"] == expected["status"],
            f"Synthetic case {name} status mismatch: {result}",
        )
        require(
            result["analysis_extraction_method"] == expected["method"],
            f"Synthetic case {name} method mismatch: {result}",
        )
        require(
            result["strict_json_valid"] is expected["strict"],
            f"Synthetic case {name} strict flag mismatch: {result}",
        )
        if expected["status"] == "valid":
            require(
                isinstance(result["analysis_distance_m"], float),
                f"Synthetic case {name} has no extracted float",
            )
            require(
                result["analysis_extraction_error"] is None,
                f"Synthetic case {name} has an unexpected error",
            )
        else:
            require(
                result["analysis_distance_m"] is None,
                f"Synthetic case {name} extracted a forbidden number",
            )
        passed.append(name)
    return passed


def validate_canary_results(
    path: Path,
    evidence: dict[str, Any],
    extractor: Any,
) -> dict[str, Any]:
    before_sha = sha256_file(path)
    require(before_sha == evidence["results_sha256"], "Canary file hash mismatch")
    source_rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    require(len(source_rows) == 12, "Canary file does not contain twelve rows")
    evidence_rows = evidence["rows"]
    statuses: Counter[str] = Counter()
    methods: Counter[str] = Counter()
    strict_valid = 0
    for source, recorded in zip(source_rows, evidence_rows):
        require(source["sequence_index"] == recorded["sequence_index"], "Index mismatch")
        require(source["pair_id"] == recorded["pair_id"], "Pair mismatch")
        require(
            source["model_key"] == evidence["model_key"], "Model key mismatch"
        )
        require(
            source["model_revision"] == evidence["model_revision"],
            "Model revision mismatch",
        )
        require(
            source["response_status"] == recorded["original_response_status"],
            "Original status mismatch",
        )
        raw_response = source["raw_response"]
        require(
            sha256_text(raw_response) == recorded["raw_response_sha256"],
            "Raw response hash mismatch",
        )
        result = extractor.extract_distance_response(raw_response)
        statuses[result["analysis_response_status"]] += 1
        methods[result["analysis_extraction_method"] or "none"] += 1
        strict_valid += int(result["strict_json_valid"] is True)
        expected_class = recorded["analysis_format_class"]
        if expected_class == "single_json_markdown_fence":
            require(result["analysis_response_status"] == "valid", "Fence not valid")
            require(
                result["analysis_extraction_method"]
                == "single_json_markdown_fence",
                "Fence method mismatch",
            )
        else:
            require(result["analysis_response_status"] == "invalid_json", "Prose accepted")
            require(result["analysis_extraction_method"] is None, "Prose method set")
    after_sha = sha256_file(path)
    require(before_sha == after_sha, "Canary results were mutated")
    require(statuses == Counter({"valid": 9, "invalid_json": 3}), "Status counts")
    require(
        methods == Counter({"single_json_markdown_fence": 9, "none": 3}),
        "Method counts",
    )
    require(strict_valid == 0, "Unexpected strict JSON response")
    return {
        "results_sha256": before_sha,
        "rows": len(source_rows),
        "strict_json_valid": strict_valid,
        "analysis_response_status_counts": dict(sorted(statuses.items())),
        "analysis_extraction_method_counts": dict(sorted(methods.items())),
        "raw_results_mutated": False,
    }


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> int:
    args = parse_args()
    root = args.project_root.resolve()
    protocol_path = resolve_under(root, args.protocol)
    evidence_path = resolve_under(root, args.evidence)
    extractor_path = resolve_under(root, args.extractor)
    output_path = resolve_under(root, args.output) if args.output else None
    canary_path = (
        resolve_under(root, args.canary_results) if args.canary_results else None
    )

    protocol = read_json(protocol_path)
    evidence = read_json(evidence_path)
    extractor = load_extractor(extractor_path)
    validate_protocol(protocol, extractor)
    validate_evidence(evidence)
    synthetic_passed = run_synthetic_tests(extractor)
    canary_validation = (
        validate_canary_results(canary_path, evidence, extractor)
        if canary_path is not None
        else None
    )

    report = {
        "artifact_name": "phase2_response_extraction_validation_v1",
        "schema_version": "1.0",
        "status": "passed",
        "validator_version": SCRIPT_VERSION,
        "protocol": {
            "path": "configs/phase2_response_extraction_protocol_v1.json",
            "sha256": sha256_file(protocol_path),
        },
        "evidence": {
            "path": "configs/phase2_response_extraction_canary_evidence_v1.json",
            "sha256": sha256_file(evidence_path),
        },
        "extractor": {
            "path": "scripts/extract_phase2_responses.py",
            "sha256": sha256_file(extractor_path),
        },
        "validator": {
            "path": "scripts/validate_phase2_response_extraction_v1.py",
            "sha256": sha256_file(Path(__file__)),
        },
        "synthetic_tests": {
            "passed": len(synthetic_passed),
            "names": synthetic_passed,
        },
        "internvl_canary_evidence_verified": canary_validation is not None,
        "internvl_canary_validation": canary_validation,
        "raw_results_mutated": False,
        "ground_truth_compared": False,
        "scientific_metrics_computed": False,
        "full_inference_remains_blocked": True,
    }
    if output_path is not None:
        write_json_atomic(output_path, report)
        print(f"Validation report: {output_path}")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    print("Phase 2 response extraction v1 validation: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
