#!/usr/bin/env python3
"""Statically validate the local-cache Phase 2 Colab canary v5."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any


SCRIPT_VERSION = "0.5.0"
EXPECTED_NOTEBOOK_SHA256 = (
    "1a2da4cc6315f5f88e33d924981aa54c9207b94c1066821bc60182e571f3e3cd"
)
EXPECTED_REVISION_SHA256 = (
    "62a929397a9aaabadbecb78087d949b700fed23f6cae359228046331da402241"
)
EXPECTED_EVIDENCE_SHA256 = (
    "6e71ac585c2a30b0207ccc09348f78772b60b10a66d2131626bef0be71b4cab2"
)
EXPECTED_SOURCE_TRANSCRIPT_SHA256 = (
    "3329bd870297b1cd5b96c38d60483489fe04de92405d62e1254472c218234647"
)
EXPECTED_BASE_COMMIT = "c3cf968744bdf6973731ade239c073ce8e2fdf93"
EXPECTED_BRANCH = "phase2/inference-protocol"
EXPECTED_CANARY_REF = "phase2-canary-v5"
EXPECTED_MODEL_KEYS = (
    "qwen3_vl_4b_instruct",
    "internvl3_5_4b_hf",
    "llava_next_mistral_7b",
)
EXPECTED_FROZEN_HASHES = {
    "colab_environment_sha256": (
        "4d2e96152a71a0105c2cdbf1ec817d1f3eaf75ed4ff98b1f05bb071deab9045c"
    ),
    "protocol_sha256": (
        "3c09ca97984f47b868758ce3b4d07bf85e38dabd3d66b7e4eae9bb474ce4b774"
    ),
    "query_manifest_sha256": (
        "61e4f9a8583dc6f375b6d4d9f72361acf1d9ff3a976485bf5b2f83736ea0360e"
    ),
    "runner_sha256": (
        "643a63625d97aa893322410cf865bd2aa190d2f9b5d4c984aa6c50ef09a38818"
    ),
    "stimulus_packager_sha256": (
        "aae1366fc05caf13a410676c7cea88df417cbeb9c82a3825288d887cba09b185"
    ),
    "stimulus_transport_descriptor_sha256": (
        "c3e88b6225d7026005c6419830fcb3f3ffda3d39ec64e0986c3e15cdac6f8fb0"
    ),
}


class ValidationError(RuntimeError):
    """Raised when a v5 invariant is absent."""


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Statically validate the single-cell Phase 2 Colab v5 notebook."
    )
    parser.add_argument("--project-root", type=Path, default=root)
    parser.add_argument(
        "--notebook",
        type=Path,
        default=Path("notebooks/phase2_colab_canary_v5.ipynb"),
    )
    parser.add_argument(
        "--revision",
        type=Path,
        default=Path("configs/phase2_colab_execution_revision_v5.json"),
    )
    parser.add_argument(
        "--evidence",
        type=Path,
        default=Path("configs/phase2_colab_llava_v4_failure_evidence_v1.json"),
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def resolve_under(root: Path, value: Path) -> Path:
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file(), f"{label} not found: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValidationError(f"Invalid {label} JSON: {error}") from error
    require(isinstance(value, dict), f"{label} is not a JSON object")
    return value


def cell_source(cell: dict[str, Any]) -> str:
    source = cell.get("source", "")
    if isinstance(source, str):
        return source
    if isinstance(source, list) and all(isinstance(item, str) for item in source):
        return "".join(source)
    raise ValidationError("A notebook cell has an invalid source field")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def validate_evidence(path: Path) -> dict[str, Any]:
    actual_sha256 = sha256_file(path)
    require(
        actual_sha256 == EXPECTED_EVIDENCE_SHA256,
        f"Unexpected failure evidence SHA-256: {actual_sha256}",
    )
    value = read_json(path, "failure evidence")
    require(value["schema_version"] == "1.0", "Unexpected evidence schema")
    require(
        value["artifact_name"] == "phase2_colab_llava_v4_failure_evidence_v1",
        "Unexpected evidence artifact name",
    )
    require(
        value["status"] == "technical_failure_recorded_no_inference",
        "Unexpected evidence status",
    )
    attempt = value["attempt"]
    require(
        attempt["attempt_id"] == "20260826T005104Z_08f97dbb",
        "Unexpected v4 attempt ID",
    )
    require(attempt["canary_git_ref"] == "phase2-canary-v4", "Unexpected v4 ref")
    require(
        attempt["canary_git_commit"]
        == "2c3daf86f69f4dab3666c24006f8002a7efba162",
        "Unexpected v4 commit",
    )
    require(
        attempt["model_key"] == "llava_next_mistral_7b",
        "Unexpected incident model",
    )
    failure = value["failure"]
    require(failure["completed_queries"] == 0, "V4 query completed")
    require(failure["inference_generation_started"] is False, "V4 generation started")
    require(failure["failed_stage"] == "model_download_before_first_query", "Unexpected failure stage")
    require(failure["error_type"] == "OSError", "Unexpected failure type")
    require("Transport endpoint is not connected" in failure["error_message"], "Drive error absent")
    require(failure["first_query_marker_observed"] is False, "First query marker present")
    require(failure["persistent_failure_record_completed"] is False, "Persistence unexpectedly completed")
    require(value["raw_model_responses_observed"] is False, "Raw response was observed")
    require(value["ground_truth_compared"] is False, "Ground truth was compared")
    require(value["scientific_metrics_computed"] is False, "Scientific metric was computed")
    require(value["full_inference_remains_blocked"] is True, "Full inference unblocked")
    require(
        value["source_evidence"]["content_sha256"]
        == EXPECTED_SOURCE_TRANSCRIPT_SHA256,
        "Unexpected source transcript hash",
    )
    return value


def validate_revision(path: Path, evidence_path: Path) -> dict[str, Any]:
    evidence = validate_evidence(evidence_path)
    actual_sha256 = sha256_file(path)
    require(
        actual_sha256 == EXPECTED_REVISION_SHA256,
        f"Unexpected technical revision SHA-256: {actual_sha256}",
    )
    value = read_json(path, "technical revision")
    require(value["schema_version"] == "1.0", "Unexpected revision schema")
    require(
        value["artifact_name"] == "phase2_colab_execution_revision_v5",
        "Unexpected revision artifact name",
    )
    require(value["scientific_design_changed"] is False, "Scientific design changed")
    require(value["scientific_outputs_analyzed"] is False, "Outputs were analyzed")
    require(value["full_inference_allowed"] is False, "Full inference was allowed")
    require(value["post_output_technical_revision"] is True, "Post-output status absent")
    require(
        value["repository"]["base_commit"] == EXPECTED_BASE_COMMIT,
        "Unexpected revision base commit",
    )
    require(value["repository"]["branch"] == EXPECTED_BRANCH, "Unexpected branch")
    require(value["canary_git_ref"] == EXPECTED_CANARY_REF, "Unexpected v5 ref")
    require(
        value["failure_evidence"]["sha256"] == EXPECTED_EVIDENCE_SHA256,
        "Revision evidence hash mismatch",
    )
    require(
        value["failure_evidence"]["path"]
        == "configs/phase2_colab_llava_v4_failure_evidence_v1.json",
        "Revision evidence path mismatch",
    )
    require(
        value["frozen_scientific_inputs"] == EXPECTED_FROZEN_HASHES,
        "Frozen scientific hashes changed",
    )
    technical = value["technical_revision"]
    for key in (
        "attempt_directories_are_unique",
        "fallback_failure_record_is_local",
        "full_inference_command_absent",
        "persistent_attempt_outputs_remain_on_google_drive",
        "single_executable_cell",
        "state_machine_is_monotonic",
    ):
        require(technical[key] is True, f"Technical invariant false: {key}")
    require(
        technical["minimum_local_cache_free_bytes"] == 25 * 1024 ** 3,
        "Unexpected local cache free-space threshold",
    )
    require(
        technical["cache_change"]["to"] == "/content/phase2_hf_cache/<model_key>",
        "Unexpected v5 cache destination",
    )
    require(
        technical["runner_or_scientific_protocol_changed"] is False,
        "Runner or scientific protocol changed",
    )
    require(
        technical["response_extraction_protocol_changed"] is False,
        "Response extraction protocol changed",
    )
    require(evidence["failure"]["completed_queries"] == 0, "Evidence inconsistency")
    return value


def validate_notebook(
    notebook_path: Path,
    revision_path: Path,
    evidence_path: Path,
    validator_path: Path,
) -> dict[str, Any]:
    revision = validate_revision(revision_path, evidence_path)
    try:
        raw = notebook_path.read_bytes()
    except FileNotFoundError as error:
        raise ValidationError(f"Notebook not found: {notebook_path}") from error
    notebook_sha256 = hashlib.sha256(raw).hexdigest()
    require(
        notebook_sha256 == EXPECTED_NOTEBOOK_SHA256,
        f"Unexpected notebook SHA-256: {notebook_sha256}",
    )
    try:
        notebook = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValidationError(f"Invalid notebook JSON: {error}") from error

    cells = notebook.get("cells")
    require(isinstance(cells, list), "Notebook cells field is not a list")
    require(len(cells) == 3, f"Expected 3 cells, found {len(cells)}")
    code_cells = [cell for cell in cells if cell.get("cell_type") == "code"]
    markdown_cells = [cell for cell in cells if cell.get("cell_type") == "markdown"]
    require(len(code_cells) == 1, f"Expected one code cell, found {len(code_cells)}")
    require(len(markdown_cells) == 2, "Expected two markdown cells")

    code_cell = code_cells[0]
    source = cell_source(code_cell)
    try:
        ast.parse(source, filename="single-v5-canary-cell")
    except SyntaxError as error:
        raise ValidationError(f"Invalid v5 Python syntax: {error}") from error
    require(code_cell.get("execution_count") is None, "Code cell was executed")
    require(code_cell.get("outputs") == [], "Code cell contains outputs")

    require('CANARY_GIT_REF = "phase2-canary-v5"' in source, "Required v5 Git ref absent")
    require(EXPECTED_REVISION_SHA256 in source, "Revision hash absent")
    require(EXPECTED_EVIDENCE_SHA256 in source, "Evidence hash absent")
    for model_key in EXPECTED_MODEL_KEYS:
        require(model_key in source, f"Frozen model key absent: {model_key}")
    for name, expected_hash in EXPECTED_FROZEN_HASHES.items():
        require(expected_hash in source, f"Frozen hash absent: {name}")

    correct_peel_source = 'f"{CANARY_GIT_REF}^{{commit}}"'
    require(correct_peel_source in source, "Correct Git peel expression absent")
    evaluated_peel = eval(  # noqa: S307 - fixed expression, no user input
        correct_peel_source,
        {"CANARY_GIT_REF": EXPECTED_CANARY_REF},
    )
    require(
        evaluated_peel == "phase2-canary-v5^{commit}",
        f"Git peel expression evaluates incorrectly: {evaluated_peel}",
    )

    forbidden_full_commands = (
        '"--mode", "full"',
        "'--mode', 'full'",
        '"--mode=full"',
        "'--mode=full'",
    )
    require(
        not any(value in source for value in forbidden_full_commands),
        "Notebook contains a full-inference command",
    )
    require(
        source.count('"--max-new-queries", "1"') == 1,
        "Notebook does not contain exactly one one-query invocation",
    )
    require(
        'DRIVE_ROOT / "v5_attempts" / MODEL_KEY / RUN_ID' in source,
        "Unique v5 attempt path absent",
    )
    require(
        'CACHE_DIR = Path("/content/phase2_hf_cache") / MODEL_KEY' in source,
        "Local VM cache path absent",
    )
    require(
        'DRIVE_ROOT / "hf_cache" / MODEL_KEY' not in source,
        "Drive-backed model cache remains",
    )
    for forbidden in (
        'DRIVE_ROOT / "inference"',
        'DRIVE_ROOT / "v2_attempts"',
        'DRIVE_ROOT / "v3_attempts"',
        'DRIVE_ROOT / "v4_attempts"',
        'DRIVE_ROOT / "provenance" / MODEL_KEY / "canary"',
    ):
        require(forbidden not in source, f"Legacy path present: {forbidden}")

    for fragment in (
        "LOCAL_CACHE_MIN_FREE_BYTES = 25 * 1024 ** 3",
        'shutil.disk_usage("/content")',
        'PROVENANCE_DIR / "local_cache_preflight.json"',
        '"cache_storage": "colab_local_vm"',
        'Path("/content") / (',
        '"persistent_recorded"',
        'print(f"Local failure fallback: {local_failure_path}"',
        'require_context("initialized")',
        'require_context("preflight_complete")',
        'require_context("first_query_complete")',
        'require_context("resume_complete")',
        '"initialized", "preflight_complete"',
        '"preflight_complete", "first_query_complete"',
        '"first_query_complete", "resume_complete"',
        '"resume_complete", "complete"',
        "assert len(first_rows) == 1",
        "assert len(current_rows) == 1",
        "assert len(completed_rows) == 12",
        '"Existing exact prefix: 0/12"',
        '"Existing exact prefix: 1/12"',
        "assert len(expected_member_names) == 588",
        'assert len(members) == transport["entry_count"] == 588',
        '"--validate-only"',
        'assert base_runtime["cuda_available"] is True',
        '"global_pip_check_blocking": False',
        '"targeted_imports_completed": True',
        '"scientific_results_analyzed": False',
        "TECHNICAL CANARY V5 COMPLETE.",
    ):
        require(fragment in source, f"Required v5 invariant absent: {fragment}")

    metadata = notebook.get("metadata")
    require(isinstance(metadata, dict), "Notebook metadata is not an object")
    require(metadata.get("accelerator") == "GPU", "Notebook accelerator is not GPU")
    colab = metadata.get("colab")
    require(isinstance(colab, dict), "Notebook Colab metadata is absent")
    require(
        colab.get("name") == "phase2_colab_canary_v5.ipynb",
        "Unexpected Colab notebook name",
    )
    require(notebook.get("nbformat") == 4, "Unexpected notebook major format")
    require(
        isinstance(notebook.get("nbformat_minor"), int)
        and notebook["nbformat_minor"] >= 5,
        "Unexpected notebook minor format",
    )

    return {
        "artifact_name": "phase2_colab_canary_notebook_validation_v5",
        "canary_execution_verified": False,
        "canary_git_ref": EXPECTED_CANARY_REF,
        "canary_git_ref_status": "pending_creation_after_preparation_commit",
        "checks": {
            "all_code_cells_have_valid_python_syntax": True,
            "all_execution_counts_cleared": True,
            "all_frozen_input_hashes_embedded": True,
            "all_outputs_cleared": True,
            "exact_prefix_resume_1_to_12_present": True,
            "exactly_one_one_query_invocation_present": True,
            "failure_record_has_local_fallback": True,
            "full_inference_command_absent": True,
            "full_inference_remains_blocked": True,
            "git_commit_peel_expression_evaluates_correctly": True,
            "global_pip_check_diagnostic_only": True,
            "legacy_attempt_paths_absent": True,
            "local_cache_free_space_check_present": True,
            "local_vm_model_cache_present": True,
            "model_keys_match_frozen_protocol": True,
            "monotonic_attempt_state_present": True,
            "notebook_json_valid": True,
            "persistent_attempt_outputs_remain_on_drive": True,
            "single_executable_cell": True,
            "stimulus_archive_588_member_check_present": True,
            "targeted_package_imports_present": True,
            "unique_attempt_directories_present": True,
            "validate_only_before_model_load_present": True,
            "v4_failure_evidence_verified": True,
        },
        "failure_evidence": {
            "path": "configs/phase2_colab_llava_v4_failure_evidence_v1.json",
            "sha256": EXPECTED_EVIDENCE_SHA256,
            "v4_completed_queries": 0,
        },
        "frozen_inputs": EXPECTED_FROZEN_HASHES,
        "notebook": {
            "code_cells": 1,
            "markdown_cells": 2,
            "nbformat": notebook["nbformat"],
            "nbformat_minor": notebook["nbformat_minor"],
            "path": "notebooks/phase2_colab_canary_v5.ipynb",
            "sha256": notebook_sha256,
            "size_bytes": len(raw),
            "total_cells": len(cells),
        },
        "recorded_date": "2026-08-26",
        "repository": {
            "base_commit": EXPECTED_BASE_COMMIT,
            "branch": EXPECTED_BRANCH,
        },
        "schema_version": "1.0",
        "scientific_design_changed": False,
        "scientific_outputs_analyzed": False,
        "status": "static_validation_passed_pending_colab_execution",
        "technical_revision": {
            "cache_storage": "colab_local_vm",
            "from_notebook": "notebooks/phase2_colab_canary_v4.ipynb",
            "path": "configs/phase2_colab_execution_revision_v5.json",
            "persistent_attempt_outputs": "google_drive",
            "reason": revision["technical_revision"]["reason"],
            "sha256": EXPECTED_REVISION_SHA256,
            "v4_llava_inference_executed": False,
        },
        "validator": {
            "path": "scripts/validate_phase2_colab_notebook_v5.py",
            "sha256": sha256_file(validator_path),
            "version": SCRIPT_VERSION,
        },
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
    notebook_path = resolve_under(root, args.notebook)
    revision_path = resolve_under(root, args.revision)
    evidence_path = resolve_under(root, args.evidence)
    output_path = resolve_under(root, args.output) if args.output else None
    validator_path = Path(__file__).resolve()
    report = validate_notebook(
        notebook_path,
        revision_path,
        evidence_path,
        validator_path,
    )
    if output_path is not None:
        write_json_atomic(output_path, report)
        print(f"Validation report: {output_path}")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    print("Phase 2 Colab notebook v5 static validation: OK")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValidationError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
