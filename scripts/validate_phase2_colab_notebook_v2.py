#!/usr/bin/env python3
"""Validate the frozen Phase 2 Colab canary notebook.

The validator performs static checks only. It does not import model packages,
access CUDA, download weights, execute notebook cells, or inspect scientific
responses. When requested, it writes a deterministic JSON validation artifact.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any


SCRIPT_VERSION = "0.2.0"
EXPECTED_NOTEBOOK_SHA256 = (
    "65183b437fcccbb252ba19e4c0d4fc71e97d6e55fc3d94c4e19249305d4e1b65"
)
EXPECTED_BASE_COMMIT = "143c9fad9a5c35ac66268731cd2431f29d7333cc"
EXPECTED_BRANCH = "phase2/inference-protocol"
EXPECTED_CANARY_REF = "phase2-canary-v2"
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
    """Raised when a notebook invariant is not satisfied."""


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Statically validate the frozen Phase 2 Colab notebook."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=project_root,
        help="Project root (default: parent of scripts/).",
    )
    parser.add_argument(
        "--notebook",
        type=Path,
        default=Path("notebooks/phase2_colab_canary_v2.ipynb"),
        help="Notebook path, relative to the project root by default.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional deterministic JSON report path.",
    )
    return parser.parse_args()


def resolve_under(root: Path, value: Path) -> Path:
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def validate_notebook(notebook_path: Path, validator_path: Path) -> dict[str, Any]:
    try:
        raw = notebook_path.read_bytes()
    except FileNotFoundError as error:
        raise ValidationError(f"Notebook not found: {notebook_path}") from error

    notebook_sha256 = sha256_bytes(raw)
    require(
        notebook_sha256 == EXPECTED_NOTEBOOK_SHA256,
        f"Unexpected notebook SHA-256: {notebook_sha256}",
    )

    try:
        notebook = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValidationError(f"Invalid notebook JSON: {error}") from error
    require(isinstance(notebook, dict), "Notebook root is not a JSON object")

    cells = notebook.get("cells")
    require(isinstance(cells, list), "Notebook cells field is not a list")
    require(len(cells) == 16, f"Expected 16 cells, found {len(cells)}")

    code_cells = [cell for cell in cells if cell.get("cell_type") == "code"]
    markdown_cells = [
        cell for cell in cells if cell.get("cell_type") == "markdown"
    ]
    require(len(code_cells) == 12, f"Expected 12 code cells, found {len(code_cells)}")
    require(
        len(markdown_cells) == 4,
        f"Expected 4 markdown cells, found {len(markdown_cells)}",
    )

    for index, cell in enumerate(code_cells, start=1):
        try:
            ast.parse(cell_source(cell), filename=f"code-cell-{index}")
        except SyntaxError as error:
            raise ValidationError(
                f"Invalid Python syntax in code cell {index}: {error}"
            ) from error
        require(
            cell.get("execution_count") is None,
            f"Code cell {index} contains an execution count",
        )
        require(cell.get("outputs") == [], f"Code cell {index} contains outputs")

    joined_source = "\n".join(cell_source(cell) for cell in cells)
    require(
        'CANARY_GIT_REF = "phase2-canary-v2"' in joined_source,
        "Required canary Git ref is absent",
    )
    for model_key in EXPECTED_MODEL_KEYS:
        require(model_key in joined_source, f"Frozen model key is absent: {model_key}")
    for name, expected_hash in EXPECTED_FROZEN_HASHES.items():
        require(expected_hash in joined_source, f"Frozen hash is absent: {name}")

    forbidden_full_commands = (
        '"--mode", "full"',
        "'--mode', 'full'",
        '"--mode=full"',
        "'--mode=full'",
    )
    require(
        not any(value in joined_source for value in forbidden_full_commands),
        "Notebook contains a full-inference command",
    )
    require(
        joined_source.count('"--max-new-queries", "1"') == 1,
        "Notebook does not contain exactly one one-query invocation",
    )

    required_resume_fragments = (
        'assert len(first_rows) == 1',
        'assert first_metadata["status"] == "partial"',
        'elif len(current_rows) == 1:',
        'assert len(completed_rows) == 12',
        'assert completed_metadata["status"] == "complete"',
        '"Existing exact prefix: 0/12"',
        '"Existing exact prefix: 1/12"',
    )
    for fragment in required_resume_fragments:
        require(fragment in joined_source, f"Resume invariant is absent: {fragment}")

    required_preflight_fragments = (
        'assert len(expected_member_names) == 588',
        'assert len(members) == transport["entry_count"] == 588',
        '"--validate-only"',
        'assert base_runtime["cuda_available"] is True',
        'assert environment_spec["canary_execution"]["full_mode_allowed"] is False',
        '"global_pip_check_blocking": False',
        '"targeted_imports_completed": True',
        '"scientific_results_analyzed": False',
    )
    for fragment in required_preflight_fragments:
        require(fragment in joined_source, f"Preflight invariant is absent: {fragment}")

    metadata = notebook.get("metadata")
    require(isinstance(metadata, dict), "Notebook metadata is not an object")
    require(metadata.get("accelerator") == "GPU", "Notebook accelerator is not GPU")
    colab = metadata.get("colab")
    require(isinstance(colab, dict), "Notebook Colab metadata is absent")
    require(
        colab.get("name") == "phase2_colab_canary_v2.ipynb",
        "Unexpected Colab notebook name",
    )
    language_info = metadata.get("language_info")
    require(isinstance(language_info, dict), "Notebook language metadata is absent")
    require(language_info.get("version") == "3.12", "Unexpected Python line")
    require(notebook.get("nbformat") == 4, "Unexpected notebook major format")
    require(
        isinstance(notebook.get("nbformat_minor"), int)
        and notebook["nbformat_minor"] >= 5,
        "Unexpected notebook minor format",
    )

    report = {
        "artifact_name": "phase2_colab_canary_notebook_validation_v2",
        "canary_execution_verified": False,
        "canary_git_ref": EXPECTED_CANARY_REF,
        "canary_git_ref_status": "pending_creation_after_preparation_commit",
        "checks": {
            "all_code_cells_have_valid_python_syntax": True,
            "all_execution_counts_cleared": True,
            "all_frozen_input_hashes_embedded": True,
            "all_outputs_cleared": True,
            "canary_git_ref_embedded": True,
            "cuda_preflight_present": True,
            "exact_prefix_resume_1_to_12_present": True,
            "exactly_one_one_query_invocation_present": True,
            "global_pip_check_diagnostic_only": True,
            "targeted_package_imports_present": True,
            "full_inference_command_absent": True,
            "full_inference_remains_blocked": True,
            "model_keys_match_frozen_protocol": True,
            "notebook_json_valid": True,
            "stimulus_archive_588_member_check_present": True,
            "validate_only_before_model_load_present": True,
        },
        "frozen_inputs": EXPECTED_FROZEN_HASHES,
        "notebook": {
            "code_cells": len(code_cells),
            "markdown_cells": len(markdown_cells),
            "nbformat": notebook["nbformat"],
            "nbformat_minor": notebook["nbformat_minor"],
            "path": "notebooks/phase2_colab_canary_v2.ipynb",
            "sha256": notebook_sha256,
            "size_bytes": len(raw),
            "total_cells": len(cells),
        },
        "recorded_date": "2026-08-24",
        "repository": {
            "base_commit": EXPECTED_BASE_COMMIT,
            "branch": EXPECTED_BRANCH,
        },
        "schema_version": "1.0",
        "scientific_design_changed": False,
        "scientific_outputs_analyzed": False,
        "status": "static_validation_passed_pending_colab_execution",
        "technical_revision": {
            "from_notebook": "notebooks/phase2_colab_canary_v1.ipynb",
            "global_pip_check_blocking": False,
            "reason": "Scope the blocking dependency checks to the inference stack.",
            "scientific_design_changed": False,
        },
        "validator": {
            "path": "scripts/validate_phase2_colab_notebook_v2.py",
            "sha256": sha256_file(validator_path),
            "version": SCRIPT_VERSION,
        },
    }
    return report


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    content = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    args = parse_args()
    root = args.project_root.resolve()
    notebook_path = resolve_under(root, args.notebook)
    output_path = resolve_under(root, args.output) if args.output else None
    validator_path = Path(__file__).resolve()
    report = validate_notebook(notebook_path, validator_path)
    if output_path is not None:
        write_json_atomic(output_path, report)
        print(f"Validation report: {output_path}")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    print("Phase 2 Colab notebook static validation: OK")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValidationError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
