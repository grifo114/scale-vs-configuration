#!/usr/bin/env python3
"""Prepare the predeclared Phase 2 inference protocol and query manifest.

The script validates the frozen v0.5 dataset and its external stimuli, resolves
immutable Hugging Face revisions for the three predeclared models, and writes:

* configs/phase2_query_manifest.jsonl
* configs/phase2_inference_protocol.json

No model weights are downloaded and no inference is executed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCRIPT_VERSION = "0.1.1"
EXPECTED_BRANCH = "phase2/inference-protocol"
DATASET_TAG = "v0.5-confirmatory-dataset"
DATASET_TAG_COMMIT = "ef785e8ffebf42418190dcebd36c8ddf11a47102"
EXPECTED_SHA256SUMS_SHA256 = (
    "b9013ed8af55f7252f294ec4b8250d08a0b52bae8e263f33b853234c1cddd7b3"
)
EXPECTED_SOURCE_LOG_SHA256 = (
    "ed8957558dd677a885e2ab753527974050e64f583098562969b8ca56bf1bb59c"
)
EXPECTED_CHECKPOINT_FILE_COUNT = 269
EXPECTED_SCENES = 49
EXPECTED_RELATIONS = 588
EXPECTED_PER_FOLD = 294
EXPECTED_PER_STRATUM = 196

PROMPT = (
    "Object A is marked with a red box and object B with a blue box. "
    "Estimate the 3D Euclidean distance between the centers of their 3D "
    "bounding boxes, in meters. Return only JSON in this format: "
    '{"distance_m": number}'
)

ORDER_SEED = "phase2-confirmatory-order-v1-20260823"
CANARY_SEED = "phase2-canary-v1-20260823"

MODEL_SPECS = (
    {
        "key": "qwen3_vl_4b_instruct",
        "model_id": "Qwen/Qwen3-VL-4B-Instruct",
        "family": "Qwen3-VL",
        "role": "contemporary_primary",
        "declared_parameters_billion": 4.0,
        "expected_license": "apache-2.0",
    },
    {
        "key": "internvl3_5_4b_hf",
        "model_id": "OpenGVLab/InternVL3_5-4B-HF",
        "family": "InternVL3.5",
        "role": "contemporary_independent",
        "declared_parameters_billion": 4.7,
        "expected_license": "apache-2.0",
    },
    {
        "key": "llava_next_mistral_7b",
        "model_id": "llava-hf/llava-v1.6-mistral-7b-hf",
        "family": "LLaVA-NeXT",
        "role": "historical_reference",
        "declared_parameters_billion": 8.0,
        "expected_license": "apache-2.0",
    },
)


class ProtocolError(RuntimeError):
    """Raised when a protocol precondition is not satisfied."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare the predeclared Phase 2 inference protocol."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Project root (default: current directory).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/v0.5"),
        help="Frozen checkpoint path, relative to project root by default.",
    )
    parser.add_argument(
        "--protocol-output",
        type=Path,
        default=Path("configs/phase2_inference_protocol.json"),
        help="Protocol output path, relative to project root by default.",
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=Path("configs/phase2_query_manifest.jsonl"),
        help="Query-manifest output path, relative to project root by default.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing protocol outputs.",
    )
    return parser.parse_args()


def resolve_under(root: Path, value: Path) -> Path:
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ProtocolError(f"Required file not found: {path}") from error
    except json.JSONDecodeError as error:
        raise ProtocolError(f"Invalid JSON in {path}: {error}") from error
    if not isinstance(value, dict):
        raise ProtocolError(f"Expected a JSON object in {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as error:
        raise ProtocolError(f"Required file not found: {path}") from error
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ProtocolError(
                f"Invalid JSONL row at {path}:{line_number}: {error}"
            ) from error
        if not isinstance(value, dict):
            raise ProtocolError(
                f"Expected a JSON object at {path}:{line_number}"
            )
        rows.append(value)
    return rows


def git(root: Path, *arguments: str) -> str:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        stderr = getattr(error, "stderr", "") or ""
        raise ProtocolError(
            f"Git command failed: git {' '.join(arguments)}\n{stderr.strip()}"
        ) from error
    return completed.stdout.strip()


def validate_git_state(root: Path, generator_path: Path) -> dict[str, Any]:
    branch = git(root, "branch", "--show-current")
    if branch != EXPECTED_BRANCH:
        raise ProtocolError(
            f"Expected branch {EXPECTED_BRANCH!r}, found {branch!r}"
        )
    try:
        generator_relative = generator_path.resolve().relative_to(root).as_posix()
    except ValueError as error:
        raise ProtocolError(
            f"Generator must be located inside the project root: {generator_path}"
        ) from error

    status_lines = [
        line
        for line in git(
            root, "status", "--porcelain", "--untracked-files=all"
        ).splitlines()
        if line.strip()
    ]
    allowed_generator_lines = {
        f"?? {generator_relative}",
        f" M {generator_relative}",
        f"M  {generator_relative}",
        f"MM {generator_relative}",
    }
    unrelated_status = [
        line for line in status_lines if line not in allowed_generator_lines
    ]
    if unrelated_status:
        raise ProtocolError(
            "The worktree contains changes unrelated to this generator:\n"
            + "\n".join(unrelated_status)
        )
    head = git(root, "rev-parse", "HEAD")
    tag_commit = git(root, "rev-list", "-n", "1", DATASET_TAG)
    if tag_commit != DATASET_TAG_COMMIT:
        raise ProtocolError(
            f"Unexpected {DATASET_TAG} target: {tag_commit}"
        )
    try:
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", DATASET_TAG_COMMIT, head],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as error:
        raise ProtocolError(
            f"Dataset commit {DATASET_TAG_COMMIT} is not an ancestor of HEAD"
        ) from error
    return {
        "protocol_source_branch": branch,
        "protocol_source_commit": head,
        "generator_path": generator_relative,
        "generator_sha256": sha256_file(generator_path),
        "worktree_other_changes_before_generation": False,
        "generator_was_the_only_possible_change": bool(status_lines),
    }


def validate_checkpoint_checksums(checkpoint: Path) -> int:
    checksum_path = checkpoint / "SHA256SUMS"
    try:
        lines = checksum_path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as error:
        raise ProtocolError(f"Missing checksum manifest: {checksum_path}") from error

    checksum_manifest_hash = sha256_file(checksum_path)
    if checksum_manifest_hash != EXPECTED_SHA256SUMS_SHA256:
        raise ProtocolError(
            "Unexpected SHA256SUMS hash: "
            f"actual={checksum_manifest_hash}, "
            f"expected={EXPECTED_SHA256SUMS_SHA256}"
        )

    verified = 0
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            raise ProtocolError(
                f"Invalid checksum line {checksum_path}:{line_number}"
            )
        expected, relative = parts
        relative = relative.lstrip("* ")
        target = checkpoint / relative
        if not target.is_file():
            raise ProtocolError(f"Checkpoint file not found: {target}")
        actual = sha256_file(target)
        if actual != expected:
            raise ProtocolError(
                f"Checkpoint hash mismatch for {relative}: "
                f"actual={actual}, expected={expected}"
            )
        verified += 1

    if verified != EXPECTED_CHECKPOINT_FILE_COUNT:
        raise ProtocolError(
            f"Expected {EXPECTED_CHECKPOINT_FILE_COUNT} checkpoint hashes, "
            f"verified {verified}"
        )
    return verified


def validate_render_protocols(checkpoint: Path) -> int:
    paths = sorted((checkpoint / "render_protocols").glob("*.json"))
    if len(paths) != EXPECTED_SCENES:
        raise ProtocolError(
            f"Expected {EXPECTED_SCENES} render protocols, found {len(paths)}"
        )
    expected = {
        "object_a": {"label": "A", "color": "#E53935"},
        "object_b": {"label": "B", "color": "#1565E8"},
        "ground_truth_in_stimulus": False,
        "categories_in_stimulus": False,
    }
    for path in paths:
        actual = read_json(path).get("stimulus_annotations")
        if actual != expected:
            raise ProtocolError(
                f"Unexpected stimulus annotation protocol in {path}: {actual!r}"
            )
    return len(paths)


def validate_dataset(
    root: Path,
    checkpoint: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    summary = read_json(checkpoint / "summary.json")
    expected_summary = {
        "status": "confirmatory_pool_exhausted_before_target",
        "inspected_captures": 102,
        "target_accepted_scenes": 50,
        "accepted_captures": EXPECTED_SCENES,
        "target_shortfall": 1,
        "numerically_rejected_captures": 51,
        "visually_rejected_captures": 2,
        "selected_relations": EXPECTED_RELATIONS,
        "relations_per_accepted_capture": 12,
    }
    for key, expected in expected_summary.items():
        actual = summary.get(key)
        if actual != expected:
            raise ProtocolError(
                f"Unexpected summary value for {key}: {actual!r} != {expected!r}"
            )
    if summary.get("source_log_sha256") != EXPECTED_SOURCE_LOG_SHA256:
        raise ProtocolError("Unexpected official source-log SHA-256")
    if summary.get("visual_rejected_capture_ids") != ["47115525", "47204605"]:
        raise ProtocolError("Unexpected visual-rejection IDs")

    rows = read_jsonl(checkpoint / "rendered_manifest.jsonl")
    if len(rows) != EXPECTED_RELATIONS:
        raise ProtocolError(
            f"Expected {EXPECTED_RELATIONS} rendered rows, found {len(rows)}"
        )

    pair_ids = [str(row.get("pair_id", "")) for row in rows]
    scenes = {str(row.get("scene_id", "")) for row in rows}
    folds = Counter(str(row.get("fold", "")) for row in rows)
    strata = Counter(str(row.get("distance_stratum", "")) for row in rows)
    if len(set(pair_ids)) != EXPECTED_RELATIONS:
        raise ProtocolError("Rendered manifest pair IDs are not unique")
    if len(scenes) != EXPECTED_SCENES:
        raise ProtocolError(f"Expected {EXPECTED_SCENES} scenes, found {len(scenes)}")
    if folds != Counter({"A": EXPECTED_PER_FOLD, "B": EXPECTED_PER_FOLD}):
        raise ProtocolError(f"Unexpected fold counts: {dict(folds)}")
    expected_strata = {
        "short": EXPECTED_PER_STRATUM,
        "medium": EXPECTED_PER_STRATUM,
        "long": EXPECTED_PER_STRATUM,
    }
    if strata != Counter(expected_strata):
        raise ProtocolError(f"Unexpected distance-stratum counts: {dict(strata)}")

    required = {
        "pair_id",
        "scene_id",
        "fold",
        "distance_stratum",
        "ground_truth_m",
        "object_a",
        "object_b",
        "stimulus_path",
        "stimulus_sha256",
    }
    missing_stimuli: list[str] = []
    mismatched_stimuli: list[str] = []
    total_bytes = 0
    for row_number, row in enumerate(rows, start=1):
        missing_fields = sorted(required - row.keys())
        if missing_fields:
            raise ProtocolError(
                f"Rendered row {row_number} lacks: {', '.join(missing_fields)}"
            )
        for object_field in ("object_a", "object_b"):
            if not isinstance(row[object_field], (str, dict)):
                raise ProtocolError(
                    f"Rendered row {row_number} has an unsupported "
                    f"{object_field} value: {row[object_field]!r}"
                )
        ground_truth = float(row["ground_truth_m"])
        if not math.isfinite(ground_truth) or ground_truth <= 0:
            raise ProtocolError(
                f"Invalid ground truth for {row['pair_id']}: {ground_truth}"
            )
        stimulus = resolve_under(root, Path(str(row["stimulus_path"])))
        if not stimulus.is_file():
            missing_stimuli.append(str(stimulus))
            continue
        total_bytes += stimulus.stat().st_size
        actual_hash = sha256_file(stimulus)
        if actual_hash != str(row["stimulus_sha256"]):
            mismatched_stimuli.append(str(row["pair_id"]))

    if missing_stimuli:
        preview = "\n".join(missing_stimuli[:5])
        raise ProtocolError(
            f"Missing {len(missing_stimuli)} stimuli. First paths:\n{preview}"
        )
    if mismatched_stimuli:
        raise ProtocolError(
            f"Stimulus hash mismatches: {', '.join(mismatched_stimuli[:5])}"
        )

    metadata = {
        "number_of_relations": len(rows),
        "number_of_scenes": len(scenes),
        "fold_counts": dict(sorted(folds.items())),
        "distance_stratum_counts": dict(sorted(strata.items())),
        "external_stimulus_bytes": total_bytes,
        "external_stimulus_mib": round(total_bytes / (1024**2), 6),
    }
    return rows, metadata


def stable_rank(seed: str, pair_id: str) -> str:
    return sha256_text(f"{seed}\0{pair_id}")


def select_canary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    used_scenes: set[str] = set()
    cell_order = (
        ("short", "A"),
        ("short", "B"),
        ("medium", "A"),
        ("medium", "B"),
        ("long", "A"),
        ("long", "B"),
    )
    for stratum, fold in cell_order:
        candidates = [
            row
            for row in rows
            if row["distance_stratum"] == stratum and row["fold"] == fold
        ]
        candidates.sort(
            key=lambda row: stable_rank(
                f"{CANARY_SEED}:{stratum}:{fold}", str(row["pair_id"])
            )
        )
        cell_selected = 0
        for row in candidates:
            scene_id = str(row["scene_id"])
            if scene_id in used_scenes:
                continue
            selected.append(row)
            used_scenes.add(scene_id)
            cell_selected += 1
            if cell_selected == 2:
                break
        if cell_selected != 2:
            raise ProtocolError(
                f"Could not select two unique-scene canary rows for {stratum}/{fold}"
            )

    cells = Counter(
        (str(row["distance_stratum"]), str(row["fold"])) for row in selected
    )
    if len(selected) != 12 or len(used_scenes) != 12 or set(cells.values()) != {2}:
        raise ProtocolError("Canary-selection invariants failed")
    return selected


def build_query_manifest(
    rows: list[dict[str, Any]],
    canary_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    canary_index = {
        str(row["pair_id"]): index
        for index, row in enumerate(canary_rows, start=1)
    }
    ordered = sorted(
        rows,
        key=lambda row: stable_rank(ORDER_SEED, str(row["pair_id"])),
    )
    manifest: list[dict[str, Any]] = []
    for run_index, row in enumerate(ordered, start=1):
        pair_id = str(row["pair_id"])
        query = dict(row)
        query.update(
            {
                "run_index": run_index,
                "pair_id": pair_id,
                "scene_id": str(row["scene_id"]),
                "fold": str(row["fold"]),
                "distance_stratum": str(row["distance_stratum"]),
                "ground_truth_m": float(row["ground_truth_m"]),
                "stimulus_path": str(row["stimulus_path"]),
                "stimulus_sha256": str(row["stimulus_sha256"]),
                "prompt": PROMPT,
                "is_canary": pair_id in canary_index,
                "canary_index": canary_index.get(pair_id),
            }
        )
        manifest.append(query)
    return manifest


def serialize_jsonl(rows: Iterable[dict[str, Any]]) -> str:
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    )


def resolve_huggingface_model(model_spec: dict[str, Any]) -> dict[str, Any]:
    model_id = str(model_spec["model_id"])
    encoded = urllib.parse.quote(model_id, safe="/")
    url = f"https://huggingface.co/api/models/{encoded}"
    request = urllib.request.Request(
        url,
        headers={"User-Agent": f"scale-vs-configuration/{SCRIPT_VERSION}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = json.load(response)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
        raise ProtocolError(
            f"Could not resolve Hugging Face model metadata for {model_id}: {error}"
        ) from error

    revision = str(payload.get("sha", ""))
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ProtocolError(
            f"Invalid Hugging Face revision for {model_id}: {revision!r}"
        )
    card_data = payload.get("cardData") or {}
    actual_license = str(card_data.get("license", "")).lower()
    expected_license = str(model_spec["expected_license"]).lower()
    if actual_license != expected_license:
        raise ProtocolError(
            f"License mismatch for {model_id}: "
            f"actual={actual_license!r}, expected={expected_license!r}"
        )

    resolved = dict(model_spec)
    resolved.update(
        {
            "revision": revision,
            "license": actual_license,
            "pipeline_tag": payload.get("pipeline_tag"),
            "library_name": payload.get("library_name"),
            "metadata_source": url,
        }
    )
    return resolved


def write_exclusive(path: Path, content: str, force: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        raise ProtocolError(
            f"Output already exists: {path}. Use --force only after reviewing it."
        )
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    args = parse_args()
    root = args.project_root.resolve()
    checkpoint = resolve_under(root, args.checkpoint)
    protocol_output = resolve_under(root, args.protocol_output)
    manifest_output = resolve_under(root, args.manifest_output)

    if protocol_output == manifest_output:
        raise ProtocolError("Protocol and manifest outputs must be different files")
    for output in (protocol_output, manifest_output):
        if output.exists() and not args.force:
            raise ProtocolError(
                f"Output already exists: {output}. Use --force only after review."
            )

    git_metadata = validate_git_state(root, Path(__file__))
    checksum_count = validate_checkpoint_checksums(checkpoint)
    render_protocol_count = validate_render_protocols(checkpoint)
    rows, dataset_metadata = validate_dataset(root, checkpoint)
    canary_rows = select_canary(rows)
    query_manifest = build_query_manifest(rows, canary_rows)
    manifest_content = serialize_jsonl(query_manifest)
    manifest_hash = sha256_text(manifest_content)

    resolved_models = [resolve_huggingface_model(spec) for spec in MODEL_SPECS]
    resolved_at = datetime.now(timezone.utc).isoformat()
    ordered_pair_ids = [row["pair_id"] for row in query_manifest]

    protocol: dict[str, Any] = {
        "schema_version": "1.0",
        "protocol_name": "phase2_confirmatory_vlm_inference",
        "protocol_status": "predeclared_pending_canary",
        "created_at_utc": resolved_at,
        "generator": {
            "path": "scripts/prepare_phase2_inference_protocol.py",
            "version": SCRIPT_VERSION,
        },
        "source_control": git_metadata,
        "dataset": {
            "checkpoint": "checkpoints/v0.5",
            "checkpoint_tag": DATASET_TAG,
            "checkpoint_tag_commit": DATASET_TAG_COMMIT,
            "checkpoint_checksums_verified": checksum_count,
            "checkpoint_sha256s_sha256": sha256_file(
                checkpoint / "SHA256SUMS"
            ),
            "rendered_manifest_sha256": sha256_file(
                checkpoint / "rendered_manifest.jsonl"
            ),
            "accepted_pairs_manifest_sha256": sha256_file(
                checkpoint / "accepted_pairs_manifest.jsonl"
            ),
            "render_protocols_verified": render_protocol_count,
            **dataset_metadata,
        },
        "stimulus_semantics": {
            "object_a_label": "A",
            "object_a_color": "#E53935",
            "object_b_label": "B",
            "object_b_color": "#1565E8",
            "ground_truth_visible": False,
            "category_names_visible": False,
        },
        "query_manifest": {
            "path": "configs/phase2_query_manifest.jsonl",
            "sha256": manifest_hash,
            "rows": len(query_manifest),
            "ordering_algorithm": "ascending_sha256(seed + NUL + pair_id)",
            "ordering_seed": ORDER_SEED,
            "ordered_pair_ids_sha256": sha256_text(
                "\n".join(ordered_pair_ids) + "\n"
            ),
        },
        "models": resolved_models,
        "inference": {
            "condition": "G-oracle",
            "prompt": PROMPT,
            "prompt_is_identical_across_models": True,
            "native_chat_template_per_model": True,
            "native_image_processor_per_model": True,
            "manual_image_resizing": False,
            "temperature": 0.0,
            "do_sample": False,
            "max_new_tokens": 64,
            "generation_seed": 0,
            "runs_per_model": 1,
            "backend": "pytorch_transformers_cuda",
            "primary_environment": "Google Colab",
            "gpu_model": "record_at_runtime",
            "software_versions": "freeze_after_successful_canary",
            "quantization": {
                "library": "bitsandbytes",
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_use_double_quant": True,
                "bnb_4bit_compute_dtype": "float16",
            },
        },
        "response_schema": {
            "expected_json": {"distance_m": "positive finite number"},
            "store_raw_response": True,
            "invalid_response_is_retained": True,
            "retry_on_invalid_response": False,
            "model_specific_prompt_repair": False,
        },
        "canary": {
            "purpose": "technical compatibility only",
            "selection_seed": CANARY_SEED,
            "selection_algorithm": (
                "two rows per distance_stratum x fold cell, ranked by SHA-256, "
                "with twelve unique scenes"
            ),
            "number_of_queries": 12,
            "number_of_scenes": 12,
            "pair_ids_in_canary_order": [
                str(row["pair_id"]) for row in canary_rows
            ],
            "acceptance_criteria": [
                "model loads without an out-of-memory error",
                "all 12 queries produce a stored raw response row",
                "result resumption accepts an exact completed prefix",
                "runtime metadata and package versions are recorded",
            ],
            "json_validity_is_not_an_inclusion_criterion": True,
            "canary_outputs_excluded_from_scientific_analysis": True,
            "full_run_reexecutes_canary_queries": True,
        },
        "failure_policy": {
            "model_replacement_after_observing_outputs": False,
            "prompt_change_after_observing_outputs": False,
            "if_free_colab_gpu_is_insufficient": (
                "use a larger CUDA GPU with the same model revision, native "
                "processor, prompt, quantization, and generation settings"
            ),
            "if_a_model_cannot_execute_under_the_predeclared_stack": (
                "record the technical failure and amend the protocol before "
                "any full confirmatory result is analyzed"
            ),
        },
        "outputs": {
            "root": "outputs/phase2/inference",
            "results_filename": "results.jsonl",
            "metadata_filename": "metadata.json",
            "checkpoint_after_each_query": True,
            "resume_requires_exact_prefix": True,
            "official_results_tracked_in_git": False,
        },
        "freeze_transition": {
            "current_state": "pending_canary",
            "requirements": [
                "three successful technical canaries",
                "exact CUDA GPU and package metadata captured",
                "query manifest and protocol committed before full inference",
            ],
            "scientific_results_may_be_analyzed_before_freeze": False,
        },
    }
    protocol_content = json.dumps(
        protocol, ensure_ascii=False, indent=2, sort_keys=True
    ) + "\n"

    write_exclusive(manifest_output, manifest_content, args.force)
    try:
        write_exclusive(protocol_output, protocol_content, args.force)
    except Exception:
        if manifest_output.exists():
            manifest_output.unlink()
        raise

    print(f"Protocol: {protocol_output}")
    print(f"Query manifest: {manifest_output}")
    print(f"Query manifest SHA-256: {manifest_hash}")
    print(f"Relations: {len(query_manifest)}")
    print(f"Canary queries: {len(canary_rows)}")
    print("Resolved models:")
    for model in resolved_models:
        print(f"  {model['model_id']} @ {model['revision']}")
    print("Status: predeclared_pending_canary")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ProtocolError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
