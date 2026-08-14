#!/usr/bin/env python3
"""Execute the frozen G-oracle manifest with a local MLX-VLM model.

The default dry-run validates the v0.1 checkpoint, the deterministic query
order, the local stimuli, the model snapshot, and the main package versions.
Inference outputs must be written outside the frozen checkpoint directory.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RUNNER_VERSION = "0.1.0"
DEFAULT_CHECKPOINT_DIR = Path("checkpoints/v0.1")
RESULTS_FILENAME = "results.jsonl"
METADATA_FILENAME = "metadata.json"


class ValidationError(RuntimeError):
    """Raised when a reproducibility check fails."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate or execute the frozen G-oracle manifest with MLX-VLM."
        )
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Project root (default: current directory).",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=DEFAULT_CHECKPOINT_DIR,
        help="Checkpoint directory, relative to project root by default.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        help="Local model snapshot. Defaults to metadata.json:model_path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for a real run. Required without --dry-run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and environment without loading the model.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Run only the first N queries in deterministic order.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an interrupted output whose rows form an exact prefix.",
    )
    parser.add_argument(
        "--allow-missing-stimuli",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--allow-missing-model",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--skip-version-check",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--skip-platform-check",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be a positive integer")
    if not args.dry_run and args.output_dir is None:
        parser.error("--output-dir is required for inference")
    if args.dry_run and args.resume:
        parser.error("--resume is not used with --dry-run")
    return args


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_under(base: Path, value: Path) -> Path:
    return value.resolve() if value.is_absolute() else (base / value).resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValidationError(f"Required file not found: {path}") from error
    except json.JSONDecodeError as error:
        raise ValidationError(f"Invalid JSON in {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValidationError(f"Expected a JSON object in {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as error:
        raise ValidationError(f"Required file not found: {path}") from error

    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValidationError(
                f"Invalid JSONL row at {path}:{line_number}: {error}"
            ) from error
        if not isinstance(row, dict):
            raise ValidationError(
                f"Expected a JSON object at {path}:{line_number}"
            )
        rows.append(row)
    return rows


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def installed_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError as error:
        raise ValidationError(
            f"Required Python distribution is not installed: {distribution}"
        ) from error


def check_environment(
    metadata: dict[str, Any],
    skip_versions: bool,
    skip_platform: bool,
) -> dict[str, str]:
    if skip_versions:
        versions = {}
        for distribution in ("mlx", "mlx-vlm", "transformers"):
            try:
                versions[distribution] = installed_version(distribution)
            except ValidationError:
                versions[distribution] = "not installed"
    else:
        versions = {
            "mlx": installed_version("mlx"),
            "mlx-vlm": installed_version("mlx-vlm"),
            "transformers": installed_version("transformers"),
        }

    if not skip_versions:
        expected = {
            "mlx": str(metadata["mlx_version"]),
            "mlx-vlm": str(metadata["mlx_vlm_version"]),
            "transformers": str(metadata["transformers_version"]),
        }
        mismatches = [
            f"{name}: installed={versions[name]}, expected={wanted}"
            for name, wanted in expected.items()
            if versions[name] != wanted
        ]
        current_python = platform.python_version()
        expected_python = str(metadata["python_version"])
        if current_python != expected_python:
            mismatches.append(
                f"Python: installed={current_python}, expected={expected_python}"
            )
        if mismatches:
            raise ValidationError(
                "Environment version mismatch:\n  " + "\n  ".join(mismatches)
            )

    current_architecture = platform.machine()
    expected_architecture = str(metadata.get("architecture", ""))
    if (
        not skip_platform
        and expected_architecture
        and current_architecture != expected_architecture
    ):
        raise ValidationError(
            "Architecture mismatch: "
            f"current={current_architecture}, expected={expected_architecture}"
        )
    return versions


def validate_manifest(
    manifest_path: Path,
    metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    actual_hash = sha256_file(manifest_path)
    expected_hash = str(metadata["manifest_sha256"])
    if actual_hash != expected_hash:
        raise ValidationError(
            "Manifest SHA-256 mismatch: "
            f"actual={actual_hash}, expected={expected_hash}"
        )

    rows = read_jsonl(manifest_path)
    required = {
        "pair_id",
        "scene_id",
        "fold",
        "stimulus",
        "ground_truth_m",
        "prompt",
    }
    pair_ids: list[str] = []
    scene_ids: set[str] = set()
    for index, row in enumerate(rows, start=1):
        missing = sorted(required - row.keys())
        if missing:
            raise ValidationError(
                f"Manifest row {index} lacks fields: {', '.join(missing)}"
            )
        pair_id = str(row["pair_id"])
        pair_ids.append(pair_id)
        scene_ids.add(str(row["scene_id"]))
        if row["fold"] not in {"A", "B"}:
            raise ValidationError(f"Invalid fold for {pair_id}: {row['fold']}")
        try:
            ground_truth = float(row["ground_truth_m"])
        except (TypeError, ValueError) as error:
            raise ValidationError(
                f"Invalid ground_truth_m for {pair_id}: {row['ground_truth_m']}"
            ) from error
        if not math.isfinite(ground_truth) or ground_truth <= 0:
            raise ValidationError(
                f"ground_truth_m must be finite and positive for {pair_id}"
            )
        if not str(row["prompt"]).strip():
            raise ValidationError(f"Empty prompt for {pair_id}")

    if len(pair_ids) != len(set(pair_ids)):
        raise ValidationError("The manifest contains duplicate pair_id values")
    if len(rows) != int(metadata["number_of_queries"]):
        raise ValidationError(
            f"Query count mismatch: manifest={len(rows)}, "
            f"metadata={metadata['number_of_queries']}"
        )
    expected_scenes = {str(value) for value in metadata["scene_ids"]}
    if scene_ids != expected_scenes:
        raise ValidationError(
            f"Scene IDs mismatch: manifest={sorted(scene_ids)}, "
            f"metadata={sorted(expected_scenes)}"
        )
    if len(scene_ids) != int(metadata["number_of_scenes"]):
        raise ValidationError(
            f"Scene count mismatch: manifest={len(scene_ids)}, "
            f"metadata={metadata['number_of_scenes']}"
        )
    return rows


def deterministic_order(
    manifest: list[dict[str, Any]],
    seed: int,
) -> list[dict[str, Any]]:
    ordered = list(manifest)
    random.Random(seed).shuffle(ordered)
    return ordered


def validate_reference_results(
    ordered: list[dict[str, Any]],
    reference_path: Path,
) -> dict[str, str]:
    reference = read_jsonl(reference_path)
    reference_ids = [str(row.get("pair_id", "")) for row in reference]
    ordered_ids = [str(row["pair_id"]) for row in ordered]
    if reference_ids != ordered_ids:
        for position, (expected, actual) in enumerate(
            zip(ordered_ids, reference_ids), start=1
        ):
            if expected != actual:
                detail = (
                    f"first mismatch at run_index={position}: "
                    f"ordered={expected}, reference={actual}"
                )
                break
        else:
            detail = (
                f"different lengths: ordered={len(ordered_ids)}, "
                f"reference={len(reference_ids)}"
            )
        raise ValidationError(f"Deterministic order mismatch ({detail})")

    hashes: dict[str, str] = {}
    for row in reference:
        pair_id = str(row["pair_id"])
        expected_hash = str(row.get("stimulus_sha256", ""))
        if not expected_hash:
            raise ValidationError(
                f"Reference result lacks stimulus_sha256 for {pair_id}"
            )
        hashes[pair_id] = expected_hash
    return hashes


def validate_stimuli(
    project_root: Path,
    manifest: list[dict[str, Any]],
    expected_hashes: dict[str, str],
    allow_missing: bool,
) -> tuple[dict[str, Path], int]:
    paths: dict[str, Path] = {}
    missing: list[Path] = []
    for row in manifest:
        pair_id = str(row["pair_id"])
        path = resolve_under(project_root, Path(str(row["stimulus"])))
        paths[pair_id] = path
        if not path.is_file():
            missing.append(path)
            continue
        actual_hash = sha256_file(path)
        expected_hash = expected_hashes[pair_id]
        if actual_hash != expected_hash:
            raise ValidationError(
                f"Stimulus SHA-256 mismatch for {pair_id}: "
                f"actual={actual_hash}, expected={expected_hash}"
            )
    if missing and not allow_missing:
        preview = "\n  ".join(str(path) for path in missing[:5])
        suffix = "" if len(missing) <= 5 else f"\n  ... and {len(missing) - 5} more"
        raise ValidationError(
            f"Missing {len(missing)} stimulus file(s):\n  {preview}{suffix}"
        )
    return paths, len(missing)


def validate_model_path(
    model_path: Path,
    revision: str,
    allow_missing: bool,
) -> None:
    if not model_path.is_dir():
        if allow_missing:
            return
        raise ValidationError(f"Local model snapshot not found: {model_path}")
    if model_path.name != revision:
        raise ValidationError(
            f"Model path revision mismatch: directory={model_path.name}, "
            f"metadata={revision}"
        )
    required = [model_path / "config.json"]
    missing = [path for path in required if not path.is_file()]
    if not list(model_path.glob("*.safetensors")):
        missing.append(model_path / "*.safetensors")
    if missing:
        rendered = ", ".join(str(path) for path in missing)
        raise ValidationError(f"Incomplete local model snapshot; missing: {rendered}")


def git_context(project_root: Path) -> dict[str, Any]:
    def command(*arguments: str) -> str | None:
        try:
            completed = subprocess.run(
                ["git", *arguments],
                cwd=project_root,
                check=True,
                capture_output=True,
                text=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            return None
        return completed.stdout.strip()

    commit = command("rev-parse", "HEAD")
    branch = command("branch", "--show-current")
    status = command("status", "--porcelain")
    return {
        "git_commit": commit,
        "git_branch": branch,
        "git_worktree_dirty": bool(status) if status is not None else None,
    }


def parse_distance(raw_response: str) -> float | None:
    text = raw_response.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    candidates = [text]
    match = re.search(r"\{.*?\}", text, flags=re.DOTALL)
    if match and match.group(0) != text:
        candidates.append(match.group(0))

    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(value, dict):
            continue
        distance = value.get("distance_m")
        if isinstance(distance, bool) or not isinstance(distance, (int, float)):
            continue
        number = float(distance)
        if math.isfinite(number) and number > 0:
            return number
    return None


def validate_resume(
    results_path: Path,
    selected: list[dict[str, Any]],
    metadata: dict[str, Any],
    output_metadata_path: Path,
) -> list[dict[str, Any]]:
    if not results_path.exists():
        return []
    previous = read_jsonl(results_path)
    if len(previous) > len(selected):
        raise ValidationError(
            f"Existing output has {len(previous)} rows, but this run selects "
            f"only {len(selected)}"
        )
    expected_ids = [str(row["pair_id"]) for row in selected[: len(previous)]]
    previous_ids = [str(row.get("pair_id", "")) for row in previous]
    if previous_ids != expected_ids:
        raise ValidationError(
            "Cannot resume: existing results are not an exact prefix of "
            "the deterministic query order"
        )
    if output_metadata_path.exists():
        previous_metadata = read_json(output_metadata_path)
        comparable = (
            "condition",
            "model_id",
            "model_revision",
            "temperature",
            "max_tokens",
            "generation_seed",
            "order_seed",
            "manifest_sha256",
        )
        mismatches = [
            key
            for key in comparable
            if previous_metadata.get(key) != metadata.get(key)
        ]
        if mismatches:
            raise ValidationError(
                "Cannot resume: metadata differs for " + ", ".join(mismatches)
            )
    return previous


def build_run_metadata(
    source: dict[str, Any],
    project_root: Path,
    checkpoint_dir: Path,
    model_path: Path,
    selected: list[dict[str, Any]],
    versions: dict[str, str],
) -> dict[str, Any]:
    scene_ids = sorted({str(row["scene_id"]) for row in selected})
    metadata = {
        "runner_version": RUNNER_VERSION,
        "run_started_utc": utc_now(),
        "run_status": "running",
        "condition": source["condition"],
        "model_id": source["model_id"],
        "model_revision": source["model_revision"],
        "model_path": str(model_path),
        "checkpoint_dir": str(checkpoint_dir),
        "mlx_version": versions["mlx"],
        "mlx_vlm_version": versions["mlx-vlm"],
        "transformers_version": versions["transformers"],
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "temperature": source["temperature"],
        "max_tokens": source["max_tokens"],
        "generation_seed": source["generation_seed"],
        "order_seed": source["order_seed"],
        "manifest_sha256": source["manifest_sha256"],
        "source_manifest_total_queries": source["number_of_queries"],
        "number_of_scenes": len(scene_ids),
        "number_of_queries": len(selected),
        "scene_ids": scene_ids,
        "valid_responses": 0,
    }
    metadata.update(git_context(project_root))
    return metadata


def run_inference(
    project_root: Path,
    checkpoint_dir: Path,
    output_dir: Path,
    model_path: Path,
    selected: list[dict[str, Any]],
    stimulus_paths: dict[str, Path],
    source_metadata: dict[str, Any],
    versions: dict[str, str],
    resume: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / RESULTS_FILENAME
    metadata_path = output_dir / METADATA_FILENAME

    run_metadata = build_run_metadata(
        source_metadata,
        project_root,
        checkpoint_dir,
        model_path,
        selected,
        versions,
    )
    if resume:
        previous = validate_resume(
            results_path, selected, run_metadata, metadata_path
        )
    else:
        existing = [path for path in (results_path, metadata_path) if path.exists()]
        if existing:
            rendered = ", ".join(str(path) for path in existing)
            raise ValidationError(
                f"Output already exists: {rendered}. Use --resume or a new directory."
            )
        previous = []

    run_metadata["valid_responses"] = sum(
        row.get("predicted_distance_m") is not None for row in previous
    )
    write_json_atomic(metadata_path, run_metadata)

    if len(previous) == len(selected):
        run_metadata["run_status"] = "completed"
        run_metadata["run_finished_utc"] = utc_now()
        write_json_atomic(metadata_path, run_metadata)
        print(f"Run already complete: {results_path}")
        return

    try:
        import mlx.core as mx
        from mlx_vlm import generate, load
        from mlx_vlm.prompt_utils import apply_chat_template
    except ImportError as error:
        raise ValidationError(
            "Could not import MLX-VLM. Activate the frozen project environment."
        ) from error

    mx.random.seed(int(source_metadata["generation_seed"]))
    print(f"Loading local model: {model_path}")
    model, processor = load(str(model_path))
    print(f"Model loaded. Queries: {len(selected)}")
    print()

    mode = "a" if previous else "w"
    with results_path.open(mode, encoding="utf-8", newline="\n") as handle:
        for zero_index in range(len(previous), len(selected)):
            row = selected[zero_index]
            pair_id = str(row["pair_id"])
            stimulus_path = stimulus_paths[pair_id]
            prompt = apply_chat_template(
                processor,
                model.config,
                str(row["prompt"]),
                num_images=1,
                num_audios=0,
                enable_thinking=False,
            )
            started = time.perf_counter()
            generation = generate(
                model,
                processor,
                prompt,
                image=[str(stimulus_path)],
                temperature=float(source_metadata["temperature"]),
                max_tokens=int(source_metadata["max_tokens"]),
                verbose=False,
            )
            elapsed = time.perf_counter() - started
            raw_response = (
                generation.text
                if hasattr(generation, "text")
                else str(generation)
            )
            predicted = parse_distance(raw_response)
            ground_truth = float(row["ground_truth_m"])

            result_row = dict(row)
            result_row.update(
                {
                    "run_index": zero_index + 1,
                    "condition": source_metadata["condition"],
                    "model_id": source_metadata["model_id"],
                    "model_revision": source_metadata["model_revision"],
                    "temperature": source_metadata["temperature"],
                    "max_tokens": source_metadata["max_tokens"],
                    "generation_seed": source_metadata["generation_seed"],
                    "stimulus_sha256": sha256_file(stimulus_path),
                    "raw_response": raw_response,
                    "predicted_distance_m": predicted,
                    "elapsed_seconds": elapsed,
                    "absolute_error_m": (
                        abs(predicted - ground_truth)
                        if predicted is not None
                        else None
                    ),
                    "relative_error": (
                        abs(predicted - ground_truth) / ground_truth
                        if predicted is not None
                        else None
                    ),
                    "prediction_to_gt_ratio": (
                        predicted / ground_truth
                        if predicted is not None
                        else None
                    ),
                    "log_error": (
                        math.log(predicted / ground_truth)
                        if predicted is not None
                        else None
                    ),
                }
            )
            handle.write(json.dumps(result_row, ensure_ascii=False) + "\n")
            handle.flush()

            if predicted is not None:
                run_metadata["valid_responses"] += 1
                prediction_text = f"{predicted:.3f} m"
            else:
                prediction_text = "invalid JSON"
            print(
                f"[{zero_index + 1:02d}/{len(selected):02d}] {pair_id} | "
                f"GT={ground_truth:.3f} m | pred={prediction_text} | "
                f"{elapsed:.2f} s"
            )

    run_metadata["run_status"] = "completed"
    run_metadata["run_finished_utc"] = utc_now()
    write_json_atomic(metadata_path, run_metadata)
    print()
    print("Run completed.")
    print(f"Valid responses: {run_metadata['valid_responses']}/{len(selected)}")
    print(f"Results: {results_path}")
    print(f"Metadata: {metadata_path}")


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    checkpoint_dir = resolve_under(project_root, args.checkpoint_dir)
    manifest_path = checkpoint_dir / "manifest.jsonl"
    source_metadata_path = checkpoint_dir / "metadata.json"
    reference_results_path = checkpoint_dir / RESULTS_FILENAME

    source_metadata = read_json(source_metadata_path)
    manifest = validate_manifest(manifest_path, source_metadata)
    ordered = deterministic_order(manifest, int(source_metadata["order_seed"]))
    reference_hashes = validate_reference_results(
        ordered, reference_results_path
    )
    stimulus_paths, missing_stimuli = validate_stimuli(
        project_root,
        manifest,
        reference_hashes,
        args.allow_missing_stimuli,
    )

    configured_model_path = (
        args.model_path
        if args.model_path is not None
        else Path(str(source_metadata["model_path"]))
    )
    model_path = resolve_under(project_root, configured_model_path)
    validate_model_path(
        model_path,
        str(source_metadata["model_revision"]),
        args.allow_missing_model,
    )
    versions = check_environment(
        source_metadata,
        args.skip_version_check,
        args.skip_platform_check,
    )

    selected = ordered[: args.limit] if args.limit is not None else ordered
    first_ids = ", ".join(str(row["pair_id"]) for row in ordered[:5])
    scene_ids = sorted({str(row["scene_id"]) for row in manifest})

    print("Validation completed.")
    print(f"Project root: {project_root}")
    print(f"Checkpoint: {checkpoint_dir}")
    print(f"Manifest SHA-256: {source_metadata['manifest_sha256']}")
    print(f"Scenes: {len(scene_ids)} ({', '.join(scene_ids)})")
    print(f"Queries: {len(manifest)}")
    print(f"Deterministic order seed: {source_metadata['order_seed']}")
    print(f"First five queries: {first_ids}")
    print(
        f"Stimuli: {len(manifest) - missing_stimuli} verified, "
        f"{missing_stimuli} missing"
    )
    print(f"Model snapshot: {model_path}")
    print(
        "Environment: "
        f"mlx={versions['mlx']}, mlx-vlm={versions['mlx-vlm']}, "
        f"transformers={versions['transformers']}, "
        f"Python={platform.python_version()}, arch={platform.machine()}"
    )

    if args.dry_run:
        print("Dry-run OK. The model was not loaded and no inference was run.")
        return 0

    output_dir = resolve_under(project_root, args.output_dir)
    if output_dir == checkpoint_dir or checkpoint_dir in output_dir.parents:
        raise ValidationError(
            "Inference output cannot be written inside the frozen checkpoint"
        )
    run_inference(
        project_root,
        checkpoint_dir,
        output_dir,
        model_path,
        selected,
        stimulus_paths,
        source_metadata,
        versions,
        args.resume,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValidationError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
