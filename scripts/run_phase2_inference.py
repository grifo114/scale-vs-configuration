#!/usr/bin/env python3
"""Run the frozen Phase 2 VLM canary or full inference sequence.

The runner consumes the committed Phase 2 protocol and query manifest. It
validates their integrity, verifies every referenced stimulus, loads exactly
one frozen model revision with the predeclared 4-bit configuration, and writes
one durable JSONL result after each successful generation.

Scientific outputs are never retried or repaired. A technical exception is
written to failures.jsonl and stops the process without advancing the exact
result prefix.
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
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCRIPT_VERSION = "0.1.0"
EXPECTED_PROTOCOL_STATUS = "predeclared_pending_canary"
EXPECTED_PRE_CANARY_AMENDMENT = "llava-total-parameter-metadata-v1"
EXPECTED_RELATIONS = 588
EXPECTED_CANARY_QUERIES = 12
EXPECTED_MANIFEST_SHA256 = (
    "61e4f9a8583dc6f375b6d4d9f72361acf1d9ff3a976485bf5b2f83736ea0360e"
)
EXPECTED_MODELS = {
    "qwen3_vl_4b_instruct": {
        "model_id": "Qwen/Qwen3-VL-4B-Instruct",
        "revision": "ebb281ec70b05090aa6165b016eac8ec08e71b17",
        "declared_parameters_billion": 4.0,
    },
    "internvl3_5_4b_hf": {
        "model_id": "OpenGVLab/InternVL3_5-4B-HF",
        "revision": "6bd4487402110ef9889ba50eb7aefeb302526fed",
        "declared_parameters_billion": 4.7,
    },
    "llava_next_mistral_7b": {
        "model_id": "llava-hf/llava-v1.6-mistral-7b-hf",
        "revision": "2424fdd47412fccc66d91719126b420e9fbd7065",
        "declared_parameters_billion": 8.0,
    },
}
MODEL_ADAPTERS = {
    "qwen3_vl_4b_instruct": {
        "model_class": "Qwen3VLForConditionalGeneration",
        "image_content_key": "image",
    },
    "internvl3_5_4b_hf": {
        "model_class": "AutoModelForImageTextToText",
        "image_content_key": "url",
    },
    "llava_next_mistral_7b": {
        "model_class": "AutoModelForImageTextToText",
        "image_content_key": "url",
    },
}
MODEL_LOADING_SPEC = {
    "attention_implementation": "sdpa",
    "device_map": {"": "cuda:0"},
    "dtype": "float16",
    "low_cpu_mem_usage": True,
    "trust_remote_code": False,
}
RECORDED_PACKAGES = (
    "accelerate",
    "bitsandbytes",
    "huggingface-hub",
    "numpy",
    "pillow",
    "safetensors",
    "torch",
    "transformers",
)


class RunnerError(RuntimeError):
    """Raised when a frozen runner precondition is not satisfied."""


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Run the frozen Phase 2 VLM inference protocol."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=project_root,
        help="Project root (default: parent of scripts/).",
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("configs/phase2_inference_protocol.json"),
        help="Frozen protocol, relative to the project root by default.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("configs/phase2_query_manifest.jsonl"),
        help="Frozen query manifest, relative to the project root by default.",
    )
    parser.add_argument(
        "--model-key",
        choices=tuple(MODEL_ADAPTERS),
        help="Frozen model key. Required unless --validate-only is used.",
    )
    parser.add_argument(
        "--mode",
        choices=("canary", "full"),
        default="canary",
        help="Run the 12-query technical canary or all 588 queries.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Override the protocol output root, relative to the project root.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Optional Hugging Face cache directory.",
    )
    parser.add_argument(
        "--max-new-queries",
        type=int,
        help=(
            "Process at most this many new queries in this invocation. "
            "Existing exact-prefix results are not counted."
        ),
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate committed inputs and all stimulus hashes without CUDA.",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
        raise RunnerError(f"Required file not found: {path}") from error
    except json.JSONDecodeError as error:
        raise RunnerError(f"Invalid JSON in {path}: {error}") from error
    if not isinstance(value, dict):
        raise RunnerError(f"Expected a JSON object in {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as error:
        raise RunnerError(f"Required file not found: {path}") from error
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise RunnerError(
                f"Invalid JSONL row at {path}:{line_number}: {error}"
            ) from error
        if not isinstance(value, dict):
            raise RunnerError(
                f"Expected a JSON object at {path}:{line_number}"
            )
        rows.append(value)
    return rows


def git(root: Path, *arguments: str, check: bool = True) -> str:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=root,
            check=check,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        stderr = getattr(error, "stderr", "") or ""
        raise RunnerError(
            f"Git command failed: git {' '.join(arguments)}\n{stderr.strip()}"
        ) from error
    return completed.stdout.strip()


def relative_to_root(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError as error:
        raise RunnerError(f"Path is outside project root: {path}") from error


def git_file_state(root: Path, path: Path) -> dict[str, Any]:
    relative = relative_to_root(root, path)
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", relative],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    ).returncode == 0
    changed = True
    if tracked:
        changed = subprocess.run(
            ["git", "diff", "--quiet", "HEAD", "--", relative],
            cwd=root,
            check=False,
        ).returncode != 0
    return {"path": relative, "tracked": tracked, "changed_from_head": changed}


def validate_git_inputs(
    root: Path,
    protocol_path: Path,
    manifest_path: Path,
    actual_inference: bool,
) -> dict[str, Any]:
    if git(root, "rev-parse", "--is-inside-work-tree") != "true":
        raise RunnerError(f"Not a Git worktree: {root}")
    files = {
        "protocol": git_file_state(root, protocol_path),
        "manifest": git_file_state(root, manifest_path),
        "runner": git_file_state(root, Path(__file__)),
    }
    for key in ("protocol", "manifest"):
        state = files[key]
        if not state["tracked"] or state["changed_from_head"]:
            raise RunnerError(
                f"{key.capitalize()} must be tracked and unchanged from HEAD: "
                f"{state['path']}"
            )
    if actual_inference:
        runner = files["runner"]
        if not runner["tracked"] or runner["changed_from_head"]:
            raise RunnerError(
                "The runner must be committed and unchanged before inference"
            )
    return {
        "branch": git(root, "branch", "--show-current"),
        "commit": git(root, "rev-parse", "HEAD"),
        "files": files,
    }


def validate_protocol(
    protocol_path: Path,
    manifest_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    protocol = read_json(protocol_path)
    if protocol.get("protocol_status") != EXPECTED_PROTOCOL_STATUS:
        raise RunnerError(
            "Unexpected protocol status: "
            f"{protocol.get('protocol_status')!r}"
        )
    amendments = protocol.get("pre_canary_amendments")
    if not isinstance(amendments, list) or len(amendments) != 1:
        raise RunnerError("Expected exactly one pre-canary protocol amendment")
    if amendments[0].get("id") != EXPECTED_PRE_CANARY_AMENDMENT:
        raise RunnerError(f"Unexpected pre-canary amendment: {amendments!r}")
    if amendments[0].get("scientific_design_changed") is not False:
        raise RunnerError("The descriptive metadata amendment is malformed")

    manifest_hash = sha256_file(manifest_path)
    if manifest_hash != EXPECTED_MANIFEST_SHA256:
        raise RunnerError(
            f"Unexpected query-manifest SHA-256: {manifest_hash}"
        )
    declared_manifest = protocol.get("query_manifest", {})
    if declared_manifest.get("sha256") != manifest_hash:
        raise RunnerError("Protocol and query-manifest hashes disagree")

    rows = read_jsonl(manifest_path)
    if len(rows) != EXPECTED_RELATIONS:
        raise RunnerError(
            f"Expected {EXPECTED_RELATIONS} queries, found {len(rows)}"
        )
    pair_ids = [str(row.get("pair_id", "")) for row in rows]
    if len(set(pair_ids)) != EXPECTED_RELATIONS:
        raise RunnerError("Query pair IDs are not unique")
    if [int(row.get("run_index", -1)) for row in rows] != list(
        range(1, EXPECTED_RELATIONS + 1)
    ):
        raise RunnerError("Global query order is not the frozen 1..588 sequence")
    if any(
        str(row.get("prompt", "")) != protocol["inference"]["prompt"]
        for row in rows
    ):
        raise RunnerError("A query prompt differs from the frozen protocol")

    canary = [row for row in rows if row.get("is_canary") is True]
    if len(canary) != EXPECTED_CANARY_QUERIES:
        raise RunnerError(
            f"Expected 12 canary rows, found {len(canary)}"
        )
    canary.sort(key=lambda row: int(row["canary_index"]))
    if [int(row["canary_index"]) for row in canary] != list(range(1, 13)):
        raise RunnerError("Canary order is not the frozen 1..12 sequence")
    if len({str(row["scene_id"]) for row in canary}) != 12:
        raise RunnerError("Canary rows do not contain twelve unique scenes")
    cell_counts = Counter(
        (str(row["fold"]), str(row["distance_stratum"]))
        for row in canary
    )
    if len(cell_counts) != 6 or set(cell_counts.values()) != {2}:
        raise RunnerError(f"Unexpected canary cell counts: {cell_counts}")
    declared_canary_ids = protocol.get("canary", {}).get(
        "pair_ids_in_canary_order"
    )
    if declared_canary_ids != [str(row["pair_id"]) for row in canary]:
        raise RunnerError("Protocol and query-manifest canary orders disagree")

    models = protocol.get("models")
    if not isinstance(models, list) or len(models) != len(MODEL_ADAPTERS):
        raise RunnerError("Unexpected frozen model list")
    by_key = {str(model.get("key", "")): model for model in models}
    if set(by_key) != set(MODEL_ADAPTERS):
        raise RunnerError(f"Unexpected frozen model keys: {sorted(by_key)}")
    for key, expected_model in EXPECTED_MODELS.items():
        model = by_key[key]
        revision = str(model.get("revision", ""))
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise RunnerError(f"Invalid revision for {key}: {revision!r}")
        for field, expected in expected_model.items():
            if model.get(field) != expected:
                raise RunnerError(
                    f"Unexpected {field} for {key}: "
                    f"{model.get(field)!r} != {expected!r}"
                )

    inference = protocol.get("inference", {})
    expected_inference = {
        "temperature": 0.0,
        "do_sample": False,
        "max_new_tokens": 64,
        "generation_seed": 0,
        "runs_per_model": 1,
        "native_chat_template_per_model": True,
        "native_image_processor_per_model": True,
        "manual_image_resizing": False,
    }
    for key, expected in expected_inference.items():
        if inference.get(key) != expected:
            raise RunnerError(
                f"Unexpected inference value {key}: {inference.get(key)!r}"
            )
    quantization = inference.get("quantization", {})
    expected_quantization = {
        "library": "bitsandbytes",
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
        "bnb_4bit_compute_dtype": "float16",
    }
    if quantization != expected_quantization:
        raise RunnerError(f"Unexpected quantization protocol: {quantization}")
    return protocol, rows


def verify_stimuli(root: Path, rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    checked = 0
    total_bytes = 0
    for row in rows:
        pair_id = str(row["pair_id"])
        path = resolve_under(root, Path(str(row["stimulus_path"])))
        if not path.is_file():
            raise RunnerError(f"Missing stimulus for {pair_id}: {path}")
        actual = sha256_file(path)
        expected = str(row["stimulus_sha256"])
        if actual != expected:
            raise RunnerError(
                f"Stimulus hash mismatch for {pair_id}: "
                f"actual={actual}, expected={expected}"
            )
        checked += 1
        total_bytes += path.stat().st_size
    return {
        "checked": checked,
        "bytes": total_bytes,
        "mib": round(total_bytes / (1024**2), 6),
    }


def selected_queries(
    rows: list[dict[str, Any]],
    mode: str,
) -> list[dict[str, Any]]:
    if mode == "full":
        return rows
    canary = [row for row in rows if row["is_canary"] is True]
    return sorted(canary, key=lambda row: int(row["canary_index"]))


def model_by_key(protocol: dict[str, Any], key: str) -> dict[str, Any]:
    matches = [model for model in protocol["models"] if model["key"] == key]
    if len(matches) != 1:
        raise RunnerError(f"Could not resolve frozen model key: {key}")
    return matches[0]


def parse_distance_response(raw_response: str) -> dict[str, Any]:
    text = raw_response.strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError as error:
        return {
            "response_status": "invalid_json",
            "parsed_distance_m": None,
            "parse_error": str(error),
        }
    if not isinstance(value, dict):
        return {
            "response_status": "invalid_schema",
            "parsed_distance_m": None,
            "parse_error": "top-level JSON value is not an object",
        }
    if set(value) != {"distance_m"}:
        return {
            "response_status": "invalid_schema",
            "parsed_distance_m": None,
            "parse_error": "expected exactly one key named distance_m",
        }
    distance = value["distance_m"]
    if isinstance(distance, bool) or not isinstance(distance, (int, float)):
        return {
            "response_status": "invalid_schema",
            "parsed_distance_m": None,
            "parse_error": "distance_m is not a JSON number",
        }
    numeric = float(distance)
    if not math.isfinite(numeric) or numeric <= 0:
        return {
            "response_status": "invalid_schema",
            "parsed_distance_m": None,
            "parse_error": "distance_m is not positive and finite",
        }
    return {
        "response_status": "valid",
        "parsed_distance_m": numeric,
        "parse_error": None,
    }


def append_jsonl_durable(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    content = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
    temporary.write_text(content + "\n", encoding="utf-8")
    os.replace(temporary, path)


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in RECORDED_PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def cuda_runtime_metadata(torch: Any) -> dict[str, Any]:
    available = bool(torch.cuda.is_available())
    devices: list[dict[str, Any]] = []
    if available:
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": properties.name,
                    "total_memory_bytes": properties.total_memory,
                    "compute_capability": [properties.major, properties.minor],
                }
            )
    return {
        "available": available,
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "device_count": len(devices),
        "devices": devices,
    }


def stable_runtime_fingerprint(
    runtime: dict[str, Any],
    model: dict[str, Any],
    adapter: dict[str, Any],
    protocol_sha256: str,
    manifest_sha256: str,
    runner_sha256: str,
) -> tuple[dict[str, Any], str]:
    cuda = runtime["cuda"]
    stable = {
        "python_version": runtime["python_version"],
        "packages": runtime["packages"],
        "torch_cuda_version": cuda["torch_cuda_version"],
        "cudnn_version": cuda["cudnn_version"],
        "cuda_devices": [
            {
                "name": item["name"],
                "total_memory_bytes": item["total_memory_bytes"],
                "compute_capability": item["compute_capability"],
            }
            for item in cuda["devices"]
        ],
        "model_id": model["model_id"],
        "model_revision": model["revision"],
        "adapter": adapter,
        "model_loading": MODEL_LOADING_SPEC,
        "protocol_sha256": protocol_sha256,
        "manifest_sha256": manifest_sha256,
        "runner_sha256": runner_sha256,
    }
    serialized = json.dumps(stable, ensure_ascii=False, sort_keys=True)
    return stable, sha256_text(serialized)


def validate_result_prefix(
    results_path: Path,
    queries: list[dict[str, Any]],
    expected: dict[str, Any],
) -> list[dict[str, Any]]:
    if not results_path.exists():
        return []
    results = read_jsonl(results_path)
    if len(results) > len(queries):
        raise RunnerError(
            f"Results contain {len(results)} rows for {len(queries)} queries"
        )
    for index, result in enumerate(results):
        query = queries[index]
        checks = {
            "sequence_index": index + 1,
            "pair_id": str(query["pair_id"]),
            "model_key": expected["model_key"],
            "model_id": expected["model_id"],
            "model_revision": expected["model_revision"],
            "mode": expected["mode"],
            "protocol_sha256": expected["protocol_sha256"],
            "query_manifest_sha256": expected["query_manifest_sha256"],
            "runner_sha256": expected["runner_sha256"],
            "runtime_fingerprint_sha256": expected[
                "runtime_fingerprint_sha256"
            ],
        }
        for field, value in checks.items():
            if result.get(field) != value:
                raise RunnerError(
                    f"Result prefix mismatch at row {index + 1}, "
                    f"field {field}: {result.get(field)!r} != {value!r}"
                )
        if not isinstance(result.get("raw_response"), str):
            raise RunnerError(
                f"Result row {index + 1} does not contain a raw response"
            )
    return results


def configure_determinism(torch: Any, seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def load_model_and_processor(
    model_spec: dict[str, Any],
    adapter: dict[str, Any],
    cache_dir: Path | None,
) -> tuple[Any, Any, Any]:
    import torch
    from transformers import (
        AutoModelForImageTextToText,
        AutoProcessor,
        BitsAndBytesConfig,
        Qwen3VLForConditionalGeneration,
    )

    model_classes = {
        "AutoModelForImageTextToText": AutoModelForImageTextToText,
        "Qwen3VLForConditionalGeneration": Qwen3VLForConditionalGeneration,
    }
    model_class = model_classes[adapter["model_class"]]
    cache_value = str(cache_dir) if cache_dir is not None else None
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    processor = AutoProcessor.from_pretrained(
        model_spec["model_id"],
        revision=model_spec["revision"],
        cache_dir=cache_value,
        trust_remote_code=False,
    )
    model = model_class.from_pretrained(
        model_spec["model_id"],
        revision=model_spec["revision"],
        cache_dir=cache_value,
        quantization_config=quantization,
        dtype=torch.float16,
        device_map={"": "cuda:0"},
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
        trust_remote_code=False,
    ).eval()
    return torch, processor, model


def prepare_inputs(
    processor: Any,
    query: dict[str, Any],
    stimulus_path: Path,
    adapter: dict[str, Any],
) -> Any:
    image_content = {
        "type": "image",
        adapter["image_content_key"]: str(stimulus_path),
    }
    messages = [
        {
            "role": "user",
            "content": [
                image_content,
                {"type": "text", "text": str(query["prompt"])},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    if adapter["model_class"] == "Qwen3VLForConditionalGeneration":
        inputs.pop("token_type_ids", None)
    return inputs


def primary_model_device(model: Any) -> Any:
    device = getattr(model, "device", None)
    if device is not None and str(device) != "meta":
        return device
    for parameter in model.parameters():
        if str(parameter.device) != "meta":
            return parameter.device
    raise RunnerError("Could not determine the model input device")


def generation_result(
    torch: Any,
    processor: Any,
    model: Any,
    inputs: Any,
    max_new_tokens: int,
) -> tuple[str, int, int]:
    device = primary_model_device(model)
    inputs = inputs.to(device)
    input_tokens = int(inputs["input_ids"].shape[-1])
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )
    generated_tokens = generated[0, input_tokens:]
    raw_response = processor.decode(
        generated_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return raw_response, input_tokens, int(generated_tokens.shape[-1])


def output_directory(
    root: Path,
    protocol: dict[str, Any],
    args: argparse.Namespace,
) -> Path:
    configured = args.output_root
    if configured is None:
        configured = Path(str(protocol["outputs"]["root"]))
    base = resolve_under(root, configured)
    if args.model_key is None:
        raise RunnerError("--model-key is required for inference")
    return base / args.model_key / args.mode


def main() -> int:
    args = parse_args()
    if args.max_new_queries is not None and args.max_new_queries <= 0:
        raise RunnerError("--max-new-queries must be positive")
    if not args.validate_only and args.model_key is None:
        raise RunnerError("--model-key is required unless --validate-only is used")

    root = args.project_root.resolve()
    protocol_path = resolve_under(root, args.protocol)
    manifest_path = resolve_under(root, args.manifest)
    cache_dir = (
        resolve_under(root, args.cache_dir) if args.cache_dir is not None else None
    )
    protocol, rows = validate_protocol(protocol_path, manifest_path)
    if not args.validate_only and args.mode == "full":
        raise RunnerError(
            "Full inference is blocked while the protocol status is "
            "predeclared_pending_canary"
        )
    git_metadata = validate_git_inputs(
        root,
        protocol_path,
        manifest_path,
        actual_inference=not args.validate_only,
    )
    stimulus_metadata = verify_stimuli(root, rows)
    protocol_sha256 = sha256_file(protocol_path)
    manifest_sha256 = sha256_file(manifest_path)
    runner_sha256 = sha256_file(Path(__file__))

    print("===== FROZEN INPUTS =====")
    print(f"Git commit: {git_metadata['commit']}")
    print(f"Protocol SHA-256: {protocol_sha256}")
    print(f"Manifest SHA-256: {manifest_sha256}")
    print(f"Runner SHA-256: {runner_sha256}")
    print(f"Relations: {len(rows)}")
    print(f"Stimuli verified: {stimulus_metadata['checked']}")
    print(f"Stimulus size: {stimulus_metadata['mib']:.2f} MiB")
    if args.validate_only:
        runner_state = git_metadata["files"]["runner"]
        print("Validation only: OK")
        print(
            "Runner committed and clean: "
            f"{runner_state['tracked'] and not runner_state['changed_from_head']}"
        )
        return 0

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise RunnerError(
            "Launch inference with PYTHONHASHSEED=0 in the process environment"
        )
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        import torch
    except ImportError as error:
        raise RunnerError("PyTorch is not installed") from error
    if not torch.cuda.is_available():
        raise RunnerError("CUDA is required by the frozen Phase 2 protocol")

    model_spec = model_by_key(protocol, str(args.model_key))
    adapter = dict(MODEL_ADAPTERS[str(args.model_key)])
    queries = selected_queries(rows, args.mode)
    output_dir = output_directory(root, protocol, args)
    results_path = output_dir / str(protocol["outputs"]["results_filename"])
    metadata_path = output_dir / str(protocol["outputs"]["metadata_filename"])
    failures_path = output_dir / "failures.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)

    runtime = {
        "captured_at_utc": utc_now(),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": package_versions(),
        "cuda": cuda_runtime_metadata(torch),
    }
    stable_runtime, runtime_fingerprint = stable_runtime_fingerprint(
        runtime,
        model_spec,
        adapter,
        protocol_sha256,
        manifest_sha256,
        runner_sha256,
    )
    expected_prefix = {
        "model_key": args.model_key,
        "model_id": model_spec["model_id"],
        "model_revision": model_spec["revision"],
        "mode": args.mode,
        "protocol_sha256": protocol_sha256,
        "query_manifest_sha256": manifest_sha256,
        "runner_sha256": runner_sha256,
        "runtime_fingerprint_sha256": runtime_fingerprint,
    }
    completed = validate_result_prefix(results_path, queries, expected_prefix)
    existing_metadata = read_json(metadata_path) if metadata_path.exists() else None
    if completed and existing_metadata is None:
        raise RunnerError("Results exist without runtime metadata")
    if (
        completed
        and existing_metadata.get("runtime_fingerprint_sha256")
        != runtime_fingerprint
    ):
        raise RunnerError("Current runtime differs from the stored result prefix")

    remaining = queries[len(completed):]
    if args.max_new_queries is not None:
        remaining = remaining[: args.max_new_queries]
    metadata: dict[str, Any] = {
        "schema_version": "1.0",
        "runner_version": SCRIPT_VERSION,
        "status": "running",
        "started_at_utc": (
            existing_metadata.get("started_at_utc")
            if existing_metadata
            else utc_now()
        ),
        "updated_at_utc": utc_now(),
        "completed_at_utc": None,
        "mode": args.mode,
        "model_key": args.model_key,
        "model_id": model_spec["model_id"],
        "model_revision": model_spec["revision"],
        "adapter": adapter,
        "model_loading": MODEL_LOADING_SPEC,
        "git": git_metadata,
        "protocol_path": relative_to_root(root, protocol_path),
        "protocol_sha256": protocol_sha256,
        "query_manifest_path": relative_to_root(root, manifest_path),
        "query_manifest_sha256": manifest_sha256,
        "runner_path": relative_to_root(root, Path(__file__)),
        "runner_sha256": runner_sha256,
        "runtime": runtime,
        "stable_runtime": stable_runtime,
        "runtime_fingerprint_sha256": runtime_fingerprint,
        "inference": protocol["inference"],
        "target_queries": len(queries),
        "completed_queries": len(completed),
        "valid_responses": sum(
            row.get("response_status") == "valid" for row in completed
        ),
        "invalid_responses": sum(
            row.get("response_status") != "valid" for row in completed
        ),
        "results_path": str(results_path),
        "failures_path": str(failures_path),
    }
    write_json_atomic(metadata_path, metadata)

    if not remaining:
        metadata["status"] = (
            "complete" if len(completed) == len(queries) else "partial"
        )
        metadata["updated_at_utc"] = utc_now()
        if metadata["status"] == "complete":
            metadata["completed_at_utc"] = utc_now()
        write_json_atomic(metadata_path, metadata)
        print(f"Existing exact prefix: {len(completed)}/{len(queries)}")
        print(f"Status: {metadata['status']}")
        return 0

    configure_determinism(torch, int(protocol["inference"]["generation_seed"]))
    print("\n===== MODEL LOAD =====")
    print(f"Model: {model_spec['model_id']}")
    print(f"Revision: {model_spec['revision']}")
    print(f"Adapter: {adapter['model_class']}")
    print(f"Existing exact prefix: {len(completed)}/{len(queries)}")
    try:
        torch.cuda.reset_peak_memory_stats()
        load_started = time.perf_counter()
        torch_module, processor, model = load_model_and_processor(
            model_spec,
            adapter,
            cache_dir,
        )
        load_seconds = time.perf_counter() - load_started
        metadata["model_load_seconds"] = load_seconds
        metadata["gpu_memory_after_load_bytes"] = torch.cuda.memory_allocated()
        metadata["gpu_peak_memory_after_load_bytes"] = (
            torch.cuda.max_memory_allocated()
        )
        metadata["updated_at_utc"] = utc_now()
        write_json_atomic(metadata_path, metadata)
    except Exception as error:
        failure = {
            "schema_version": "1.0",
            "recorded_at_utc": utc_now(),
            "stage": "model_load",
            "model_key": args.model_key,
            "model_id": model_spec["model_id"],
            "model_revision": model_spec["revision"],
            "mode": args.mode,
            "exception_type": type(error).__name__,
            "exception_message": str(error),
            "traceback": traceback.format_exc(),
            "runtime_fingerprint_sha256": runtime_fingerprint,
        }
        append_jsonl_durable(failures_path, failure)
        metadata["status"] = "failed_model_load"
        metadata["updated_at_utc"] = utc_now()
        metadata["last_failure"] = failure
        write_json_atomic(metadata_path, metadata)
        raise

    print(f"Model load time: {metadata['model_load_seconds']:.2f} s")
    max_new_tokens = int(protocol["inference"]["max_new_tokens"])
    for query in remaining:
        sequence_index = len(completed) + 1
        stimulus_path = resolve_under(root, Path(str(query["stimulus_path"])))
        print(
            f"[{sequence_index:03d}/{len(queries):03d}] "
            f"{query['pair_id']}"
        )
        try:
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            inputs = prepare_inputs(
                processor,
                query,
                stimulus_path,
                adapter,
            )
            raw_response, input_tokens, generated_tokens = generation_result(
                torch_module,
                processor,
                model,
                inputs,
                max_new_tokens,
            )
            elapsed = time.perf_counter() - started
            parsed = parse_distance_response(raw_response)
            result = {
                "schema_version": "1.0",
                "recorded_at_utc": utc_now(),
                "sequence_index": sequence_index,
                "run_index": int(query["run_index"]),
                "canary_index": query.get("canary_index"),
                "mode": args.mode,
                "model_key": args.model_key,
                "model_id": model_spec["model_id"],
                "model_revision": model_spec["revision"],
                "pair_id": str(query["pair_id"]),
                "scene_id": str(query["scene_id"]),
                "fold": str(query["fold"]),
                "distance_stratum": str(query["distance_stratum"]),
                "ground_truth_m": float(query["ground_truth_m"]),
                "stimulus_path": str(query["stimulus_path"]),
                "stimulus_sha256": str(query["stimulus_sha256"]),
                "prompt": str(query["prompt"]),
                "prompt_sha256": sha256_text(str(query["prompt"])),
                "raw_response": raw_response,
                **parsed,
                "input_tokens": input_tokens,
                "generated_tokens": generated_tokens,
                "elapsed_seconds": elapsed,
                "gpu_peak_memory_bytes": torch.cuda.max_memory_allocated(),
                "protocol_sha256": protocol_sha256,
                "query_manifest_sha256": manifest_sha256,
                "runner_sha256": runner_sha256,
                "runtime_fingerprint_sha256": runtime_fingerprint,
            }
            append_jsonl_durable(results_path, result)
            completed.append(result)
            metadata["completed_queries"] = len(completed)
            metadata["valid_responses"] = sum(
                row["response_status"] == "valid" for row in completed
            )
            metadata["invalid_responses"] = (
                len(completed) - metadata["valid_responses"]
            )
            metadata["updated_at_utc"] = utc_now()
            write_json_atomic(metadata_path, metadata)
        except Exception as error:
            failure = {
                "schema_version": "1.0",
                "recorded_at_utc": utc_now(),
                "stage": "query",
                "sequence_index": sequence_index,
                "pair_id": str(query["pair_id"]),
                "model_key": args.model_key,
                "model_id": model_spec["model_id"],
                "model_revision": model_spec["revision"],
                "mode": args.mode,
                "exception_type": type(error).__name__,
                "exception_message": str(error),
                "traceback": traceback.format_exc(),
                "runtime_fingerprint_sha256": runtime_fingerprint,
            }
            append_jsonl_durable(failures_path, failure)
            metadata["status"] = "failed_query"
            metadata["updated_at_utc"] = utc_now()
            metadata["last_failure"] = failure
            write_json_atomic(metadata_path, metadata)
            raise

    metadata["status"] = (
        "complete" if len(completed) == len(queries) else "partial"
    )
    metadata["updated_at_utc"] = utc_now()
    if metadata["status"] == "complete":
        metadata["completed_at_utc"] = utc_now()
    write_json_atomic(metadata_path, metadata)
    print("\n===== RUN RESULT =====")
    print(f"Status: {metadata['status']}")
    print(f"Completed: {len(completed)}/{len(queries)}")
    print(f"Valid JSON responses: {metadata['valid_responses']}")
    print(f"Invalid responses retained: {metadata['invalid_responses']}")
    print(f"Results: {results_path}")
    print(f"Metadata: {metadata_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RunnerError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
