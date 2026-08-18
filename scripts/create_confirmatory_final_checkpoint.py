#!/usr/bin/env python3
"""Create the final v0.5 checkpoint for the confirmatory CA-1M sample."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED_LOG_SHA256 = (
    "ed8957558dd677a885e2ab753527974050e64f583098562969b8ca56bf1bb59c"
)
EXPECTED_INSPECTED = 102
EXPECTED_ACCEPTED = 49
EXPECTED_NUMERICAL_REJECTED = 51
EXPECTED_VISUAL_REJECTED = 2
EXPECTED_VISUAL_REJECTION_IDS = {"47115525", "47204605"}


def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Build and validate the final confirmatory checkpoint."
    )
    parser.add_argument("--project-root", type=Path, default=default_root)
    parser.add_argument("--checkpoint", default="v0.5")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--refresh-readme",
        action="store_true",
        help=(
            "Replace the README of an existing checkpoint with the current "
            "generated version and rebuild SHA256SUMS."
        ),
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return data


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"Expected JSON object at {path}:{line_number}")
            rows.append(row)
    return rows


def jsonl_text(path: Path, expected_rows: int) -> str:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} JSONL rows in {path}, found {len(lines)}"
        )
    for line_number, line in enumerate(lines, 1):
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError(f"Expected JSON object at {path}:{line_number}")
    return "\n".join(lines) + "\n"


def git_value(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=root, text=True
    ).strip()


def final_accepted(row: dict[str, Any]) -> bool:
    return bool(row.get("selection_accepted")) and row.get("visual_qc") == "pass"


def numerical_rejected(row: dict[str, Any]) -> bool:
    return not bool(row.get("selection_accepted"))


def visual_rejected(row: dict[str, Any]) -> bool:
    return bool(row.get("selection_accepted")) and row.get("visual_qc") == "fail"


def artifact_path(root: Path, row: dict[str, Any], key: str) -> Path:
    artifacts = row.get("artifacts")
    if not isinstance(artifacts, dict) or key not in artifacts:
        raise KeyError(f"Missing artifact {key!r} for capture {row['capture_id']}")
    entry = artifacts[key]
    if isinstance(entry, str):
        relative = entry
        expected_size = None
        expected_sha = None
    elif isinstance(entry, dict):
        relative = entry.get("path")
        expected_size = entry.get("size_bytes")
        expected_sha = entry.get("sha256")
    else:
        raise TypeError(f"Invalid artifact entry {key!r} for {row['capture_id']}")
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"Invalid artifact path {key!r} for {row['capture_id']}")
    path = Path(relative)
    if not path.is_absolute():
        path = root / path
    if not path.is_file():
        raise FileNotFoundError(path)
    if expected_size is not None and path.stat().st_size != int(expected_size):
        raise ValueError(f"Size mismatch for {path}")
    if expected_sha is not None and sha256(path) != str(expected_sha):
        raise ValueError(f"SHA-256 mismatch for {path}")
    return path


def copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def sanitize_validation_report(source: Path, root: Path) -> str:
    text = source.read_text(encoding="utf-8")
    return text.replace(str(root) + os.sep, "")


def accepted_summary(row: dict[str, Any]) -> dict[str, Any]:
    decision = row["selection_decision"]
    summary = decision["selection_summary"]
    return {
        "order_index": int(row["order_index"]),
        "capture_id": str(row["capture_id"]),
        "candidate_rank": summary["candidate_rank"],
        "frame_index": summary["frame_index"],
        "strict_pair_count": decision["best_strict_pair_count"],
        "selected_pairs": summary["selected_pairs"],
        "selected_minimum_m": summary["selected_minimum_m"],
        "selected_maximum_m": summary["selected_maximum_m"],
        "distance_ratio": (
            summary["selected_maximum_m"] / summary["selected_minimum_m"]
        ),
        "unique_objects_used": summary["unique_objects_used"],
        "maximum_object_uses": summary["maximum_object_uses"],
        "visual_qc": row["visual_qc"],
        "visual_qc_note": row["visual_qc_note"],
    }


def rejected_summary(row: dict[str, Any]) -> dict[str, Any]:
    decision = row["selection_decision"]
    result: dict[str, Any] = {
        "order_index": int(row["order_index"]),
        "capture_id": str(row["capture_id"]),
        "capture_decision": (
            "rejected_visual" if visual_rejected(row) else "rejected_numerical"
        ),
        "candidate_frames": decision.get("candidate_frames"),
        "best_candidate_rank": decision.get("best_candidate_rank"),
        "best_frame_index": decision.get("best_frame_index"),
        "best_strict_pair_count": decision.get("best_strict_pair_count"),
        "rejection_reasons": decision.get("rejection_reasons", []),
    }
    if visual_rejected(row):
        result["visual_qc"] = row["visual_qc"]
        result["visual_qc_note"] = row["visual_qc_note"]
    return result


def build_readme(summary: dict[str, Any]) -> str:
    visual_ids = ", ".join(f"`{item}`" for item in summary["visual_rejected_capture_ids"])
    return f"""# Checkpoint {summary['checkpoint']} - Final Confirmatory Dataset

This checkpoint freezes the final outcome of the confirmatory selection after
all {summary['inspected_captures']} captures in the predefined order were
inspected.

The pool was exhausted with {summary['accepted_captures']} accepted scenes,
one fewer than the target of {summary['target_accepted_scenes']}. Under the
frozen stopping rule, the thresholds were not relaxed and no capture outside
the pool was used as a replacement.

## Final Outcome

- captures inspected: {summary['inspected_captures']};
- captures accepted: {summary['accepted_captures']};
- numerical rejections: {summary['numerically_rejected_captures']};
- visual rejections: {summary['visually_rejected_captures']};
- selected relations: {summary['selected_relations']};
- final acceptance rate: {summary['observed_final_acceptance_rate']:.3%};
- visually rejected captures: {visual_ids};
- official log SHA-256: `{summary['source_log_sha256']}`;
- source commit: `{summary['source_commit']}`;
- source branch: `{summary['source_branch']}`.

## Contents

| File or directory | Purpose |
|---|---|
| `summary.json` | structured summary of the final outcome |
| `confirmatory_capture_log.jsonl` | 102 sequential decisions and artifact hashes |
| `accepted_pairs_manifest.jsonl` | 588 relations from the 49 accepted scenes |
| `rendered_manifest.jsonl` | rendered manifests for the accepted scenes |
| `confirmatory_capture_protocol.json` | frozen confirmatory protocol |
| `confirmatory_visual_qc_amendment_v1.json` | predeclared rule for visual-QC failures |
| `confirmatory_capture_order.csv` | complete deterministic capture order |
| `confirmatory_capture_order_metadata.json` | capture-order metadata and hash |
| `development_capture_ids.txt` | captures excluded because they were used in development |
| `selection_protocols/` | numerical decisions for all 102 captures |
| `validation_reports/` | validation reports for the 49 accepted scenes |
| `render_protocols/` | rendering parameters for the accepted scenes |
| `contact_sheets/` | visual evidence for the 49 accepted scenes |
| `visual_rejections/` | pairs, renderings, and contact sheets for the two visual-QC failures |
| `SHA256SUMS` | integrity checksums for every checkpoint file |

## Scope and Portability

The CA-1M TAR archives, full frame audits, candidate lists, and large
pair-feasibility files remain outside Git. Their paths, sizes, and hashes are
preserved in the official log.

Only the absolute local project-root prefix was removed from the validation
reports copied into this checkpoint. The scientific values were not changed,
and the hashes of the original reports remain in the log.

This checkpoint contains no model results and does not represent a 50-scene
sample. The final confirmatory dataset contains 49 scenes and 588 relations.
"""


def verify_checksum_file(checkpoint: Path) -> int:
    checksum_path = checkpoint / "SHA256SUMS"
    if not checksum_path.is_file():
        raise FileNotFoundError(checksum_path)
    lines = checksum_path.read_text(encoding="utf-8").splitlines()
    checked = 0
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            expected, relative = line.split("  ", 1)
        except ValueError as error:
            raise ValueError(
                f"Invalid SHA256SUMS entry at line {line_number}"
            ) from error
        path = checkpoint / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        observed = sha256(path)
        if observed != expected:
            raise ValueError(
                f"Checksum mismatch before README refresh: {relative}"
            )
        checked += 1
    return checked


def refresh_checkpoint_readme(checkpoint: Path) -> None:
    if not checkpoint.is_dir():
        raise FileNotFoundError(checkpoint)
    summary_path = checkpoint / "summary.json"
    readme_path = checkpoint / "README.md"
    checksum_path = checkpoint / "SHA256SUMS"
    if not readme_path.is_file():
        raise FileNotFoundError(readme_path)

    checked_before = verify_checksum_file(checkpoint)
    summary = read_json(summary_path)
    new_readme = build_readme(summary).encode("utf-8")

    checksum_files = sorted(
        path
        for path in checkpoint.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    )
    checksum_lines: list[str] = []
    for path in checksum_files:
        relative = path.relative_to(checkpoint)
        digest = (
            hashlib.sha256(new_readme).hexdigest()
            if path == readme_path
            else sha256(path)
        )
        checksum_lines.append(f"{digest}  {relative}\n")
    new_checksums = "".join(checksum_lines).encode("utf-8")

    old_readme = readme_path.read_bytes()
    old_checksums = checksum_path.read_bytes()
    readme_temporary = checkpoint / ".README.md.tmp"
    checksums_temporary = checkpoint / ".SHA256SUMS.tmp"
    try:
        readme_temporary.write_bytes(new_readme)
        checksums_temporary.write_bytes(new_checksums)
        os.replace(readme_temporary, readme_path)
        os.replace(checksums_temporary, checksum_path)
        checked_after = verify_checksum_file(checkpoint)
    except Exception:
        readme_path.write_bytes(old_readme)
        checksum_path.write_bytes(old_checksums)
        readme_temporary.unlink(missing_ok=True)
        checksums_temporary.unlink(missing_ok=True)
        raise

    print(f"Updated: {readme_path}")
    print(f"Checksums verified before update: {checked_before}")
    print(f"Checksums verified after update: {checked_after}")
    print(f"SHA256SUMS: {sha256(checksum_path)}")


def main() -> int:
    args = parse_args()
    root = args.project_root.expanduser().resolve()
    destination = root / "checkpoints" / args.checkpoint
    if args.refresh_readme:
        if args.dry_run:
            raise ValueError("--refresh-readme and --dry-run cannot be used together")
        refresh_checkpoint_readme(destination)
        return 0
    if destination.exists():
        raise FileExistsError(
            f"Checkpoint already exists; inspect it instead of overwriting: {destination}"
        )

    base = root / "outputs/phase1/confirmatory_dataset"
    log_path = base / "confirmatory_capture_log.jsonl"
    rows = read_jsonl(log_path)
    log_hash = sha256(log_path)
    if log_hash != EXPECTED_LOG_SHA256:
        raise ValueError(
            f"Unexpected official log SHA-256: {log_hash}; expected {EXPECTED_LOG_SHA256}"
        )

    orders = [int(row["order_index"]) for row in rows]
    capture_ids = [str(row["capture_id"]) for row in rows]
    accepted = [row for row in rows if final_accepted(row)]
    numerical = [row for row in rows if numerical_rejected(row)]
    visual = [row for row in rows if visual_rejected(row)]
    if len(rows) != EXPECTED_INSPECTED:
        raise ValueError(f"Expected 102 log rows, found {len(rows)}")
    if orders != list(range(1, EXPECTED_INSPECTED + 1)):
        raise ValueError("Log order is not the complete frozen sequence 1..102")
    if len(set(capture_ids)) != EXPECTED_INSPECTED:
        raise ValueError("Capture IDs are not unique")
    if len(accepted) != EXPECTED_ACCEPTED:
        raise ValueError(f"Expected 49 accepted captures, found {len(accepted)}")
    if len(numerical) != EXPECTED_NUMERICAL_REJECTED:
        raise ValueError(f"Expected 51 numerical rejections, found {len(numerical)}")
    if len(visual) != EXPECTED_VISUAL_REJECTED:
        raise ValueError(f"Expected 2 visual rejections, found {len(visual)}")
    if {str(row["capture_id"]) for row in visual} != EXPECTED_VISUAL_REJECTION_IDS:
        raise ValueError("Unexpected visual-rejection capture IDs")

    config_sources = {
        "confirmatory_capture_protocol.json": root / "configs/confirmatory_capture_protocol.json",
        "confirmatory_visual_qc_amendment_v1.json": root / "configs/confirmatory_visual_qc_amendment_v1.json",
        "confirmatory_capture_order.csv": root / "configs/confirmatory_capture_order.csv",
        "confirmatory_capture_order_metadata.json": root / "configs/confirmatory_capture_order_metadata.json",
        "development_capture_ids.txt": root / "configs/development_capture_ids.txt",
    }
    for source in config_sources.values():
        if not source.is_file():
            raise FileNotFoundError(source)

    protocol = read_json(config_sources["confirmatory_capture_protocol.json"])
    stopping = protocol["stopping_rule"]
    if int(stopping["target_accepted_scenes"]) != 50:
        raise ValueError("Unexpected confirmatory target")
    if int(stopping["maximum_captures_inspected"]) != EXPECTED_INSPECTED:
        raise ValueError("Unexpected maximum capture count")

    with config_sources["confirmatory_capture_order.csv"].open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        frozen_order = list(csv.DictReader(handle))
    if [str(row["capture_id"]) for row in frozen_order] != capture_ids:
        raise ValueError("Official log does not match the frozen capture order")

    source_branch = git_value(root, "branch", "--show-current")
    source_commit = git_value(root, "rev-parse", "HEAD")

    accepted_rows: list[str] = []
    rendered_rows: list[str] = []
    accepted_summaries: list[dict[str, Any]] = []
    rejected_summaries = [rejected_summary(row) for row in rows if not final_accepted(row)]
    contact_sheet_bytes = 0
    rejection_reason_counts: Counter[str] = Counter()
    for row in numerical:
        reasons = row["selection_decision"].get("rejection_reasons", [])
        if not reasons:
            rejection_reason_counts["unspecified_numerical_rejection"] += 1
        for reason in reasons:
            rejection_reason_counts[str(reason)] += 1
    rejection_reason_counts["visual_qc_failure"] = len(visual)

    plans: list[tuple[Path, str, str]] = []
    # Tuple entries are source path, destination relative path, transform.
    for row in rows:
        capture_id = str(row["capture_id"])
        protocol_source = artifact_path(root, row, "selection_protocol")
        plans.append((protocol_source, f"selection_protocols/{capture_id}.json", "copy"))

        if not bool(row.get("selection_accepted")):
            continue

        selection_source = artifact_path(root, row, "selection")
        rendered_source = artifact_path(root, row, "rendered_manifest")
        render_protocol_source = artifact_path(root, row, "render_protocol")
        contact_source = artifact_path(root, row, "contact_sheet")
        validation_source = artifact_path(root, row, "validation_report")
        selection_text = jsonl_text(selection_source, 12)
        rendered_text = jsonl_text(rendered_source, 12)
        contact_sheet_bytes += contact_source.stat().st_size

        if final_accepted(row):
            accepted_rows.extend(selection_text.splitlines())
            rendered_rows.extend(rendered_text.splitlines())
            accepted_summaries.append(accepted_summary(row))
            plans.extend(
                [
                    (render_protocol_source, f"render_protocols/{capture_id}.json", "copy"),
                    (contact_source, f"contact_sheets/{capture_id}.png", "copy"),
                    (validation_source, f"validation_reports/{capture_id}.txt", "sanitize"),
                ]
            )
        else:
            plans.extend(
                [
                    (selection_source, f"visual_rejections/selected_pairs/{capture_id}.jsonl", "copy"),
                    (rendered_source, f"visual_rejections/rendered_manifests/{capture_id}.jsonl", "copy"),
                    (render_protocol_source, f"visual_rejections/render_protocols/{capture_id}.json", "copy"),
                    (contact_source, f"visual_rejections/contact_sheets/{capture_id}.png", "copy"),
                    (validation_source, f"visual_rejections/validation_reports/{capture_id}.txt", "sanitize"),
                ]
            )

    if len(accepted_rows) != EXPECTED_ACCEPTED * 12:
        raise ValueError(f"Expected 588 accepted-pair rows, found {len(accepted_rows)}")
    if len(rendered_rows) != EXPECTED_ACCEPTED * 12:
        raise ValueError(f"Expected 588 rendered rows, found {len(rendered_rows)}")

    selected_minimums = [item["selected_minimum_m"] for item in accepted_summaries]
    selected_maximums = [item["selected_maximum_m"] for item in accepted_summaries]
    summary: dict[str, Any] = {
        "schema_version": "1.0",
        "checkpoint": args.checkpoint,
        "status": "confirmatory_pool_exhausted_before_target",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_branch": source_branch,
        "source_commit": source_commit,
        "source_log": str(log_path.relative_to(root)),
        "source_log_sha256": log_hash,
        "inspected_order_range": [1, EXPECTED_INSPECTED],
        "inspected_captures": len(rows),
        "target_accepted_scenes": 50,
        "accepted_captures": len(accepted),
        "target_shortfall": 50 - len(accepted),
        "numerically_accepted_captures": len(accepted) + len(visual),
        "numerically_rejected_captures": len(numerical),
        "visually_rejected_captures": len(visual),
        "observed_final_acceptance_rate": len(accepted) / len(rows),
        "visual_pass_rate_among_numerically_accepted": len(accepted) / (len(accepted) + len(visual)),
        "selected_relations": len(accepted_rows),
        "relations_per_accepted_capture": 12,
        "next_order_index": None,
        "next_capture_id": None,
        "accepted_capture_ids": [str(row["capture_id"]) for row in accepted],
        "numerically_rejected_capture_ids": [str(row["capture_id"]) for row in numerical],
        "visual_rejected_capture_ids": [str(row["capture_id"]) for row in visual],
        "rejection_reason_counts": dict(sorted(rejection_reason_counts.items())),
        "global_selected_minimum_m": min(selected_minimums),
        "global_selected_maximum_m": max(selected_maximums),
        "accepted_capture_summaries": accepted_summaries,
        "rejected_capture_summaries": rejected_summaries,
        "checkpoint_artifact_counts": {
            "selection_protocols": len(rows),
            "accepted_validation_reports": len(accepted),
            "accepted_render_protocols": len(accepted),
            "accepted_contact_sheets": len(accepted),
            "visual_rejection_evidence_sets": len(visual),
            "contact_sheet_bytes": contact_sheet_bytes,
        },
        "checkpoint_transformations": [
            {
                "files": "validation_reports/*.txt and visual_rejections/validation_reports/*.txt",
                "operation": "removed_absolute_project_root_prefix",
                "scientific_values_changed": False,
                "reason": "repository_portability",
            }
        ],
    }

    print("Final confirmatory checkpoint validation")
    print(f"Official log: {log_hash}")
    print(f"Inspected: {len(rows)}/102")
    print(f"Accepted: {len(accepted)}/50")
    print(f"Numerical rejections: {len(numerical)}")
    print(f"Visual rejections: {len(visual)}")
    print(f"Selected relations: {len(accepted_rows)}")
    print(f"Files planned from recorded artifacts: {len(plans)}")
    print(f"Contact-sheet bytes: {contact_sheet_bytes}")
    if args.dry_run:
        print("Dry-run OK. No checkpoint was written.")
        return 0

    checkpoints_root = destination.parent
    checkpoints_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{args.checkpoint}.", dir=checkpoints_root))
    try:
        for filename, source in config_sources.items():
            copy_file(source, temporary / filename)
        copy_file(log_path, temporary / "confirmatory_capture_log.jsonl")

        (temporary / "accepted_pairs_manifest.jsonl").write_text(
            "\n".join(accepted_rows) + "\n", encoding="utf-8"
        )
        (temporary / "rendered_manifest.jsonl").write_text(
            "\n".join(rendered_rows) + "\n", encoding="utf-8"
        )

        for source, relative, transform in plans:
            target = temporary / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if transform == "copy":
                copy_file(source, target)
            elif transform == "sanitize":
                target.write_text(
                    sanitize_validation_report(source, root), encoding="utf-8"
                )
            else:
                raise ValueError(f"Unknown transformation: {transform}")

        write_json(temporary / "summary.json", summary)
        (temporary / "README.md").write_text(build_readme(summary), encoding="utf-8")

        checksum_files = sorted(
            path for path in temporary.rglob("*")
            if path.is_file() and path.name != "SHA256SUMS"
        )
        checksum_text = "".join(
            f"{sha256(path)}  {path.relative_to(temporary)}\n"
            for path in checksum_files
        )
        (temporary / "SHA256SUMS").write_text(checksum_text, encoding="utf-8")

        for line in checksum_text.splitlines():
            expected, relative = line.split("  ", 1)
            if sha256(temporary / relative) != expected:
                raise ValueError(f"Internal checksum verification failed: {relative}")

        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    final_files = [path for path in destination.rglob("*") if path.is_file()]
    total_bytes = sum(path.stat().st_size for path in final_files)
    print(f"Created: {destination.relative_to(root)}")
    print(f"Files: {len(final_files)}")
    print(f"Total bytes: {total_bytes}")
    print(f"SHA256SUMS: {sha256(destination / 'SHA256SUMS')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
