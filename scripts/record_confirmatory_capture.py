#!/usr/bin/env python3
"""Append one inspected capture to the sequential confirmatory audit log."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Record one confirmatory capture in frozen processing order."
    )
    parser.add_argument("--capture-id", required=True)
    parser.add_argument(
        "--order",
        type=Path,
        default=Path("configs/confirmatory_capture_order.csv"),
    )
    parser.add_argument(
        "--confirmatory-protocol",
        type=Path,
        default=Path("configs/confirmatory_capture_protocol.json"),
    )
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--audit-metadata", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--candidate-protocol", type=Path, required=True)
    parser.add_argument("--feasibility", type=Path, required=True)
    parser.add_argument("--feasibility-protocol", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--selection-protocol", type=Path, required=True)
    parser.add_argument("--validation-report", type=Path)
    parser.add_argument("--rendered-manifest", type=Path)
    parser.add_argument("--render-protocol", type=Path)
    parser.add_argument("--contact-sheet", type=Path)
    parser.add_argument(
        "--visual-qc",
        choices=("pass", "fail", "not_applicable"),
        required=True,
    )
    parser.add_argument("--visual-qc-note", required=True)
    parser.add_argument(
        "--log",
        type=Path,
        default=Path(
            "outputs/phase1/confirmatory_dataset/confirmatory_capture_log.jsonl"
        ),
    )
    parser.add_argument("--project-root", type=Path, default=project_root)
    return parser.parse_args()


def resolve(project_root: Path, path: Path) -> Path:
    path = path.expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def display(project_root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(project_root))
    except ValueError:
        return str(path)


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
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"Expected object at {path}:{line_number}")
            rows.append(row)
    return rows


def count_jsonl(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def artifact(project_root: Path, path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": display(project_root, path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def capture_accepted(record: dict[str, Any]) -> bool:
    """Return the final decision, with fallback for schema-1.0 records."""
    inferred = bool(record.get("selection_accepted")) and (
        record.get("visual_qc") == "pass"
    )
    if "capture_accepted" in record:
        stored = bool(record["capture_accepted"])
        if stored != inferred:
            raise ValueError(
                "Log record has inconsistent selection, visual QC, and "
                "final acceptance"
            )
        return stored
    return inferred


def main() -> int:
    args = parse_args()
    root = args.project_root.expanduser().resolve()
    order_path = resolve(root, args.order)
    config_path = resolve(root, args.confirmatory_protocol)
    log_path = resolve(root, args.log)
    if not order_path.is_file() or not config_path.is_file():
        raise FileNotFoundError("Frozen order or confirmatory protocol is missing")

    with order_path.open("r", encoding="utf-8", newline="") as handle:
        order = list(csv.DictReader(handle))
    existing = read_jsonl(log_path)
    for index, row in enumerate(existing, start=1):
        if int(row["order_index"]) != index:
            raise ValueError("Existing log order indices are not contiguous")
        if row["capture_id"] != order[index - 1]["capture_id"]:
            raise ValueError("Existing log does not match the frozen capture order")
    next_index = len(existing) + 1
    if next_index > len(order):
        raise ValueError("All captures in the frozen order are already recorded")
    expected_id = order[next_index - 1]["capture_id"]
    if args.capture_id != expected_id:
        raise ValueError(
            f"Next capture must be {expected_id}; received {args.capture_id}"
        )

    paths = {
        "archive": resolve(root, args.archive),
        "audit": resolve(root, args.audit),
        "audit_metadata": resolve(root, args.audit_metadata),
        "candidates": resolve(root, args.candidates),
        "candidate_protocol": resolve(root, args.candidate_protocol),
        "feasibility": resolve(root, args.feasibility),
        "feasibility_protocol": resolve(root, args.feasibility_protocol),
        "selection": resolve(root, args.selection),
        "selection_protocol": resolve(root, args.selection_protocol),
    }
    optional = {
        "validation_report": args.validation_report,
        "rendered_manifest": args.rendered_manifest,
        "render_protocol": args.render_protocol,
        "contact_sheet": args.contact_sheet,
    }
    for name, value in optional.items():
        if value is not None:
            paths[name] = resolve(root, value)

    selection_protocol = read_json(paths["selection_protocol"])
    decisions = selection_protocol.get("capture_decisions", {})
    if args.capture_id not in decisions:
        raise ValueError("Selection protocol has no decision for capture")
    decision = decisions[args.capture_id]
    selection_accepted = bool(decision["accepted"])
    expected_pairs = int(
        read_json(config_path)["scene_acceptance"]["selected_pairs"]
    )
    observed_pairs = count_jsonl(paths["selection"])
    if selection_accepted and observed_pairs != expected_pairs:
        raise ValueError(
            f"Accepted capture has {observed_pairs} pairs; expected {expected_pairs}"
        )
    if not selection_accepted and observed_pairs != 0:
        raise ValueError("Rejected capture must have an empty selection manifest")
    if selection_accepted:
        required = (
            "validation_report",
            "rendered_manifest",
            "render_protocol",
            "contact_sheet",
        )
        missing = [name for name in required if name not in paths]
        if missing:
            raise ValueError("Accepted capture is missing: " + ", ".join(missing))
        validation = paths["validation_report"].read_text(encoding="utf-8")
        if "Result: PASSED" not in validation:
            raise ValueError("Accepted capture validation report did not pass")
        if args.visual_qc not in ("pass", "fail"):
            raise ValueError(
                "Numerically accepted capture requires visual-qc=pass or fail"
            )
    elif args.visual_qc != "not_applicable":
        raise ValueError(
            "Numerically rejected capture requires visual-qc=not_applicable"
        )

    final_accepted = selection_accepted and args.visual_qc == "pass"
    if final_accepted:
        capture_decision = "accepted"
    elif selection_accepted:
        capture_decision = "rejected_visual"
    else:
        capture_decision = "rejected_numerical"

    config = read_json(config_path)
    record = {
        "schema_version": "1.1",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "order_index": next_index,
        "capture_id": args.capture_id,
        "order_key_sha256": order[next_index - 1]["order_key_sha256"],
        "selection_accepted": selection_accepted,
        "selection_pair_count": observed_pairs,
        "selection_decision": decision,
        "visual_qc": args.visual_qc,
        "visual_qc_note": args.visual_qc_note,
        "capture_accepted": final_accepted,
        "capture_decision": capture_decision,
        "artifacts": {
            name: artifact(root, path) for name, path in sorted(paths.items())
        },
    }
    updated = existing + [record]
    write_jsonl_atomic(log_path, updated)

    accepted_count = sum(capture_accepted(row) for row in updated)
    rejected_count = len(updated) - accepted_count
    numerical_rejected_count = sum(
        not bool(row.get("selection_accepted")) for row in updated
    )
    visual_rejected_count = sum(
        bool(row.get("selection_accepted"))
        and row.get("visual_qc") == "fail"
        for row in updated
    )
    target = int(config["stopping_rule"]["target_accepted_scenes"])
    next_capture = (
        order[len(updated)]["capture_id"]
        if len(updated) < len(order) and accepted_count < target
        else None
    )
    print("Confirmatory capture log")
    print(f"Recorded: order={next_index} capture={args.capture_id}")
    print(f"Decision: {capture_decision.upper()}")
    print(f"Inspected: {len(updated)}")
    print(f"Accepted: {accepted_count}/{target}")
    print(
        f"Rejected: {rejected_count} "
        f"(numerical={numerical_rejected_count}, visual={visual_rejected_count})"
    )
    print(f"Next capture: {next_capture or 'none'}")
    print(f"Log: {log_path}")
    print(f"Log SHA-256: {sha256(log_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
