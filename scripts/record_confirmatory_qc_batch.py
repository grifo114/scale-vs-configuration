#!/usr/bin/env python3
"""Record a reviewed confirmatory QC batch in frozen order, with resume support."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shlex
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Validate manual visual-QC decisions and record a preprocessed "
            "batch sequentially in the official confirmatory log. Rerunning "
            "after an interruption resumes after verified recorded rows."
        )
    )
    parser.add_argument("--batch-manifest", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=root)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve(root: Path, path: Path) -> Path:
    expanded = path.expanduser()
    resolved = expanded.resolve() if expanded.is_absolute() else (root / expanded).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Path must remain inside project root: {path}") from exc
    return resolved


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


def write_text_atomic(path: Path, text: str) -> None:
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
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_json_atomic(path: Path, data: dict[str, Any]) -> None:
    write_text_atomic(
        path,
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
    )


def capture_accepted(record: dict[str, Any]) -> bool:
    inferred = bool(record.get("selection_accepted")) and (
        record.get("visual_qc") == "pass"
    )
    if "capture_accepted" in record:
        stored = bool(record["capture_accepted"])
        if stored != inferred:
            raise ValueError(
                "Official log has inconsistent selection, visual QC, and "
                "final acceptance"
            )
        return stored
    return inferred


def validate_frozen_log(
    log: list[dict[str, Any]], order: list[dict[str, str]]
) -> None:
    if len(log) > len(order):
        raise ValueError("Official log is longer than the frozen order")
    for index, row in enumerate(log, start=1):
        if int(row["order_index"]) != index:
            raise ValueError("Official log order indices are not contiguous")
        if row["capture_id"] != order[index - 1]["capture_id"]:
            raise ValueError("Official log does not match the frozen order")
        capture_accepted(row)


def expected_visual_decision(
    record: dict[str, Any], decisions: dict[str, dict[str, str]]
) -> tuple[str, str]:
    if not record["numerical_accepted"]:
        reasons = record.get("selection_decision", {}).get(
            "rejection_reasons", []
        )
        suffix = ", ".join(str(reason) for reason in reasons) or "unspecified"
        return (
            "not_applicable",
            f"Rejected by the frozen numerical protocol: {suffix}.",
        )
    decision = decisions[record["capture_id"]]
    return decision["visual_qc"], decision["visual_qc_note"]


def validate_decisions(
    manifest: dict[str, Any], decisions_document: dict[str, Any]
) -> dict[str, dict[str, str]]:
    if decisions_document.get("schema_version") != "1.0":
        raise ValueError("Unsupported decisions schema_version")
    if decisions_document.get("batch_id") != manifest.get("batch_id"):
        raise ValueError("Decisions batch_id does not match batch manifest")
    raw = decisions_document.get("decisions")
    if not isinstance(raw, dict):
        raise TypeError("decisions must be a JSON object keyed by capture ID")
    expected_ids = {
        row["capture_id"]
        for row in manifest["records"]
        if row["numerical_accepted"]
    }
    if set(raw) != expected_ids:
        missing = sorted(expected_ids - set(raw))
        extra = sorted(set(raw) - expected_ids)
        raise ValueError(
            f"Visual decision IDs differ from pending captures; "
            f"missing={missing}, extra={extra}"
        )
    validated: dict[str, dict[str, str]] = {}
    for capture_id in sorted(expected_ids):
        value = raw[capture_id]
        if not isinstance(value, dict):
            raise TypeError(f"Decision for {capture_id} must be an object")
        visual_qc = value.get("visual_qc")
        note = value.get("visual_qc_note")
        if visual_qc not in ("pass", "fail"):
            raise ValueError(
                f"Decision for {capture_id} must use visual_qc=pass or fail"
            )
        if not isinstance(note, str) or not note.strip():
            raise ValueError(
                f"Decision for {capture_id} requires a non-empty visual_qc_note"
            )
        validated[capture_id] = {
            "visual_qc": visual_qc,
            "visual_qc_note": note.strip(),
        }
    return validated


def recorder_command(
    python: str,
    root: Path,
    record: dict[str, Any],
    visual_qc: str,
    note: str,
) -> list[str]:
    artifacts = record["artifacts"]
    required = (
        "archive",
        "audit",
        "audit_metadata",
        "candidates",
        "candidate_protocol",
        "feasibility",
        "feasibility_protocol",
        "selection",
        "selection_protocol",
    )
    missing = [name for name in required if name not in artifacts]
    if missing:
        raise ValueError(
            f"Batch record {record['capture_id']} is missing artifacts: {missing}"
        )
    command = [
        python,
        "scripts/record_confirmatory_capture.py",
        "--capture-id",
        record["capture_id"],
        "--archive",
        str(resolve(root, Path(artifacts["archive"]))),
        "--audit",
        str(resolve(root, Path(artifacts["audit"]))),
        "--audit-metadata",
        str(resolve(root, Path(artifacts["audit_metadata"]))),
        "--candidates",
        str(resolve(root, Path(artifacts["candidates"]))),
        "--candidate-protocol",
        str(resolve(root, Path(artifacts["candidate_protocol"]))),
        "--feasibility",
        str(resolve(root, Path(artifacts["feasibility"]))),
        "--feasibility-protocol",
        str(resolve(root, Path(artifacts["feasibility_protocol"]))),
        "--selection",
        str(resolve(root, Path(artifacts["selection"]))),
        "--selection-protocol",
        str(resolve(root, Path(artifacts["selection_protocol"]))),
    ]
    if record["numerical_accepted"]:
        optional = (
            ("validation_report", "--validation-report"),
            ("rendered_manifest", "--rendered-manifest"),
            ("render_protocol", "--render-protocol"),
            ("contact_sheet", "--contact-sheet"),
        )
        missing = [name for name, _ in optional if name not in artifacts]
        if missing:
            raise ValueError(
                f"Accepted batch record {record['capture_id']} is missing: {missing}"
            )
        for name, flag in optional:
            command.extend([flag, str(resolve(root, Path(artifacts[name])))])
    command.extend(
        [
            "--visual-qc",
            visual_qc,
            "--visual-qc-note",
            note,
            "--project-root",
            str(root),
        ]
    )
    return command


def verify_already_recorded(
    official_row: dict[str, Any],
    batch_row: dict[str, Any],
    visual_qc: str,
    note: str,
) -> None:
    if official_row["capture_id"] != batch_row["capture_id"]:
        raise ValueError("Official log diverged from this batch")
    if bool(official_row.get("selection_accepted")) != bool(
        batch_row["numerical_accepted"]
    ):
        raise ValueError("Recorded numerical decision differs from batch")
    if official_row.get("visual_qc") != visual_qc:
        raise ValueError("Recorded visual decision differs from decisions file")
    if official_row.get("visual_qc_note") != note:
        raise ValueError("Recorded visual-QC note differs from decisions file")


def receipt(
    manifest: dict[str, Any],
    log_path: Path,
    rows_before: int,
    records: list[dict[str, Any]],
    recorded_count: int,
    status: str,
) -> dict[str, Any]:
    official = read_jsonl(log_path)
    return {
        "schema_version": "1.0",
        "status": status,
        "batch_id": manifest["batch_id"],
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "official_log": str(log_path),
        "official_log_rows": len(official),
        "official_log_sha256": sha256(log_path),
        "batch_rows_before": rows_before,
        "batch_records_total": len(records),
        "batch_records_recorded": recorded_count,
        "accepted_total": sum(capture_accepted(row) for row in official),
    }


def main() -> int:
    args = parse_args()
    root = args.project_root.expanduser().resolve()
    manifest_path = resolve(root, args.batch_manifest)
    decisions_path = resolve(root, args.decisions)
    manifest = read_json(manifest_path)
    if manifest.get("schema_version") != "1.0":
        raise ValueError("Unsupported batch manifest schema_version")
    if manifest.get("status") != "awaiting_visual_qc":
        raise ValueError("Batch is not in awaiting_visual_qc state")
    records = manifest.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("Batch manifest has no records")
    start, end = manifest["order_range"]
    if len(records) != end - start + 1:
        raise ValueError("Batch manifest is incomplete")
    for offset, row in enumerate(records):
        if int(row["order_index"]) != int(start) + offset:
            raise ValueError("Batch order indices are not contiguous")

    decisions = validate_decisions(manifest, read_json(decisions_path))
    order_path = root / "configs/confirmatory_capture_order.csv"
    config_path = root / "configs/confirmatory_capture_protocol.json"
    log_path = resolve(root, Path(manifest["official_log"]))
    snapshot_path = resolve(root, Path(manifest["official_log_snapshot"]))
    with order_path.open("r", encoding="utf-8", newline="") as handle:
        order = list(csv.DictReader(handle))
    config = read_json(config_path)
    target = int(config["stopping_rule"]["target_accepted_scenes"])
    source_rows = read_jsonl(snapshot_path)
    rows_before = int(manifest["official_log_rows_before"])
    if len(source_rows) != rows_before:
        raise ValueError("Official-log snapshot row count does not match manifest")
    if sha256(snapshot_path) != manifest["official_log_sha256_before"]:
        raise ValueError("Official-log snapshot hash does not match manifest")
    official = read_jsonl(log_path)
    validate_frozen_log(official, order)
    if official[:rows_before] != source_rows:
        raise ValueError("Official log prefix changed after batch preprocessing")
    already_recorded = len(official) - rows_before
    if already_recorded < 0 or already_recorded > len(records):
        raise ValueError("Official log position is outside this batch")
    for offset in range(already_recorded):
        visual_qc, note = expected_visual_decision(records[offset], decisions)
        verify_already_recorded(
            official[rows_before + offset],
            records[offset],
            visual_qc,
            note,
        )

    accepted_now = sum(capture_accepted(row) for row in official)
    if accepted_now >= target and already_recorded < len(records):
        raise ValueError("Target was reached before all batch rows were recorded")

    print("Confirmatory QC batch registration")
    print(f"Batch ID: {manifest['batch_id']}")
    print(f"Frozen order range: {start}-{end}")
    print(f"Already recorded and verified: {already_recorded}/{len(records)}")
    print(f"Accepted before remaining rows: {accepted_now}/{target}")
    for row in records[already_recorded:]:
        visual_qc, _ = expected_visual_decision(row, decisions)
        print(
            f"  order={row['order_index']} capture={row['capture_id']} "
            f"numerical={'pass' if row['numerical_accepted'] else 'fail'} "
            f"visual={visual_qc}"
        )
    if args.dry_run:
        print("Dry-run OK. Official log was not modified.")
        return 0

    receipt_path = manifest_path.parent / "recording_receipt.json"
    python = sys.executable
    recorded_count = already_recorded
    try:
        for row in records[already_recorded:]:
            visual_qc, note = expected_visual_decision(row, decisions)
            command = recorder_command(
                python, root, row, visual_qc, note
            )
            print("$ " + shlex.join(command), flush=True)
            subprocess.run(command, cwd=root, check=True)
            recorded_count += 1
            updated = read_jsonl(log_path)
            if len(updated) != rows_before + recorded_count:
                raise RuntimeError("Recorder advanced the official log unexpectedly")
            verify_already_recorded(
                updated[-1], row, visual_qc, note
            )
            write_json_atomic(
                receipt_path,
                receipt(
                    manifest,
                    log_path,
                    rows_before,
                    records,
                    recorded_count,
                    "recording",
                ),
            )
    except Exception:
        if log_path.exists():
            write_json_atomic(
                receipt_path,
                receipt(
                    manifest,
                    log_path,
                    rows_before,
                    records,
                    recorded_count,
                    "interrupted",
                ),
            )
        raise

    final = receipt(
        manifest,
        log_path,
        rows_before,
        records,
        recorded_count,
        "completed",
    )
    final["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    final["decisions_file"] = str(decisions_path)
    final["decisions_sha256"] = sha256(decisions_path)
    write_json_atomic(receipt_path, final)
    print()
    print("Batch registration completed")
    print(f"Recorded: {recorded_count}/{len(records)}")
    print(f"Accepted total: {final['accepted_total']}/{target}")
    print(f"Official log: {log_path}")
    print(f"Log SHA-256: {final['official_log_sha256']}")
    print(f"Receipt: {receipt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
