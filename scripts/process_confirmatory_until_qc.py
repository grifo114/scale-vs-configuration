#!/usr/bin/env python3
"""Advance the frozen confirmatory order until manual visual QC is required."""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_BASE = Path("outputs/phase1/confirmatory_dataset")


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Process consecutive confirmatory captures in frozen order, "
            "record numerical rejections, and stop at the next capture "
            "that requires manual visual QC."
        )
    )
    parser.add_argument("--project-root", type=Path, default=project_root)
    parser.add_argument(
        "--max-captures",
        type=int,
        default=10,
        help="Maximum captures to process in one invocation (default: 10).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow the underlying runner to replace existing stage outputs.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


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


def read_order(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def validate_log(
    log: list[dict[str, Any]], order: list[dict[str, str]]
) -> None:
    if len(log) > len(order):
        raise ValueError("Confirmatory log is longer than the frozen order")
    for index, row in enumerate(log, start=1):
        if int(row["order_index"]) != index:
            raise ValueError("Confirmatory log order indices are not contiguous")
        if row["capture_id"] != order[index - 1]["capture_id"]:
            raise ValueError("Confirmatory log does not match the frozen order")


def pending_decision(
    root: Path, capture_id: str
) -> tuple[bool, Path] | None:
    protocol = (
        root
        / DEFAULT_BASE
        / "selection"
        / f"{capture_id}_protocol.json"
    )
    if not protocol.exists():
        return None
    decisions = read_json(protocol).get("capture_decisions", {})
    decision = decisions.get(capture_id)
    if not isinstance(decision, dict):
        raise ValueError(f"Selection protocol lacks decision: {protocol}")
    contact_sheet = (
        root
        / DEFAULT_BASE
        / "stimuli"
        / capture_id
        / "contact_sheet_all.png"
    )
    return bool(decision.get("accepted")), contact_sheet


def run_and_tee(command: list[str], cwd: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    print("$ " + shlex.join(command), flush=True)
    with output.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        if process.stdout is None:
            raise RuntimeError("Runner stdout pipe was not created")
        for line in process.stdout:
            print(line, end="")
            handle.write(line)
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def print_handoff(
    capture_id: str,
    contact_sheet: Path,
    inspected: int,
    accepted: int,
    target: int,
) -> None:
    if not contact_sheet.is_file():
        raise FileNotFoundError(
            "Numerically accepted capture has no contact sheet: "
            f"{contact_sheet}"
        )
    print()
    print("Manual visual QC required")
    print(f"Capture ID: {capture_id}")
    print(f"Officially inspected: {inspected}")
    print(f"Accepted progress: {accepted}/{target}")
    print(f"Contact sheet: {contact_sheet}")
    print("The official order was not advanced for this capture.")


def main() -> int:
    args = parse_args()
    if args.max_captures < 1:
        raise ValueError("--max-captures must be at least 1")

    root = args.project_root.expanduser().resolve()
    order_path = root / "configs/confirmatory_capture_order.csv"
    config_path = root / "configs/confirmatory_capture_protocol.json"
    log_path = root / DEFAULT_BASE / "confirmatory_capture_log.jsonl"
    runner = root / "scripts/process_next_confirmatory_capture.py"
    if not runner.is_file():
        raise FileNotFoundError(runner)

    order = read_order(order_path)
    config = read_json(config_path)
    target = int(config["stopping_rule"]["target_accepted_scenes"])
    log = read_jsonl(log_path)
    validate_log(log, order)
    accepted = sum(capture_accepted(row) for row in log)

    print("Confirmatory automatic advance")
    print(f"Officially inspected: {len(log)}/{len(order)}")
    print(f"Accepted: {accepted}/{target}")
    print(f"Per-run limit: {args.max_captures}")
    if accepted >= target:
        print("Target already reached. Nothing was processed.")
        return 0
    if len(log) >= len(order):
        print("Frozen capture pool is exhausted.")
        return 0

    next_capture = order[len(log)]["capture_id"]
    existing = pending_decision(root, next_capture)
    if existing is not None:
        is_accepted, contact_sheet = existing
        if is_accepted:
            print_handoff(
                next_capture, contact_sheet, len(log), accepted, target
            )
            return 0
        raise RuntimeError(
            "The next capture has a rejected selection protocol but is not "
            "present in the official log. Review this interrupted state "
            f"before continuing: {next_capture}"
        )

    if args.dry_run:
        print(f"Next capture: {next_capture}")
        print("Dry-run OK. No download or processing was performed.")
        return 0

    processed = 0
    while processed < args.max_captures:
        log_before = read_jsonl(log_path)
        validate_log(log_before, order)
        accepted_before = sum(capture_accepted(row) for row in log_before)
        if accepted_before >= target or len(log_before) >= len(order):
            break

        capture_id = order[len(log_before)]["capture_id"]
        console_log = (
            root / DEFAULT_BASE / "runner_logs" / f"{capture_id}_console.txt"
        )
        if console_log.exists() and not args.overwrite:
            raise FileExistsError(
                "Runner log already exists. Review the interrupted run or "
                f"rerun with --overwrite: {console_log}"
            )
        command = [
            sys.executable,
            str(runner.relative_to(root)),
            "--capture-id",
            capture_id,
        ]
        if args.overwrite:
            command.append("--overwrite")
        print()
        print(
            f"Processing {processed + 1}/{args.max_captures}: "
            f"{capture_id}"
        )
        run_and_tee(command, root, console_log)
        processed += 1

        log_after = read_jsonl(log_path)
        validate_log(log_after, order)
        if len(log_after) == len(log_before) + 1:
            record = log_after[-1]
            if record["capture_id"] != capture_id:
                raise RuntimeError("Runner recorded an unexpected capture")
            if bool(record["selection_accepted"]):
                raise RuntimeError(
                    "Underlying runner unexpectedly recorded an accepted "
                    "capture without manual visual QC"
                )
            print(f"Numerical rejection recorded: {capture_id}")
            continue
        if len(log_after) != len(log_before):
            raise RuntimeError("Unexpected change in the official log length")

        decision = pending_decision(root, capture_id)
        if decision is None:
            raise RuntimeError(
                "Runner neither recorded a rejection nor produced a "
                f"selection decision for {capture_id}"
            )
        is_accepted, contact_sheet = decision
        if not is_accepted:
            raise RuntimeError(
                "Runner produced an unrecorded numerical rejection for "
                f"{capture_id}"
            )
        print_handoff(
            capture_id,
            contact_sheet,
            len(log_after),
            accepted_before,
            target,
        )
        print(f"Runner log: {console_log}")
        return 0

    final_log = read_jsonl(log_path)
    final_accepted = sum(capture_accepted(row) for row in final_log)
    print()
    print("Batch limit reached")
    print(f"Captures processed in this run: {processed}")
    print(f"Officially inspected: {len(final_log)}/{len(order)}")
    print(f"Accepted: {final_accepted}/{target}")
    if final_accepted < target and len(final_log) < len(order):
        print(f"Next capture: {order[len(final_log)]['capture_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
