#!/usr/bin/env python3
"""Process the next frozen-order confirmatory capture through visual-QC handoff."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shlex
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Run the non-model confirmatory pipeline for the next capture."
    )
    parser.add_argument("--capture-id")
    parser.add_argument("--project-root", type=Path, default=root)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
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
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise TypeError(f"Expected JSON object: {path}")
                rows.append(row)
    return rows


def count_jsonl(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    write_text_atomic(path, json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def run(command: list[str], cwd: Path) -> None:
    print("$ " + " ".join(shlex.quote(part) for part in command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def run_validation(command: list[str], cwd: Path, report: Path) -> None:
    print("$ " + " ".join(shlex.quote(part) for part in command), flush=True)
    completed = subprocess.run(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    print(completed.stdout, end="")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode:
        raise subprocess.CalledProcessError(completed.returncode, command)


def ensure_destinations(paths: list[Path], overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        listing = "\n".join(str(path) for path in existing)
        raise FileExistsError(
            "Stage outputs already exist. Review them or rerun with --overwrite:\n"
            + listing
        )


def main() -> int:
    args = parse_args()
    root = args.project_root.expanduser().resolve()
    order_path = root / "configs/confirmatory_capture_order.csv"
    protocol_path = root / "configs/confirmatory_capture_protocol.json"
    log_path = (
        root
        / "outputs/phase1/confirmatory_dataset/confirmatory_capture_log.jsonl"
    )
    with order_path.open("r", encoding="utf-8", newline="") as handle:
        order = list(csv.DictReader(handle))
    log = read_jsonl(log_path)
    config = read_json(protocol_path)
    target = int(config["stopping_rule"]["target_accepted_scenes"])
    accepted_count = sum(bool(row["selection_accepted"]) for row in log)
    if accepted_count >= target:
        raise ValueError(f"Confirmatory target already reached: {accepted_count}/{target}")
    next_index = len(log) + 1
    if next_index > len(order):
        raise ValueError("Frozen capture pool is exhausted")
    entry = order[next_index - 1]
    capture_id = entry["capture_id"]
    if args.capture_id is not None and args.capture_id != capture_id:
        raise ValueError(
            f"Next capture is {capture_id}; received --capture-id {args.capture_id}"
        )

    archive = root / f"data/ca1m-confirmatory/ca1m-val-{capture_id}.tar"
    base = root / "outputs/phase1/confirmatory_dataset"
    audit = base / f"audits/{capture_id}.jsonl"
    audit_metadata = base / f"audits/{capture_id}_metadata.json"
    candidates = base / f"candidates/{capture_id}.jsonl"
    candidate_protocol = base / f"candidates/{capture_id}_protocol.json"
    feasibility = base / f"pair_feasibility/{capture_id}.jsonl"
    feasibility_protocol = base / f"pair_feasibility/{capture_id}_protocol.json"
    selection = base / f"selection/{capture_id}.jsonl"
    selection_protocol = base / f"selection/{capture_id}_protocol.json"
    validation = base / f"validation/{capture_id}_validation.txt"
    stimuli = base / f"stimuli/{capture_id}"
    rendered_manifest = stimuli / "rendered_manifest.jsonl"
    render_protocol = stimuli / "render_protocol.json"
    contact_sheet = stimuli / "contact_sheet_all.png"
    stage_outputs = [
        audit,
        audit_metadata,
        candidates,
        candidate_protocol,
        feasibility,
        feasibility_protocol,
        selection,
        selection_protocol,
        validation,
        rendered_manifest,
        render_protocol,
        contact_sheet,
    ]

    print("Next confirmatory capture")
    print(f"Order index: {next_index}")
    print(f"Capture ID: {capture_id}")
    print(f"URL: {entry['url']}")
    print(f"Accepted progress: {accepted_count}/{target}")
    print(f"Archive: {archive}")
    if args.dry_run:
        print("Dry-run OK. No download or processing was performed.")
        return 0

    ensure_destinations(stage_outputs, args.overwrite)
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        run(
            [
                "curl",
                "--fail",
                "--location",
                "--continue-at",
                "-",
                "--progress-bar",
                entry["url"],
                "--output",
                str(archive),
            ],
            root,
        )
    print("Validating TAR structure ...", flush=True)
    with tarfile.open(archive, "r") as handle:
        member_count = sum(1 for _ in handle)
    print(f"TAR members: {member_count}")

    python = sys.executable
    overwrite = ["--overwrite"]
    run(
        [
            python,
            "scripts/audit_ca1m_frames.py",
            str(archive),
            "--output",
            str(audit),
            "--metadata",
            str(audit_metadata),
            "--progress-every",
            "250",
            *overwrite,
        ],
        root,
    )
    frame_count = count_jsonl(audit)
    run(
        [
            python,
            "scripts/rank_scene_candidates.py",
            str(audit),
            "--minimum-oracle",
            "8",
            "--minimum-natural",
            "6",
            "--minimum-natural-interior",
            "3",
            "--sharpness-quantile",
            "0",
            "--candidates-per-temporal-bin",
            str(frame_count),
            "--candidates-per-capture",
            str(frame_count),
            "--output",
            str(candidates),
            "--protocol",
            str(candidate_protocol),
            *overwrite,
        ],
        root,
    )
    if count_jsonl(candidates) == 0:
        write_text_atomic(feasibility, "")
        write_json_atomic(
            feasibility_protocol,
            {
                "schema_version": "1.0",
                "status": "confirmatory_early_rejection",
                "capture_id": capture_id,
                "reason": "no_frame_candidates",
                "input": str(candidates.relative_to(root)),
                "input_sha256": sha256(candidates),
                "output": str(feasibility.relative_to(root)),
                "output_sha256": sha256(feasibility),
            },
        )
        write_text_atomic(selection, "")
        write_json_atomic(
            selection_protocol,
            {
                "schema_version": "1.0",
                "status": "confirmatory_pair_selection",
                "capture_decisions": {
                    capture_id: {
                        "capture_id": capture_id,
                        "candidate_frames": 0,
                        "best_candidate_rank": None,
                        "best_frame_index": None,
                        "best_strict_pair_count": 0,
                        "accepted": False,
                        "rejection_reasons": ["no_frame_candidates"],
                    }
                },
                "accepted_captures": [],
                "rejected_captures": [capture_id],
                "number_of_pairs": 0,
                "output": str(selection.relative_to(root)),
                "output_sha256": sha256(selection),
            },
        )
        run(
            [
                python,
                "scripts/record_confirmatory_capture.py",
                "--capture-id",
                capture_id,
                "--archive",
                str(archive),
                "--audit",
                str(audit),
                "--audit-metadata",
                str(audit_metadata),
                "--candidates",
                str(candidates),
                "--candidate-protocol",
                str(candidate_protocol),
                "--feasibility",
                str(feasibility),
                "--feasibility-protocol",
                str(feasibility_protocol),
                "--selection",
                str(selection),
                "--selection-protocol",
                str(selection_protocol),
                "--visual-qc",
                "not_applicable",
                "--visual-qc-note",
                "Rejected because no frame met the frozen candidate rule.",
            ],
            root,
        )
        print("Capture rejected at frame-candidate stage and recorded.")
        return 0
    run(
        [
            python,
            "scripts/evaluate_pair_feasibility.py",
            str(candidates),
            "--output",
            str(feasibility),
            "--protocol",
            str(feasibility_protocol),
            *overwrite,
        ],
        root,
    )
    run(
        [
            python,
            "scripts/select_confirmatory_pairs.py",
            str(feasibility),
            "--output",
            str(selection),
            "--protocol",
            str(selection_protocol),
            *overwrite,
        ],
        root,
    )
    decision = read_json(selection_protocol)["capture_decisions"][capture_id]
    if not decision["accepted"]:
        run(
            [
                python,
                "scripts/record_confirmatory_capture.py",
                "--capture-id",
                capture_id,
                "--archive",
                str(archive),
                "--audit",
                str(audit),
                "--audit-metadata",
                str(audit_metadata),
                "--candidates",
                str(candidates),
                "--candidate-protocol",
                str(candidate_protocol),
                "--feasibility",
                str(feasibility),
                "--feasibility-protocol",
                str(feasibility_protocol),
                "--selection",
                str(selection),
                "--selection-protocol",
                str(selection_protocol),
                "--visual-qc",
                "not_applicable",
                "--visual-qc-note",
                "Rejected by the frozen numerical protocol before rendering.",
            ],
            root,
        )
        print("Capture rejected and recorded. Run this command again for the next ID.")
        return 0

    run_validation(
        [
            python,
            "scripts/validate_pair_manifest.py",
            str(selection),
            "--protocol",
            str(selection_protocol),
        ],
        root,
        validation,
    )
    run(
        [
            python,
            "scripts/render_pair_stimuli.py",
            str(selection),
            "--output-dir",
            str(stimuli),
            "--columns",
            "3",
            "--cell-width",
            "360",
            *overwrite,
        ],
        root,
    )
    print()
    print("Capture accepted numerically; manual visual QC is required.")
    print(f"Contact sheet: {contact_sheet}")
    print("Do not process the next capture until this capture is recorded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
