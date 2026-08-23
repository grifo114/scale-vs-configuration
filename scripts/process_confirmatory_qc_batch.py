#!/usr/bin/env python3
"""Preprocess a frozen-order confirmatory batch without touching the official log."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


OFFICIAL_BASE = Path("outputs/phase1/confirmatory_dataset")
SAFE_BATCH_ID = re.compile(r"^[A-Za-z0-9._-]+$")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=(
            "Process several captures in frozen order, queue numerical "
            "decisions, and create one PDF packet for manual visual QC. "
            "The official confirmatory log is never modified."
        )
    )
    parser.add_argument("--project-root", type=Path, default=root)
    parser.add_argument("--max-captures", type=int, default=10)
    parser.add_argument("--batch-id")
    parser.add_argument("--dry-run", action="store_true")
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


def validate_log(
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


def relative(root: Path, path: Path) -> str:
    return str(path.resolve().relative_to(root))


def run_logged(command: list[str], root: Path, console_log: Path) -> None:
    rendered = "$ " + shlex.join(command)
    print(rendered, flush=True)
    console_log.parent.mkdir(parents=True, exist_ok=True)
    with console_log.open("a", encoding="utf-8") as handle:
        handle.write(rendered + "\n")
        process = subprocess.Popen(
            command,
            cwd=root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        if process.stdout is None:
            raise RuntimeError("Subprocess stdout pipe was not created")
        for line in process.stdout:
            print(line, end="")
            handle.write(line)
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def run_validation_logged(
    command: list[str], root: Path, console_log: Path, report: Path
) -> None:
    rendered = "$ " + shlex.join(command)
    print(rendered, flush=True)
    completed = subprocess.run(
        command,
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    print(completed.stdout, end="")
    console_log.parent.mkdir(parents=True, exist_ok=True)
    with console_log.open("a", encoding="utf-8") as handle:
        handle.write(rendered + "\n")
        handle.write(completed.stdout)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode:
        raise subprocess.CalledProcessError(completed.returncode, command)


def git_value(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


def stage_paths(work: Path, capture_id: str) -> dict[str, Path]:
    stimuli = work / "stimuli" / capture_id
    return {
        "audit": work / "audits" / f"{capture_id}.jsonl",
        "audit_metadata": work / "audits" / f"{capture_id}_metadata.json",
        "candidates": work / "candidates" / f"{capture_id}.jsonl",
        "candidate_protocol": work
        / "candidates"
        / f"{capture_id}_protocol.json",
        "feasibility": work / "pair_feasibility" / f"{capture_id}.jsonl",
        "feasibility_protocol": work
        / "pair_feasibility"
        / f"{capture_id}_protocol.json",
        "selection": work / "selection" / f"{capture_id}.jsonl",
        "selection_protocol": work
        / "selection"
        / f"{capture_id}_protocol.json",
        "validation_report": work
        / "validation"
        / f"{capture_id}_validation.txt",
        "rendered_manifest": stimuli / "rendered_manifest.jsonl",
        "render_protocol": stimuli / "render_protocol.json",
        "contact_sheet": stimuli / "contact_sheet_all.png",
        "stimuli": stimuli,
    }


def early_rejection(
    root: Path,
    capture_id: str,
    candidates: Path,
    feasibility: Path,
    feasibility_protocol: Path,
    selection: Path,
    selection_protocol: Path,
) -> None:
    write_text_atomic(feasibility, "")
    write_json_atomic(
        feasibility_protocol,
        {
            "schema_version": "1.0",
            "status": "confirmatory_early_rejection",
            "capture_id": capture_id,
            "reason": "no_frame_candidates",
            "input": relative(root, candidates),
            "input_sha256": sha256(candidates),
            "output": relative(root, feasibility),
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
            "output": relative(root, selection),
            "output_sha256": sha256(selection),
        },
    )


def artifact_paths(
    root: Path,
    archive: Path,
    paths: dict[str, Path],
    numerical_accepted: bool,
) -> dict[str, str]:
    names = [
        "audit",
        "audit_metadata",
        "candidates",
        "candidate_protocol",
        "feasibility",
        "feasibility_protocol",
        "selection",
        "selection_protocol",
    ]
    if numerical_accepted:
        names.extend(
            [
                "validation_report",
                "rendered_manifest",
                "render_protocol",
                "contact_sheet",
            ]
        )
    result = {"archive": relative(root, archive)}
    for name in names:
        path = paths[name]
        if not path.is_file():
            raise FileNotFoundError(path)
        result[name] = relative(root, path)
    return result


def create_qc_packet(
    root: Path,
    batch_root: Path,
    records: list[dict[str, Any]],
) -> tuple[str | None, str, str]:
    packet = batch_root / "qc_packet"
    packet.mkdir(parents=True, exist_ok=True)
    accepted = [row for row in records if row["numerical_accepted"]]
    index_path = packet / "qc_index.csv"
    with index_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["page", "order_index", "capture_id", "contact_sheet"],
        )
        writer.writeheader()
        copied: list[Path] = []
        for page, row in enumerate(accepted, start=1):
            source = root / row["artifacts"]["contact_sheet"]
            name = (
                f"page-{page:02d}_order-{row['order_index']:03d}_"
                f"capture-{row['capture_id']}.png"
            )
            destination = packet / name
            shutil.copy2(source, destination)
            copied.append(destination)
            writer.writerow(
                {
                    "page": page,
                    "order_index": row["order_index"],
                    "capture_id": row["capture_id"],
                    "contact_sheet": name,
                }
            )

    pdf_path: Path | None = None
    if copied:
        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError(
                "Pillow is required to create the batch QC PDF"
            ) from exc
        pages = []
        for path in copied:
            with Image.open(path) as image:
                pages.append(image.convert("RGB"))
        pdf_path = packet / "qc_packet.pdf"
        pages[0].save(
            pdf_path,
            "PDF",
            save_all=True,
            append_images=pages[1:],
            resolution=150.0,
        )
        for image in pages:
            image.close()

    decisions_template = batch_root / "visual_qc_decisions.template.json"
    write_json_atomic(
        decisions_template,
        {
            "schema_version": "1.0",
            "batch_id": batch_root.name,
            "decisions": {
                row["capture_id"]: {
                    "visual_qc": "pass_or_fail",
                    "visual_qc_note": "",
                }
                for row in accepted
            },
        },
    )
    return (
        relative(root, pdf_path) if pdf_path is not None else None,
        relative(root, index_path),
        relative(root, decisions_template),
    )


def main() -> int:
    args = parse_args()
    if args.max_captures < 1:
        raise ValueError("--max-captures must be at least 1")

    root = args.project_root.expanduser().resolve()
    order_path = root / "configs/confirmatory_capture_order.csv"
    config_path = root / "configs/confirmatory_capture_protocol.json"
    log_path = root / OFFICIAL_BASE / "confirmatory_capture_log.jsonl"
    with order_path.open("r", encoding="utf-8", newline="") as handle:
        order = list(csv.DictReader(handle))
    config = read_json(config_path)
    target = int(config["stopping_rule"]["target_accepted_scenes"])
    official_log = read_jsonl(log_path)
    validate_log(official_log, order)
    accepted_before = sum(capture_accepted(row) for row in official_log)
    if accepted_before >= target:
        raise ValueError("Confirmatory target is already reached")
    if len(official_log) >= len(order):
        raise ValueError("Frozen capture pool is exhausted")

    available = len(order) - len(official_log)
    remaining_acceptances = target - accepted_before
    batch_size = min(args.max_captures, available, remaining_acceptances)
    start_index = len(official_log) + 1
    end_index = start_index + batch_size - 1
    batch_id = args.batch_id or (
        f"order-{start_index:03d}-to-{end_index:03d}"
    )
    if not SAFE_BATCH_ID.fullmatch(batch_id):
        raise ValueError("--batch-id contains unsupported characters")
    batch_root = root / OFFICIAL_BASE / "qc_batches" / batch_id
    work = batch_root / "work"
    manifest_path = batch_root / "batch_manifest.json"
    if batch_root.exists():
        raise FileExistsError(
            f"Batch workspace already exists; review it before retrying: {batch_root}"
        )

    entries = order[start_index - 1 : end_index]
    official_log_hash = sha256(log_path)
    print("Confirmatory QC batch dry-run" if args.dry_run else "Confirmatory QC batch")
    print(f"Officially inspected: {len(official_log)}/{len(order)}")
    print(f"Accepted: {accepted_before}/{target}")
    print(f"Batch ID: {batch_id}")
    print(f"Frozen order range: {start_index}-{end_index}")
    print("Capture IDs: " + ", ".join(row["capture_id"] for row in entries))
    print("Official log will not be modified.")
    if args.dry_run:
        return 0

    batch_root.mkdir(parents=True)
    log_snapshot = batch_root / "official_log_before.jsonl"
    shutil.copy2(log_path, log_snapshot)
    manifest: dict[str, Any] = {
        "schema_version": "1.0",
        "status": "processing",
        "batch_id": batch_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_branch": git_value(root, "branch", "--show-current"),
        "source_commit": git_value(root, "rev-parse", "HEAD"),
        "official_log": relative(root, log_path),
        "official_log_snapshot": relative(root, log_snapshot),
        "official_log_sha256_before": official_log_hash,
        "official_log_rows_before": len(official_log),
        "accepted_before": accepted_before,
        "target_accepted_scenes": target,
        "order_range": [start_index, end_index],
        "records": [],
    }
    write_json_atomic(manifest_path, manifest)

    python = sys.executable
    records: list[dict[str, Any]] = []
    try:
        for offset, entry in enumerate(entries):
            order_index = start_index + offset
            capture_id = entry["capture_id"]
            archive = root / f"data/ca1m-confirmatory/ca1m-val-{capture_id}.tar"
            paths = stage_paths(work, capture_id)
            console_log = batch_root / "runner_logs" / f"{capture_id}.txt"
            print()
            print(
                f"Processing batch capture {offset + 1}/{batch_size}: "
                f"order={order_index} capture={capture_id}"
            )
            archive.parent.mkdir(parents=True, exist_ok=True)
            downloaded_part: Path | None = None
            if not archive.exists():
                downloaded_part = archive.with_name(archive.name + ".part")
                run_logged(
                    [
                        "curl",
                        "--fail",
                        "--location",
                        "--continue-at",
                        "-",
                        "--progress-bar",
                        entry["url"],
                        "--output",
                        str(downloaded_part),
                    ],
                    root,
                    console_log,
                )
            print("Validating TAR structure ...", flush=True)
            archive_to_validate = downloaded_part or archive
            with tarfile.open(archive_to_validate, "r") as handle:
                member_count = sum(1 for _ in handle)
            print(f"TAR members: {member_count}")
            if downloaded_part is not None:
                os.replace(downloaded_part, archive)

            run_logged(
                [
                    python,
                    "scripts/audit_ca1m_frames.py",
                    str(archive),
                    "--output",
                    str(paths["audit"]),
                    "--metadata",
                    str(paths["audit_metadata"]),
                    "--progress-every",
                    "250",
                    "--overwrite",
                ],
                root,
                console_log,
            )
            frame_count = count_jsonl(paths["audit"])
            run_logged(
                [
                    python,
                    "scripts/rank_scene_candidates.py",
                    str(paths["audit"]),
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
                    str(paths["candidates"]),
                    "--protocol",
                    str(paths["candidate_protocol"]),
                    "--overwrite",
                ],
                root,
                console_log,
            )

            if count_jsonl(paths["candidates"]) == 0:
                early_rejection(
                    root,
                    capture_id,
                    paths["candidates"],
                    paths["feasibility"],
                    paths["feasibility_protocol"],
                    paths["selection"],
                    paths["selection_protocol"],
                )
            else:
                run_logged(
                    [
                        python,
                        "scripts/evaluate_pair_feasibility.py",
                        str(paths["candidates"]),
                        "--output",
                        str(paths["feasibility"]),
                        "--protocol",
                        str(paths["feasibility_protocol"]),
                        "--overwrite",
                    ],
                    root,
                    console_log,
                )
                run_logged(
                    [
                        python,
                        "scripts/select_confirmatory_pairs.py",
                        str(paths["feasibility"]),
                        "--output",
                        str(paths["selection"]),
                        "--protocol",
                        str(paths["selection_protocol"]),
                        "--overwrite",
                    ],
                    root,
                    console_log,
                )

            decision = read_json(paths["selection_protocol"])[
                "capture_decisions"
            ][capture_id]
            numerical_accepted = bool(decision["accepted"])
            if numerical_accepted:
                run_validation_logged(
                    [
                        python,
                        "scripts/validate_pair_manifest.py",
                        str(paths["selection"]),
                        "--protocol",
                        str(paths["selection_protocol"]),
                    ],
                    root,
                    console_log,
                    paths["validation_report"],
                )
                run_logged(
                    [
                        python,
                        "scripts/render_pair_stimuli.py",
                        str(paths["selection"]),
                        "--output-dir",
                        str(paths["stimuli"]),
                        "--columns",
                        "3",
                        "--cell-width",
                        "360",
                        "--overwrite",
                    ],
                    root,
                    console_log,
                )

            record = {
                "order_index": order_index,
                "capture_id": capture_id,
                "numerical_accepted": numerical_accepted,
                "selection_pair_count": count_jsonl(paths["selection"]),
                "selection_decision": decision,
                "visual_qc": "pending" if numerical_accepted else "not_applicable",
                "capture_decision": (
                    "pending_visual_qc"
                    if numerical_accepted
                    else "rejected_numerical"
                ),
                "artifacts": artifact_paths(
                    root, archive, paths, numerical_accepted
                ),
            }
            records.append(record)
            manifest["records"] = records
            write_json_atomic(manifest_path, manifest)
            print(
                "Queued for visual QC"
                if numerical_accepted
                else "Numerical rejection queued"
            )

        if sha256(log_path) != official_log_hash:
            raise RuntimeError("Official log changed during batch processing")
        if len(read_jsonl(log_path)) != len(official_log):
            raise RuntimeError("Official log length changed during batch processing")
        pdf_path, index_path, decisions_template = create_qc_packet(
            root, batch_root, records
        )
        manifest["status"] = "awaiting_visual_qc"
        manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["numerically_accepted"] = sum(
            row["numerical_accepted"] for row in records
        )
        manifest["numerically_rejected"] = len(records) - int(
            manifest["numerically_accepted"]
        )
        manifest["qc_packet_pdf"] = pdf_path
        manifest["qc_index"] = index_path
        manifest["visual_qc_decisions_template"] = decisions_template
        manifest["official_log_sha256_after"] = sha256(log_path)
        manifest["official_log_rows_after"] = len(read_jsonl(log_path))
        write_json_atomic(manifest_path, manifest)
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["failed_at_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        manifest["records"] = records
        manifest["official_log_sha256_after_failure"] = sha256(log_path)
        manifest["official_log_rows_after_failure"] = len(read_jsonl(log_path))
        write_json_atomic(manifest_path, manifest)
        raise

    print()
    print("Batch preprocessing completed")
    print(f"Captures processed: {len(records)}")
    print(f"Numerically accepted: {manifest['numerically_accepted']}")
    print(f"Numerically rejected: {manifest['numerically_rejected']}")
    print(f"Manifest: {manifest_path}")
    print(f"QC packet: {root / pdf_path if pdf_path else 'none'}")
    print("Official log unchanged; manual batch QC is now required.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
