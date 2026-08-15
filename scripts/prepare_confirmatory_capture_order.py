#!/usr/bin/env python3
"""Create the frozen deterministic processing order for confirmatory captures."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pathlib
import re
import tempfile
from typing import Any


def parse_args() -> argparse.Namespace:
    project_root = pathlib.Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Build and verify the confirmatory CA-1M capture order."
    )
    parser.add_argument(
        "--project-root",
        type=pathlib.Path,
        default=project_root,
    )
    parser.add_argument(
        "--validation-list",
        type=pathlib.Path,
        default=pathlib.Path("ml-cubifyanything/data/val.txt"),
    )
    parser.add_argument(
        "--protocol",
        type=pathlib.Path,
        default=pathlib.Path("configs/confirmatory_capture_protocol.json"),
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("configs/confirmatory_capture_order.csv"),
    )
    parser.add_argument(
        "--metadata",
        type=pathlib.Path,
        default=pathlib.Path("configs/confirmatory_capture_order_metadata.json"),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve(project_root: pathlib.Path, path: pathlib.Path) -> pathlib.Path:
    path = path.expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def display(project_root: pathlib.Path, path: pathlib.Path) -> str:
    try:
        return str(path.relative_to(project_root))
    except ValueError:
        return str(path)


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_protocol(path: pathlib.Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Protocol root must be a JSON object")
    return data


def load_excluded_ids(path: pathlib.Path) -> list[str]:
    ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    ids = [capture_id for capture_id in ids if capture_id]
    if len(ids) != len(set(ids)):
        raise ValueError("Development exclusion file contains duplicate IDs")
    if any(not capture_id.isdigit() for capture_id in ids):
        raise ValueError("Development capture IDs must contain digits only")
    return ids


def load_validation_records(
    path: pathlib.Path,
    url_pattern: str,
) -> list[tuple[str, str]]:
    pattern = re.compile(url_pattern)
    records: list[tuple[str, str]] = []
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        url = raw_line.strip()
        if not url:
            continue
        match = pattern.fullmatch(url)
        if match is None:
            raise ValueError(f"Invalid validation URL on line {line_number}: {url}")
        records.append((match.group(1), url))
    capture_ids = [capture_id for capture_id, _ in records]
    urls = [url for _, url in records]
    if len(capture_ids) != len(set(capture_ids)):
        raise ValueError("Validation list contains duplicate capture IDs")
    if len(urls) != len(set(urls)):
        raise ValueError("Validation list contains duplicate URLs")
    return records


def write_csv_atomic(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        newline="",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = pathlib.Path(handle.name)
        writer = csv.DictWriter(
            handle,
            fieldnames=("order_index", "capture_id", "order_key_sha256", "url"),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_json_atomic(path: pathlib.Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = pathlib.Path(handle.name)
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main() -> int:
    args = parse_args()
    project_root = args.project_root.expanduser().resolve()
    validation_path = resolve(project_root, args.validation_list)
    protocol_path = resolve(project_root, args.protocol)
    output_path = resolve(project_root, args.output)
    metadata_path = resolve(project_root, args.metadata)

    for source in (validation_path, protocol_path):
        if not source.is_file():
            raise FileNotFoundError(f"Required file not found: {source}")
    for destination in (output_path, metadata_path):
        if destination.exists() and not args.overwrite:
            raise FileExistsError(
                f"Destination exists; pass --overwrite to replace it: {destination}"
            )

    protocol = load_protocol(protocol_path)
    source_rule = protocol["source"]
    exclusion_rule = protocol["development_exclusion"]
    pool_rule = protocol["confirmatory_pool"]
    order_rule = protocol["capture_order"]
    stopping_rule = protocol["stopping_rule"]

    exclusion_path = resolve(project_root, pathlib.Path(exclusion_rule["file"]))
    if not exclusion_path.is_file():
        raise FileNotFoundError(f"Exclusion file not found: {exclusion_path}")
    excluded_ids = load_excluded_ids(exclusion_path)
    expected_excluded = [str(value) for value in exclusion_rule["capture_ids"]]
    if excluded_ids != expected_excluded:
        raise ValueError("Exclusion file does not exactly match the frozen protocol")

    records = load_validation_records(validation_path, str(source_rule["url_pattern"]))
    expected_records = int(source_rule["expected_records"])
    expected_unique = int(source_rule["expected_unique_capture_ids"])
    if len(records) != expected_records or len(records) != expected_unique:
        raise ValueError(
            f"Validation records={len(records)}; expected {expected_records} unique records"
        )

    record_ids = {capture_id for capture_id, _ in records}
    missing_exclusions = sorted(set(excluded_ids) - record_ids)
    if missing_exclusions:
        raise ValueError(
            "Development IDs missing from validation list: "
            + ", ".join(missing_exclusions)
        )

    namespace = str(order_rule["namespace"])
    confirmatory = []
    for capture_id, url in records:
        if capture_id in excluded_ids:
            continue
        key = hashlib.sha256(f"{namespace}:{capture_id}".encode("utf-8")).hexdigest()
        confirmatory.append((key, capture_id, url))
    confirmatory.sort(key=lambda item: (item[0], item[1]))

    expected_pool = int(pool_rule["expected_capture_count"])
    if len(confirmatory) != expected_pool:
        raise ValueError(
            f"Confirmatory pool={len(confirmatory)}; expected {expected_pool}"
        )
    target = int(stopping_rule["target_accepted_scenes"])
    maximum = int(stopping_rule["maximum_captures_inspected"])
    if target <= 0 or target > len(confirmatory):
        raise ValueError("Invalid target_accepted_scenes")
    if maximum != len(confirmatory):
        raise ValueError("maximum_captures_inspected must equal the frozen pool size")

    rows = [
        {
            "order_index": index,
            "capture_id": capture_id,
            "order_key_sha256": key,
            "url": url,
        }
        for index, (key, capture_id, url) in enumerate(confirmatory, start=1)
    ]
    write_csv_atomic(output_path, rows)
    metadata = {
        "schema_version": "1.0",
        "protocol": display(project_root, protocol_path),
        "protocol_sha256": sha256(protocol_path),
        "validation_list": display(project_root, validation_path),
        "validation_list_sha256": sha256(validation_path),
        "development_exclusion": display(project_root, exclusion_path),
        "development_exclusion_sha256": sha256(exclusion_path),
        "source_records": len(records),
        "excluded_records": len(excluded_ids),
        "confirmatory_pool": len(rows),
        "target_accepted_scenes": target,
        "order_method": order_rule["method"],
        "order_namespace": namespace,
        "output": display(project_root, output_path),
        "output_sha256": sha256(output_path),
        "first_ten_capture_ids": [row["capture_id"] for row in rows[:10]],
    }
    write_json_atomic(metadata_path, metadata)

    print("Confirmatory capture order")
    print(f"Source records: {len(records)}")
    print(f"Development exclusions: {len(excluded_ids)}")
    print(f"Confirmatory pool: {len(rows)}")
    print(f"Target accepted scenes: {target}")
    print(f"Namespace: {namespace}")
    print("First ten: " + ", ".join(metadata["first_ten_capture_ids"]))
    print(f"Output: {output_path}")
    print(f"Output SHA-256: {metadata['output_sha256']}")
    print(f"Metadata: {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
