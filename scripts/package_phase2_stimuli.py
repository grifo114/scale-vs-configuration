#!/usr/bin/env python3
"""Build and verify the deterministic Phase 2 stimulus transport archive.

The archive contains exactly the 588 stimulus files referenced by the frozen
Phase 2 query manifest, preserving their repository-relative paths. Protocol,
manifest, and runner stay in Git and are intentionally not duplicated here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Iterable


SCRIPT_VERSION = "0.1.0"
EXPECTED_BRANCH = "phase2/inference-protocol"
EXPECTED_RELATIONS = 588
EXPECTED_CANARY_QUERIES = 12
EXPECTED_PROTOCOL_SHA256 = (
    "3c09ca97984f47b868758ce3b4d07bf85e38dabd3d66b7e4eae9bb474ce4b774"
)
EXPECTED_MANIFEST_SHA256 = (
    "61e4f9a8583dc6f375b6d4d9f72361acf1d9ff3a976485bf5b2f83736ea0360e"
)
EXPECTED_RUNNER_SHA256 = (
    "643a63625d97aa893322410cf865bd2aa190d2f9b5d4c984aa6c50ef09a38818"
)


class PackageError(RuntimeError):
    """Raised when an input or package invariant is violated."""


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Build a deterministic tar archive of Phase 2 stimuli."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=project_root,
        help="Project root (default: parent of scripts/).",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("configs/phase2_query_manifest.jsonl"),
        help="Frozen query manifest, relative to the project root by default.",
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("configs/phase2_inference_protocol.json"),
        help="Frozen inference protocol, relative to the project root by default.",
    )
    parser.add_argument(
        "--runner",
        type=Path,
        default=Path("scripts/run_phase2_inference.py"),
        help="Frozen inference runner, relative to the project root by default.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Destination .tar path. Required unless --validate-only is used.",
    )
    parser.add_argument(
        "--descriptor",
        type=Path,
        help="Destination JSON descriptor (default: OUTPUT.json).",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate frozen inputs and all stimulus hashes without packaging.",
    )
    return parser.parse_args()


def resolve_under(root: Path, value: Path) -> Path:
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def relative_to_root(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError as error:
        raise PackageError(f"Path is outside project root: {path}") from error


def sha256_stream(handle: BinaryIO) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return sha256_stream(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as error:
        raise PackageError(f"Manifest not found: {path}") from error
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise PackageError(
                f"Invalid JSON at {path}:{line_number}: {error}"
            ) from error
        if not isinstance(value, dict):
            raise PackageError(f"Expected JSON object at {path}:{line_number}")
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
        raise PackageError(
            f"Git command failed: git {' '.join(arguments)}\n{stderr.strip()}"
        ) from error
    return completed.stdout.strip()


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
    return {
        "path": relative,
        "tracked": tracked,
        "changed_from_head": changed,
    }


def validate_git_state(
    root: Path,
    protocol_path: Path,
    manifest_path: Path,
    runner_path: Path,
    actual_packaging: bool,
) -> dict[str, Any]:
    if git(root, "rev-parse", "--is-inside-work-tree") != "true":
        raise PackageError(f"Not a Git worktree: {root}")
    branch = git(root, "branch", "--show-current")
    if branch != EXPECTED_BRANCH:
        raise PackageError(f"Unexpected branch: {branch!r}")
    required = {
        "protocol": git_file_state(root, protocol_path),
        "manifest": git_file_state(root, manifest_path),
        "runner": git_file_state(root, runner_path),
        "packager": git_file_state(root, Path(__file__)),
    }
    for key in ("protocol", "manifest", "runner"):
        state = required[key]
        if not state["tracked"] or state["changed_from_head"]:
            raise PackageError(
                f"{key.capitalize()} must be tracked and unchanged: {state['path']}"
            )
    worktree_status = git(root, "status", "--porcelain=v1", "--untracked-files=all")
    if actual_packaging and worktree_status:
        raise PackageError("The complete worktree must be clean before packaging")
    if actual_packaging:
        packager = required["packager"]
        if not packager["tracked"] or packager["changed_from_head"]:
            raise PackageError("The packager must be committed and unchanged")
    return {
        "branch": branch,
        "commit": git(root, "rev-parse", "HEAD"),
        "worktree_clean": not bool(worktree_status),
        "files": required,
    }


def validate_frozen_hashes(
    protocol_path: Path,
    manifest_path: Path,
    runner_path: Path,
) -> dict[str, str]:
    actual = {
        "protocol_sha256": sha256_file(protocol_path),
        "query_manifest_sha256": sha256_file(manifest_path),
        "runner_sha256": sha256_file(runner_path),
    }
    expected = {
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "query_manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "runner_sha256": EXPECTED_RUNNER_SHA256,
    }
    for key, expected_value in expected.items():
        if actual[key] != expected_value:
            raise PackageError(
                f"Unexpected {key}: {actual[key]} != {expected_value}"
            )
    return actual


def validate_manifest(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(rows) != EXPECTED_RELATIONS:
        raise PackageError(f"Expected 588 rows, found {len(rows)}")
    pair_ids = [str(row.get("pair_id", "")) for row in rows]
    if len(set(pair_ids)) != EXPECTED_RELATIONS:
        raise PackageError("Pair IDs are not unique")
    if [int(row.get("run_index", -1)) for row in rows] != list(
        range(1, EXPECTED_RELATIONS + 1)
    ):
        raise PackageError("Manifest order is not the frozen 1..588 sequence")
    canary = [row for row in rows if row.get("is_canary") is True]
    if len(canary) != EXPECTED_CANARY_QUERIES:
        raise PackageError(f"Expected 12 canary rows, found {len(canary)}")
    if len({str(row.get("scene_id", "")) for row in canary}) != 12:
        raise PackageError("Canary does not contain twelve unique scenes")
    paths = [str(row.get("stimulus_path", "")) for row in rows]
    if len(set(paths)) != EXPECTED_RELATIONS:
        raise PackageError("Stimulus paths are not unique")
    for relative in paths:
        pure = PurePosixPath(relative)
        if not relative or pure.is_absolute() or ".." in pure.parts:
            raise PackageError(f"Unsafe stimulus path: {relative!r}")
    return rows


def verify_source_stimuli(
    root: Path,
    rows: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int, str]:
    verified: list[dict[str, Any]] = []
    total_bytes = 0
    set_digest = hashlib.sha256()
    for row in rows:
        relative = str(row["stimulus_path"])
        source = resolve_under(root, Path(relative))
        if not source.is_file():
            raise PackageError(f"Missing stimulus: {relative}")
        actual_hash = sha256_file(source)
        expected_hash = str(row["stimulus_sha256"])
        if actual_hash != expected_hash:
            raise PackageError(
                f"Stimulus hash mismatch for {row['pair_id']}: "
                f"{actual_hash} != {expected_hash}"
            )
        size = source.stat().st_size
        total_bytes += size
        set_digest.update(f"{expected_hash}  {relative}\n".encode("utf-8"))
        verified.append(
            {
                "pair_id": str(row["pair_id"]),
                "path": relative,
                "sha256": expected_hash,
                "size_bytes": size,
            }
        )
    return verified, total_bytes, set_digest.hexdigest()


def normalized_tarinfo(source: Path, arcname: str) -> tarfile.TarInfo:
    information = tarfile.TarInfo(name=arcname)
    information.size = source.stat().st_size
    information.mode = 0o644
    information.uid = 0
    information.gid = 0
    information.uname = ""
    information.gname = ""
    information.mtime = 0
    information.type = tarfile.REGTYPE
    information.pax_headers = {}
    return information


def build_archive(
    root: Path,
    entries: Iterable[dict[str, Any]],
    destination: Path,
) -> None:
    with tarfile.open(destination, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for entry in entries:
            source = resolve_under(root, Path(str(entry["path"])))
            information = normalized_tarinfo(source, str(entry["path"]))
            with source.open("rb") as handle:
                archive.addfile(information, fileobj=handle)


def verify_archive(
    archive_path: Path,
    entries: list[dict[str, Any]],
) -> None:
    expected_names = [str(entry["path"]) for entry in entries]
    expected_by_name = {str(entry["path"]): entry for entry in entries}
    with tarfile.open(archive_path, mode="r:") as archive:
        members = archive.getmembers()
        actual_names = [member.name for member in members]
        if actual_names != expected_names:
            raise PackageError("Archive member order or membership is unexpected")
        for member in members:
            if not member.isfile():
                raise PackageError(f"Archive contains a non-file member: {member.name}")
            expected = expected_by_name[member.name]
            if member.size != int(expected["size_bytes"]):
                raise PackageError(f"Archive size mismatch: {member.name}")
            handle = archive.extractfile(member)
            if handle is None:
                raise PackageError(f"Could not read archive member: {member.name}")
            actual_hash = sha256_stream(handle)
            if actual_hash != str(expected["sha256"]):
                raise PackageError(f"Archive hash mismatch: {member.name}")


def write_json_exclusive(path: Path, value: dict[str, Any]) -> None:
    content = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o644)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise


def main() -> int:
    args = parse_args()
    if not args.validate_only and args.output is None:
        raise PackageError("--output is required unless --validate-only is used")
    if args.validate_only and args.descriptor is not None:
        raise PackageError("--descriptor cannot be used with --validate-only")

    root = args.project_root.resolve()
    manifest_path = resolve_under(root, args.manifest)
    protocol_path = resolve_under(root, args.protocol)
    runner_path = resolve_under(root, args.runner)
    git_metadata = validate_git_state(
        root,
        protocol_path,
        manifest_path,
        runner_path,
        actual_packaging=not args.validate_only,
    )
    hashes = validate_frozen_hashes(protocol_path, manifest_path, runner_path)
    rows = validate_manifest(read_jsonl(manifest_path))
    entries, total_bytes, stimulus_set_sha256 = verify_source_stimuli(root, rows)
    packager_sha256 = sha256_file(Path(__file__))

    print("===== PHASE 2 STIMULUS PACKAGE INPUTS =====")
    print(f"Git commit: {git_metadata['commit']}")
    print(f"Protocol SHA-256: {hashes['protocol_sha256']}")
    print(f"Manifest SHA-256: {hashes['query_manifest_sha256']}")
    print(f"Runner SHA-256: {hashes['runner_sha256']}")
    print(f"Packager SHA-256: {packager_sha256}")
    print(f"Stimuli verified: {len(entries)}")
    print(f"Payload size: {total_bytes / (1024**2):.2f} MiB")
    print(f"Stimulus-set SHA-256: {stimulus_set_sha256}")
    if args.validate_only:
        state = git_metadata["files"]["packager"]
        print("Validation only: OK")
        print(
            "Packager committed and clean: "
            f"{state['tracked'] and not state['changed_from_head']}"
        )
        return 0

    output_path = args.output.resolve()
    descriptor_path = (
        args.descriptor.resolve()
        if args.descriptor is not None
        else output_path.with_name(output_path.name + ".json")
    )
    if output_path.suffix != ".tar":
        raise PackageError("The output filename must end with .tar")
    if output_path == descriptor_path:
        raise PackageError("Archive and descriptor paths must differ")
    for path in (output_path, descriptor_path):
        if path.exists():
            raise PackageError(f"Refusing to overwrite existing path: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)

    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            dir=output_path.parent,
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
        temporary_path = Path(temporary_name)
        build_archive(root, entries, temporary_path)
        verify_archive(temporary_path, entries)
        archive_sha256 = sha256_file(temporary_path)
        archive_size = temporary_path.stat().st_size
        os.replace(temporary_path, output_path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)

    descriptor = {
        "schema_version": "1.0",
        "artifact_name": "phase2_stimuli_transport_v1",
        "archive": {
            "filename": output_path.name,
            "format": "uncompressed_posix_pax_tar",
            "sha256": archive_sha256,
            "size_bytes": archive_size,
            "entry_count": len(entries),
            "payload_size_bytes": total_bytes,
            "entry_order": "phase2_query_manifest.run_index_ascending",
            "normalized_metadata": {
                "gid": 0,
                "gname": "",
                "mode": "0644",
                "mtime": 0,
                "uid": 0,
                "uname": "",
            },
        },
        "source": {
            "git_branch": git_metadata["branch"],
            "git_commit": git_metadata["commit"],
            "protocol_path": relative_to_root(root, protocol_path),
            "protocol_sha256": hashes["protocol_sha256"],
            "query_manifest_path": relative_to_root(root, manifest_path),
            "query_manifest_sha256": hashes["query_manifest_sha256"],
            "runner_path": relative_to_root(root, runner_path),
            "runner_sha256": hashes["runner_sha256"],
            "packager_path": relative_to_root(root, Path(__file__)),
            "packager_sha256": packager_sha256,
            "stimulus_set_sha256": stimulus_set_sha256,
        },
    }
    write_json_exclusive(descriptor_path, descriptor)
    print("\n===== PACKAGE RESULT =====")
    print(f"Archive: {output_path}")
    print(f"Archive SHA-256: {archive_sha256}")
    print(f"Archive size: {archive_size / (1024**2):.2f} MiB")
    print(f"Descriptor: {descriptor_path}")
    print("Archive verification: OK")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PackageError as error:
        print(f"ERROR: {error}", file=os.sys.stderr)
        raise SystemExit(1)
