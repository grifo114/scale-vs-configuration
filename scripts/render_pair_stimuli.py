#!/usr/bin/env python3
"""Render red/blue pair stimuli and human-audit contact sheets."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tarfile
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps


OBJECT_A_COLOR = "#E53935"
OBJECT_B_COLOR = "#1565E8"
BACKGROUND_COLOR = "#FFFFFF"
TEXT_COLOR = "#111111"


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Render selected CA-1M pairs as red/blue visual stimuli."
    )
    parser.add_argument("manifest", type=Path, help="Selected-pair JSONL manifest.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=project_root,
        help="Project root. Defaults to the parent of scripts/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Destination directory for stimuli and audit sheets.",
    )
    parser.add_argument("--columns", type=int, default=3)
    parser.add_argument("--cell-width", type=int, default=360)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing generated files.",
    )
    return parser.parse_args()


def resolve_path(project_root: Path, path: Path) -> Path:
    path = path.expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def display_path(project_root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(project_root))
    except ValueError:
        return str(path)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"Expected an object at {path}:{line_number}")
            rows.append(row)
    return rows


def read_member_bytes(archive: tarfile.TarFile, name: str) -> bytes:
    handle = archive.extractfile(name)
    if handle is None:
        raise FileNotFoundError(f"Member not found in {archive.name}: {name}")
    with handle:
        return handle.read()


def draw_label(
    draw: ImageDraw.ImageDraw,
    box: list[float],
    label: str,
    color: str,
    font: ImageFont.ImageFont,
) -> None:
    left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
    text_width = right - left
    text_height = bottom - top
    x = max(0, int(round(box[0])))
    y = max(0, int(round(box[1])) - text_height - 8)
    draw.rectangle(
        (x, y, x + text_width + 10, y + text_height + 8), fill=color
    )
    draw.text((x + 5, y + 4), label, fill="white", font=font)


def render_stimulus(source: Image.Image, row: dict[str, Any]) -> Image.Image:
    stimulus = source.convert("RGB").copy()
    draw = ImageDraw.Draw(stimulus)
    font = ImageFont.load_default(size=max(16, min(stimulus.size) // 40))
    line_width = max(3, min(stimulus.size) // 180)
    for object_key, label, color in (
        ("object_a", "A", OBJECT_A_COLOR),
        ("object_b", "B", OBJECT_B_COLOR),
    ):
        box = row[object_key]["box_2d_rend"]
        if not isinstance(box, list) or len(box) != 4:
            raise ValueError(f"Invalid box for {row['pair_id']}/{object_key}")
        numeric_box = [float(value) for value in box]
        draw.rectangle(tuple(numeric_box), outline=color, width=line_width)
        draw_label(draw, numeric_box, label, color, font)
    return stimulus


def save_png_atomic(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "wb",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".png",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        image.save(temporary, format="PNG")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> str:
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
    return sha256_file(path)


def write_json_atomic(path: Path, data: dict[str, Any]) -> None:
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
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def audit_title(row: dict[str, Any]) -> tuple[str, str]:
    first = (
        f"{row['pair_id']} | fold {row['fold']} | "
        f"{row['distance_stratum']} | GT={row['ground_truth_m']:.3f} m"
    )
    second = (
        f"A={row['object_a']['category']} | "
        f"B={row['object_b']['category']}"
    )
    return first, second


def build_scene_sheet(
    rendered: list[tuple[dict[str, Any], Image.Image]],
    columns: int,
    cell_width: int,
) -> Image.Image:
    if not rendered:
        raise ValueError("Cannot build an empty scene sheet")
    source_width, source_height = rendered[0][1].size
    thumbnail_height = max(1, round(cell_width * source_height / source_width))
    title_height = 48
    cell_height = title_height + thumbnail_height
    number_of_rows = math.ceil(len(rendered) / columns)
    sheet = Image.new(
        "RGB",
        (columns * cell_width, number_of_rows * cell_height),
        BACKGROUND_COLOR,
    )
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=12)
    for index, (row, image) in enumerate(rendered):
        column = index % columns
        sheet_row = index // columns
        x = column * cell_width
        y = sheet_row * cell_height
        first, second = audit_title(row)
        draw.text((x + 5, y + 5), first, fill=TEXT_COLOR, font=font)
        draw.text((x + 5, y + 23), second, fill=TEXT_COLOR, font=font)
        thumbnail = ImageOps.contain(image, (cell_width, thumbnail_height))
        paste_x = x + (cell_width - thumbnail.width) // 2
        paste_y = y + title_height + (thumbnail_height - thumbnail.height) // 2
        sheet.paste(thumbnail, (paste_x, paste_y))
    return sheet


def build_master_sheet(
    scene_sheets: list[tuple[str, Image.Image]], columns: int = 2
) -> Image.Image:
    if not scene_sheets:
        raise ValueError("Cannot build an empty master sheet")
    cell_width = 900
    preview_height = 1080
    title_height = 30
    cell_height = title_height + preview_height
    number_of_rows = math.ceil(len(scene_sheets) / columns)
    master = Image.new(
        "RGB", (columns * cell_width, number_of_rows * cell_height), BACKGROUND_COLOR
    )
    draw = ImageDraw.Draw(master)
    font = ImageFont.load_default(size=14)
    for index, (scene_id, sheet) in enumerate(scene_sheets):
        column = index % columns
        master_row = index // columns
        x = column * cell_width
        y = master_row * cell_height
        draw.text((x + 5, y + 7), f"Scene {scene_id}", fill=TEXT_COLOR, font=font)
        preview = ImageOps.contain(sheet, (cell_width, preview_height))
        paste_x = x + (cell_width - preview.width) // 2
        paste_y = y + title_height + (preview_height - preview.height) // 2
        master.paste(preview, (paste_x, paste_y))
    return master


def main() -> int:
    args = parse_args()
    project_root = args.project_root.expanduser().resolve()
    manifest_path = resolve_path(project_root, args.manifest)
    output_dir = resolve_path(project_root, args.output_dir)
    rendered_manifest_path = output_dir / "rendered_manifest.jsonl"
    protocol_path = output_dir / "render_protocol.json"
    master_sheet_path = output_dir / "contact_sheet_all.png"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    if args.columns <= 0 or args.cell_width < 240:
        raise ValueError("--columns must be positive and --cell-width at least 240")

    rows = load_jsonl(manifest_path)
    required = (
        "scene_id",
        "archive",
        "image_member",
        "pair_id",
        "pair_index",
        "fold",
        "distance_stratum",
        "ground_truth_m",
        "object_a",
        "object_b",
    )
    for row in rows:
        missing = [field for field in required if field not in row]
        if missing:
            raise KeyError(f"Missing stimulus fields: {', '.join(missing)}")
    pair_ids = [str(row["pair_id"]) for row in rows]
    if len(pair_ids) != len(set(pair_ids)):
        raise ValueError("Duplicate pair IDs in manifest")
    scene_counts = Counter(str(row["scene_id"]) for row in rows)
    unexpected = {
        scene: count for scene, count in scene_counts.items() if count != 12
    }
    if unexpected:
        raise ValueError(f"Expected 12 pairs per scene: {unexpected}")

    stimuli_paths = {
        str(row["pair_id"]): output_dir
        / "stimuli"
        / str(row["scene_id"])
        / f"{row['pair_id']}.png"
        for row in rows
    }
    scene_sheet_paths = {
        scene_id: output_dir / "contact_sheets" / f"{scene_id}.png"
        for scene_id in scene_counts
    }
    targets = (
        list(stimuli_paths.values())
        + list(scene_sheet_paths.values())
        + [rendered_manifest_path, protocol_path, master_sheet_path]
    )
    if not args.overwrite:
        existing = [path for path in targets if path.exists()]
        if existing:
            raise FileExistsError(
                "Generated destination exists; pass --overwrite: "
                + ", ".join(str(path) for path in existing[:3])
            )

    grouped: dict[tuple[Path, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        archive_path = resolve_path(project_root, Path(str(row["archive"])))
        key = (archive_path, str(row["scene_id"]), str(row["image_member"]))
        grouped[key].append(row)

    rendered_by_scene: dict[str, list[tuple[dict[str, Any], Image.Image]]] = defaultdict(list)
    rendered_manifest: list[dict[str, Any]] = []
    for (archive_path, scene_id, image_member), scene_rows in sorted(
        grouped.items(), key=lambda item: (item[0][1], item[0][2])
    ):
        if not archive_path.is_file():
            raise FileNotFoundError(f"Archive not found: {archive_path}")
        print(f"Rendering scene {scene_id} from {archive_path.name} ...")
        with tarfile.open(archive_path, "r") as archive:
            source = Image.open(BytesIO(read_member_bytes(archive, image_member)))
            source.load()
        for row in sorted(scene_rows, key=lambda value: int(value["pair_index"])):
            stimulus = render_stimulus(source, row)
            stimulus_path = stimuli_paths[str(row["pair_id"])]
            save_png_atomic(stimulus, stimulus_path)
            rendered_by_scene[scene_id].append((row, stimulus))
            rendered_row = dict(row)
            rendered_row["stimulus_path"] = display_path(
                project_root, stimulus_path
            )
            rendered_row["stimulus_sha256"] = sha256_file(stimulus_path)
            rendered_manifest.append(rendered_row)

    scene_sheets: list[tuple[str, Image.Image]] = []
    scene_sheet_hashes: dict[str, str] = {}
    for scene_id, rendered in sorted(rendered_by_scene.items()):
        rendered.sort(key=lambda item: int(item[0]["pair_index"]))
        sheet = build_scene_sheet(rendered, args.columns, args.cell_width)
        sheet_path = scene_sheet_paths[scene_id]
        save_png_atomic(sheet, sheet_path)
        scene_sheets.append((scene_id, sheet))
        scene_sheet_hashes[scene_id] = sha256_file(sheet_path)

    master = build_master_sheet(scene_sheets)
    save_png_atomic(master, master_sheet_path)
    rendered_manifest.sort(
        key=lambda row: (str(row["scene_id"]), int(row["pair_index"]))
    )
    rendered_manifest_sha256 = write_jsonl_atomic(
        rendered_manifest_path, rendered_manifest
    )
    protocol = {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_manifest": display_path(project_root, manifest_path),
        "source_manifest_sha256": sha256_file(manifest_path),
        "number_of_scenes": len(rendered_by_scene),
        "number_of_stimuli": len(rendered_manifest),
        "stimulus_annotations": {
            "object_a": {"label": "A", "color": OBJECT_A_COLOR},
            "object_b": {"label": "B", "color": OBJECT_B_COLOR},
            "ground_truth_in_stimulus": False,
            "categories_in_stimulus": False,
        },
        "human_audit_sheets": {
            "ground_truth_in_title": True,
            "categories_in_title": True,
            "scene_sheet_sha256": scene_sheet_hashes,
            "master_sheet": display_path(project_root, master_sheet_path),
            "master_sheet_sha256": sha256_file(master_sheet_path),
        },
        "rendered_manifest": display_path(project_root, rendered_manifest_path),
        "rendered_manifest_sha256": rendered_manifest_sha256,
    }
    write_json_atomic(protocol_path, protocol)

    print(f"Scenes: {len(rendered_by_scene)}")
    print(f"Stimuli: {len(rendered_manifest)}")
    print(f"Rendered manifest: {rendered_manifest_path}")
    print(f"Master contact sheet: {master_sheet_path}")
    print(f"Protocol: {protocol_path}")
    print(f"Master SHA-256: {protocol['human_audit_sheets']['master_sheet_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
