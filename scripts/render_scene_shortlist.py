#!/usr/bin/env python3
"""Render annotated CA-1M scene candidates and a contact sheet."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tarfile
import tempfile
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps


INTERIOR_COLOR = "#00B86B"
NEAR_BORDER_COLOR = "#F28E1C"
TITLE_COLOR = "#111111"
BACKGROUND_COLOR = "#FFFFFF"


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Render a pair-feasibility shortlist with natural-object boxes."
    )
    parser.add_argument("feasibility", type=Path, help="Pair-feasibility JSONL.")
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
        help="Directory for annotated images and render_manifest.json.",
    )
    parser.add_argument(
        "--contact-sheet",
        type=Path,
        help="Contact-sheet PNG. Defaults to <output-dir>/contact_sheet.png.",
    )
    parser.add_argument(
        "--selected-only",
        action="store_true",
        help="Render only candidate_rank=1 frames.",
    )
    parser.add_argument(
        "--columns",
        type=int,
        default=3,
        help="Number of contact-sheet columns.",
    )
    parser.add_argument(
        "--cell-width",
        type=int,
        default=480,
        help="Contact-sheet cell width in pixels.",
    )
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


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    return right - left, bottom - top


def draw_label(
    draw: ImageDraw.ImageDraw,
    x: float,
    y: float,
    text: str,
    color: str,
    font: ImageFont.ImageFont,
) -> None:
    width, height = text_size(draw, text, font)
    left = max(0, int(round(x)))
    top = max(0, int(round(y)) - height - 4)
    draw.rectangle((left, top, left + width + 6, top + height + 4), fill=color)
    draw.text((left + 3, top + 2), text, fill="white", font=font)


def annotate_scene(image: Image.Image, row: dict[str, Any]) -> Image.Image:
    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    font = ImageFont.load_default()
    line_width = max(2, min(annotated.size) // 250)
    for obj in row["natural_objects"]:
        box = obj["box_2d_rend"]
        if not isinstance(box, list) or len(box) != 4:
            raise ValueError("Invalid box_2d_rend in natural_objects")
        color = INTERIOR_COLOR if obj["interior"] else NEAR_BORDER_COLOR
        draw.rectangle(tuple(float(value) for value in box), outline=color, width=line_width)
        label = f"{obj['instance_index']} {obj['category']}"
        draw_label(draw, float(box[0]), float(box[1]), label, color, font)
    return annotated


def candidate_title(row: dict[str, Any]) -> str:
    return (
        f"{row['capture_id']} | rank {row['candidate_rank']} | "
        f"frame {row['frame_index']} | bin {row['temporal_bin']} | "
        f"N={row['natural']} I={row['natural_interior']} "
        f"P={row['number_of_eligible_pairs']}"
    )


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


def build_contact_sheet(
    rendered: list[tuple[dict[str, Any], Image.Image]],
    columns: int,
    cell_width: int,
) -> Image.Image:
    if not rendered:
        raise ValueError("No rendered candidates for the contact sheet")
    first_width, first_height = rendered[0][1].size
    image_height = max(1, round(cell_width * first_height / first_width))
    title_height = 48
    cell_height = title_height + image_height
    rows = math.ceil(len(rendered) / columns)
    sheet = Image.new(
        "RGB", (columns * cell_width, rows * cell_height), BACKGROUND_COLOR
    )
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for index, (row, image) in enumerate(rendered):
        column = index % columns
        sheet_row = index // columns
        x = column * cell_width
        y = sheet_row * cell_height
        title = candidate_title(row)
        draw.text((x + 6, y + 6), title, fill=TITLE_COLOR, font=font)
        legend = "green=interior | orange=within 5 px of border"
        draw.text((x + 6, y + 24), legend, fill=TITLE_COLOR, font=font)
        thumbnail = ImageOps.contain(image, (cell_width, image_height))
        paste_x = x + (cell_width - thumbnail.width) // 2
        paste_y = y + title_height + (image_height - thumbnail.height) // 2
        sheet.paste(thumbnail, (paste_x, paste_y))
    return sheet


def main() -> int:
    args = parse_args()
    project_root = args.project_root.expanduser().resolve()
    feasibility_path = resolve_path(project_root, args.feasibility)
    output_dir = resolve_path(project_root, args.output_dir)
    contact_sheet_path = (
        resolve_path(project_root, args.contact_sheet)
        if args.contact_sheet
        else output_dir / "contact_sheet.png"
    )
    manifest_path = output_dir / "render_manifest.json"
    if not feasibility_path.is_file():
        raise FileNotFoundError(f"Feasibility file not found: {feasibility_path}")
    if args.columns <= 0 or args.cell_width < 200:
        raise ValueError("--columns must be positive and --cell-width at least 200")

    rows = load_jsonl(feasibility_path)
    if args.selected_only:
        rows = [row for row in rows if bool(row.get("selected_by_frame_rule"))]
    required = (
        "archive",
        "capture_id",
        "candidate_rank",
        "frame_index",
        "temporal_bin",
        "image_member",
        "natural_objects",
        "natural",
        "natural_interior",
        "number_of_eligible_pairs",
    )
    for row in rows:
        missing = [field for field in required if field not in row]
        if missing:
            raise KeyError(f"Missing render fields: {', '.join(missing)}")
    rows.sort(key=lambda row: (str(row["capture_id"]), int(row["candidate_rank"])))

    generated_paths = [
        output_dir
        / f"{row['capture_id']}-r{int(row['candidate_rank']):02d}"
        f"-f{int(row['frame_index']):04d}.png"
        for row in rows
    ]
    targets = generated_paths + [contact_sheet_path, manifest_path]
    if not args.overwrite:
        existing = [path for path in targets if path.exists()]
        if existing:
            raise FileExistsError(
                "Generated destination exists; pass --overwrite: "
                + ", ".join(str(path) for path in existing[:3])
            )

    grouped: dict[Path, list[tuple[dict[str, Any], Path]]] = defaultdict(list)
    for row, generated_path in zip(rows, generated_paths):
        archive_path = resolve_path(project_root, Path(str(row["archive"])))
        grouped[archive_path].append((row, generated_path))
    rendered: list[tuple[dict[str, Any], Image.Image]] = []
    manifest_rows: list[dict[str, Any]] = []
    for archive_path, archive_rows in sorted(grouped.items(), key=lambda item: str(item[0])):
        if not archive_path.is_file():
            raise FileNotFoundError(f"Archive not found: {archive_path}")
        print(f"Reading {archive_path.name} ...")
        with tarfile.open(archive_path, "r") as archive:
            for row, generated_path in archive_rows:
                source = Image.open(BytesIO(read_member_bytes(archive, str(row["image_member"]))))
                source.load()
                annotated = annotate_scene(source, row)
                save_png_atomic(annotated, generated_path)
                rendered.append((row, annotated))
                manifest_rows.append(
                    {
                        "capture_id": str(row["capture_id"]),
                        "candidate_rank": int(row["candidate_rank"]),
                        "frame_index": int(row["frame_index"]),
                        "image_member": str(row["image_member"]),
                        "rendered_image": display_path(project_root, generated_path),
                        "sha256": sha256_file(generated_path),
                    }
                )
    rendered.sort(key=lambda item: (str(item[0]["capture_id"]), int(item[0]["candidate_rank"])))
    sheet = build_contact_sheet(rendered, args.columns, args.cell_width)
    save_png_atomic(sheet, contact_sheet_path)
    manifest = {
        "schema_version": "1.0",
        "source": display_path(project_root, feasibility_path),
        "source_sha256": sha256_file(feasibility_path),
        "selected_only": args.selected_only,
        "number_of_images": len(rendered),
        "box_colors": {
            "interior": INTERIOR_COLOR,
            "within_5px_of_border": NEAR_BORDER_COLOR,
        },
        "images": manifest_rows,
        "contact_sheet": display_path(project_root, contact_sheet_path),
        "contact_sheet_sha256": sha256_file(contact_sheet_path),
    }
    write_json_atomic(manifest_path, manifest)

    print(f"Rendered images: {len(rendered)}")
    print(f"Output directory: {output_dir}")
    print(f"Contact sheet: {contact_sheet_path}")
    print(f"Manifest: {manifest_path}")
    print(f"Contact-sheet SHA-256: {manifest['contact_sheet_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
