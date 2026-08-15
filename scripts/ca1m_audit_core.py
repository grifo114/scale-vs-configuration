#!/usr/bin/env python3
"""Shared, explicit rules for auditing CA-1M frames."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import asdict, dataclass
from io import BytesIO
from typing import Any


@dataclass(frozen=True)
class AuditRules:
    box_field: str = "box_2d_rend"
    minimum_box_area_px2: float = 1000.0
    minimum_box_side_px: float = 20.0
    maximum_box_coverage: float = 0.5
    interior_margin_px: float = 5.0
    generic_categories: tuple[str, ...] = (
        "baseboard",
        "ceiling",
        "floor",
        "object",
        "wall",
    )
    category_uniqueness_scope: str = "all_instances"
    caption_required: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def finite_vector(value: Any, length: int) -> bool:
    return (
        isinstance(value, list)
        and len(value) == length
        and all(finite_number(item) for item in value)
    )


def geometric_valid(instance: dict[str, Any]) -> bool:
    position = instance.get("position")
    scale = instance.get("scale")
    corners = instance.get("corners")
    if not finite_vector(position, 3) or not finite_vector(scale, 3):
        return False
    if not all(float(value) > 0 for value in scale):
        return False
    if not isinstance(corners, list) or len(corners) != 8:
        return False
    return all(finite_vector(corner, 3) for corner in corners)


def normalized_category(instance: dict[str, Any]) -> str:
    return str(instance.get("category", "")).strip().casefold()


def valid_box(
    instance: dict[str, Any],
    box_field: str,
) -> tuple[float, float, float, float] | None:
    box = instance.get(box_field)
    if not finite_vector(box, 4):
        return None
    return tuple(float(value) for value in box)


def oracle_valid(
    instance: dict[str, Any],
    width: int,
    height: int,
    rules: AuditRules,
) -> bool:
    if not geometric_valid(instance) or width <= 0 or height <= 0:
        return False
    box = valid_box(instance, rules.box_field)
    if box is None:
        return False
    box_width = box[2] - box[0]
    box_height = box[3] - box[1]
    if (
        box_width < rules.minimum_box_side_px
        or box_height < rules.minimum_box_side_px
    ):
        return False
    area = box_width * box_height
    coverage = area / (width * height)
    return (
        area >= rules.minimum_box_area_px2
        and coverage <= rules.maximum_box_coverage
    )


def interior_valid(
    instance: dict[str, Any],
    width: int,
    height: int,
    rules: AuditRules,
) -> bool:
    box = valid_box(instance, rules.box_field)
    if box is None:
        return False
    margin = rules.interior_margin_px
    return (
        box[0] >= margin
        and box[1] >= margin
        and box[2] <= width - margin
        and box[3] <= height - margin
    )


def classify_instances(
    instances: list[dict[str, Any]],
    width: int,
    height: int,
    rules: AuditRules,
) -> dict[str, list[dict[str, Any]]]:
    geometric = [
        instance for instance in instances if geometric_valid(instance)
    ]
    oracle = [
        instance
        for instance in instances
        if oracle_valid(instance, width, height, rules)
    ]
    category_counts = Counter(
        normalized_category(instance) for instance in instances
    )
    generic = set(rules.generic_categories)
    natural: list[dict[str, Any]] = []
    for instance in oracle:
        category = normalized_category(instance)
        if not category or category in generic:
            continue
        if category_counts[category] != 1:
            continue
        if rules.caption_required and not str(
            instance.get("caption", "")
        ).strip():
            continue
        natural.append(instance)
    natural_interior = [
        instance
        for instance in natural
        if interior_valid(instance, width, height, rules)
    ]
    return {
        "geometric": geometric,
        "oracle": oracle,
        "natural": natural,
        "natural_interior": natural_interior,
    }


def sharpness_variance_of_laplacian(image_bytes: bytes) -> tuple[int, int, float]:
    import cv2
    import numpy as np
    from PIL import Image

    with Image.open(BytesIO(image_bytes)) as image:
        rgb = np.asarray(image.convert("RGB"))
    height, width = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return width, height, sharpness
