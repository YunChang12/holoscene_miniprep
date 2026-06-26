"""Instance mask normalization for HoloScene."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .writer import ensure_dir, reset_dir, write_json
from .wrappers.sam2_wrapper import SAM2Wrapper

LOGGER = logging.getLogger(__name__)


def load_provided_masks(provided_dir: str | Path, frame_count: int, resolution: tuple[int, int]) -> list[np.ndarray]:
    """Load provided masks, resize nearest, and return uint8 arrays."""

    root = Path(provided_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Mask directory not found: {root}")
    paths = [p for p in sorted(root.iterdir()) if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}]
    if len(paths) != frame_count:
        raise ValueError(f"Mask count {len(paths)} does not match frame count {frame_count}")
    width, height = resolution
    masks = []
    for path in paths:
        with Image.open(path) as im:
            mask = im.convert("L").resize((width, height), Image.Resampling.NEAREST)
            masks.append(np.asarray(mask, dtype=np.uint8))
    return masks


def generate_masks_with_sam2_placeholder(*args: Any, **kwargs: Any) -> list[np.ndarray]:
    """Call the SAM2 wrapper placeholder."""

    return SAM2Wrapper().run(*args, **kwargs)


def create_dummy_masks(frame_count: int, resolution: tuple[int, int], background_value: int = 255) -> list[np.ndarray]:
    """Create one rectangular foreground instance over most of each frame."""

    width, height = resolution
    masks = []
    margin_x = max(1, width // 20)
    margin_y = max(1, height // 20)
    for _ in range(frame_count):
        mask = np.full((height, width), int(background_value), dtype=np.uint8)
        mask[margin_y : height - margin_y, margin_x : width - margin_x] = 0
        masks.append(mask)
    return masks


def remap_masks_for_holoscene(
    masks: list[np.ndarray],
    background_value: int = 255,
    min_area_ratio: float = 0.001,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    """Ensure HoloScene mask convention: output background=255, object raw ids=0..N-1.

    ``background_value`` describes the background value in the source masks.
    The written HoloScene masks always use 255 for background.
    """

    if not masks:
        raise ValueError("No masks to remap")
    h, w = masks[0].shape
    valid_labels: set[int] = set()
    total_pixels = h * w * len(masks)
    counts: dict[int, int] = {}
    for mask in masks:
        labels, cnts = np.unique(mask, return_counts=True)
        for label, count in zip(labels.tolist(), cnts.tolist()):
            label_int = int(label)
            if label_int == int(background_value):
                continue
            counts[label_int] = counts.get(label_int, 0) + int(count)
    for label, count in counts.items():
        if count / float(total_pixels) >= float(min_area_ratio):
            valid_labels.add(label)
    ordered = sorted(valid_labels)
    label_to_raw = {label: idx for idx, label in enumerate(ordered)}
    remapped = []
    for mask in masks:
        out = np.full(mask.shape, 255, dtype=np.uint8)
        for old, new in label_to_raw.items():
            out[mask == old] = int(new)
        remapped.append(out)
    mapping = {
        "background_value": 255,
        "source_background_value": int(background_value),
        "objects": [
            {
                "source_label": int(old),
                "raw_mask_value": int(new),
                "holoscene_node_id": int(new + 1),
                "label": f"object_{new}",
            }
            for old, new in label_to_raw.items()
        ],
    }
    if not mapping["objects"]:
        LOGGER.warning("No foreground objects survived min_area_ratio; masks are all background")
    return remapped, mapping


def write_instance_masks(scene_dir: str | Path, masks: list[np.ndarray], overwrite: bool = True) -> Path:
    """Write instance_mask/frameXXXXXX.png files."""

    out_dir = reset_dir(Path(scene_dir) / "instance_mask") if overwrite else ensure_dir(Path(scene_dir) / "instance_mask")
    for idx, mask in enumerate(masks):
        Image.fromarray(mask.astype(np.uint8), mode="L").save(out_dir / f"frame{idx:06d}.png")
    return out_dir


def write_id_mapping(scene_dir: str | Path, mapping: dict[str, Any]) -> Path:
    """Write meta/id_mapping.json."""

    return write_json(Path(scene_dir) / "meta" / "id_mapping.json", mapping)


def load_id_mapping(scene_dir: str | Path) -> dict[str, Any]:
    """Load id mapping from scene meta."""

    return json.loads((Path(scene_dir) / "meta" / "id_mapping.json").read_text(encoding="utf-8"))
