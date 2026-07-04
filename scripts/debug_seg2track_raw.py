#!/usr/bin/env python3
"""Visualize raw Seg2Track instances before MiniPrep mask composition."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from holoprep.writer import ensure_dir, write_json


PALETTE = np.asarray(
    [
        [255, 80, 80],
        [80, 255, 120],
        [80, 160, 255],
        [255, 220, 80],
        [220, 80, 255],
        [80, 255, 240],
        [255, 150, 80],
        [150, 80, 255],
        [120, 220, 120],
        [220, 120, 120],
        [120, 120, 220],
        [220, 220, 120],
    ],
    dtype=np.uint8,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Debug raw Seg2Track masks for one or more frames.")
    parser.add_argument("--scene_dir", required=True, help="MiniPrep scene directory.")
    parser.add_argument("--frames", default="all", help="Comma-separated 0-based frame ids, ranges like 3-8, or all.")
    parser.add_argument("--max_instances", type=int, default=80, help="Maximum raw instances to draw per frame.")
    args = parser.parse_args()

    scene = Path(args.scene_dir).expanduser().resolve()
    result_path = scene / "raw_outputs" / "seg2track_sam2" / "result.json"
    if not result_path.is_file():
        raise SystemExit(f"Missing raw Seg2Track result: {result_path}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    frame_ids = _parse_frames(args.frames, result)

    out_dir = ensure_dir(scene / "review" / "seg2track_raw_debug")
    summaries = []
    for frame_id in frame_ids:
        frame = _find_result_frame(result, frame_id)
        if frame is None:
            continue
        summaries.append(_debug_frame(scene, frame, frame_id, out_dir, max_instances=int(args.max_instances)))
    report = {
        "scene_dir": str(scene),
        "result_path": str(result_path),
        "frames": summaries,
    }
    write_json(scene / "meta" / "seg2track_raw_debug_report.json", report)
    print(json.dumps({"out_dir": str(out_dir), "frames": len(summaries)}, ensure_ascii=False, indent=2))
    return 0


def _parse_frames(value: str, result: dict[str, Any]) -> list[int]:
    value = str(value).strip().lower()
    if value == "all":
        ids = []
        for frame in result.get("frames", []):
            raw_idx = int(frame.get("frame_idx", frame.get("frame_index", 0)))
            ids.append(raw_idx - 1 if raw_idx >= 1 else raw_idx)
        return sorted(set(ids))
    out: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return sorted(out)


def _find_result_frame(result: dict[str, Any], frame_id: int) -> dict[str, Any] | None:
    for frame in result.get("frames", []):
        raw_idx = int(frame.get("frame_idx", frame.get("frame_index", 0)))
        idx = raw_idx - 1 if raw_idx >= 1 else raw_idx
        if idx == int(frame_id):
            return frame
    return None


def _debug_frame(scene: Path, frame: dict[str, Any], frame_id: int, out_dir: Path, *, max_instances: int) -> dict[str, Any]:
    image_path = scene / "images" / f"frame{frame_id:06d}.jpg"
    final_mask_path = scene / "instance_mask" / f"frame{frame_id:06d}.png"
    with Image.open(image_path) as image:
        image = image.convert("RGB")
    final_mask = np.asarray(Image.open(final_mask_path).convert("L"), dtype=np.uint8) if final_mask_path.is_file() else None

    instances = [inst for inst in frame.get("instances", []) if isinstance(inst, dict)]
    decoded = []
    empty_or_missing_mask = []
    for idx, inst in enumerate(instances):
        mask = _decode_rle(inst.get("mask_rle"))
        if mask is None:
            empty_or_missing_mask.append(_instance_row(inst, idx, area_pixels=0, survived_pixels=0, has_mask=False))
            continue
        survived_pixels = 0
        if final_mask is not None:
            survived_pixels = int(np.count_nonzero(mask & (final_mask != 255)))
        row = _instance_row(inst, idx, area_pixels=int(mask.sum()), survived_pixels=survived_pixels, has_mask=bool(mask.any()))
        if not mask.any():
            empty_or_missing_mask.append(row)
            continue
        row["mask"] = mask
        decoded.append(row)
    decoded.sort(key=lambda item: float(item["score"]), reverse=True)
    drawn = decoded[:max_instances]

    raw_overlay = _overlay_instances(image, drawn, empty_or_missing_mask, draw_bbox=True)
    score_overlay = _score_order_overlay(image, drawn)
    final_overlay = _overlay_final(image, final_mask) if final_mask is not None else np.asarray(image)
    grid_path = out_dir / f"frame{frame_id:06d}_raw_grid.jpg"
    _write_grid(
        [np.asarray(image), raw_overlay, final_overlay, score_overlay],
        ["image", "raw masks+bbox; white boxes=no valid mask", "final MiniPrep mask", "score-order raw masks"],
        grid_path,
        footer=(
            f"frame{frame_id:06d} instances={len(instances)} valid_masks={len(decoded)} "
            f"empty_or_missing_masks={len(empty_or_missing_mask)} drawn={len(drawn)}"
        ),
    )

    inst_dir = ensure_dir(out_dir / f"frame{frame_id:06d}_instances")
    rows = []
    for item in drawn:
        single = _overlay_instances(image, [item], [], draw_bbox=True)
        name = f"{item['index']:03d}_{_safe_name(item['track_id'])}_{_safe_name(item['concept_label'])}.jpg"
        Image.fromarray(single).save(inst_dir / name, quality=92)
        row = {k: v for k, v in item.items() if k != "mask"}
        rows.append(row)
    rows.extend(empty_or_missing_mask)
    write_json(out_dir / f"frame{frame_id:06d}_instances.json", rows)
    return {
        "frame": int(frame_id),
        "raw_instance_count": len(instances),
        "valid_mask_count": len(decoded),
        "empty_or_missing_mask_count": len(empty_or_missing_mask),
        "drawn_instance_count": len(drawn),
        "grid": str(grid_path),
        "instances_dir": str(inst_dir),
        "instances": rows,
    }


def _instance_row(
    inst: dict[str, Any],
    idx: int,
    *,
    area_pixels: int,
    survived_pixels: int,
    has_mask: bool,
) -> dict[str, Any]:
    return {
        "index": idx,
        "track_id": str(inst.get("track_id", "")),
        "concept_label": str(inst.get("concept_label", inst.get("label", ""))),
        "score": float(inst.get("score", 0.0) or 0.0),
        "bbox": inst.get("bbox"),
        "area_pixels": int(area_pixels),
        "survived_pixels_in_final_foreground": int(survived_pixels),
        "has_valid_mask": bool(has_mask),
    }


def _decode_rle(mask_rle: Any) -> np.ndarray | None:
    if not mask_rle:
        return None
    from pycocotools import mask as mask_util  # type: ignore

    if isinstance(mask_rle, str):
        obj = json.loads(mask_rle)
    elif isinstance(mask_rle, dict):
        obj = dict(mask_rle)
    else:
        return None
    if isinstance(obj.get("counts"), str):
        obj["counts"] = obj["counts"].encode("utf-8")
    mask = mask_util.decode(obj).astype(bool)
    if mask.ndim == 3:
        mask = mask[..., 0]
    return mask


def _overlay_instances(
    image: Image.Image,
    instances: list[dict[str, Any]],
    empty_instances: list[dict[str, Any]],
    *,
    draw_bbox: bool,
) -> np.ndarray:
    base = np.asarray(image, dtype=np.float32)
    out = base.copy()
    for idx, item in enumerate(instances):
        color = PALETTE[idx % len(PALETTE)].astype(np.float32)
        mask = item["mask"]
        out[mask] = out[mask] * 0.55 + color * 0.45
    result = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(result)
    if draw_bbox:
        for idx, item in enumerate(instances):
            bbox = item.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            color = tuple(int(v) for v in PALETTE[idx % len(PALETTE)])
            draw.rectangle([float(v) for v in bbox], outline=color, width=2)
            label = f"{item['track_id']} {item['concept_label']} {item['score']:.2f}"
            x0, y0 = float(bbox[0]), float(bbox[1])
            draw.rectangle([x0, max(0, y0 - 14), x0 + min(260, 7 * len(label) + 8), y0], fill=(0, 0, 0))
            draw.text((x0 + 3, max(0, y0 - 13)), label, fill=(255, 255, 255))
        for item in empty_instances:
            bbox = item.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            draw.rectangle([float(v) for v in bbox], outline=(255, 255, 255), width=2)
            label = f"{item['track_id']} {item['concept_label']} no-mask {item['score']:.2f}"
            x0, y0 = float(bbox[0]), float(bbox[1])
            draw.rectangle([x0, max(0, y0 - 14), x0 + min(300, 7 * len(label) + 8), y0], fill=(255, 255, 255))
            draw.text((x0 + 3, max(0, y0 - 13)), label, fill=(0, 0, 0))
    return np.asarray(result)


def _score_order_overlay(image: Image.Image, instances: list[dict[str, Any]]) -> np.ndarray:
    base = np.asarray(image, dtype=np.float32) * 0.35
    out = base.copy()
    for idx, item in enumerate(reversed(instances)):
        color = PALETTE[(len(instances) - idx - 1) % len(PALETTE)].astype(np.float32)
        out[item["mask"]] = color
    return np.clip(out, 0, 255).astype(np.uint8)


def _overlay_final(image: Image.Image, final_mask: np.ndarray | None) -> np.ndarray:
    if final_mask is None:
        return np.asarray(image)
    base = np.asarray(image, dtype=np.float32)
    out = base.copy()
    for label in sorted(int(v) for v in np.unique(final_mask) if int(v) != 255):
        color = PALETTE[label % len(PALETTE)].astype(np.float32)
        out[final_mask == label] = out[final_mask == label] * 0.55 + color * 0.45
    return np.clip(out, 0, 255).astype(np.uint8)


def _write_grid(panels: list[np.ndarray], titles: list[str], path: Path, *, footer: str) -> None:
    h, w = panels[0].shape[:2]
    title_h = 30
    footer_h = 24
    canvas = Image.new("RGB", (w * len(panels), h + title_h + footer_h), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, (panel, title) in enumerate(zip(panels, titles)):
        x = idx * w
        canvas.paste(Image.fromarray(panel.astype(np.uint8)), (x, title_h))
        draw.rectangle([x, 0, x + w, title_h], fill=(245, 245, 245))
        draw.text((x + 8, 8), title, fill=(0, 0, 0))
    draw.text((8, h + title_h + 5), footer, fill=(0, 0, 0))
    canvas.save(path, quality=92)


def _safe_name(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(value).strip())
    return cleaned[:60] or "item"


if __name__ == "__main__":
    raise SystemExit(main())
