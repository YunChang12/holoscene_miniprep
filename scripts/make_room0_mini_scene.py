#!/usr/bin/env python3
"""Create a small immutable test scene from HoloScene Replica room_0."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from holoprep.writer import ensure_dir, write_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Sample HoloScene Replica room_0 into a MiniPrep test scene.")
    parser.add_argument(
        "--source_scene",
        default="/root/autodl-fs/Zaiwu/third_party/HoloScene/data_dir/replica/room_0",
        help="Path to HoloScene data_dir/replica/room_0.",
    )
    parser.add_argument(
        "--output_scene",
        default="tmp_tests/room0_mini",
        help="Output mini scene path. This directory is recreated.",
    )
    parser.add_argument("--num_frames", type=int, default=30, help="Number of sampled frames.")
    parser.add_argument("--start_index", type=int, default=0, help="First source transform index to sample.")
    parser.add_argument("--stride", type=int, default=5, help="Stride over official transforms.frames.")
    parser.add_argument(
        "--resolution",
        type=int,
        nargs=2,
        default=[512, 512],
        metavar=("WIDTH", "HEIGHT"),
        help="Output image resolution.",
    )
    args = parser.parse_args()

    source_scene = Path(args.source_scene).expanduser().resolve()
    output_scene = Path(args.output_scene).expanduser()
    if not output_scene.is_absolute():
        output_scene = (PROJECT_ROOT / output_scene).resolve()
    make_room0_mini_scene(
        source_scene=source_scene,
        output_scene=output_scene,
        num_frames=int(args.num_frames),
        start_index=int(args.start_index),
        stride=int(args.stride),
        resolution=(int(args.resolution[0]), int(args.resolution[1])),
    )
    print(f"output_scene={output_scene}")
    return 0


def make_room0_mini_scene(
    *,
    source_scene: Path,
    output_scene: Path,
    num_frames: int,
    start_index: int,
    stride: int,
    resolution: tuple[int, int],
) -> None:
    if num_frames <= 0:
        raise ValueError("--num_frames must be positive")
    if stride <= 0:
        raise ValueError("--stride must be positive")
    if start_index < 0:
        raise ValueError("--start_index must be non-negative")
    transforms_path = source_scene / "transforms.json"
    images_root = source_scene / "images"
    if not transforms_path.is_file():
        raise FileNotFoundError(f"Missing source transforms.json: {transforms_path}")
    if not images_root.is_dir():
        raise FileNotFoundError(f"Missing source images directory: {images_root}")

    data = json.loads(transforms_path.read_text(encoding="utf-8"))
    frames = data.get("frames")
    if not isinstance(frames, list) or not frames:
        raise RuntimeError(f"Source transforms has no frames: {transforms_path}")
    selected = list(enumerate(frames))[start_index::stride][:num_frames]
    if len(selected) < num_frames:
        raise RuntimeError(f"Only selected {len(selected)} frames, requested {num_frames}")

    if output_scene.exists():
        shutil.rmtree(output_scene)
    images_out = ensure_dir(output_scene / "images")
    meta_out = ensure_dir(output_scene / "meta")

    width, height = resolution
    src_w = int(data.get("w") or width)
    src_h = int(data.get("h") or height)
    sx = float(width) / max(float(src_w), 1e-6)
    sy = float(height) / max(float(src_h), 1e-6)

    out_frames: list[dict[str, Any]] = []
    mapping: dict[str, Any] = {}
    for out_idx, (source_transform_index, frame) in enumerate(selected):
        rel = str(frame.get("file_path", ""))
        source_file = source_scene / rel
        if not source_file.is_file():
            raise FileNotFoundError(f"Frame file from transforms does not exist: {source_file}")
        out_name = f"frame{out_idx:06d}.jpg"
        with Image.open(source_file) as im:
            rgb = im.convert("RGB")
            original_size = [rgb.width, rgb.height]
            if rgb.size != (width, height):
                rgb = rgb.resize((width, height), Image.Resampling.BILINEAR)
            rgb.save(images_out / out_name, quality=95)

        out_frames.append(
            {
                "file_path": f"images/{out_name}",
                "transform_matrix": frame["transform_matrix"],
            }
        )
        mapping[f"frame{out_idx:06d}"] = {
            "source_file": rel,
            "source_index": _frame_index_from_name(rel),
            "source_transform_index": int(source_transform_index),
            "original_size": original_size,
            "output_size": [int(width), int(height)],
        }

    out_transforms = {
        "camera_model": data.get("camera_model", "OPENCV"),
        "fl_x": float(data.get("fl_x", width)) * sx,
        "fl_y": float(data.get("fl_y", height)) * sy,
        "cx": float(data.get("cx", src_w / 2.0)) * sx,
        "cy": float(data.get("cy", src_h / 2.0)) * sy,
        "h": int(height),
        "w": int(width),
        "frames": out_frames,
    }
    write_json(output_scene / "transforms_official.json", out_transforms)
    write_json(meta_out / "source_frame_mapping.json", mapping)


def _frame_index_from_name(path: str) -> int | None:
    match = re.search(r"(\d+)(?=\.[^.]+$)", Path(path).name)
    return int(match.group(1)) if match else None


if __name__ == "__main__":
    raise SystemExit(main())
