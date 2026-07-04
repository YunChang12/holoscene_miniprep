#!/usr/bin/env python3
"""Prepare aligned Wild_SLAM_Mocap depth and camera inputs for MiniPrep."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


def main() -> int:
    parser = argparse.ArgumentParser(description="Create strided Wild_SLAM_Mocap provided inputs.")
    parser.add_argument("--dataset-root", required=True, help="Scene directory containing rgb/depth txt files.")
    parser.add_argument("--output-dir", required=True, help="Directory for generated provided inputs.")
    parser.add_argument("--stride", type=int, required=True, help="Frame stride over rgb.txt/depth.txt/groundtruth.txt.")
    parser.add_argument("--resolution", nargs=2, type=int, metavar=("WIDTH", "HEIGHT"), required=True)
    parser.add_argument("--max-frames", type=int, default=None, help="Optional cap after stride sampling.")
    parser.add_argument("--overwrite", action="store_true", help="Remove output-dir before writing.")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if args.stride <= 0:
        raise ValueError("--stride must be positive")
    width, height = int(args.resolution[0]), int(args.resolution[1])
    if width <= 0 or height <= 0:
        raise ValueError("--resolution must contain positive width and height")
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output directory exists: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rgb_rows = _read_table(dataset_root / "rgb.txt", expected_min_cols=2)
    depth_rows = _read_table(dataset_root / "depth.txt", expected_min_cols=2)
    gt_rows = _read_table(dataset_root / "groundtruth.txt", expected_min_cols=8)
    if not (len(rgb_rows) == len(depth_rows) == len(gt_rows)):
        raise ValueError(f"Count mismatch: rgb={len(rgb_rows)} depth={len(depth_rows)} gt={len(gt_rows)}")

    intrinsics = json.loads((dataset_root / "intrinsics.json").read_text(encoding="utf-8"))
    color = intrinsics["color"]
    depth_scale = float(intrinsics.get("depth_scale", 1.0))
    source_width = int(color["width"])
    source_height = int(color["height"])
    sx = width / float(source_width)
    sy = height / float(source_height)

    selected_indices = list(range(0, len(rgb_rows), args.stride))
    if args.max_frames:
        selected_indices = selected_indices[: int(args.max_frames)]
    depth_out = output_dir / "depth"
    depth_out.mkdir(parents=True, exist_ok=True)

    frames = []
    manifest_frames = []
    for out_idx, source_idx in enumerate(selected_indices):
        rgb = rgb_rows[source_idx]
        depth = depth_rows[source_idx]
        gt = gt_rows[source_idx]
        timestamp = rgb[0]
        if depth[0] != timestamp or gt[0] != timestamp:
            raise ValueError(
                f"Timestamp mismatch at row {source_idx}: rgb={timestamp} depth={depth[0]} gt={gt[0]}"
            )

        depth_path = dataset_root / depth[1]
        depth_m = _load_depth_meters(depth_path, width=width, height=height, depth_scale=depth_scale)
        np.save(depth_out / f"frame{out_idx:06d}.npy", depth_m)

        tx, ty, tz = (float(gt[1]), float(gt[2]), float(gt[3]))
        qx, qy, qz, qw = (float(gt[4]), float(gt[5]), float(gt[6]), float(gt[7]))
        transform = _pose_matrix(tx, ty, tz, qx, qy, qz, qw)
        frames.append({"file_path": f"images/frame{out_idx:06d}.jpg", "transform_matrix": transform})
        manifest_frames.append(
            {
                "frame_index": out_idx,
                "source_frame_index": source_idx,
                "timestamp": float(timestamp),
                "rgb_path": rgb[1],
                "depth_path": depth[1],
            }
        )

    transforms = {
        "camera_model": "OPENCV",
        "fl_x": float(color["fx"]) * sx,
        "fl_y": float(color["fy"]) * sy,
        "cx": float(color["ppx"]) * sx,
        "cy": float(color["ppy"]) * sy,
        "k1": float(color.get("coeffs", [0.0])[0]) if len(color.get("coeffs", [])) > 0 else 0.0,
        "k2": float(color.get("coeffs", [0.0, 0.0])[1]) if len(color.get("coeffs", [])) > 1 else 0.0,
        "p1": float(color.get("coeffs", [0.0, 0.0, 0.0])[2]) if len(color.get("coeffs", [])) > 2 else 0.0,
        "p2": float(color.get("coeffs", [0.0, 0.0, 0.0, 0.0])[3]) if len(color.get("coeffs", [])) > 3 else 0.0,
        "k3": float(color.get("coeffs", [0.0, 0.0, 0.0, 0.0, 0.0])[4])
        if len(color.get("coeffs", [])) > 4
        else 0.0,
        "w": width,
        "h": height,
        "frames": frames,
    }
    (output_dir / "transforms.json").write_text(json.dumps(transforms, indent=2), encoding="utf-8")
    manifest = {
        "dataset_root": str(dataset_root),
        "stride": int(args.stride),
        "max_frames": args.max_frames,
        "source_frame_count": len(rgb_rows),
        "frame_count": len(selected_indices),
        "source_resolution": [source_width, source_height],
        "output_resolution": [width, height],
        "depth_scale": depth_scale,
        "frames": manifest_frames,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"provided_dir={output_dir}")
    print(f"frame_count={len(selected_indices)}")
    return 0


def _read_table(path: Path, *, expected_min_cols: int) -> list[list[str]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < expected_min_cols:
            raise ValueError(f"Expected at least {expected_min_cols} columns in {path}: {line}")
        rows.append(parts)
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def _load_depth_meters(path: Path, *, width: int, height: int, depth_scale: float) -> np.ndarray:
    with Image.open(path) as image:
        depth_raw = image.resize((width, height), Image.Resampling.NEAREST)
        arr = np.asarray(depth_raw, dtype=np.float32)
    depth_m = arr * float(depth_scale)
    depth_m[arr <= 0] = np.nan
    return depth_m.astype(np.float32)


def _pose_matrix(tx: float, ty: float, tz: float, qx: float, qy: float, qz: float, qw: float) -> list[list[float]]:
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm <= 1e-12:
        raise ValueError("Quaternion norm is zero")
    qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm
    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz
    return [
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy), tx],
        [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx), ty],
        [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy), tz],
        [0.0, 0.0, 0.0, 1.0],
    ]


if __name__ == "__main__":
    raise SystemExit(main())
