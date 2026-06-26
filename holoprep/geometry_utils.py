"""Geometry utilities for instance point clouds and bboxes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .writer import ensure_dir, reset_dir, write_json


def backproject_depth_to_points(depth: np.ndarray, fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """Backproject a depth map to camera-space points of shape H,W,3."""

    h, w = depth.shape
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    z = depth.astype(np.float32)
    x = (xs - float(cx)) * z / float(fx)
    y = (ys - float(cy)) * z / float(fy)
    return np.stack([x, y, z], axis=-1)


def transform_points_to_world(points: np.ndarray, transform_matrix: list[list[float]]) -> np.ndarray:
    """Transform Nx3 camera-space points to world-space using camera-to-world pose."""

    pts = points.reshape(-1, 3)
    pose = np.asarray(transform_matrix, dtype=np.float32)
    homog = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float32)], axis=1)
    world = (pose @ homog.T).T[:, :3]
    return world.reshape(points.shape)


def build_instance_pointclouds(
    scene_dir: str | Path,
    transforms: dict[str, Any],
    id_mapping: dict[str, Any],
    max_points_per_instance: int = 200000,
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    """Build world-space point clouds for each raw object id.

    ``transforms.json`` is treated as camera-to-world, matching the NeRF-style
    convention expected by HoloScene. If a provided dataset uses world-to-camera
    matrices, convert them before writing transforms.json.
    """

    scene = Path(scene_dir)
    fx = float(transforms["fl_x"])
    fy = float(transforms["fl_y"])
    cx = float(transforms["cx"])
    cy = float(transforms["cy"])
    object_raw_ids = [int(obj["raw_mask_value"]) for obj in id_mapping.get("objects", [])]
    clouds: dict[int, list[np.ndarray]] = {rid: [] for rid in object_raw_ids}
    visible_counts: dict[int, int] = {rid: 0 for rid in object_raw_ids}
    for idx, frame in enumerate(transforms.get("frames", [])):
        depth = np.load(scene / "depth" / f"frame{idx:06d}.npy")
        with Image.open(scene / "instance_mask" / f"frame{idx:06d}.png") as im:
            mask = np.asarray(im.convert("L"), dtype=np.uint8)
        cam_points = backproject_depth_to_points(depth, fx, fy, cx, cy)
        world_points = transform_points_to_world(cam_points, frame["transform_matrix"])
        for rid in object_raw_ids:
            selected = world_points[mask == rid]
            if selected.size:
                visible_counts[rid] += 1
                if selected.shape[0] > 5000:
                    step = max(1, selected.shape[0] // 5000)
                    selected = selected[::step]
                clouds[rid].append(selected.astype(np.float32))
    out_clouds: dict[int, np.ndarray] = {}
    for rid, chunks in clouds.items():
        if chunks:
            pts = np.concatenate(chunks, axis=0)
            if pts.shape[0] > max_points_per_instance:
                idxs = np.linspace(0, pts.shape[0] - 1, max_points_per_instance).astype(np.int64)
                pts = pts[idxs]
            out_clouds[rid] = pts.astype(np.float32)
        else:
            out_clouds[rid] = np.zeros((0, 3), dtype=np.float32)
    meta = {"visible_frame_counts": {str(k): int(v) for k, v in visible_counts.items()}}
    return out_clouds, meta


def estimate_instance_bbox(clouds: dict[int, np.ndarray], id_mapping: dict[str, Any], visible_meta: dict[str, Any]) -> dict[str, Any]:
    """Estimate bbox statistics for each instance cloud."""

    objects = []
    raw_to_info = {int(obj["raw_mask_value"]): obj for obj in id_mapping.get("objects", [])}
    visible_counts = visible_meta.get("visible_frame_counts", {})
    for rid, pts in clouds.items():
        info = raw_to_info.get(rid, {})
        if pts.shape[0] == 0:
            bbox_min = bbox_max = center = [0.0, 0.0, 0.0]
            xy_extent = [0.0, 0.0]
        else:
            min_v = pts.min(axis=0)
            max_v = pts.max(axis=0)
            bbox_min = min_v.astype(float).tolist()
            bbox_max = max_v.astype(float).tolist()
            center = ((min_v + max_v) * 0.5).astype(float).tolist()
            xy_extent = (max_v[:2] - min_v[:2]).astype(float).tolist()
        loaded_id = int(info.get("holoscene_node_id", rid + 1))
        objects.append(
            {
                "raw_mask_value": int(rid),
                "holoscene_node_id": loaded_id,
                "loaded_instance_id": loaded_id,
                "label": info.get("label", f"object_{rid}"),
                "num_points": int(pts.shape[0]),
                "point_count": int(pts.shape[0]),
                "visible_frame_count": int(visible_counts.get(str(rid), 0)),
                "visible_frames": int(visible_counts.get(str(rid), 0)),
                "bbox_min": bbox_min,
                "bbox_max": bbox_max,
                "z_min": float(bbox_min[2]),
                "z_max": float(bbox_max[2]),
                "center": center,
                "xy_extent": xy_extent,
            }
        )
    return {"objects": objects}


def write_instance_clouds(scene_dir: str | Path, clouds: dict[int, np.ndarray]) -> Path:
    """Export simple ASCII PLY point clouds for review.

    File names use HoloScene loaded ids, so raw mask id 0 is written as
    ``object_001.ply``.
    """

    out_dir = reset_dir(Path(scene_dir) / "review" / "instance_clouds")
    for rid, pts in clouds.items():
        path = out_dir / f"object_{rid + 1:03d}.ply"
        with path.open("w", encoding="utf-8") as fh:
            fh.write("ply\nformat ascii 1.0\n")
            fh.write(f"element vertex {pts.shape[0]}\n")
            fh.write("property float x\nproperty float y\nproperty float z\n")
            fh.write("end_header\n")
            for x, y, z in pts:
                fh.write(f"{float(x)} {float(y)} {float(z)}\n")
    return out_dir


def build_geometry(scene_dir: str | Path, transforms: dict[str, Any], id_mapping: dict[str, Any]) -> dict[str, Any]:
    """Build point clouds, bbox report, and PLY review files."""

    clouds, visible_meta = build_instance_pointclouds(scene_dir, transforms, id_mapping)
    bbox = estimate_instance_bbox(clouds, id_mapping, visible_meta)
    write_json(Path(scene_dir) / "meta" / "instance_bbox.json", bbox)
    write_instance_clouds(scene_dir, clouds)
    return bbox
