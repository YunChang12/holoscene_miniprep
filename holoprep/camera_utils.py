"""Camera transform helpers for HoloScene/NeRF style transforms.json."""

from __future__ import annotations

import json
import logging
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from .writer import write_json
from .wrappers.vggt_wrapper import VGGTWrapper

LOGGER = logging.getLogger(__name__)


def _validate_matrix_4x4(matrix: Any, index: int) -> list[list[float]]:
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.shape != (4, 4):
        raise ValueError(f"transform_matrix at frame {index} must be 4x4, got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"transform_matrix at frame {index} contains NaN/Inf")
    return arr.tolist()


def load_provided_transforms(path: str | Path, frame_count: int, resolution: tuple[int, int]) -> dict[str, Any]:
    """Load, validate, and normalize provided transforms.json."""

    src = Path(path).expanduser().resolve()
    if not src.is_file():
        raise FileNotFoundError(f"provided_transforms not found: {src}")
    data = json.loads(src.read_text(encoding="utf-8"))
    frames = data.get("frames")
    if not isinstance(frames, list):
        raise ValueError("provided transforms.json must contain frames list")
    if len(frames) != frame_count:
        raise ValueError(f"transforms frame count {len(frames)} does not match image count {frame_count}")
    width, height = resolution
    old_w = float(data.get("w") or width)
    old_h = float(data.get("h") or height)
    sx = float(width) / max(old_w, 1e-6)
    sy = float(height) / max(old_h, 1e-6)
    out_frames = []
    for idx, frame in enumerate(frames):
        matrix = _validate_matrix_4x4(frame.get("transform_matrix"), idx)
        out_frames.append({"file_path": f"images/frame{idx:06d}.jpg", "transform_matrix": matrix})
    data["camera_model"] = data.get("camera_model", "OPENCV")
    data["fl_x"] = float(data.get("fl_x", width)) * sx
    data["fl_y"] = float(data.get("fl_y", height)) * sy
    data["cx"] = float(data.get("cx", old_w / 2.0)) * sx
    data["cy"] = float(data.get("cy", old_h / 2.0)) * sy
    data["w"] = int(width)
    data["h"] = int(height)
    data["frames"] = out_frames
    return data


def create_fixed_camera_transforms(
    frame_count: int,
    resolution: tuple[int, int],
    assume_fov_deg: float = 60.0,
) -> dict[str, Any]:
    """Create a simple static camera transform for format testing."""

    width, height = resolution
    fov = math.radians(float(assume_fov_deg))
    fx = 0.5 * width / math.tan(0.5 * fov)
    fy = 0.5 * height / math.tan(0.5 * fov)
    frames = []
    for idx in range(frame_count):
        pose = np.eye(4, dtype=np.float64)
        frames.append({"file_path": f"images/frame{idx:06d}.jpg", "transform_matrix": pose.tolist()})
    return {
        "camera_model": "OPENCV",
        "fl_x": float(fx),
        "fl_y": float(fy),
        "cx": float(width / 2.0),
        "cy": float(height / 2.0),
        "h": int(height),
        "w": int(width),
        "frames": frames,
    }


def estimate_transforms_with_vggt_placeholder(image_dir: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Call the VGGT wrapper placeholder."""

    wrapper = VGGTWrapper()
    return wrapper.run(image_dir=image_dir, output_dir=output_dir)


def write_transforms_json(scene_dir: str | Path, transforms: dict[str, Any]) -> Path:
    """Write transforms.json and a small camera report."""

    scene = Path(scene_dir)
    camera_report_extra = transforms.get("_camera_report_extra")
    transforms_to_write = {k: v for k, v in transforms.items() if not str(k).startswith("_")}
    for idx, frame in enumerate(transforms_to_write.get("frames", [])):
        _validate_matrix_4x4(frame.get("transform_matrix"), idx)
    out = write_json(scene / "transforms.json", transforms_to_write)
    report = {
        "frame_count": len(transforms_to_write.get("frames", [])),
        "camera_model": transforms_to_write.get("camera_model"),
        "width": transforms_to_write.get("w"),
        "height": transforms_to_write.get("h"),
        "fl_x": transforms_to_write.get("fl_x"),
        "fl_y": transforms_to_write.get("fl_y"),
        "cx": transforms_to_write.get("cx"),
        "cy": transforms_to_write.get("cy"),
    }
    if isinstance(camera_report_extra, dict):
        report.update(camera_report_extra)
    write_json(scene / "meta" / "camera_report.json", report)
    return out


def copy_manual_transforms(src: str | Path, dst: str | Path) -> None:
    """Copy a transforms file, used by callers that only need raw copying."""

    shutil.copy2(Path(src), Path(dst))


def intrinsics_from_transforms(transforms: dict[str, Any]) -> dict[str, float]:
    """Extract intrinsics from transforms.json."""

    return {
        "fx": float(transforms["fl_x"]),
        "fy": float(transforms["fl_y"]),
        "cx": float(transforms["cx"]),
        "cy": float(transforms["cy"]),
        "width": int(transforms["w"]),
        "height": int(transforms["h"]),
    }
