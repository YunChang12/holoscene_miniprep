"""Depth loading, cleaning, saving, and visualization."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .writer import ensure_dir, reset_dir, write_json
from .wrappers.depth_wrapper import DepthWrapper


def load_provided_depth(provided_dir: str | Path, frame_count: int, resolution: tuple[int, int]) -> list[np.ndarray]:
    """Load provided .npy or image depth files and resize to resolution."""

    root = Path(provided_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Depth directory not found: {root}")
    paths = [p for p in sorted(root.iterdir()) if p.suffix.lower() in {".npy", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}]
    if len(paths) != frame_count:
        raise ValueError(f"Depth count {len(paths)} does not match frame count {frame_count}")
    width, height = resolution
    depths = []
    for path in paths:
        if path.suffix.lower() == ".npy":
            arr = np.load(path).astype(np.float32)
        else:
            with Image.open(path) as im:
                arr = np.asarray(im.convert("F"), dtype=np.float32)
        if arr.ndim == 3:
            arr = arr[..., 0]
        if arr.shape != (height, width):
            pil = Image.fromarray(arr.astype(np.float32), mode="F").resize((width, height), Image.Resampling.BILINEAR)
            arr = np.asarray(pil, dtype=np.float32)
        depths.append(arr.astype(np.float32))
    return depths


def generate_depth_with_model_placeholder(*args: Any, **kwargs: Any) -> list[np.ndarray]:
    """Call the external depth wrapper placeholder."""

    return DepthWrapper().run(*args, **kwargs)


def create_dummy_depth(frame_count: int, resolution: tuple[int, int], value: float = 1.0) -> list[np.ndarray]:
    """Create constant depth maps."""

    width, height = resolution
    return [np.full((height, width), float(value), dtype=np.float32) for _ in range(frame_count)]


def clean_depth(depth: np.ndarray, clip_min: float, clip_max: float) -> np.ndarray:
    """Replace NaN/Inf and clip depth range."""

    arr = np.asarray(depth, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=float(clip_max), posinf=float(clip_max), neginf=float(clip_min))
    return np.clip(arr, float(clip_min), float(clip_max)).astype(np.float32)


def save_depth_npy(scene_dir: str | Path, depths: list[np.ndarray], clip_min: float, clip_max: float, overwrite: bool = True) -> Path:
    """Save depth maps as HoloScene .npy files and write depth report."""

    out_dir = reset_dir(Path(scene_dir) / "depth") if overwrite else ensure_dir(Path(scene_dir) / "depth")
    report = {"frame_count": len(depths), "clip_min": float(clip_min), "clip_max": float(clip_max), "frames": []}
    for idx, depth in enumerate(depths):
        arr = clean_depth(depth, clip_min, clip_max)
        np.save(out_dir / f"frame{idx:06d}.npy", arr)
        report["frames"].append(
            {
                "frame_index": idx,
                "min": float(arr.min()),
                "max": float(arr.max()),
                "mean": float(arr.mean()),
                "has_nan": bool(np.isnan(arr).any()),
                "has_inf": bool(np.isinf(arr).any()),
            }
        )
    write_json(Path(scene_dir) / "meta" / "depth_report.json", report)
    return out_dir


def depth_to_vis(depth: np.ndarray) -> Image.Image:
    """Convert a depth map to a colorized percentile-normalized visualization."""

    arr = np.asarray(depth, dtype=np.float32)
    finite = np.isfinite(arr)
    if not finite.any():
        norm = np.zeros(arr.shape, dtype=np.float32)
    else:
        vals = arr[finite]
        lo, hi = np.percentile(vals, [2, 98])
        if hi <= lo:
            hi = lo + 1e-6
        norm = np.clip((arr - lo) / (hi - lo), 0, 1)
    rgb = _magma_like(norm)
    return Image.fromarray(rgb, mode="RGB")


def _magma_like(norm: np.ndarray) -> np.ndarray:
    """Small dependency-free color ramp for depth review images."""

    x = np.clip(norm.astype(np.float32), 0.0, 1.0)
    stops = np.asarray(
        [
            [0.001, 0.000, 0.014],
            [0.171, 0.067, 0.373],
            [0.445, 0.122, 0.506],
            [0.716, 0.215, 0.475],
            [0.944, 0.377, 0.365],
            [0.997, 0.760, 0.529],
            [0.987, 0.991, 0.749],
        ],
        dtype=np.float32,
    )
    pos = x * (len(stops) - 1)
    lo = np.floor(pos).astype(np.int32)
    hi = np.clip(lo + 1, 0, len(stops) - 1)
    t = (pos - lo)[..., None]
    rgb = stops[lo] * (1.0 - t) + stops[hi] * t
    return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)


def make_depth_visualization(scene_dir: str | Path, overwrite: bool = True) -> Path:
    """Write review/depth_vis/*.png from scene depth files."""

    scene = Path(scene_dir)
    out_dir = reset_dir(scene / "review" / "depth_vis") if overwrite else ensure_dir(scene / "review" / "depth_vis")
    for path in sorted((scene / "depth").glob("*.npy")):
        depth_to_vis(np.load(path)).save(out_dir / f"{path.stem}.png")
    return out_dir
