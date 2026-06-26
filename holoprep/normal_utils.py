"""Normal map generation and visualization."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .writer import ensure_dir, reset_dir
from .wrappers.normal_wrapper import NormalWrapper


def load_provided_normal(provided_dir: str | Path, frame_count: int, resolution: tuple[int, int]) -> list[np.ndarray]:
    """Load normal PNG/JPG files encoded as RGB normals."""

    root = Path(provided_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Normal directory not found: {root}")
    paths = [p for p in sorted(root.iterdir()) if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}]
    if len(paths) != frame_count:
        raise ValueError(f"Normal count {len(paths)} does not match frame count {frame_count}")
    width, height = resolution
    normals = []
    for path in paths:
        with Image.open(path) as im:
            rgb = im.convert("RGB").resize((width, height), Image.Resampling.BILINEAR)
        arr = np.asarray(rgb, dtype=np.float32) / 255.0
        normals.append(arr * 2.0 - 1.0)
    return normals


def generate_normals_with_model_placeholder(*args: Any, **kwargs: Any) -> list[np.ndarray]:
    """Call external normal model placeholder."""

    return NormalWrapper().run(*args, **kwargs)


def normal_from_depth(depth: np.ndarray, fx: float, fy: float) -> np.ndarray:
    """Estimate camera-space normals from a depth map."""

    z = np.asarray(depth, dtype=np.float32)
    dzdx = np.gradient(z, axis=1)
    dzdy = np.gradient(z, axis=0)
    nx = -dzdx * float(fx)
    ny = -dzdy * float(fy)
    nz = np.ones_like(z)
    normal = np.stack([nx, ny, nz], axis=-1)
    norm = np.linalg.norm(normal, axis=-1, keepdims=True)
    normal = normal / np.maximum(norm, 1e-6)
    return normal.astype(np.float32)


def create_dummy_normal(frame_count: int, resolution: tuple[int, int]) -> list[np.ndarray]:
    """Create [0, 0, 1] normals."""

    width, height = resolution
    n = np.zeros((height, width, 3), dtype=np.float32)
    n[..., 2] = 1.0
    return [n.copy() for _ in range(frame_count)]


def encode_normal_png(normal: np.ndarray) -> Image.Image:
    """Encode normal [-1, 1] to RGB PNG."""

    arr = np.asarray(normal, dtype=np.float32)
    arr = arr / np.maximum(np.linalg.norm(arr, axis=-1, keepdims=True), 1e-6)
    rgb = np.clip((arr + 1.0) * 0.5 * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def save_normal_png(scene_dir: str | Path, normals: list[np.ndarray], overwrite: bool = True) -> Path:
    """Save normal maps as RGB PNG files."""

    out_dir = reset_dir(Path(scene_dir) / "normal") if overwrite else ensure_dir(Path(scene_dir) / "normal")
    for idx, normal in enumerate(normals):
        encode_normal_png(normal).save(out_dir / f"frame{idx:06d}.png")
    return out_dir


def make_normal_visualization(scene_dir: str | Path, overwrite: bool = True) -> Path:
    """Copy encoded normal maps into review/normal_vis."""

    scene = Path(scene_dir)
    out_dir = reset_dir(scene / "review" / "normal_vis") if overwrite else ensure_dir(scene / "review" / "normal_vis")
    for path in sorted((scene / "normal").glob("*.png")):
        with Image.open(path) as im:
            im.convert("RGB").save(out_dir / path.name)
    return out_dir
