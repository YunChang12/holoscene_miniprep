"""Frame extraction and normalization utilities."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .writer import ensure_dir, reset_dir, write_json

LOGGER = logging.getLogger(__name__)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _load_cv2():
    try:
        import cv2  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "Video input requires opencv-python. Install requirements.txt or use input_type=images."
        ) from exc
    return cv2


def extract_frames_from_video(
    video_path: str | Path,
    target_fps: float,
    max_frames: int | None,
    resolution: tuple[int, int],
    output_dir: str | Path,
    overwrite: bool = True,
) -> list[dict[str, Any]]:
    """Extract frames from a video, resize them, and write HoloScene names."""

    cv2 = _load_cv2()
    src = Path(video_path).expanduser().resolve()
    if not src.is_file():
        raise FileNotFoundError(f"Video not found: {src}")
    images_dir = reset_dir(output_dir) if overwrite else ensure_dir(output_dir)
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {src}")
    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    if source_fps <= 1e-6:
        source_fps = 30.0
    stride = max(1, int(round(source_fps / max(float(target_fps), 1e-6))))
    frame_meta: list[dict[str, Any]] = []
    frame_idx = 0
    out_idx = 0
    width, height = resolution
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % stride != 0:
                frame_idx += 1
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            original_size = [pil.width, pil.height]
            pil = pil.resize((width, height), Image.Resampling.BILINEAR)
            filename = f"frame{out_idx:06d}.jpg"
            pil.save(images_dir / filename, quality=95)
            frame_meta.append(
                {
                    "frame_index": out_idx,
                    "source_frame_index": frame_idx,
                    "timestamp": float(frame_idx / source_fps),
                    "source_path": str(src),
                    "file_path": f"images/{filename}",
                    "original_size": original_size,
                    "output_size": [width, height],
                }
            )
            out_idx += 1
            frame_idx += 1
            if max_frames and out_idx >= int(max_frames):
                break
    finally:
        cap.release()
    if not frame_meta:
        raise RuntimeError(f"No frames extracted from video: {src}")
    LOGGER.info("Extracted %d frames from %s", len(frame_meta), src)
    return frame_meta


def load_image_sequence(input_dir: str | Path) -> list[Path]:
    """Return sorted image paths from a directory."""

    root = Path(input_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Image directory not found: {root}")
    paths = [p for p in sorted(root.iterdir()) if p.suffix.lower() in IMAGE_EXTS and p.is_file()]
    if not paths:
        raise RuntimeError(f"No images found in {root}")
    return paths


def write_images(
    image_paths: list[Path],
    resolution: tuple[int, int],
    output_dir: str | Path,
    max_frames: int | None = None,
    overwrite: bool = True,
) -> list[dict[str, Any]]:
    """Normalize image sequence naming and resolution."""

    images_dir = reset_dir(output_dir) if overwrite else ensure_dir(output_dir)
    width, height = resolution
    selected = image_paths[: int(max_frames)] if max_frames else image_paths
    frame_meta: list[dict[str, Any]] = []
    for idx, src in enumerate(selected):
        with Image.open(src) as im:
            rgb = im.convert("RGB")
            original_size = [rgb.width, rgb.height]
            rgb = rgb.resize((width, height), Image.Resampling.BILINEAR)
            filename = f"frame{idx:06d}.jpg"
            rgb.save(images_dir / filename, quality=95)
        frame_meta.append(
            {
                "frame_index": idx,
                "source_frame_index": idx,
                "timestamp": None,
                "source_path": str(src),
                "file_path": f"images/{filename}",
                "original_size": original_size,
                "output_size": [width, height],
            }
        )
    LOGGER.info("Wrote %d normalized images to %s", len(frame_meta), images_dir)
    return frame_meta


def write_frame_meta(scene_dir: str | Path, frame_meta: list[dict[str, Any]]) -> Path:
    """Write frame metadata under meta/frame_meta.json."""

    return write_json(Path(scene_dir) / "meta" / "frame_meta.json", frame_meta)


def list_scene_images(scene_dir: str | Path) -> list[Path]:
    """List normalized scene images."""

    images_dir = Path(scene_dir) / "images"
    return [p for p in sorted(images_dir.iterdir()) if p.suffix.lower() in IMAGE_EXTS and p.is_file()]


def read_image_array(path: str | Path) -> np.ndarray:
    """Read RGB image as uint8 numpy array."""

    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)
