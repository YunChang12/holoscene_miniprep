"""Review visualization utilities."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .depth_utils import make_depth_visualization
from .geometry_utils import build_geometry
from .graph_utils import visualize_graph
from .normal_utils import make_normal_visualization
from .writer import ensure_dir, read_json

LOGGER = logging.getLogger(__name__)


def _palette(label: int) -> tuple[int, int, int]:
    colors = [
        (255, 80, 80),
        (80, 255, 120),
        (80, 160, 255),
        (255, 220, 80),
        (220, 80, 255),
        (80, 255, 240),
    ]
    return colors[label % len(colors)]


def overlay_mask(image: Image.Image, mask: Image.Image, frame_id: str | None = None, alpha: float = 0.45) -> Image.Image:
    """Overlay a HoloScene label mask on an RGB image."""

    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    m = np.asarray(mask.convert("L"), dtype=np.uint8)
    overlay = rgb.copy()
    labels = sorted(int(v) for v in np.unique(m) if int(v) != 255)
    for label in sorted(int(v) for v in np.unique(m) if int(v) != 255):
        color = np.asarray(_palette(label), dtype=np.float32)
        overlay[m == label] = overlay[m == label] * (1 - alpha) + color * alpha
    out = Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8), mode="RGB")
    draw = ImageDraw.Draw(out)
    title = frame_id or ""
    if labels:
        title = f"{title} ids:{','.join(str(v) for v in labels[:8])}".strip()
    if title:
        box_w = min(out.width, max(180, 9 * len(title) + 12))
        draw.rectangle([0, 0, box_w, 24], fill=(0, 0, 0))
        draw.text((6, 5), title, fill=(255, 255, 255))
    return out


def make_mask_overlay_video(scene_dir: str | Path, fps: float = 8.0) -> Path:
    """Create review/mask_overlay.mp4 if cv2 is available; otherwise write PNG frames."""

    scene = Path(scene_dir)
    frames_dir = ensure_dir(scene / "review" / "mask_overlay_frames")
    image_paths = sorted((scene / "images").glob("frame*.jpg"))
    mask_paths = sorted((scene / "instance_mask").glob("frame*.png"))
    overlay_paths = []
    for img_path, mask_path in zip(image_paths, mask_paths):
        with Image.open(img_path) as image, Image.open(mask_path) as mask:
            out = overlay_mask(image, mask, frame_id=img_path.stem)
        out_path = frames_dir / f"{img_path.stem}.png"
        out.save(out_path)
        overlay_paths.append(out_path)
    video_path = scene / "review" / "mask_overlay.mp4"
    try:
        import cv2  # type: ignore

        first = cv2.imread(str(overlay_paths[0]))
        height, width = first.shape[:2]
        writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height))
        for path in overlay_paths:
            writer.write(cv2.imread(str(path)))
        writer.release()
        return video_path
    except Exception as exc:  # pragma: no cover - depends on optional cv2
        LOGGER.warning("Could not create MP4 overlay (%s); PNG frames are in %s", exc, frames_dir)
        return frames_dir


def make_depth_vis(scene_dir: str | Path) -> Path:
    """Create depth visualizations."""

    return make_depth_visualization(scene_dir)


def make_normal_vis(scene_dir: str | Path) -> Path:
    """Create normal visualizations."""

    return make_normal_visualization(scene_dir)


def visualize_scene(scene_dir: str | Path) -> None:
    """Create standard review artifacts."""

    scene = Path(scene_dir)
    make_mask_overlay_video(scene)
    make_depth_vis(scene)
    make_normal_vis(scene)
    if (scene / "graph.json").is_file():
        visualize_graph(scene, read_json(scene / "graph.json"))
    if (scene / "transforms.json").is_file():
        _write_camera_trajectory_from_transforms(scene)
    if (scene / "meta" / "id_mapping.json").is_file() and (scene / "transforms.json").is_file():
        try:
            build_geometry(scene, read_json(scene / "transforms.json"), read_json(scene / "meta" / "id_mapping.json"))
        except Exception as exc:  # pragma: no cover - review should be best effort
            LOGGER.warning("Could not build instance clouds for review: %s", exc)


def _write_camera_trajectory_from_transforms(scene: Path) -> Path:
    """Export camera centers from transforms.json to review/camera_trajectory.ply."""

    transforms = read_json(scene / "transforms.json")
    centers = []
    for frame in transforms.get("frames", []):
        mat = np.asarray(frame.get("transform_matrix"), dtype=np.float64)
        if mat.shape == (4, 4) and np.isfinite(mat).all():
            centers.append(mat[:3, 3])
    out = scene / "review" / "camera_trajectory.ply"
    ensure_dir(out.parent)
    with out.open("w", encoding="utf-8") as fh:
        fh.write("ply\nformat ascii 1.0\n")
        fh.write(f"element vertex {len(centers)}\n")
        fh.write("property float x\nproperty float y\nproperty float z\n")
        fh.write("end_header\n")
        for c in centers:
            fh.write(f"{float(c[0])} {float(c[1])} {float(c[2])}\n")
    return out
