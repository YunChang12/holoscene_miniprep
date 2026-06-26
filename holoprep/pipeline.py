"""Pipeline orchestration for the minimal HoloScene preprocessing flow."""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .camera_utils import (
    create_fixed_camera_transforms,
    estimate_transforms_with_vggt_placeholder,
    load_provided_transforms,
    write_transforms_json,
)
from .config import PrepConfig, section
from .depth_utils import (
    create_dummy_depth,
    generate_depth_with_model_placeholder,
    load_provided_depth,
    make_depth_visualization,
    save_depth_npy,
)
from .frame_utils import extract_frames_from_video, list_scene_images, load_image_sequence, write_frame_meta, write_images
from .geometry_utils import build_geometry
from .graph_utils import infer_graph_simple, visualize_graph, write_graph_debug, write_graph_json
from .mask_utils import (
    create_dummy_masks,
    generate_masks_with_sam2_placeholder,
    load_id_mapping,
    load_provided_masks,
    remap_masks_for_holoscene,
    write_id_mapping,
    write_instance_masks,
)
from .normal_utils import (
    create_dummy_normal,
    generate_normals_with_model_placeholder,
    load_provided_normal,
    normal_from_depth,
    save_normal_png,
)
from .validation import validate_scene
from .visualization import visualize_scene
from .writer import ensure_dir, read_json, write_json
from .wrappers.depth_wrapper import ZaiwuDepthAnythingWrapper
from .wrappers.seg2track_sam2_wrapper import ZaiwuSeg2TrackSAM2Wrapper
from .wrappers.vggt_wrapper import ZaiwuVGGTWrapper

LOGGER = logging.getLogger(__name__)

DEFAULT_STAGES = ["frames", "camera", "mask", "depth", "normal", "geometry", "graph", "validate", "review"]
KNOWN_STAGES = set(DEFAULT_STAGES)


def parse_stages(value: str | None) -> list[str]:
    """Parse a comma separated stage list."""

    if not value or value.strip().lower() == "all":
        return list(DEFAULT_STAGES)
    stages = [part.strip().lower() for part in value.split(",") if part.strip()]
    unknown = sorted(set(stages) - KNOWN_STAGES)
    if unknown:
        raise ValueError(f"Unknown stage(s): {', '.join(unknown)}. Known stages: {', '.join(DEFAULT_STAGES)}")
    return stages


def run_pipeline(config: PrepConfig, stages: Iterable[str] | None = None, resume: bool = False) -> dict[str, Any]:
    """Run selected preprocessing stages and return a small summary."""

    selected = list(stages) if stages is not None else list(DEFAULT_STAGES)
    scene_dir = config.output_dir
    ensure_dir(scene_dir)
    ensure_dir(scene_dir / "meta")
    ensure_dir(scene_dir / "review")

    LOGGER.info("Scene: %s", config.scene_name)
    LOGGER.info("Output: %s", scene_dir)
    LOGGER.info("Stages: %s", ",".join(selected))

    frame_meta: list[dict[str, Any]] | None = None
    frame_count = 0
    transforms: dict[str, Any] | None = None
    bbox_report: dict[str, Any] | None = None
    validation_report: dict[str, Any] | None = None

    for stage in selected:
        if stage == "frames":
            frame_meta = _stage_frames(config, resume=resume)
            frame_count = len(frame_meta)
            if frame_count <= 0:
                raise RuntimeError("No normalized frames are available. Run the frames stage first.")
            continue

        if frame_meta is None:
            frame_meta = _load_or_create_frame_meta(scene_dir)
            frame_count = len(frame_meta)
            if frame_count <= 0:
                raise RuntimeError("No normalized frames are available. Run the frames stage first.")

        if stage == "camera":
            transforms = _stage_camera(config, frame_count, resume=resume)
        elif stage == "mask":
            _stage_mask(config, frame_count, resume=resume)
        elif stage == "depth":
            _stage_depth(config, frame_count, resume=resume)
        elif stage == "normal":
            transforms = transforms or read_json(scene_dir / "transforms.json")
            _stage_normal(config, frame_count, transforms, resume=resume)
        elif stage == "geometry":
            transforms = transforms or read_json(scene_dir / "transforms.json")
            bbox_report = _stage_geometry(config, transforms, resume=resume)
        elif stage == "graph":
            transforms = transforms or read_json(scene_dir / "transforms.json")
            bbox_report = bbox_report or _ensure_bbox_for_graph(config, transforms, resume=resume)
            _stage_graph(config, bbox_report, resume=resume)
        elif stage == "validate":
            validation_report = validate_scene(scene_dir)
            if not validation_report.get("ok", False):
                raise RuntimeError(f"Validation failed. See {scene_dir / 'meta' / 'validation_report.md'}")
        elif stage == "review":
            _stage_review(config, resume=resume)
        else:  # parse_stages should already prevent this.
            raise ValueError(f"Unknown stage: {stage}")

    if frame_meta is None:
        frame_meta = _load_or_create_frame_meta(scene_dir)
        frame_count = len(frame_meta)
        if frame_count <= 0:
            raise RuntimeError("No normalized frames are available. Run the frames stage first.")

    metadata = _write_preprocess_metadata(config, frame_count, selected, validation_report)
    return {"scene_dir": str(scene_dir), "frame_count": frame_count, "metadata": metadata}


def _stage_frames(config: PrepConfig, resume: bool) -> list[dict[str, Any]]:
    scene = config.output_dir
    frame_cfg = section(config, "frame")
    images_dir = scene / "images"
    if resume and _has_files(images_dir, "frame*.jpg"):
        LOGGER.info("[frames] resume: using existing images")
        return _load_or_create_frame_meta(scene)

    input_type = str(config.data["scene"]["input_type"]).lower()
    input_path = Path(config.data["scene"]["input_path"]).expanduser()
    resolution = config.resolution
    max_frames = frame_cfg.get("max_frames")
    overwrite = bool(frame_cfg.get("overwrite", True))
    if input_type == "video":
        frame_meta = extract_frames_from_video(
            video_path=input_path,
            target_fps=float(frame_cfg.get("target_fps", 3)),
            max_frames=int(max_frames) if max_frames else None,
            resolution=resolution,
            output_dir=images_dir,
            overwrite=overwrite,
        )
    elif input_type == "images":
        image_paths = load_image_sequence(input_path)
        frame_meta = write_images(
            image_paths=image_paths,
            resolution=resolution,
            output_dir=images_dir,
            max_frames=int(max_frames) if max_frames else None,
            overwrite=overwrite,
        )
    else:
        raise ValueError("scene.input_type must be 'video' or 'images'")
    write_frame_meta(scene, frame_meta)
    LOGGER.info("[frames] wrote %d frames", len(frame_meta))
    return frame_meta


def _stage_camera(config: PrepConfig, frame_count: int, resume: bool) -> dict[str, Any]:
    scene = config.output_dir
    if resume and (scene / "transforms.json").is_file():
        LOGGER.info("[camera] resume: using existing transforms.json")
        return read_json(scene / "transforms.json")
    cam_cfg = section(config, "camera")
    mode = str(cam_cfg.get("mode", "fixed")).lower()
    provided = _optional_path(cam_cfg.get("provided_transforms"))
    if mode == "provided" or (mode == "provided_or_vggt" and provided and provided.is_file()):
        if not provided:
            raise ValueError("camera.mode=provided requires camera.provided_transforms")
        transforms = load_provided_transforms(provided, frame_count, config.resolution)
    elif mode == "fixed":
        transforms = create_fixed_camera_transforms(
            frame_count=frame_count,
            resolution=config.resolution,
            assume_fov_deg=float(cam_cfg.get("assume_fov_deg", 60.0)),
        )
    elif mode in {"vggt", "provided_or_vggt"}:
        transforms = estimate_transforms_with_vggt_placeholder(scene / "images", scene / "meta" / "vggt")
    elif mode == "zaiwu_vggt":
        transforms = ZaiwuVGGTWrapper().run(
            image_dir=scene / "images",
            output_dir=scene,
            config=cam_cfg,
            frame_count=frame_count,
            resolution=config.resolution,
            scene_id=config.scene_name,
        )
    else:
        raise ValueError(f"Unsupported camera.mode: {mode}")
    write_transforms_json(scene, transforms)
    LOGGER.info("[camera] wrote transforms.json using mode=%s", mode)
    return transforms


def _stage_mask(config: PrepConfig, frame_count: int, resume: bool) -> dict[str, Any]:
    scene = config.output_dir
    if resume and _has_files(scene / "instance_mask", "frame*.png") and (scene / "meta" / "id_mapping.json").is_file():
        LOGGER.info("[mask] resume: using existing instance_mask")
        return load_id_mapping(scene)
    mask_cfg = section(config, "mask")
    mode = str(mask_cfg.get("mode", "dummy")).lower()
    provided = _optional_path(mask_cfg.get("provided_dir"))
    background_value = int(mask_cfg.get("background_value", 255))
    if mode == "provided" or (mode == "provided_or_sam2" and provided and provided.is_dir()):
        if not provided:
            raise ValueError("mask.mode=provided requires mask.provided_dir")
        masks = load_provided_masks(provided, frame_count, config.resolution)
        source_background_value = background_value
    elif mode == "dummy":
        masks = create_dummy_masks(frame_count, config.resolution, background_value=255)
        source_background_value = 255
    elif mode in {"sam2", "provided_or_sam2"}:
        masks = generate_masks_with_sam2_placeholder(image_dir=scene / "images", output_dir=scene / "meta" / "sam2")
        source_background_value = background_value
    elif mode == "zaiwu_seg2track_sam2":
        masks, mapping = ZaiwuSeg2TrackSAM2Wrapper().run(
            image_dir=scene / "images",
            output_dir=scene,
            config=mask_cfg,
            frame_count=frame_count,
            resolution=config.resolution,
        )
        write_instance_masks(scene, masks, overwrite=True)
        write_id_mapping(scene, mapping)
        LOGGER.info("[mask] wrote %d masks with %d tracked objects", len(masks), len(mapping.get("objects", [])))
        return mapping
    else:
        raise ValueError(f"Unsupported mask.mode: {mode}")
    remapped, mapping = remap_masks_for_holoscene(
        masks,
        background_value=source_background_value,
        min_area_ratio=float(mask_cfg.get("min_area_ratio", 0.001)),
    )
    write_instance_masks(scene, remapped, overwrite=True)
    write_id_mapping(scene, mapping)
    LOGGER.info("[mask] wrote %d masks with %d objects", len(remapped), len(mapping.get("objects", [])))
    return mapping


def _stage_depth(config: PrepConfig, frame_count: int, resume: bool) -> Path:
    scene = config.output_dir
    if resume and _has_files(scene / "depth", "frame*.npy"):
        LOGGER.info("[depth] resume: using existing depth")
        return scene / "depth"
    depth_cfg = section(config, "depth")
    mode = str(depth_cfg.get("mode", "dummy")).lower()
    provided = _optional_path(depth_cfg.get("provided_dir"))
    if mode == "provided" or (mode == "provided_or_model" and provided and provided.is_dir()):
        if not provided:
            raise ValueError("depth.mode=provided requires depth.provided_dir")
        depths = load_provided_depth(provided, frame_count, config.resolution)
    elif mode == "dummy":
        depths = create_dummy_depth(frame_count, config.resolution)
    elif mode in {"da3", "marigold", "model", "provided_or_model"}:
        depths = generate_depth_with_model_placeholder(image_dir=scene / "images", output_dir=scene / "meta" / "depth_model")
    elif mode == "zaiwu_da3":
        depths = ZaiwuDepthAnythingWrapper().run(
            image_dir=scene / "images",
            output_dir=scene,
            config=depth_cfg,
            frame_count=frame_count,
            resolution=config.resolution,
        )
    else:
        raise ValueError(f"Unsupported depth.mode: {mode}")
    out = save_depth_npy(
        scene,
        depths,
        clip_min=float(depth_cfg.get("clip_min", 0.05)),
        clip_max=float(depth_cfg.get("clip_max", 20.0)),
        overwrite=True,
    )
    LOGGER.info("[depth] wrote %d depth maps", len(depths))
    return out


def _stage_normal(config: PrepConfig, frame_count: int, transforms: dict[str, Any], resume: bool) -> Path:
    scene = config.output_dir
    if resume and _has_files(scene / "normal", "frame*.png"):
        LOGGER.info("[normal] resume: using existing normals")
        return scene / "normal"
    normal_cfg = section(config, "normal")
    mode = str(normal_cfg.get("mode", "depth_to_normal")).lower()
    provided = _optional_path(normal_cfg.get("provided_dir"))
    if mode == "provided":
        if not provided:
            raise ValueError("normal.mode=provided requires normal.provided_dir")
        normals = load_provided_normal(provided, frame_count, config.resolution)
    elif mode == "depth_to_normal":
        normals = []
        fx = float(transforms["fl_x"])
        fy = float(transforms["fl_y"])
        for idx in range(frame_count):
            normals.append(normal_from_depth(np.load(scene / "depth" / f"frame{idx:06d}.npy"), fx=fx, fy=fy))
    elif mode == "dummy":
        normals = create_dummy_normal(frame_count, config.resolution)
    elif mode in {"model", "omnidata", "marigold"}:
        normals = generate_normals_with_model_placeholder(image_dir=scene / "images", output_dir=scene / "meta" / "normal_model")
    else:
        raise ValueError(f"Unsupported normal.mode: {mode}")
    out = save_normal_png(scene, normals, overwrite=True)
    LOGGER.info("[normal] wrote %d normal maps", len(normals))
    return out


def _stage_geometry(config: PrepConfig, transforms: dict[str, Any], resume: bool) -> dict[str, Any]:
    scene = config.output_dir
    bbox_path = scene / "meta" / "instance_bbox.json"
    if resume and bbox_path.is_file():
        LOGGER.info("[geometry] resume: using existing instance_bbox.json")
        return read_json(bbox_path)
    mapping = load_id_mapping(scene)
    bbox = build_geometry(scene, transforms, mapping)
    LOGGER.info("[geometry] wrote bbox report for %d objects", len(bbox.get("objects", [])))
    return bbox


def _stage_graph(config: PrepConfig, bbox_report: dict[str, Any], resume: bool) -> Path:
    scene = config.output_dir
    if resume and (scene / "graph.json").is_file():
        LOGGER.info("[graph] resume: using existing graph.json")
        return scene / "graph.json"
    graph_cfg = section(config, "graph")
    mode = str(graph_cfg.get("mode", "auto_simple")).lower()
    if mode == "manual":
        manual = _optional_path(graph_cfg.get("manual_graph"))
        if not manual or not manual.is_file():
            raise ValueError("graph.mode=manual requires graph.manual_graph")
        shutil.copy2(manual, scene / "graph.json")
        write_json(scene / "meta" / "graph_debug.json", {"mode": "manual", "source": str(manual)})
        graph = read_json(scene / "graph.json")
    elif mode == "auto_simple":
        graph, debug = infer_graph_simple(
            bbox_report,
            vertical_gap_threshold=float(graph_cfg.get("vertical_gap_threshold", 0.08)),
            xy_overlap_threshold=float(graph_cfg.get("xy_overlap_threshold", 0.15)),
            root_id=int(graph_cfg.get("root_id", 0)),
        )
        write_graph_json(scene, graph)
        write_graph_debug(scene, debug)
    else:
        raise ValueError(f"Unsupported graph.mode: {mode}")
    visualize_graph(scene, graph)
    LOGGER.info("[graph] wrote graph.json with %d nodes", len(graph))
    return scene / "graph.json"


def _stage_review(config: PrepConfig, resume: bool) -> None:
    scene = config.output_dir
    visualize_scene(scene)
    LOGGER.info("[review] wrote review artifacts")


def _ensure_bbox_for_graph(config: PrepConfig, transforms: dict[str, Any], resume: bool) -> dict[str, Any]:
    bbox_path = config.output_dir / "meta" / "instance_bbox.json"
    if bbox_path.is_file():
        return read_json(bbox_path)
    LOGGER.info("[graph] instance_bbox.json missing; running geometry prerequisite")
    return _stage_geometry(config, transforms, resume=resume)


def _load_or_create_frame_meta(scene_dir: Path) -> list[dict[str, Any]]:
    meta_path = scene_dir / "meta" / "frame_meta.json"
    if meta_path.is_file():
        return read_json(meta_path)
    images = list_scene_images(scene_dir)
    frame_meta = [
        {
            "frame_index": idx,
            "source_frame_index": None,
            "timestamp": None,
            "source_path": None,
            "file_path": f"images/{path.name}",
            "original_size": None,
            "output_size": None,
        }
        for idx, path in enumerate(images)
    ]
    if frame_meta:
        write_frame_meta(scene_dir, frame_meta)
    return frame_meta


def _write_preprocess_metadata(
    config: PrepConfig,
    frame_count: int,
    stages: list[str],
    validation_report: dict[str, Any] | None,
) -> dict[str, Any]:
    scene = config.output_dir
    metadata = {
        "scene_id": config.scene_name,
        "scene_dir": str(scene),
        "source_type": config.data["scene"]["input_type"],
        "source_path": str(config.data["scene"]["input_path"]),
        "image_resolution": list(config.resolution),
        "frame_count": int(frame_count),
        "stages": stages,
        "camera_mode": section(config, "camera").get("mode"),
        "mask_mode": section(config, "mask").get("mode"),
        "depth_mode": section(config, "depth").get("mode"),
        "normal_mode": section(config, "normal").get("mode"),
        "graph_mode": section(config, "graph").get("mode"),
        "validation_ok": None if validation_report is None else bool(validation_report.get("ok")),
    }
    mapping_path = scene / "meta" / "id_mapping.json"
    if mapping_path.is_file():
        mapping = read_json(mapping_path)
        metadata["track_to_node"] = {
            str(obj.get("label", f"object_{obj['raw_mask_value']}")): {
                "raw_mask_value": int(obj["raw_mask_value"]),
                "holoscene_node_id": int(obj["holoscene_node_id"]),
                "source_label": obj.get("source_label"),
            }
            for obj in mapping.get("objects", [])
        }
    write_json(scene / "meta" / "preprocess_metadata.json", metadata)
    return metadata


def _has_files(path: Path, pattern: str) -> bool:
    return path.is_dir() and any(path.glob(pattern))


def _optional_path(value: Any) -> Path | None:
    if value in (None, ""):
        return None
    return Path(value).expanduser()
