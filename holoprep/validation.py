"""Deep validation for HoloScene MiniPrep outputs."""

from __future__ import annotations

import json
import math
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .writer import ensure_dir, write_json

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
DEFAULT_CLIP_MIN = 0.05
DEFAULT_CLIP_MAX = 20.0


def validate_scene(scene_dir: str | Path, clip_min: float = DEFAULT_CLIP_MIN, clip_max: float = DEFAULT_CLIP_MAX) -> dict[str, Any]:
    """Validate whether a prepared scene can be safely read by HoloScene.

    The validator writes modality-specific reports under ``meta/`` and review
    files under ``review/``. Serious loader-breaking issues are errors; weak
    dummy-like data such as a fixed camera or constant normals are warnings.
    """

    scene = Path(scene_dir).expanduser().resolve()
    ensure_dir(scene / "meta")
    ensure_dir(scene / "review")

    errors: list[str] = []
    warnings: list[str] = []
    checks: dict[str, Any] = {}

    files_report = _check_required_files(scene, errors)
    checks["files"] = files_report
    if errors:
        return _write_validation(scene, errors, warnings, checks, summary={})

    images = _list_images(scene / "images")
    masks = {p.stem: p for p in sorted((scene / "instance_mask").glob("frame*.png"))}
    depths = {p.stem: p for p in sorted((scene / "depth").glob("frame*.npy"))}
    normals = {p.stem: p for p in sorted((scene / "normal").glob("frame*.png"))}
    image_map = {p.stem: p for p in images}

    transforms = _read_json_file(scene / "transforms.json", errors, "transforms.json")
    graph = _read_json_file(scene / "graph.json", errors, "graph.json")
    id_mapping = _read_json_file(scene / "meta" / "id_mapping.json", errors, "meta/id_mapping.json")
    if errors:
        return _write_validation(scene, errors, warnings, checks, summary={})

    frame_report = _check_frame_consistency(scene, image_map, masks, depths, normals, transforms, errors)
    checks["frames"] = frame_report
    write_json(scene / "meta" / "frame_consistency_report.json", frame_report["frames"])

    resolution_report = _check_resolution(scene, image_map, masks, depths, normals, transforms, errors)
    checks["resolution"] = resolution_report
    resolution = resolution_report.get("expected_resolution")

    camera_report = _check_camera(scene, transforms, image_map, resolution, errors, warnings)
    camera_source_report_path = scene / "meta" / "camera_source_report.json"
    if camera_source_report_path.is_file():
        try:
            source_report = json.loads(camera_source_report_path.read_text(encoding="utf-8"))
            if isinstance(source_report, dict):
                camera_report.update({f"source_{key}": value for key, value in source_report.items()})
                for key in ["source", "input_convention", "resolved_input_convention", "output_convention", "inverted", "scale_aligned", "scale_factor", "scale_warning"]:
                    if key in source_report:
                        camera_report[key] = source_report[key]
        except Exception as exc:
            _warn(warnings, "camera", f"无法读取 camera_source_report.json: {exc}", "检查 camera wrapper 输出。")
    checks["camera"] = camera_report
    write_json(scene / "meta" / "camera_report.json", camera_report)
    _write_camera_trajectory_ply(scene / "review" / "camera_trajectory.ply", camera_report.get("centers", []))

    mask_report, id_mapping_check = _check_masks(scene, masks, id_mapping, resolution, errors, warnings)
    checks["mask"] = mask_report
    write_json(scene / "meta" / "mask_report.json", mask_report)
    write_json(scene / "meta" / "id_mapping_check.json", id_mapping_check)

    depth_report = _check_depth(scene, depths, resolution, float(clip_min), float(clip_max), errors)
    checks["depth"] = depth_report
    write_json(scene / "meta" / "depth_report.json", depth_report)

    normal_report = _check_normals(scene, normals, resolution, errors, warnings)
    checks["normal"] = normal_report
    write_json(scene / "meta" / "normal_report.json", normal_report)

    graph_report = _check_graph(scene, graph, id_mapping, errors, warnings)
    checks["graph"] = graph_report
    write_json(scene / "meta" / "graph_report.json", graph_report)

    summary = {
        "num_frames": int(len(image_map)),
        "num_instances": int(len(id_mapping.get("objects", [])) if isinstance(id_mapping, dict) else 0),
        "resolution": resolution,
        "camera_type": camera_report.get("camera_type", "unknown"),
        "graph_status": graph_report.get("status", "unknown"),
    }
    return _write_validation(scene, errors, warnings, checks, summary=summary)


def _check_required_files(scene: Path, errors: list[str]) -> dict[str, Any]:
    required_dirs = ["images", "instance_mask", "depth", "normal"]
    required_files = ["transforms.json", "graph.json", "meta/id_mapping.json"]
    report = {"directories": {}, "files": {}, "ok": True}
    for name in required_dirs:
        exists = (scene / name).is_dir()
        report["directories"][name] = exists
        if not exists:
            _error(errors, "files", f"缺少目录: {name}", f"重新运行对应阶段生成 {name}/。")
    for name in required_files:
        exists = (scene / name).is_file()
        report["files"][name] = exists
        if not exists:
            _error(errors, "files", f"缺少文件: {name}", f"重新运行对应阶段或手动提供 {name}。")
    report["ok"] = not errors
    return report


def _check_frame_consistency(
    scene: Path,
    images: dict[str, Path],
    masks: dict[str, Path],
    depths: dict[str, Path],
    normals: dict[str, Path],
    transforms: dict[str, Any],
    errors: list[str],
) -> dict[str, Any]:
    transform_by_stem: dict[str, bool] = {}
    transform_paths = []
    for frame in transforms.get("frames", []) if isinstance(transforms, dict) else []:
        rel = str(frame.get("file_path", ""))
        stem = Path(rel).stem
        if stem:
            transform_by_stem[stem] = True
            transform_paths.append(rel)
    all_stems = sorted(set(images) | set(masks) | set(depths) | set(normals) | set(transform_by_stem))
    frames: dict[str, Any] = {}
    missing_by_modality = {"image": [], "mask": [], "depth": [], "normal": [], "transform": []}
    for stem in all_stems:
        row = {
            "image": stem in images,
            "mask": stem in masks,
            "depth": stem in depths,
            "normal": stem in normals,
            "transform": stem in transform_by_stem,
        }
        frames[stem] = row
        for key, ok in row.items():
            if not ok:
                missing_by_modality[key].append(stem)
    for key, missing in missing_by_modality.items():
        if missing:
            _error(
                errors,
                "frames",
                f"{key} 缺失 {len(missing)} 帧: {missing[:10]}",
                "检查文件命名是否为 frameXXXXXX，并重新运行缺失模态对应阶段。",
            )
    counts = {
        "images": len(images),
        "instance_mask": len(masks),
        "depth": len(depths),
        "normal": len(normals),
        "transforms": len(transform_by_stem),
    }
    if len(set(counts.values())) != 1:
        _error(errors, "frames", f"帧数量不一致: {counts}", "确保所有模态和 transforms.json.frames 数量一致。")
    return {
        "ok": not any(missing_by_modality.values()) and len(set(counts.values())) == 1,
        "counts": counts,
        "missing_by_modality": missing_by_modality,
        "transform_file_paths": transform_paths,
        "frames": frames,
    }


def _check_resolution(
    scene: Path,
    images: dict[str, Path],
    masks: dict[str, Path],
    depths: dict[str, Path],
    normals: dict[str, Path],
    transforms: dict[str, Any],
    errors: list[str],
) -> dict[str, Any]:
    report: dict[str, Any] = {"frames": {}, "ok": True, "expected_resolution": None}
    expected: tuple[int, int] | None = None
    if images:
        first = images[sorted(images)[0]]
        with Image.open(first) as im:
            expected = im.size
            report["expected_resolution"] = [int(expected[0]), int(expected[1])]
    if expected is None:
        _error(errors, "resolution", "没有可读取的 image 帧。", "先运行 frames 阶段。")
        report["ok"] = False
        return report
    expected_wh = expected
    expected_hw = (expected[1], expected[0])
    if int(transforms.get("w", -1)) != expected_wh[0] or int(transforms.get("h", -1)) != expected_wh[1]:
        _error(
            errors,
            "resolution",
            f"transforms.json 分辨率为 {(transforms.get('w'), transforms.get('h'))}，期望 {expected_wh}。",
            "重新生成 transforms.json，或确保 provided transforms 的内参已按输出分辨率缩放。",
        )
    for stem in sorted(images):
        row: dict[str, Any] = {}
        with Image.open(images[stem]) as im:
            row["image"] = list(im.size)
            if im.size != expected_wh:
                _error(errors, "resolution", f"{stem} image 尺寸 {im.size} != {expected_wh}", "重新运行 frames 阶段统一分辨率。")
        if stem in masks:
            with Image.open(masks[stem]) as im:
                row["mask"] = list(im.size)
                if im.size != expected_wh:
                    _error(errors, "resolution", f"{stem} mask 尺寸 {im.size} != {expected_wh}", "重新运行 mask 阶段或 resize provided mask。")
        if stem in depths:
            try:
                arr = np.load(depths[stem], mmap_mode="r")
                row["depth"] = list(arr.shape)
                if tuple(arr.shape) != expected_hw:
                    _error(errors, "resolution", f"{stem} depth shape {arr.shape} != {expected_hw}", "重新运行 depth 阶段并输出 HxW 的 .npy。")
            except Exception as exc:
                _error(errors, "depth", f"{stem} depth 无法读取: {exc}", "确认 depth 文件是 .npy。")
        if stem in normals:
            with Image.open(normals[stem]) as im:
                row["normal"] = list(im.size)
                if im.size != expected_wh:
                    _error(errors, "resolution", f"{stem} normal 尺寸 {im.size} != {expected_wh}", "重新运行 normal 阶段或 resize provided normal。")
        report["frames"][stem] = row
    report["ok"] = not any(err.startswith("[resolution]") for err in errors)
    return report


def _check_camera(
    scene: Path,
    transforms: dict[str, Any],
    images: dict[str, Path],
    resolution: list[int] | None,
    errors: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    required = ["camera_model", "fl_x", "fl_y", "cx", "cy", "h", "w", "frames"]
    report: dict[str, Any] = {"required_fields": {}, "frames": [], "centers": [], "ok": True}
    for key in required:
        exists = key in transforms
        report["required_fields"][key] = exists
        if not exists:
            _error(errors, "camera", f"transforms.json 缺少顶层字段 {key}", "重新生成 transforms.json。")
    for key in ["fl_x", "fl_y", "cx", "cy"]:
        try:
            value = float(transforms.get(key))
            if not math.isfinite(value) or value <= 0 and key in {"fl_x", "fl_y"}:
                raise ValueError(value)
        except Exception:
            _error(errors, "camera", f"transforms.json 字段 {key} 不是有效数值: {transforms.get(key)}", "检查相机内参。")
    frames = transforms.get("frames", [])
    if not isinstance(frames, list):
        _error(errors, "camera", "transforms.json.frames 不是 list。", "重新生成 transforms.json。")
        frames = []
    centers = []
    rotations = []
    for idx, frame in enumerate(frames):
        frame_report: dict[str, Any] = {"frame_index": idx, "ok": True}
        rel = str(frame.get("file_path", ""))
        frame_report["file_path"] = rel
        if not rel or not (scene / rel).is_file():
            _error(errors, "camera", f"第 {idx} 帧 file_path 不存在: {rel}", "确保 file_path 使用 images/frameXXXXXX.jpg 相对路径。")
            frame_report["ok"] = False
        mat = np.asarray(frame.get("transform_matrix"), dtype=np.float64)
        if mat.shape != (4, 4):
            _error(errors, "camera", f"第 {idx} 帧 transform_matrix shape 为 {mat.shape}，不是 4x4。", "修复相机 pose 矩阵。")
            frame_report["ok"] = False
            report["frames"].append(frame_report)
            continue
        if not np.isfinite(mat).all():
            _error(errors, "camera", f"第 {idx} 帧 transform_matrix 包含 NaN/Inf。", "修复相机 pose 矩阵。")
            frame_report["ok"] = False
        if not np.allclose(mat[3], np.array([0.0, 0.0, 0.0, 1.0]), atol=1e-5):
            _error(errors, "camera", f"第 {idx} 帧 transform_matrix 最后一行不是 [0,0,0,1]: {mat[3].tolist()}", "修复为标准齐次矩阵。")
            frame_report["ok"] = False
        r = mat[:3, :3]
        ortho_error = float(np.linalg.norm(r.T @ r - np.eye(3)))
        det = float(np.linalg.det(r))
        frame_report["orthogonality_error"] = ortho_error
        frame_report["determinant"] = det
        if ortho_error > 1e-2:
            _error(errors, "camera", f"第 {idx} 帧 R 不接近正交矩阵，误差 {ortho_error:.4g}。", "检查 transforms 坐标系或矩阵格式。")
            frame_report["ok"] = False
        if abs(det - 1.0) > 1e-2:
            _error(errors, "camera", f"第 {idx} 帧 det(R)={det:.4g}，不接近 1。", "检查是否混入缩放、反射或 world-to-camera 矩阵。")
            frame_report["ok"] = False
        center = mat[:3, 3].astype(float)
        centers.append(center)
        rotations.append(r)
        frame_report["center"] = center.tolist()
        report["frames"].append(frame_report)
    if centers:
        centers_arr = np.stack(centers, axis=0)
        report["centers"] = centers_arr.astype(float).tolist()
        if len(centers) > 1:
            translations = np.linalg.norm(np.diff(centers_arr, axis=0), axis=1)
            rotations_deg = []
            for a, b in zip(rotations[:-1], rotations[1:]):
                rel = a.T @ b
                cos = np.clip((np.trace(rel) - 1.0) * 0.5, -1.0, 1.0)
                rotations_deg.append(float(np.degrees(np.arccos(cos))))
            max_t = float(translations.max()) if translations.size else 0.0
            med_t = float(np.median(translations)) if translations.size else 0.0
            max_r = float(max(rotations_deg)) if rotations_deg else 0.0
            report["trajectory"] = {
                "translation_steps": translations.astype(float).tolist(),
                "rotation_steps_deg": rotations_deg,
                "max_translation_step": max_t,
                "median_translation_step": med_t,
                "max_rotation_step_deg": max_r,
            }
            fixed = max_t < 1e-8 and max_r < 1e-6
            report["fixed_camera_detected"] = bool(fixed)
            report["camera_type"] = "fixed" if fixed else "moving"
            if fixed:
                _warn(warnings, "camera", "fixed camera detected; HoloScene multi-view reconstruction may be weak", "提供真实 transforms 或使用 VGGT 估计相机。")
            if translations.size >= 3 and med_t > 1e-8 and max_t > max(10.0 * med_t, med_t + 1.0):
                _warn(warnings, "camera", f"相机轨迹存在异常平移跳变，max={max_t:.4g}, median={med_t:.4g}", "检查对应帧 pose 是否错位。")
            if max_r > 90.0:
                _warn(warnings, "camera", f"相邻帧旋转跳变较大，max={max_r:.2f} deg", "检查 transforms 坐标系和帧顺序。")
        else:
            report["fixed_camera_detected"] = True
            report["camera_type"] = "fixed"
            _warn(warnings, "camera", "只有一帧相机，HoloScene 多视角重建会很弱。", "增加多帧输入。")
    else:
        report["camera_type"] = "unknown"
    report["ok"] = not any(err.startswith("[camera]") for err in errors)
    return report


def _check_masks(
    scene: Path,
    masks: dict[str, Path],
    id_mapping: dict[str, Any],
    resolution: list[int] | None,
    errors: list[str],
    warnings: list[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    report: dict[str, Any] = {"frames": {}, "instances": {}, "ok": True}
    expected_wh = tuple(resolution) if resolution else None
    area_by_id: dict[int, list[float]] = {}
    frame_count_by_id: dict[int, int] = {}
    all_labels: set[int] = set()
    for stem, path in sorted(masks.items()):
        with Image.open(path) as im:
            mode = im.mode
            arr_raw = np.asarray(im)
            if arr_raw.ndim != 2:
                _error(errors, "mask", f"{stem} mask 不是单通道，mode={mode}, shape={arr_raw.shape}", "输出单通道 label PNG。")
                arr = np.asarray(im.convert("L"), dtype=np.uint8)
            else:
                arr = arr_raw.astype(np.uint8)
        labels = sorted(int(v) for v in np.unique(arr))
        object_labels = [v for v in labels if v != 255]
        if 255 not in labels:
            _error(errors, "mask", f"{stem} mask 不包含背景值 255。", "确保背景像素为 255。")
        total = float(arr.shape[0] * arr.shape[1])
        frame_stats = {"mode": mode, "labels": labels, "instances": {}}
        for label in object_labels:
            ratio = float(np.count_nonzero(arr == label) / max(total, 1.0))
            area_by_id.setdefault(label, []).append(ratio)
            frame_count_by_id[label] = frame_count_by_id.get(label, 0) + 1
            all_labels.add(label)
            frame_stats["instances"][str(label)] = {"area_ratio": ratio}
        report["frames"][stem] = frame_stats
    if all_labels and sorted(all_labels) != list(range(max(all_labels) + 1)):
        _error(
            errors,
            "mask",
            f"全局前景实例 ID 不连续: {sorted(all_labels)}",
            "重新运行 mask remap，使全局前景 ID 为 0..N-1。单帧只出现可见子集是允许的。",
        )
    for label in sorted(all_labels):
        ratios = area_by_id.get(label, [])
        stats = {
            "raw_mask_value": int(label),
            "holoscene_loaded_id": int(label + 1),
            "visible_frames": int(frame_count_by_id.get(label, 0)),
            "mean_area_ratio": float(np.mean(ratios)) if ratios else 0.0,
            "min_area_ratio": float(np.min(ratios)) if ratios else 0.0,
            "max_area_ratio": float(np.max(ratios)) if ratios else 0.0,
        }
        if stats["visible_frames"] <= 1:
            _warn(warnings, "mask", f"实例 raw_id={label} 只出现 {stats['visible_frames']} 帧。", "检查跟踪是否断裂或过滤该实例。")
        if stats["mean_area_ratio"] < 0.001:
            _warn(warnings, "mask", f"实例 raw_id={label} 平均面积过小: {stats['mean_area_ratio']:.6f}", "提高分割质量或过滤过小实例。")
        report["instances"][str(label)] = stats
    id_check = _check_id_mapping(id_mapping, all_labels, errors, warnings)
    report["ok"] = not any(err.startswith("[mask]") for err in errors)
    return report, id_check


def _check_id_mapping(
    id_mapping: dict[str, Any],
    actual_raw_ids: set[int],
    errors: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    report = {"ok": True, "actual_raw_ids": sorted(actual_raw_ids), "mapped_raw_ids": [], "objects": []}
    objects = id_mapping.get("objects", []) if isinstance(id_mapping, dict) else []
    if not isinstance(objects, list):
        _error(errors, "mask", "id_mapping.json 中 objects 不是 list。", "重新生成 id_mapping.json。")
        return report
    mapped_raw = set()
    for obj in objects:
        raw = int(obj.get("raw_mask_value", -999))
        node = int(obj.get("holoscene_node_id", -999))
        mapped_raw.add(raw)
        ok = node == raw + 1
        row = {"raw_mask_value": raw, "holoscene_node_id": node, "expected_loaded_id": raw + 1, "ok": ok, "label": obj.get("label")}
        if not ok:
            _error(errors, "mask", f"id_mapping raw_id={raw} 的 holoscene_node_id={node}，期望 {raw + 1}。", "按 HoloScene 加载规则修正 node_id。")
        report["objects"].append(row)
    report["mapped_raw_ids"] = sorted(mapped_raw)
    missing = actual_raw_ids - mapped_raw
    extra = mapped_raw - actual_raw_ids
    if missing:
        _error(errors, "mask", f"id_mapping 缺少实际 mask raw ids: {sorted(missing)}", "重新运行 mask 阶段生成 id_mapping。")
    if extra:
        _warn(warnings, "mask", f"id_mapping 包含未在 mask 中出现的 raw ids: {sorted(extra)}", "检查是否有实例被过滤或没有可见帧。")
    report["missing_mapping"] = sorted(missing)
    report["extra_mapping"] = sorted(extra)
    report["ok"] = not missing and all(item["ok"] for item in report["objects"])
    return report


def _check_depth(
    scene: Path,
    depths: dict[str, Path],
    resolution: list[int] | None,
    clip_min: float,
    clip_max: float,
    errors: list[str],
) -> dict[str, Any]:
    expected_hw = (resolution[1], resolution[0]) if resolution else None
    report: dict[str, Any] = {
        "num_frames": len(depths),
        "clip_min": clip_min,
        "clip_max": clip_max,
        "frames": {},
        "ok": True,
    }
    finite_values = []
    total_pixels = nan_count = inf_count = neg_count = below_clip = above_clip = 0
    for stem, path in sorted(depths.items()):
        try:
            raw = np.load(path)
        except Exception as exc:
            _error(errors, "depth", f"{stem} depth 无法读取: {exc}", "确认 depth 是 .npy 文件。")
            continue
        if raw.ndim != 2:
            _error(errors, "depth", f"{stem} depth shape={raw.shape}，不是 HxW。", "输出单通道 HxW depth .npy。")
        if expected_hw and tuple(raw.shape[:2]) != expected_hw:
            _error(errors, "depth", f"{stem} depth shape={raw.shape}，期望 {expected_hw}。", "重新运行 depth 阶段。")
        arr = np.asarray(raw, dtype=np.float64)
        finite = np.isfinite(arr)
        nan = int(np.isnan(arr).sum())
        inf = int(np.isinf(arr).sum())
        neg = int((arr < 0).sum())
        total = int(arr.size)
        total_pixels += total
        nan_count += nan
        inf_count += inf
        neg_count += neg
        below_clip += int((arr < clip_min).sum())
        above_clip += int((arr > clip_max).sum())
        vals = arr[finite]
        if vals.size:
            finite_values.append(vals.reshape(-1))
        frame_report = {
            "dtype": str(raw.dtype),
            "shape": list(raw.shape),
            "nan_count": nan,
            "inf_count": inf,
            "negative_count": neg,
            "min": float(np.min(vals)) if vals.size else None,
            "max": float(np.max(vals)) if vals.size else None,
            "mean": float(np.mean(vals)) if vals.size else None,
            "median": float(np.median(vals)) if vals.size else None,
            "below_clip_ratio": float(np.count_nonzero(arr < clip_min) / max(total, 1)),
            "above_clip_ratio": float(np.count_nonzero(arr > clip_max) / max(total, 1)),
        }
        if nan or inf:
            _error(errors, "depth", f"{stem} depth 存在 NaN/Inf: nan={nan}, inf={inf}", "清理 depth 后重新保存。")
        if neg:
            _error(errors, "depth", f"{stem} depth 存在负数: {neg} pixels", "检查深度单位和归一化。")
        report["frames"][stem] = frame_report
    all_vals = np.concatenate(finite_values) if finite_values else np.asarray([], dtype=np.float64)
    report.update(
        {
            "global_min": float(np.min(all_vals)) if all_vals.size else None,
            "global_max": float(np.max(all_vals)) if all_vals.size else None,
            "global_mean": float(np.mean(all_vals)) if all_vals.size else None,
            "global_median": float(np.median(all_vals)) if all_vals.size else None,
            "nan_ratio": float(nan_count / max(total_pixels, 1)),
            "inf_ratio": float(inf_count / max(total_pixels, 1)),
            "negative_ratio": float(neg_count / max(total_pixels, 1)),
            "below_clip_ratio": float(below_clip / max(total_pixels, 1)),
            "above_clip_ratio": float(above_clip / max(total_pixels, 1)),
        }
    )
    report["ok"] = not any(err.startswith("[depth]") for err in errors)
    return report


def _check_normals(
    scene: Path,
    normals: dict[str, Path],
    resolution: list[int] | None,
    errors: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    expected_wh = tuple(resolution) if resolution else None
    report: dict[str, Any] = {"num_frames": len(normals), "frames": {}, "ok": True}
    constant_frames = []
    length_means = []
    for stem, path in sorted(normals.items()):
        with Image.open(path) as im:
            mode = im.mode
            size = im.size
            arr = np.asarray(im.convert("RGB"), dtype=np.float32)
        if mode != "RGB":
            _error(errors, "normal", f"{stem} normal mode={mode}，不是 RGB。", "输出 RGB normal PNG。")
        if expected_wh and size != expected_wh:
            _error(errors, "normal", f"{stem} normal 尺寸 {size} != {expected_wh}", "重新运行 normal 阶段。")
        decoded = arr / 255.0 * 2.0 - 1.0
        length = np.linalg.norm(decoded, axis=-1)
        if not np.isfinite(decoded).all():
            _error(errors, "normal", f"{stem} normal 解码后存在 NaN/Inf。", "重新生成 normal。")
        spatial_std = np.std(arr.reshape(-1, 3), axis=0)
        constant = bool(float(np.max(spatial_std)) < 1e-6)
        if constant:
            constant_frames.append(stem)
        length_mean = float(np.mean(length))
        length_means.append(length_mean)
        bad_len_ratio = float(np.count_nonzero(np.abs(length - 1.0) > 0.25) / max(length.size, 1))
        if bad_len_ratio > 0.2:
            _warn(warnings, "normal", f"{stem} normal 向量长度异常比例较高: {bad_len_ratio:.3f}", "检查 normal 编码是否为 ((n+1)*0.5*255)。")
        report["frames"][stem] = {
            "mode": mode,
            "size": list(size),
            "raw_min": int(arr.min()),
            "raw_max": int(arr.max()),
            "length_mean": length_mean,
            "length_median": float(np.median(length)),
            "bad_length_ratio": bad_len_ratio,
            "spatial_std_rgb": spatial_std.astype(float).tolist(),
            "is_constant": constant,
        }
    if len(constant_frames) == len(normals) and normals:
        _warn(warnings, "normal", "所有 normal 帧都是常量图，可能是 dummy normal。", "接入真实 normal 或使用 depth_to_normal/Marigold。")
    report["constant_frames"] = constant_frames
    report["global_length_mean"] = float(np.mean(length_means)) if length_means else None
    report["ok"] = not any(err.startswith("[normal]") for err in errors)
    return report


def _check_graph(
    scene: Path,
    graph: list[dict[str, Any]] | Any,
    id_mapping: dict[str, Any],
    errors: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    report: dict[str, Any] = {"status": "valid", "nodes": [], "edges": [], "bfs_parent": {}, "ok": True}
    if not isinstance(graph, list):
        _error(errors, "graph", "graph.json 不是 list。", "按 HoloScene graph list 格式重新生成。")
        report["status"] = "invalid"
        report["ok"] = False
        return report
    adjacency: dict[int, set[int]] = {}
    seen_item_nodes: set[int] = set()
    duplicate_nodes = []
    for item in graph:
        node = int(item.get("node_id", -999))
        if node in seen_item_nodes:
            duplicate_nodes.append(node)
        seen_item_nodes.add(node)
        adjacency.setdefault(node, set())
        for adj in item.get("adj_nodes", []):
            adjacency.setdefault(node, set()).add(int(adj))
            adjacency.setdefault(int(adj), set()).add(node)
    report["nodes"] = sorted(adjacency)
    edges = sorted({tuple(sorted((a, b))) for a, values in adjacency.items() for b in values if a != b})
    report["edges"] = [list(e) for e in edges]
    if duplicate_nodes:
        _error(errors, "graph", f"graph 中存在重复 node: {sorted(set(duplicate_nodes))}", "去重 graph 节点。")
    if 0 not in adjacency:
        _error(errors, "graph", "graph 缺少 root node 0。", "添加 root 节点 0。")
    expected_nodes = {0}
    for obj in id_mapping.get("objects", []) if isinstance(id_mapping, dict) else []:
        expected_nodes.add(int(obj.get("holoscene_node_id", -999)))
    unknown = set(adjacency) - expected_nodes
    missing = expected_nodes - set(adjacency)
    if unknown:
        _error(errors, "graph", f"graph node_id 不在 id_mapping loaded ids 中: {sorted(unknown)}", "graph 使用 HoloScene loaded id，不要使用 raw mask id。")
    if missing:
        _error(errors, "graph", f"id_mapping 中的节点未出现在 graph 中: {sorted(missing)}", "补齐 graph 节点。")
    if 0 in adjacency:
        parent, reachable = _bfs_tree(adjacency, 0)
        report["bfs_parent"] = {str(k): int(v) if v is not None else None for k, v in parent.items()}
        unreachable = sorted(set(adjacency) - reachable)
        if unreachable:
            _error(errors, "graph", f"graph 存在无法从 root 到达的孤立节点: {unreachable}", "确保 graph 连通。")
        edge_count = len(edges)
        node_count = len(adjacency)
        has_cycle = edge_count >= node_count and node_count > 0
        report["has_cycle"] = bool(has_cycle)
        if has_cycle:
            _error(errors, "graph", "graph 存在环，无法形成干净 parent tree。", "删除多余边，使 graph 成为树或森林并从 root 连通。")
        object_nodes = sorted(n for n in adjacency if n != 0)
        all_root = object_nodes and all(parent.get(n) == 0 for n in object_nodes)
        report["all_objects_attached_to_root"] = bool(all_root)
        if all_root:
            _warn(warnings, "graph", "graph is valid but all objects are attached to root; support relation is weak", "若有明确支撑关系，可提供 manual graph。")
    report["expected_nodes_from_id_mapping"] = sorted(expected_nodes)
    report["status"] = "invalid" if any(err.startswith("[graph]") for err in errors) else "valid"
    report["ok"] = report["status"] == "valid"
    return report


def _bfs_tree(adjacency: dict[int, set[int]], root: int) -> tuple[dict[int, int | None], set[int]]:
    parent: dict[int, int | None] = {root: None}
    reached = {root}
    queue: deque[int] = deque([root])
    while queue:
        node = queue.popleft()
        for nxt in sorted(adjacency.get(node, set())):
            if nxt in reached:
                continue
            reached.add(nxt)
            parent[nxt] = node
            queue.append(nxt)
    return parent, reached


def _write_camera_trajectory_ply(path: Path, centers: list[list[float]]) -> Path:
    ensure_dir(path.parent)
    pts = np.asarray(centers, dtype=np.float64).reshape(-1, 3) if centers else np.zeros((0, 3), dtype=np.float64)
    with path.open("w", encoding="utf-8") as fh:
        fh.write("ply\nformat ascii 1.0\n")
        fh.write(f"element vertex {pts.shape[0]}\n")
        fh.write("property float x\nproperty float y\nproperty float z\n")
        fh.write("end_header\n")
        for x, y, z in pts:
            fh.write(f"{float(x)} {float(y)} {float(z)}\n")
    return path


def _write_validation(
    scene: Path,
    errors: list[str],
    warnings: list[str],
    checks: dict[str, Any],
    summary: dict[str, Any],
) -> dict[str, Any]:
    status = "fail" if errors else "warning" if warnings else "pass"
    report = {
        "ok": not errors,
        "scene_dir": str(scene),
        "status": status,
        "errors": errors,
        "warnings": warnings,
        "summary": summary,
        "checks": checks,
    }
    write_json(scene / "meta" / "validation_report.json", report)
    lines = ["# Validation Report", "", f"status: {status}", f"ok: {not errors}", ""]
    if summary:
        lines.extend(["## Summary", "```json", json.dumps(summary, ensure_ascii=False, indent=2), "```", ""])
    if errors:
        lines.append("## Errors")
        lines.extend(f"- {e}" for e in errors)
        lines.append("")
    if warnings:
        lines.append("## Warnings")
        lines.extend(f"- {w}" for w in warnings)
        lines.append("")
    lines.append("## Reports")
    lines.extend(
        [
            "- `meta/frame_consistency_report.json`",
            "- `meta/camera_report.json`",
            "- `meta/mask_report.json`",
            "- `meta/id_mapping_check.json`",
            "- `meta/depth_report.json`",
            "- `meta/normal_report.json`",
            "- `meta/graph_report.json`",
        ]
    )
    (scene / "meta" / "validation_report.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def _list_images(path: Path) -> list[Path]:
    return [p for p in sorted(path.glob("frame*")) if p.suffix.lower() in IMAGE_EXTS and p.is_file()]


def _read_json_file(path: Path, errors: list[str], label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        _error(errors, "files", f"{label} 无法读取 JSON: {exc}", "检查 JSON 格式。")
        return {}


def _error(errors: list[str], section: str, message: str, suggestion: str) -> None:
    errors.append(f"[{section}] {message} 修复建议: {suggestion}")


def _warn(warnings: list[str], section: str, message: str, suggestion: str) -> None:
    warnings.append(f"[{section}] {message} 建议: {suggestion}")
