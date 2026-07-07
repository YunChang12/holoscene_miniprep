"""VGGT integrations."""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from ..writer import ensure_dir, write_json
from .depth_wrapper import _extract_job_id
from .zaiwu_client import ZaiwuClient

LOGGER = logging.getLogger(__name__)


class VGGTWrapper:
    """External VGGT wrapper placeholder."""

    def run(self, image_dir, output_dir):
        """Estimate camera transforms with VGGT.

        This minimal project deliberately does not bundle VGGT. Provide
        transforms.json or implement this method to call the deployed Zaiwu
        ``services.vggt.reconstruct_scene_from_dir`` service.
        """

        raise RuntimeError(
            "VGGT is not installed in holoscene_miniprep. Provide camera.provided_transforms "
            "or implement VGGTWrapper.run() to call Zaiwu services.vggt."
        )


class ZaiwuVGGTWrapper:
    """Call Zaiwu VGGT and convert its output to HoloScene transforms.json."""

    handler = "services.vggt.reconstruct_scene_from_dir"

    def run(
        self,
        image_dir: str | Path,
        output_dir: str | Path,
        config: dict[str, Any],
        *,
        frame_count: int,
        resolution: tuple[int, int],
        scene_id: str,
    ) -> dict[str, Any]:
        scene = Path(output_dir)
        raw_dir = scene / "raw_outputs" / "vggt"
        if raw_dir.exists():
            shutil.rmtree(raw_dir)
        ensure_dir(raw_dir)

        service_url = str(config.get("service_url", ""))
        client = ZaiwuClient(service_url, timeout=float(config.get("request_timeout_sec", 30.0)))
        try:
            health = client.health_check()
            write_json(raw_dir / "health.json", health)
        except Exception as exc:
            write_json(raw_dir / "health.json", {"ok": False, "error": str(exc)})

        result = self._run_direct_or_gateway(client, Path(image_dir), raw_dir, config, scene_id=scene_id)
        write_json(raw_dir / "result.json", result)

        fallback = None
        fallback_path = config.get("fallback_transforms")
        if fallback_path:
            path = Path(str(fallback_path)).expanduser()
            if path.is_file():
                fallback = _load_fallback_transforms(path, frame_count, resolution)

        transforms, camera_report, compare_report = _convert_vggt_to_transforms(
            result,
            frame_count=frame_count,
            resolution=resolution,
            input_convention=str(config.get("input_convention", "auto")).lower(),
            fallback_transforms=fallback,
        )
        if bool(config.get("scale_align_with_depth", False)):
            scale_report = _scale_align_with_depth(scene, transforms, result)
            camera_report.update(scale_report)

        transforms["_camera_report_extra"] = camera_report
        write_json(raw_dir / "converted_transforms.json", {k: v for k, v in transforms.items() if not str(k).startswith("_")})
        write_json(scene / "meta" / "camera_source_report.json", camera_report)
        write_json(scene / "meta" / "camera_report.json", camera_report)
        if compare_report is not None:
            write_json(scene / "meta" / "camera_compare_report.json", compare_report)
        return transforms

    def _run_direct_or_gateway(
        self,
        client: ZaiwuClient,
        image_dir: Path,
        raw_dir: Path,
        config: dict[str, Any],
        *,
        scene_id: str,
    ) -> dict[str, Any]:
        capture_dir = str(image_dir.expanduser().resolve())
        prefer_direct = bool(config.get("prefer_direct_http", True))
        if prefer_direct:
            try:
                result = client.post_json(
                    "/reconstruct/scene_from_dir",
                    {"capture_dir": capture_dir, "scene_id": scene_id},
                )
                write_json(raw_dir / "direct_http.json", {"used": True})
                return result
            except Exception as exc:
                write_json(raw_dir / "direct_http.json", {"used": False, "error": str(exc)})

        try:
            record = client.submit_job(
                self.handler,
                {"capture_dir": capture_dir, "scene_id": scene_id},
                labels={"service_id": "services.vggt"},
            )
            write_json(raw_dir / "job_submit.json", record)
            job_id = _extract_job_id(record)
            final = client.poll_job(
                job_id,
                timeout_sec=float(config.get("job_timeout_sec", 1800.0)),
                poll_interval=float(config.get("poll_interval_sec", 2.0)),
            )
            write_json(raw_dir / "job_record.json", final)
            result = final.get("result")
            if not isinstance(result, dict):
                raise RuntimeError(f"VGGT job {job_id} succeeded but result is not an object: {result}")
            return result
        except Exception as exc:
            write_json(raw_dir / "gateway_job_error.json", {"error": str(exc)})
            return client.call_mcp_tool(
                "reconstruct_scene_from_dir",
                {"capture_dir": capture_dir, "scene_id": scene_id},
                sse_read_timeout=float(config.get("job_timeout_sec", 1800.0)),
            )


class LongSequenceVGGTWrapper:
    """Run vggt_long_sequence_pose and adapt its output for MiniPrep."""

    def run(
        self,
        image_dir: str | Path,
        output_dir: str | Path,
        config: dict[str, Any],
        *,
        frame_count: int,
        resolution: tuple[int, int],
        scene_id: str,
    ) -> dict[str, Any]:
        scene = Path(output_dir).expanduser().resolve()
        raw_dir = scene / "raw_outputs" / "vggt_long_sequence"
        if _config_bool(config, "overwrite", False) and raw_dir.exists():
            shutil.rmtree(raw_dir)
        ensure_dir(raw_dir)
        project_dir = _resolve_long_sequence_project_dir(config)
        script = project_dir / "scripts" / "run_vggt_long_sequence_pose.py"
        if not script.is_file():
            raise FileNotFoundError(f"long-sequence VGGT runner not found: {script}")

        cmd = self._build_command(
            script=script,
            image_dir=Path(image_dir),
            raw_dir=raw_dir,
            config=config,
            scene_id=scene_id,
        )
        write_json(raw_dir / "command.json", {"cwd": str(project_dir), "command": cmd})
        log_path = raw_dir / "run.log"
        LOGGER.info("[camera] running long-sequence VGGT: %s", " ".join(cmd))
        with log_path.open("w", encoding="utf-8") as log_file:
            proc = subprocess.run(  # noqa: S603 - command is constructed from config and local script path
                cmd,
                cwd=str(project_dir),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        write_json(raw_dir / "run_record.json", {"returncode": int(proc.returncode), "log": str(log_path)})
        if proc.returncode != 0:
            raise RuntimeError(f"long-sequence VGGT failed with exit code {proc.returncode}. See {log_path}")

        final_path = raw_dir / "final_transforms.json"
        if not final_path.is_file():
            raise RuntimeError(f"long-sequence VGGT did not write final_transforms.json: {final_path}")
        long_sequence_transforms = json.loads(final_path.read_text(encoding="utf-8"))
        transforms = _convert_long_sequence_transforms(
            long_sequence_transforms,
            frame_count=frame_count,
            resolution=resolution,
            camera_model=str(config.get("output_camera_model", config.get("camera_model", "OPENCV"))),
        )
        camera_report = _build_long_sequence_camera_report(
            raw_dir=raw_dir,
            project_dir=project_dir,
            final_path=final_path,
            command=cmd,
            source_transforms=long_sequence_transforms,
            output_transforms=transforms,
        )
        transforms["_camera_report_extra"] = camera_report
        write_json(raw_dir / "converted_transforms.json", {k: v for k, v in transforms.items() if not str(k).startswith("_")})
        write_json(scene / "meta" / "camera_source_report.json", camera_report)
        write_json(scene / "meta" / "camera_report.json", camera_report)
        return transforms

    def _build_command(
        self,
        *,
        script: Path,
        image_dir: Path,
        raw_dir: Path,
        config: dict[str, Any],
        scene_id: str,
    ) -> list[str]:
        service_url = str(config.get("service_url") or config.get("vggt_service_url") or "")
        if not service_url:
            raise ValueError("camera.mode=long_sequence_vggt requires camera.service_url or camera.vggt_service_url")
        python_executable = str(config.get("python_executable") or sys.executable)
        cmd = [
            python_executable,
            str(script),
            "--frames-dir",
            str(image_dir.expanduser().resolve()),
            "--output-dir",
            str(raw_dir),
            "--vggt-service-url",
            service_url,
            "--camera-model",
            str(config.get("long_sequence_camera_model", "OPENCV")),
        ]
        _append_cli_value(cmd, config, "max_image_size", "--max-image-size", 518)
        _append_cli_value(cmd, config, "keyframe_interval", "--keyframe-interval", 5)
        _append_cli_value(cmd, config, "key_window_size", "--key-window-size", 80)
        _append_cli_value(cmd, config, "key_overlap", "--key-overlap", 20)
        _append_cli_value(cmd, config, "all_window_size", "--all-window-size", 80)
        _append_cli_value(cmd, config, "all_overlap", "--all-overlap", 30)
        _append_cli_value(cmd, config, "dpt_chunk_size", "--dpt-chunk-size", 4)
        _append_cli_value(cmd, config, "dtype", "--dtype", "bf16")
        _append_cli_value(cmd, config, "input_convention", "--input-convention", "auto")
        _append_cli_value(cmd, config, "request_timeout_sec", "--request-timeout-sec", 30.0)
        _append_cli_value(cmd, config, "job_timeout_sec", "--job-timeout-sec", 1800.0)
        _append_cli_value(cmd, config, "poll_interval_sec", "--poll-interval-sec", 2.0)
        _append_cli_value(cmd, config, "min_anchor_frames", "--min-anchor-frames", 6)
        _append_cli_value(cmd, config, "ransac_threshold", "--ransac-threshold", 0.05)
        _append_cli_value(cmd, config, "ransac_iters", "--ransac-iters", 128)
        _append_cli_value(cmd, config, "max_anchor_translation_rmse", "--max-anchor-translation-rmse", 0.15)
        _append_cli_value(cmd, config, "max_anchor_rotation_deg", "--max-anchor-rotation-deg", 10.0)
        _append_cli_value(cmd, config, "max_scale_ratio", "--max-scale-ratio", 3.0)
        _append_cli_value(cmd, config, "quality_good_anchor_rmse", "--quality-good-anchor-rmse", 0.05)
        _append_cli_value(cmd, config, "quality_good_anchor_rotation_deg", "--quality-good-anchor-rotation-deg", 5.0)
        _append_cli_value(cmd, config, "quality_min_overlap_frames", "--quality-min-overlap-frames", 10)
        _append_cli_value(cmd, config, "max_overlap_translation_p90", "--max-overlap-translation-p90", None)
        _append_cli_value(cmd, config, "max_overlap_rotation_p90_deg", "--max-overlap-rotation-p90-deg", None)
        _append_cli_value(cmd, config, "min_overlap_rotation_p90_deg", "--min-overlap-rotation-p90-deg", 15.0)
        _append_cli_value(cmd, config, "min_window_quality_score", "--min-window-quality-score", 0.05)
        _append_cli_value(cmd, config, "low_quality_fallback_weight", "--low-quality-fallback-weight", 0.25)
        if _config_bool(config, "prefer_direct_http", not service_url.rstrip("/").endswith("/sse")):
            cmd.append("--prefer-direct-http")
        if _config_bool(config, "disable_depth", True):
            cmd.append("--disable-depth")
        if _config_bool(config, "enable_pose_graph", False):
            cmd.append("--enable-pose-graph")
        resume_default = not _config_bool(config, "overwrite", False)
        if _config_bool(config, "resume", resume_default):
            cmd.append("--resume")
        if _config_bool(config, "debug", False):
            cmd.append("--debug")
        if _config_bool(config, "disable_window_quality_gating", False):
            cmd.append("--disable-window-quality-gating")
        intrinsics_json = config.get("intrinsics_json")
        if intrinsics_json:
            cmd.extend(["--intrinsics-json", str(Path(str(intrinsics_json)).expanduser())])
        return cmd


def _convert_vggt_to_transforms(
    result: dict[str, Any],
    *,
    frame_count: int,
    resolution: tuple[int, int],
    input_convention: str,
    fallback_transforms: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
    frames_in = result.get("frames")
    if not isinstance(frames_in, list):
        raise RuntimeError(f"VGGT result missing frames list: {result.keys()}")
    if len(frames_in) != int(frame_count):
        raise RuntimeError(f"VGGT frame count {len(frames_in)} does not match images {frame_count}")

    width, height = resolution
    intrinsics = [_as_matrix(frame.get("intrinsic"), (3, 3), f"intrinsic[{idx}]") for idx, frame in enumerate(frames_in)]
    fx = float(np.median([k[0, 0] for k in intrinsics]))
    fy = float(np.median([k[1, 1] for k in intrinsics]))
    cx = float(np.median([k[0, 2] for k in intrinsics]))
    cy = float(np.median([k[1, 2] for k in intrinsics]))

    extrinsics = [_as_pose_matrix(frame.get("extrinsic"), f"extrinsic[{idx}]") for idx, frame in enumerate(frames_in)]
    convention, inverted, compare_for_choice = _choose_convention(
        extrinsics,
        input_convention=input_convention,
        fallback_transforms=fallback_transforms,
    )

    pose_mats = []
    for mat in extrinsics:
        pose = _invert_pose(mat) if convention == "world_to_camera" else mat.copy()
        pose[3] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        pose_mats.append(pose)

    transforms = {
        "camera_model": "OPENCV",
        "fl_x": fx,
        "fl_y": fy,
        "cx": cx,
        "cy": cy,
        "h": int(height),
        "w": int(width),
        "frames": [
            {
                "file_path": f"images/frame{idx:06d}.jpg",
                "transform_matrix": pose.tolist(),
            }
            for idx, pose in enumerate(pose_mats)
        ],
    }

    camera_report: dict[str, Any] = {
        "source": "zaiwu_vggt",
        "input_convention": input_convention,
        "resolved_input_convention": convention,
        "output_convention": "camera_to_world",
        "inverted": bool(inverted),
        "scale_aligned": False,
        "scale_factor": 1.0,
        "model_name": result.get("model_name"),
        "backend": result.get("backend"),
        "num_images": result.get("num_images"),
        "depth_confidence_mean": _safe_mean([f.get("depth_confidence_mean") for f in frames_in]),
        "point_confidence_mean": _safe_mean([f.get("point_confidence_mean") for f in frames_in]),
    }
    if compare_for_choice:
        camera_report["convention_choice"] = compare_for_choice

    compare_report = None
    if fallback_transforms is not None:
        compare_report = _compare_transforms(transforms, fallback_transforms)
    return transforms, camera_report, compare_report


def _resolve_long_sequence_project_dir(config: dict[str, Any]) -> Path:
    value = config.get("long_sequence_project_dir") or config.get("project_dir") or os.environ.get("HOLOSCENE_LONG_SEQUENCE_VGGT_DIR")
    if value:
        path = Path(str(value)).expanduser().resolve()
    else:
        default = Path("/autodl-fs/data/Chengpeng/vggt_long_sequence_pose")
        if default.is_dir():
            path = default
        else:
            raise ValueError(
                "camera.mode=long_sequence_vggt requires camera.long_sequence_project_dir "
                "or camera.project_dir pointing to vggt_long_sequence_pose."
            )
    if not path.is_dir():
        raise FileNotFoundError(f"long-sequence VGGT project_dir not found: {path}")
    return path


def _append_cli_value(cmd: list[str], config: dict[str, Any], key: str, option: str, default: Any) -> None:
    value = config.get(key, default)
    if value is None:
        return
    cmd.extend([option, str(value)])


def _config_bool(config: dict[str, Any], key: str, default: bool) -> bool:
    value = config.get(key, default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _convert_long_sequence_transforms(
    data: dict[str, Any],
    *,
    frame_count: int,
    resolution: tuple[int, int],
    camera_model: str,
) -> dict[str, Any]:
    frames_in = data.get("frames")
    if not isinstance(frames_in, list):
        raise RuntimeError(f"long-sequence final_transforms.json missing frames list: {data.keys()}")
    if len(frames_in) != int(frame_count):
        raise RuntimeError(f"long-sequence frame count {len(frames_in)} does not match images {frame_count}")
    width, height = resolution
    old_w = float(data.get("w") or width)
    old_h = float(data.get("h") or height)
    sx = float(width) / max(old_w, 1e-6)
    sy = float(height) / max(old_h, 1e-6)
    frames = []
    for idx, frame in enumerate(frames_in):
        pose = _as_pose_matrix(frame.get("transform_matrix"), f"long_sequence[{idx}]").copy()
        pose[3] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        frames.append(
            {
                "file_path": f"images/frame{idx:06d}.jpg",
                "transform_matrix": pose.tolist(),
            }
        )
    return {
        "camera_model": camera_model or "OPENCV",
        "fl_x": float(data.get("fl_x", width)) * sx,
        "fl_y": float(data.get("fl_y", height)) * sy,
        "cx": float(data.get("cx", old_w / 2.0)) * sx,
        "cy": float(data.get("cy", old_h / 2.0)) * sy,
        "h": int(height),
        "w": int(width),
        "frames": frames,
    }


def _build_long_sequence_camera_report(
    *,
    raw_dir: Path,
    project_dir: Path,
    final_path: Path,
    command: list[str],
    source_transforms: dict[str, Any],
    output_transforms: dict[str, Any],
) -> dict[str, Any]:
    pose_report = _read_json_if_exists(raw_dir / "pose_quality_report.json")
    window_quality = _read_json_if_exists(raw_dir / "all_frame_alignment" / "window_quality_report.json")
    frames = output_transforms.get("frames", [])
    pose_mats = [_as_pose_matrix(frame.get("transform_matrix"), f"output[{idx}]") for idx, frame in enumerate(frames)]
    translation_steps = _translation_steps(pose_mats)
    rotation_steps = _rotation_steps_deg(pose_mats)
    failed_key = pose_report.get("failed_key_windows", []) if isinstance(pose_report, dict) else []
    failed_all = pose_report.get("failed_all_windows", []) if isinstance(pose_report, dict) else []
    suspicious = pose_report.get("suspicious_frames", []) if isinstance(pose_report, dict) else []
    dropped = window_quality.get("dropped_windows", []) if isinstance(window_quality, dict) else []
    downweighted = window_quality.get("downweighted_windows", []) if isinstance(window_quality, dict) else []
    missing = source_transforms.get("missing_pose_filled_frames", [])
    return {
        "source": "long_sequence_vggt",
        "long_sequence_project_dir": str(project_dir),
        "long_sequence_output_dir": str(raw_dir),
        "source_final_transforms": str(final_path),
        "command": command,
        "frame_count": len(frames),
        "camera_model": output_transforms.get("camera_model"),
        "width": output_transforms.get("w"),
        "height": output_transforms.get("h"),
        "fl_x": output_transforms.get("fl_x"),
        "fl_y": output_transforms.get("fl_y"),
        "cx": output_transforms.get("cx"),
        "cy": output_transforms.get("cy"),
        "output_convention": "camera_to_world",
        "missing_pose_filled_count": len(missing) if isinstance(missing, list) else 0,
        "failed_key_window_count": len(failed_key) if isinstance(failed_key, list) else 0,
        "failed_all_window_count": len(failed_all) if isinstance(failed_all, list) else 0,
        "suspicious_frame_count": len(suspicious) if isinstance(suspicious, list) else 0,
        "window_quality_gating_enabled": window_quality.get("enabled") if isinstance(window_quality, dict) else None,
        "dropped_window_count": len(dropped) if isinstance(dropped, list) else 0,
        "downweighted_window_count": len(downweighted) if isinstance(downweighted, list) else 0,
        "trajectory": {
            "max_translation_step": float(np.max(translation_steps)) if translation_steps.size else 0.0,
            "median_translation_step": float(np.median(translation_steps)) if translation_steps.size else 0.0,
            "max_rotation_step_deg": float(max(rotation_steps)) if rotation_steps else 0.0,
        },
    }


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _load_fallback_transforms(path: Path, frame_count: int, resolution: tuple[int, int]) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    frames = data.get("frames")
    if not isinstance(frames, list) or len(frames) != int(frame_count):
        raise RuntimeError(f"fallback_transforms frame count does not match images: {path}")
    width, height = resolution
    old_w = float(data.get("w") or width)
    old_h = float(data.get("h") or height)
    sx = float(width) / max(old_w, 1e-6)
    sy = float(height) / max(old_h, 1e-6)
    return {
        "camera_model": data.get("camera_model", "OPENCV"),
        "fl_x": float(data.get("fl_x", width)) * sx,
        "fl_y": float(data.get("fl_y", height)) * sy,
        "cx": float(data.get("cx", old_w / 2.0)) * sx,
        "cy": float(data.get("cy", old_h / 2.0)) * sy,
        "w": int(width),
        "h": int(height),
        "frames": [
            {
                "file_path": f"images/frame{idx:06d}.jpg",
                "transform_matrix": _as_pose_matrix(frame.get("transform_matrix"), f"fallback[{idx}]").tolist(),
            }
            for idx, frame in enumerate(frames)
        ],
    }


def _choose_convention(
    extrinsics: list[np.ndarray],
    *,
    input_convention: str,
    fallback_transforms: dict[str, Any] | None,
) -> tuple[str, bool, dict[str, Any] | None]:
    if input_convention in {"camera_to_world", "c2w"}:
        return "camera_to_world", False, None
    if input_convention in {"world_to_camera", "w2c"}:
        return "world_to_camera", True, None
    if input_convention != "auto":
        raise ValueError("camera.input_convention must be auto, camera_to_world, or world_to_camera")
    if fallback_transforms is None:
        # VGGT core uses pose_encoding_to_extri_intri, whose extrinsic is normally world-to-camera.
        return "world_to_camera", True, {"reason": "no fallback_transforms; defaulted to VGGT world_to_camera"}

    as_c2w = [m.copy() for m in extrinsics]
    as_w2c = [_invert_pose(m) for m in extrinsics]
    fallback = [_as_pose_matrix(f["transform_matrix"], f"fallback[{idx}]") for idx, f in enumerate(fallback_transforms.get("frames", []))]
    c2w_score = _trajectory_distance(as_c2w, fallback)
    w2c_score = _trajectory_distance(as_w2c, fallback)
    if w2c_score <= c2w_score:
        return "world_to_camera", True, {"c2w_score": c2w_score, "w2c_score": w2c_score}
    return "camera_to_world", False, {"c2w_score": c2w_score, "w2c_score": w2c_score}


def _compare_transforms(candidate: dict[str, Any], fallback: dict[str, Any]) -> dict[str, Any]:
    cand_frames = candidate.get("frames", [])
    ref_frames = fallback.get("frames", [])
    n = min(len(cand_frames), len(ref_frames))
    cand = [_as_pose_matrix(cand_frames[idx]["transform_matrix"], f"candidate[{idx}]") for idx in range(n)]
    ref = [_as_pose_matrix(ref_frames[idx]["transform_matrix"], f"fallback[{idx}]") for idx in range(n)]
    center_dists = _aligned_center_distances(cand, ref)
    cand_steps = _translation_steps(cand)
    ref_steps = _translation_steps(ref)
    cand_rot = _rotation_steps_deg(cand)
    ref_rot = _rotation_steps_deg(ref)
    return {
        "frame_count": int(n),
        "intrinsics": {
            "candidate": {k: candidate.get(k) for k in ["fl_x", "fl_y", "cx", "cy", "w", "h"]},
            "fallback": {k: fallback.get(k) for k in ["fl_x", "fl_y", "cx", "cy", "w", "h"]},
            "absolute_diff": {
                k: float(abs(float(candidate.get(k, 0.0)) - float(fallback.get(k, 0.0))))
                for k in ["fl_x", "fl_y", "cx", "cy"]
            },
        },
        "center_distance_after_similarity_align": {
            "median": float(np.median(center_dists)) if center_dists.size else 0.0,
            "max": float(np.max(center_dists)) if center_dists.size else 0.0,
            "per_frame": center_dists.astype(float).tolist(),
        },
        "translation_step": {
            "candidate": cand_steps.astype(float).tolist(),
            "fallback": ref_steps.astype(float).tolist(),
            "median_abs_diff": float(np.median(np.abs(cand_steps[: len(ref_steps)] - ref_steps[: len(cand_steps)])))
            if len(cand_steps) and len(ref_steps)
            else 0.0,
        },
        "rotation_step_deg": {
            "candidate": cand_rot,
            "fallback": ref_rot,
            "median_abs_diff": float(np.median(np.abs(np.asarray(cand_rot[: len(ref_rot)]) - np.asarray(ref_rot[: len(cand_rot)]))))
            if cand_rot and ref_rot
            else 0.0,
        },
        "warnings": _camera_compare_warnings(center_dists, cand_steps, cand_rot),
    }


def _scale_align_with_depth(scene: Path, transforms: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    # Current Zaiwu VGGT result does not expose dense VGGT depth, only point maps
    # written as artifacts. Keep this explicit so the user knows scale alignment
    # was requested but skipped instead of silently pretending it happened.
    frames = result.get("frames") if isinstance(result, dict) else None
    has_depth = isinstance(frames, list) and any("depth" in frame or "depth_map" in frame for frame in frames if isinstance(frame, dict))
    if not has_depth:
        return {
            "scale_aligned": False,
            "scale_factor": 1.0,
            "scale_warning": "scale_align_with_depth requested, but current VGGT result has no dense depth field to compare with DA3.",
        }
    return {
        "scale_aligned": False,
        "scale_factor": 1.0,
        "scale_warning": "VGGT dense-depth scale alignment is reserved; no stable depth field schema was detected.",
    }


def _as_matrix(value: Any, shape: tuple[int, int], name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != shape:
        raise RuntimeError(f"Expected {name} shape {shape}, got {arr.shape}")
    if not np.isfinite(arr).all():
        raise RuntimeError(f"{name} contains NaN/Inf")
    return arr


def _as_pose_matrix(value: Any, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape == (3, 4):
        out = np.eye(4, dtype=np.float64)
        out[:3, :4] = arr
        arr = out
    if arr.shape != (4, 4):
        raise RuntimeError(f"Expected {name} shape 4x4 or 3x4, got {arr.shape}")
    if not np.isfinite(arr).all():
        raise RuntimeError(f"{name} contains NaN/Inf")
    return arr


def _invert_pose(mat: np.ndarray) -> np.ndarray:
    inv = np.linalg.inv(mat)
    inv[3] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return inv


def _centers(mats: list[np.ndarray]) -> np.ndarray:
    if not mats:
        return np.zeros((0, 3), dtype=np.float64)
    return np.stack([mat[:3, 3] for mat in mats], axis=0).astype(np.float64)


def _trajectory_distance(a: list[np.ndarray], b: list[np.ndarray]) -> float:
    dist = _aligned_center_distances(a, b)
    return float(np.median(dist)) if dist.size else float("inf")


def _aligned_center_distances(a: list[np.ndarray], b: list[np.ndarray]) -> np.ndarray:
    n = min(len(a), len(b))
    if n <= 0:
        return np.zeros((0,), dtype=np.float64)
    ca = _centers(a[:n])
    cb = _centers(b[:n])
    ca0 = ca - ca.mean(axis=0, keepdims=True)
    cb0 = cb - cb.mean(axis=0, keepdims=True)
    scale = float(np.linalg.norm(cb0) / max(np.linalg.norm(ca0), 1e-8))
    ca_aligned = ca0 * scale + cb.mean(axis=0, keepdims=True)
    return np.linalg.norm(ca_aligned - cb, axis=1)


def _translation_steps(mats: list[np.ndarray]) -> np.ndarray:
    centers = _centers(mats)
    if len(centers) < 2:
        return np.zeros((0,), dtype=np.float64)
    return np.linalg.norm(np.diff(centers, axis=0), axis=1)


def _rotation_steps_deg(mats: list[np.ndarray]) -> list[float]:
    values = []
    for a, b in zip(mats[:-1], mats[1:]):
        rel = a[:3, :3].T @ b[:3, :3]
        cos = np.clip((np.trace(rel) - 1.0) * 0.5, -1.0, 1.0)
        values.append(float(np.degrees(np.arccos(cos))))
    return values


def _safe_mean(values: list[Any]) -> float | None:
    nums = [float(v) for v in values if v is not None]
    return float(np.mean(nums)) if nums else None


def _camera_compare_warnings(center_dists: np.ndarray, trans_steps: np.ndarray, rot_steps: list[float]) -> list[str]:
    warnings = []
    if center_dists.size and float(np.median(center_dists)) > 0.25:
        warnings.append("VGGT camera centers differ from fallback by a large median distance after similarity alignment.")
    if trans_steps.size >= 3:
        med = float(np.median(trans_steps))
        mx = float(np.max(trans_steps))
        if med > 1e-8 and mx > max(10.0 * med, med + 1.0):
            warnings.append("VGGT trajectory has a large translation jump.")
    if rot_steps and max(rot_steps) > 90.0:
        warnings.append("VGGT trajectory has a large rotation jump.")
    return warnings
