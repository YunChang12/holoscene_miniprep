"""Zaiwu Seg2Track-SAM2 integration for stable instance masks."""

from __future__ import annotations

import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from ..frame_utils import load_image_sequence
from ..writer import ensure_dir, write_json
from .depth_wrapper import _extract_job_id
from .zaiwu_client import ZaiwuClient


class ZaiwuSeg2TrackSAM2Wrapper:
    """Call Zaiwu ``services.seg2track_sam2.seg2track_parse_video``."""

    handler = "services.seg2track_sam2.seg2track_parse_video"

    def run(
        self,
        image_dir: str | Path,
        output_dir: str | Path,
        config: dict[str, Any],
        *,
        frame_count: int,
        resolution: tuple[int, int],
    ) -> tuple[list[np.ndarray], dict[str, Any]]:
        scene = Path(output_dir)
        raw_dir = scene / "raw_outputs" / "seg2track_sam2"
        if raw_dir.exists():
            shutil.rmtree(raw_dir)
        ensure_dir(raw_dir)

        client = ZaiwuClient(
            str(config.get("service_url", "")),
            timeout=float(config.get("request_timeout_sec", 30.0)),
        )
        try:
            health = client.health_check()
            write_json(raw_dir / "health.json", health)
        except Exception as exc:
            write_json(raw_dir / "health.json", {"ok": False, "error": str(exc)})

        video_path = raw_dir / "input.mp4"
        _make_video_from_images(Path(image_dir), video_path, fps=float(config.get("fps", 8.0)), resolution=resolution)
        payload = {
            "text_prompt": str(config.get("text_prompt", "object.")),
            "detect_interval": int(config.get("detect_interval", 5)),
            "window_size": int(config.get("window_size", 300)),
            "box_threshold": float(config.get("box_threshold", 0.3)),
            "text_threshold": float(config.get("text_threshold", 0.25)),
        }
        result: dict[str, Any]
        try:
            video_file_id = client.upload_file(video_path)
            write_json(raw_dir / "upload.json", {"video_file_id": video_file_id, "video_path": str(video_path)})
            job_payload = dict(payload)
            job_payload["video_file_id"] = video_file_id
            record = client.submit_job(self.handler, job_payload, labels={"service_id": "services.seg2track_sam2"})
            write_json(raw_dir / "job_submit.json", record)
            job_id = _extract_job_id(record)
            final = client.poll_job(
                job_id,
                timeout_sec=float(config.get("job_timeout_sec", 1800.0)),
                poll_interval=float(config.get("poll_interval_sec", 2.0)),
            )
            write_json(raw_dir / "job_record.json", final)
            job_result = final.get("result")
            if not isinstance(job_result, dict):
                raise RuntimeError(f"Seg2Track job {job_id} succeeded but result is not an object: {job_result}")
            result = job_result
        except Exception as exc:
            write_json(raw_dir / "gateway_job_error.json", {"error": str(exc)})
            # Direct worker file service has its own /upload endpoint; retry there.
            video_file_id = client.upload_file(video_path)
            write_json(raw_dir / "direct_upload.json", {"video_file_id": video_file_id, "video_path": str(video_path)})
            direct_payload = dict(payload)
            direct_payload["video_file_id"] = video_file_id
            result = client.call_mcp_tool(
                "seg2track_parse_video",
                direct_payload,
                sse_read_timeout=float(config.get("job_timeout_sec", 1800.0)),
            )
        if not isinstance(result, dict):
            raise RuntimeError(f"Seg2Track result is not an object: {result}")
        write_json(raw_dir / "result.json", result)

        masks, mapping = _convert_seg2track_result(
            result,
            frame_count=frame_count,
            resolution=resolution,
            background_value=int(config.get("background_value", 255)),
            min_area_ratio=float(config.get("min_area_ratio", 0.001)),
            min_visible_frames=int(config.get("min_visible_frames", 3)),
        )
        write_json(raw_dir / "converted_id_mapping.json", mapping)
        return masks, mapping


def _make_video_from_images(image_dir: Path, output_path: Path, *, fps: float, resolution: tuple[int, int]) -> None:
    try:
        import cv2  # type: ignore
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("seg2track_sam2 requires opencv-python to create a temporary MP4 input") from exc

    width, height = resolution
    paths = load_image_sequence(image_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (int(width), int(height)),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create temporary video: {output_path}")
    try:
        for path in paths:
            with Image.open(path) as im:
                rgb = im.convert("RGB").resize((width, height), Image.Resampling.BILINEAR)
            bgr = cv2.cvtColor(np.asarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR)
            writer.write(bgr)
    finally:
        writer.release()


def _convert_seg2track_result(
    result: dict[str, Any],
    *,
    frame_count: int,
    resolution: tuple[int, int],
    background_value: int,
    min_area_ratio: float,
    min_visible_frames: int,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    width, height = resolution
    frames = result.get("frames")
    if not isinstance(frames, list):
        raise RuntimeError(f"Seg2Track result missing frames list: {result.keys()}")

    track_stats: dict[str, dict[str, Any]] = {}
    per_frame_instances: list[list[dict[str, Any]]] = [[] for _ in range(frame_count)]
    for frame in frames:
        raw_frame_idx = int(frame.get("frame_idx", frame.get("frame_index", 0)))
        # Seg2Track service reports frame_idx as 1-based; tolerate 0-based too.
        frame_idx = raw_frame_idx - 1 if raw_frame_idx >= 1 else raw_frame_idx
        if frame_idx < 0 or frame_idx >= frame_count:
            continue
        instances = frame.get("instances") or []
        if not isinstance(instances, list):
            continue
        for inst in instances:
            if not isinstance(inst, dict):
                continue
            track_id = str(inst.get("track_id", ""))
            if not track_id:
                continue
            mask = _decode_mask(inst.get("mask_rle"), width=width, height=height)
            if mask is None:
                continue
            area = int(mask.sum())
            if area <= 0:
                continue
            score = float(inst.get("score", 0.0) or 0.0)
            label = str(inst.get("concept_label", inst.get("label", track_id)))
            per_frame_instances[frame_idx].append(
                {
                    "track_id": track_id,
                    "label": label,
                    "score": score,
                    "area": area,
                    "mask": mask,
                    "bbox": inst.get("bbox"),
                }
            )
            stats = track_stats.setdefault(
                track_id,
                {"track_id": track_id, "label": label, "visible_frames": set(), "scores": [], "area_pixels": 0},
            )
            stats["visible_frames"].add(frame_idx)
            stats["scores"].append(score)
            stats["area_pixels"] += area

    total_pixels = max(1, int(width) * int(height) * int(frame_count))
    kept_track_ids = []
    for track_id, stats in track_stats.items():
        visible = len(stats["visible_frames"])
        area_ratio = float(stats["area_pixels"]) / float(total_pixels)
        if visible >= int(min_visible_frames) and area_ratio >= float(min_area_ratio):
            kept_track_ids.append(track_id)
    kept_track_ids.sort(key=lambda tid: (min(track_stats[tid]["visible_frames"]), tid))
    track_to_raw = {track_id: idx for idx, track_id in enumerate(kept_track_ids)}

    masks = [np.full((height, width), int(background_value), dtype=np.uint8) for _ in range(frame_count)]
    overlap_pixels_by_frame: dict[str, int] = {}
    discarded_by_filter = sorted(set(track_stats) - set(kept_track_ids))
    for frame_idx, instances in enumerate(per_frame_instances):
        canvas_score = np.full((height, width), -np.inf, dtype=np.float32)
        overlap_pixels = 0
        for inst in sorted(instances, key=lambda item: float(item["score"])):
            raw_id = track_to_raw.get(str(inst["track_id"]))
            if raw_id is None:
                continue
            binary = np.asarray(inst["mask"], dtype=bool)
            score = float(inst["score"])
            overlap_pixels += int((binary & (masks[frame_idx] != background_value)).sum())
            write_region = binary & (score >= canvas_score)
            masks[frame_idx][write_region] = int(raw_id)
            canvas_score[write_region] = score
        if overlap_pixels:
            overlap_pixels_by_frame[f"frame{frame_idx:06d}"] = int(overlap_pixels)

    objects = []
    tracks: dict[str, Any] = {}
    for track_id, raw_id in track_to_raw.items():
        stats = track_stats[track_id]
        scores = [float(v) for v in stats["scores"]]
        item = {
            "track_id": track_id,
            "raw_mask_value": int(raw_id),
            "holoscene_node_id": int(raw_id + 1),
            "label": str(stats["label"]),
            "source_label": str(stats["label"]),
            "visible_frames": int(len(stats["visible_frames"])),
            "mean_score": float(np.mean(scores)) if scores else 0.0,
            "area_ratio": float(stats["area_pixels"]) / float(total_pixels),
        }
        objects.append(item)
        tracks[track_id] = item

    mapping = {
        "loader_rule": "background_255_to_0_and_object_id_plus_1",
        "background_value": int(background_value),
        "source": "zaiwu_seg2track_sam2",
        "objects": objects,
        "tracks": tracks,
        "filtering": {
            "min_area_ratio": float(min_area_ratio),
            "min_visible_frames": int(min_visible_frames),
            "discarded_tracks": discarded_by_filter,
        },
        "overlap": {
            "overlap_pixels_by_frame": overlap_pixels_by_frame,
            "total_overlap_pixels": int(sum(overlap_pixels_by_frame.values())),
        },
    }
    return masks, mapping


def _decode_mask(mask_rle: Any, *, width: int, height: int) -> np.ndarray | None:
    if not mask_rle:
        return None
    try:
        from pycocotools import mask as mask_util  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("seg2track_sam2 mask_rle decoding requires pycocotools in this environment") from exc

    if isinstance(mask_rle, str):
        obj = json.loads(mask_rle)
    elif isinstance(mask_rle, dict):
        obj = dict(mask_rle)
    else:
        raise RuntimeError(f"Unsupported mask_rle type: {type(mask_rle).__name__}")
    counts = obj.get("counts")
    if isinstance(counts, str):
        obj["counts"] = counts.encode("utf-8")
    mask = mask_util.decode(obj).astype(np.uint8)
    if mask.ndim == 3:
        mask = mask[..., 0]
    if mask.shape != (height, width):
        mask_img = Image.fromarray((mask > 0).astype(np.uint8) * 255, mode="L")
        mask_img = mask_img.resize((width, height), Image.Resampling.NEAREST)
        mask = (np.asarray(mask_img, dtype=np.uint8) > 127).astype(np.uint8)
    return mask.astype(bool)
