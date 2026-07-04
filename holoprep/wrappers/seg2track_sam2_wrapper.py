"""Zaiwu Seg2Track-SAM2 integration for stable instance masks."""

from __future__ import annotations

import json
import shutil
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
    default_ignored_labels = ("floor", "wall", "ceiling")

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
            min_area_ratio=float(config.get("min_area_ratio", 0.001)),
            min_visible_frames=int(config.get("min_visible_frames", 3)),
            max_area_ratio=_optional_float(config.get("max_area_ratio")),
            ignored_labels=_string_list(
                config.get("ignored_labels", config.get("background_labels", self.default_ignored_labels))
            ),
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
    min_area_ratio: float,
    min_visible_frames: int,
    max_area_ratio: float | None = None,
    ignored_labels: list[str] | None = None,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    width, height = resolution
    frames = result.get("frames")
    if not isinstance(frames, list):
        raise RuntimeError(f"Seg2Track result missing frames list: {result.keys()}")

    track_stats: dict[str, dict[str, Any]] = {}
    per_frame_instances: list[list[dict[str, Any]]] = [[] for _ in range(frame_count)]
    ignored_track_ids: set[str] = set()
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
            if _label_is_ignored(label, ignored_labels or []):
                ignored_track_ids.add(track_id)
                continue
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
    discarded_by_reason: dict[str, list[str]] = {
        "too_small_or_short": [],
        "too_large": [],
        "ignored_label": sorted(ignored_track_ids),
    }
    for track_id, stats in track_stats.items():
        visible = len(stats["visible_frames"])
        area_ratio = float(stats["area_pixels"]) / float(total_pixels)
        if visible < int(min_visible_frames) or area_ratio < float(min_area_ratio):
            discarded_by_reason["too_small_or_short"].append(track_id)
            continue
        if max_area_ratio is not None and area_ratio > float(max_area_ratio):
            discarded_by_reason["too_large"].append(track_id)
            continue
        kept_track_ids.append(track_id)
    kept_track_ids.sort(key=lambda tid: (min(track_stats[tid]["visible_frames"]), tid))
    track_to_raw = {track_id: idx for idx, track_id in enumerate(kept_track_ids)}

    background_value = 255
    masks = [np.full((height, width), background_value, dtype=np.uint8) for _ in range(frame_count)]
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

    masks, raw_to_final, removed_after_overlap = _remap_visible_labels_contiguous(
        masks,
        expected_raw_ids=track_to_raw.values(),
        background_value=background_value,
    )

    objects = []
    tracks: dict[str, Any] = {}
    for track_id, original_raw_id in track_to_raw.items():
        raw_id = raw_to_final.get(original_raw_id)
        if raw_id is None:
            continue
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
            "source_raw_mask_value": int(original_raw_id),
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
            "max_area_ratio": None if max_area_ratio is None else float(max_area_ratio),
            "ignored_labels": ignored_labels or [],
            "discarded_tracks": discarded_by_filter,
            "discarded_by_reason": {key: sorted(value) for key, value in discarded_by_reason.items()},
            "discarded_after_overlap": removed_after_overlap,
        },
        "overlap": {
            "overlap_pixels_by_frame": overlap_pixels_by_frame,
            "total_overlap_pixels": int(sum(overlap_pixels_by_frame.values())),
        },
    }
    return masks, mapping


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _string_list(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, str):
        return [part.strip().lower() for part in value.replace(",", ".").split(".") if part.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(part).strip().lower() for part in value if str(part).strip()]
    raise ValueError("ignored_labels/background_labels must be a string or list")


def _label_is_ignored(label: str, ignored_labels: list[str]) -> bool:
    normalized = " ".join(str(label).lower().replace("_", " ").replace("-", " ").split())
    if not normalized:
        return False
    return any(ignored and ignored in normalized for ignored in ignored_labels)


def _remap_visible_labels_contiguous(
    masks: list[np.ndarray],
    *,
    expected_raw_ids: Any | None = None,
    background_value: int = 255,
) -> tuple[list[np.ndarray], dict[int, int], list[int]]:
    """Drop labels that have no surviving pixels and remap visible ids to 0..N-1."""

    visible = sorted(
        int(value)
        for mask in masks
        for value in np.unique(mask).tolist()
        if int(value) != int(background_value)
    )
    visible_unique = sorted(set(visible))
    raw_to_final = {old: new for new, old in enumerate(visible_unique)}
    if expected_raw_ids is None:
        expected = set(range(max(visible_unique) + 1)) if visible_unique else set()
    else:
        expected = {int(value) for value in expected_raw_ids}
    removed = sorted(expected - set(visible_unique))

    remapped: list[np.ndarray] = []
    for mask in masks:
        out = np.full(mask.shape, int(background_value), dtype=np.uint8)
        for old, new in raw_to_final.items():
            out[mask == int(old)] = int(new)
        remapped.append(out)
    return remapped, raw_to_final, removed


def _decode_mask(mask_rle: Any, *, width: int, height: int) -> np.ndarray | None:
    if not mask_rle:
        return None
    if isinstance(mask_rle, str):
        obj = json.loads(mask_rle)
    elif isinstance(mask_rle, dict):
        obj = dict(mask_rle)
    else:
        raise RuntimeError(f"Unsupported mask_rle type: {type(mask_rle).__name__}")
    try:
        from pycocotools import mask as mask_util  # type: ignore

        rle_obj = dict(obj)
        counts = rle_obj.get("counts")
        if isinstance(counts, str):
            rle_obj["counts"] = counts.encode("utf-8")
        mask = mask_util.decode(rle_obj).astype(np.uint8)
    except Exception:
        mask = _decode_coco_rle(obj)
    if mask.ndim == 3:
        mask = mask[..., 0]
    if mask.shape != (height, width):
        mask_img = Image.fromarray((mask > 0).astype(np.uint8) * 255, mode="L")
        mask_img = mask_img.resize((width, height), Image.Resampling.NEAREST)
        mask = (np.asarray(mask_img, dtype=np.uint8) > 127).astype(np.uint8)
    return mask.astype(bool)


def _decode_coco_rle(obj: dict[str, Any]) -> np.ndarray:
    size = obj.get("size")
    if not isinstance(size, (list, tuple)) or len(size) != 2:
        raise RuntimeError(f"COCO RLE missing size: {obj}")
    height, width = int(size[0]), int(size[1])
    counts_value = obj.get("counts")
    if isinstance(counts_value, str):
        counts = _decode_compressed_coco_counts(counts_value)
    elif isinstance(counts_value, bytes):
        counts = _decode_compressed_coco_counts(counts_value.decode("utf-8"))
    elif isinstance(counts_value, list):
        counts = [int(value) for value in counts_value]
    else:
        raise RuntimeError(f"Unsupported COCO RLE counts type: {type(counts_value).__name__}")
    flat = np.zeros(height * width, dtype=np.uint8)
    offset = 0
    value = 0
    for count in counts:
        next_offset = min(offset + int(count), flat.size)
        if value == 1 and next_offset > offset:
            flat[offset:next_offset] = 1
        offset = next_offset
        value = 1 - value
        if offset >= flat.size:
            break
    return flat.reshape((height, width), order="F")


def _decode_compressed_coco_counts(text: str) -> list[int]:
    counts: list[int] = []
    cursor = 0
    while cursor < len(text):
        value = 0
        shift = 0
        while True:
            if cursor >= len(text):
                raise RuntimeError("Malformed compressed COCO RLE counts")
            char_value = ord(text[cursor]) - 48
            cursor += 1
            value |= (char_value & 0x1F) << shift
            more = char_value & 0x20
            shift += 5
            if not more:
                if char_value & 0x10:
                    value |= -1 << shift
                break
        if len(counts) > 2:
            value += counts[-2]
        counts.append(int(value))
    return counts
