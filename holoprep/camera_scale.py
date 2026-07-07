"""Depth-based camera scale alignment for VGGT-style trajectories."""

from __future__ import annotations

import copy
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


def align_camera_scale_with_depth(
    scene_dir: str | Path,
    transforms: dict[str, Any],
    config: dict[str, Any],
    *,
    frame_count: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Estimate a global camera-center scale from depth correspondences.

    The returned transforms keep the original NeRF/HoloScene schema. Only
    ``frames[*].transform_matrix[:3, 3]`` is changed when alignment succeeds.
    """

    method = str(config.get("method", "depth_correspondence")).lower()
    if method not in {"depth_correspondence", "depth_correspondences"}:
        return dict(transforms), {
            "scale_aligned": False,
            "method": method,
            "reason": f"unsupported_method:{method}",
        }

    try:
        frames = _validate_transforms(transforms, frame_count)
        observations, pair_reports = collect_depth_scale_observations(scene_dir, transforms, config, frames=frames)
        pair_selection = _summarize_pair_selection(pair_reports, config)
        num_pairs_used = int(pair_selection["used_pairs"])
        min_used_pairs = int(config.get("min_used_pairs", _default_min_used_pairs(config)))
        estimate = estimate_scale_from_observations(observations, config)
        if not estimate.get("ok", False):
            report = {
                "scale_aligned": False,
                "method": method,
                "reason": estimate.get("reason", "scale_estimation_failed"),
                "frame_count": int(frame_count),
                "num_pairs": len(pair_reports),
                "num_pairs_used": num_pairs_used,
                "min_used_pairs": min_used_pairs,
                "num_observations": len(observations),
                "pair_selection": pair_selection,
                "pair_reports": pair_reports,
            }
            report.update({k: v for k, v in estimate.items() if k != "ok"})
            return dict(transforms), report
        if num_pairs_used < min_used_pairs:
            report = {
                "scale_aligned": False,
                "method": method,
                "reason": "not_enough_used_frame_pairs",
                "frame_count": int(frame_count),
                "num_pairs": len(pair_reports),
                "num_pairs_used": num_pairs_used,
                "min_used_pairs": min_used_pairs,
                "num_observations": len(observations),
                "pair_selection": pair_selection,
                "pair_reports": pair_reports,
            }
            report.update({k: v for k, v in estimate.items() if k != "ok"})
            return dict(transforms), report
        spread_failure = _pair_scale_spread_failure(estimate, config)
        if spread_failure:
            report = {
                "scale_aligned": False,
                "method": method,
                "reason": "pair_scale_candidates_inconsistent",
                "frame_count": int(frame_count),
                "num_pairs": len(pair_reports),
                "num_pairs_used": num_pairs_used,
                "min_used_pairs": min_used_pairs,
                "num_observations": len(observations),
                "pair_selection": pair_selection,
                "pair_reports": pair_reports,
            }
            report.update(spread_failure)
            report.update({k: v for k, v in estimate.items() if k != "ok"})
            return dict(transforms), report

        scale_factor = float(estimate["scale_factor"])
        aligned = apply_camera_scale_to_transforms(
            transforms,
            scale_factor,
            anchor=config.get("anchor", "first"),
        )
        report = {
            "scale_aligned": True,
            "method": method,
            "scale_factor": scale_factor,
            "anchor": config.get("anchor", "first"),
            "frame_count": int(frame_count),
            "num_pairs": len(pair_reports),
            "num_pairs_used": num_pairs_used,
            "min_used_pairs": min_used_pairs,
            "num_observations": len(observations),
            "pair_selection": pair_selection,
            "pair_reports": pair_reports,
            "output_format": "transforms_json_schema_preserved",
            "modified_fields": ["frames[].transform_matrix[:3][3]"],
        }
        report.update({k: v for k, v in estimate.items() if k not in {"ok", "scale_factor"}})
        return aligned, report
    except Exception as exc:
        return dict(transforms), {
            "scale_aligned": False,
            "method": method,
            "reason": str(exc),
            "frame_count": int(frame_count),
        }


def collect_depth_scale_observations(
    scene_dir: str | Path,
    transforms: dict[str, Any],
    config: dict[str, Any],
    *,
    frames: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Collect per-match scalar scale observations from RGB/depth pairs."""

    scene = Path(scene_dir)
    frames = frames if frames is not None else _validate_transforms(transforms, len(transforms.get("frames", [])))
    fx = float(transforms["fl_x"])
    fy = float(transforms["fl_y"])
    cx = float(transforms["cx"])
    cy = float(transforms["cy"])
    depth_dir = scene / str(config.get("depth_dir", "depth"))
    if not depth_dir.is_dir():
        raise FileNotFoundError(f"depth_dir not found: {depth_dir}")

    pairs = _select_frame_pairs(len(frames), config=config, frames=frames)
    observations: list[dict[str, Any]] = []
    pair_reports: list[dict[str, Any]] = []
    for i, j in pairs:
        pair_obs, pair_report = _collect_pair_observations(
            scene,
            depth_dir,
            frames,
            i,
            j,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            config=config,
        )
        observations.extend(pair_obs)
        pair_reports.append(pair_report)
    return observations, pair_reports


def estimate_scale_from_observations(observations: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    """Robustly estimate one global scale from scalar observations."""

    min_observations = int(config.get("min_observations", config.get("min_matches", 50)))
    pair_scale_stats = _pair_scale_statistics(observations)
    if len(observations) < min_observations:
        report = {
            "ok": False,
            "reason": "not_enough_valid_depth_correspondences",
            "min_observations": min_observations,
            "num_observations": len(observations),
        }
        report.update(pair_scale_stats)
        return report

    scales = np.asarray([float(obs["scale_candidate"]) for obs in observations], dtype=np.float64)
    valid = np.isfinite(scales) & (scales > 0.0)
    min_scale = config.get("min_scale_factor")
    max_scale = config.get("max_scale_factor")
    if min_scale is not None:
        valid &= scales >= float(min_scale)
    if max_scale is not None:
        valid &= scales <= float(max_scale)
    if int(np.sum(valid)) < min_observations:
        report = {
            "ok": False,
            "reason": "not_enough_positive_scale_candidates",
            "min_observations": min_observations,
            "num_observations": int(np.sum(valid)),
        }
        report.update(pair_scale_stats)
        return report

    percentile = config.get("robust_percentile", [10, 90])
    if isinstance(percentile, (list, tuple)) and len(percentile) == 2 and int(np.sum(valid)) >= 10:
        lo_pct = float(percentile[0])
        hi_pct = float(percentile[1])
        lo, hi = np.percentile(scales[valid], [lo_pct, hi_pct])
        valid &= (scales >= float(lo)) & (scales <= float(hi))

    if int(np.sum(valid)) < min_observations:
        report = {
            "ok": False,
            "reason": "not_enough_candidates_after_percentile_filter",
            "min_observations": min_observations,
            "num_observations": int(np.sum(valid)),
        }
        report.update(pair_scale_stats)
        return report

    initial_scale = float(np.median(scales[valid]))
    residuals = _residuals_for_scale(observations, initial_scale)
    threshold = float(config.get("ransac_threshold", 0.15))
    inliers = valid & np.isfinite(residuals) & (residuals <= threshold)
    if int(np.sum(inliers)) < min_observations:
        # Keep the failure explicit; a loose bad scale is worse than no scale.
        report = {
            "ok": False,
            "reason": "not_enough_inliers_after_depth_residual_filter",
            "scale_factor_initial": initial_scale,
            "ransac_threshold": threshold,
            "min_observations": min_observations,
            "num_observations": int(np.sum(valid)),
            "num_inliers": int(np.sum(inliers)),
            "residual_median_initial": float(np.median(residuals[valid])) if np.any(valid) else None,
        }
        report.update(pair_scale_stats)
        return report

    scale_factor = float(np.median(scales[inliers]))
    final_residuals = _residuals_for_scale(observations, scale_factor)
    inlier_residuals = final_residuals[inliers]
    inlier_scales = scales[inliers]
    report = {
        "ok": True,
        "scale_factor": scale_factor,
        "scale_factor_initial": initial_scale,
        "num_inliers": int(np.sum(inliers)),
        "inlier_ratio": float(np.sum(inliers) / max(np.sum(valid), 1)),
        "scale_candidate_median": float(np.median(inlier_scales)),
        "scale_candidate_p10": float(np.percentile(inlier_scales, 10)),
        "scale_candidate_p90": float(np.percentile(inlier_scales, 90)),
        "residual_median": float(np.median(inlier_residuals)) if inlier_residuals.size else None,
        "residual_p90": float(np.percentile(inlier_residuals, 90)) if inlier_residuals.size else None,
        "ransac_threshold": threshold,
    }
    report.update(pair_scale_stats)
    return report


def apply_camera_scale_to_transforms(
    transforms: dict[str, Any],
    scale_factor: float,
    *,
    anchor: Any = "first",
) -> dict[str, Any]:
    """Return a copy with camera centers scaled about an anchor point."""

    out = copy.deepcopy(transforms)
    frames = out.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("transforms.frames must be a non-empty list")
    centers = []
    for idx, frame in enumerate(frames):
        mat = _pose_matrix(frame.get("transform_matrix"), f"transform_matrix[{idx}]")
        centers.append(mat[:3, 3])
    anchor_center = _resolve_anchor_center(np.stack(centers, axis=0), anchor)
    for idx, frame in enumerate(frames):
        mat = _pose_matrix(frame.get("transform_matrix"), f"transform_matrix[{idx}]")
        mat[:3, 3] = anchor_center + float(scale_factor) * (mat[:3, 3] - anchor_center)
        mat[3] = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        frame["transform_matrix"] = mat.astype(float).tolist()
    return out


def _collect_pair_observations(
    scene: Path,
    depth_dir: Path,
    frames: list[dict[str, Any]],
    i: int,
    j: int,
    *,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pose_i = _pose_matrix(frames[i].get("transform_matrix"), f"transform_matrix[{i}]")
    pose_j = _pose_matrix(frames[j].get("transform_matrix"), f"transform_matrix[{j}]")
    center_i = pose_i[:3, 3]
    center_j = pose_j[:3, 3]
    baseline = float(np.linalg.norm(center_i - center_j))
    report: dict[str, Any] = {
        "frame_i": int(i),
        "frame_j": int(j),
        "gap": int(j - i),
        "baseline": baseline,
        "used": False,
        "num_matches": 0,
        "num_observations": 0,
    }
    if baseline < float(config.get("min_baseline", 0.02)):
        report["reason"] = "baseline_too_small"
        return [], report

    matches = _match_image_pair(
        scene / frames[i]["file_path"],
        scene / frames[j]["file_path"],
        max_features=int(config.get("max_features", 4000)),
        ratio=float(config.get("match_ratio", 0.75)),
        max_matches=int(config.get("max_matches_per_pair", 800)),
    )
    report["num_matches"] = len(matches)
    if not matches:
        report["reason"] = "no_feature_matches"
        return [], report

    depth_i = np.load(depth_dir / f"frame{i:06d}.npy")
    depth_j = np.load(depth_dir / f"frame{j:06d}.npy")
    min_depth = float(config.get("min_depth", 0.05))
    max_depth = float(config.get("max_depth", 20.0))
    r_ji = pose_j[:3, :3].T @ pose_i[:3, :3]
    t_ji = pose_j[:3, :3].T @ (center_i - center_j)
    denom = float(np.dot(t_ji, t_ji))
    if denom < 1e-12:
        report["reason"] = "relative_translation_too_small"
        return [], report

    observations: list[dict[str, Any]] = []
    residual_limit = float(config.get("candidate_residual_threshold", max(float(config.get("ransac_threshold", 0.15)) * 3.0, 0.3)))
    for match in matches:
        u_i, v_i = match["pt_i"]
        u_j, v_j = match["pt_j"]
        z_i = _sample_depth(depth_i, u_i, v_i)
        z_j = _sample_depth(depth_j, u_j, v_j)
        if not (_valid_depth(z_i, min_depth, max_depth) and _valid_depth(z_j, min_depth, max_depth)):
            continue
        x_i = _backproject_point(float(u_i), float(v_i), float(z_i), fx, fy, cx, cy)
        x_j = _backproject_point(float(u_j), float(v_j), float(z_j), fx, fy, cx, cy)
        rotated = r_ji @ x_i
        rhs = x_j - rotated
        scale = float(np.dot(t_ji, rhs) / denom)
        if not math.isfinite(scale) or scale <= 0.0:
            continue
        residual = float(np.linalg.norm(rotated + scale * t_ji - x_j))
        if residual > residual_limit:
            continue
        observations.append(
            {
                "frame_i": int(i),
                "frame_j": int(j),
                "scale_candidate": scale,
                "candidate_residual": residual,
                "match_distance": float(match["distance"]),
                "t": t_ji.astype(float).tolist(),
                "rotated_point": rotated.astype(float).tolist(),
                "target_point": x_j.astype(float).tolist(),
            }
        )

    report["num_observations"] = len(observations)
    min_matches = int(config.get("min_matches", 50))
    if len(observations) < min_matches:
        report["reason"] = "not_enough_depth_valid_matches"
        return [], report
    report["used"] = True
    report["scale_candidate_median"] = float(np.median([obs["scale_candidate"] for obs in observations]))
    report["residual_median"] = float(np.median([obs["candidate_residual"] for obs in observations]))
    return observations, report


def _match_image_pair(
    image_i: Path,
    image_j: Path,
    *,
    max_features: int,
    ratio: float,
    max_matches: int,
) -> list[dict[str, Any]]:
    try:
        import cv2  # type: ignore
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("camera_scale requires opencv-python for feature matching") from exc

    img_i = cv2.imread(str(image_i), cv2.IMREAD_GRAYSCALE)
    img_j = cv2.imread(str(image_j), cv2.IMREAD_GRAYSCALE)
    if img_i is None:
        raise FileNotFoundError(f"image not readable: {image_i}")
    if img_j is None:
        raise FileNotFoundError(f"image not readable: {image_j}")
    orb = cv2.ORB_create(nfeatures=max(100, int(max_features)))
    kp_i, des_i = orb.detectAndCompute(img_i, None)
    kp_j, des_j = orb.detectAndCompute(img_j, None)
    if des_i is None or des_j is None or not kp_i or not kp_j:
        return []
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    raw = matcher.knnMatch(des_i, des_j, k=2)
    matches = []
    for item in raw:
        if len(item) < 2:
            continue
        first, second = item
        if float(first.distance) > float(ratio) * float(second.distance):
            continue
        pt_i = kp_i[first.queryIdx].pt
        pt_j = kp_j[first.trainIdx].pt
        matches.append(
            {
                "pt_i": (float(pt_i[0]), float(pt_i[1])),
                "pt_j": (float(pt_j[0]), float(pt_j[1])),
                "distance": float(first.distance),
            }
        )
    matches.sort(key=lambda item: item["distance"])
    return matches[: max(1, int(max_matches))]


def _select_frame_pairs(
    num_frames: int,
    *,
    pair_gap: int | None = None,
    max_pairs: int | None = None,
    config: dict[str, Any] | None = None,
    frames: list[dict[str, Any]] | None = None,
) -> list[tuple[int, int]]:
    """Select image pairs for scale estimation.

    Direct ``pair_gap`` calls keep the legacy single-gap behavior. Pipeline
    calls pass ``config`` and default to a multi-gap, baseline-aware strategy so
    slow trajectories still contribute enough usable frame pairs.
    """

    if num_frames < 2:
        return []
    cfg = dict(config or {})
    if pair_gap is not None and config is None:
        cfg["pair_strategy"] = "single_gap"
        cfg["pair_gap"] = pair_gap
    if max_pairs is not None:
        cfg.setdefault("max_pairs", max_pairs)

    strategy = _pair_strategy(cfg)
    max_pair_count = int(cfg.get("max_pairs", 80))
    if strategy == "single_gap":
        gap = int(cfg.get("pair_gap", cfg.get("pair_stride", 5)))
        return _select_single_gap_pairs(num_frames, pair_gap=gap, max_pairs=max_pair_count)
    return _select_multi_gap_pairs(num_frames, config=cfg, frames=frames)


def _select_single_gap_pairs(num_frames: int, *, pair_gap: int, max_pairs: int) -> list[tuple[int, int]]:
    gap = max(1, int(pair_gap))
    if gap >= num_frames:
        gap = num_frames - 1
    starts = list(range(0, num_frames - gap))
    if not starts:
        return []
    if max_pairs > 0 and len(starts) > int(max_pairs):
        idxs = np.linspace(0, len(starts) - 1, int(max_pairs)).round().astype(int)
        starts = [starts[int(idx)] for idx in idxs]
    return [(int(i), int(i + gap)) for i in starts]


def _select_multi_gap_pairs(
    num_frames: int,
    *,
    config: dict[str, Any],
    frames: list[dict[str, Any]] | None,
) -> list[tuple[int, int]]:
    gaps = _resolve_pair_gaps(num_frames, config)
    if not gaps:
        return []

    centers = _frame_centers(frames) if frames is not None else None
    candidates: list[dict[str, Any]] = []
    for gap in gaps:
        for i in range(0, num_frames - gap):
            baseline = None
            if centers is not None:
                baseline = float(np.linalg.norm(centers[i] - centers[i + gap]))
            candidates.append({"frame_i": int(i), "frame_j": int(i + gap), "gap": int(gap), "baseline": baseline})
    if not candidates:
        return []

    selection_min_baseline = _optional_float(config.get("selection_min_baseline", config.get("min_baseline")))
    eligible = candidates
    if selection_min_baseline is not None and centers is not None:
        eligible = [item for item in candidates if float(item["baseline"]) >= selection_min_baseline]
        if not eligible:
            eligible = candidates

    max_pairs = int(config.get("max_pairs", 80))
    if max_pairs <= 0:
        return [(int(item["frame_i"]), int(item["frame_j"])) for item in sorted(eligible, key=_pair_sort_key)]

    selected: list[dict[str, Any]] = []
    selected_keys: set[tuple[int, int]] = set()
    gap_quota = _gap_quotas(gaps, max_pairs)

    # First pass: preserve temporal coverage inside every gap.
    for gap in gaps:
        group = [item for item in eligible if int(item["gap"]) == int(gap)]
        if not group:
            continue
        group.sort(key=lambda item: int(item["frame_i"]))
        quota = min(int(gap_quota.get(gap, 0)), len(group))
        for idx in _linspace_indices(len(group), quota):
            item = group[idx]
            key = (int(item["frame_i"]), int(item["frame_j"]))
            if key not in selected_keys:
                selected.append(item)
                selected_keys.add(key)

    # Second pass: fill remaining slots with strongest-baseline candidates.
    remaining = [item for item in eligible if (int(item["frame_i"]), int(item["frame_j"])) not in selected_keys]
    remaining.sort(key=lambda item: _pair_priority_key(item, centers is not None))
    for item in remaining:
        if len(selected) >= max_pairs:
            break
        selected.append(item)
        selected_keys.add((int(item["frame_i"]), int(item["frame_j"])))

    selected.sort(key=_pair_sort_key)
    return [(int(item["frame_i"]), int(item["frame_j"])) for item in selected[:max_pairs]]


def _pair_strategy(config: dict[str, Any]) -> str:
    value = str(config.get("pair_strategy", "multi_gap")).strip().lower()
    if value in {"single", "single_gap", "fixed", "fixed_gap"}:
        return "single_gap"
    return "multi_gap"


def _default_min_used_pairs(config: dict[str, Any]) -> int:
    return 1 if _pair_strategy(config) == "single_gap" else 3


def _resolve_pair_gaps(num_frames: int, config: dict[str, Any]) -> list[int]:
    explicit = config.get("pair_gaps")
    if explicit not in (None, ""):
        gaps = _int_list(explicit)
    else:
        base_gap = max(1, int(config.get("pair_gap", config.get("pair_stride", 5))))
        multipliers = _int_list(config.get("pair_gap_multipliers", [1, 2, 4, 8, 16]))
        gaps = [base_gap * max(1, int(mult)) for mult in multipliers]
    max_pair_gap = config.get("max_pair_gap")
    if max_pair_gap not in (None, ""):
        max_gap = max(1, int(max_pair_gap))
        gaps = [gap for gap in gaps if gap <= max_gap]
    gaps = sorted({int(gap) for gap in gaps if 1 <= int(gap) < int(num_frames)})
    if not gaps and num_frames >= 2:
        gaps = [min(max(1, int(config.get("pair_gap", config.get("pair_stride", 5)))), num_frames - 1)]
    return gaps


def _int_list(value: Any) -> list[int]:
    if isinstance(value, str):
        parts = [part.strip() for part in value.replace(",", " ").split() if part.strip()]
        return [int(part) for part in parts]
    if isinstance(value, (list, tuple, set)):
        return [int(item) for item in value]
    return [int(value)]


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _frame_centers(frames: list[dict[str, Any]]) -> np.ndarray:
    centers = []
    for idx, frame in enumerate(frames):
        centers.append(_pose_matrix(frame.get("transform_matrix"), f"transform_matrix[{idx}]")[:3, 3])
    return np.stack(centers, axis=0)


def _gap_quotas(gaps: list[int], max_pairs: int) -> dict[int, int]:
    if not gaps:
        return {}
    base = int(max_pairs) // len(gaps)
    extra = int(max_pairs) % len(gaps)
    return {gap: base + (1 if idx < extra else 0) for idx, gap in enumerate(gaps)}


def _linspace_indices(length: int, count: int) -> list[int]:
    if length <= 0 or count <= 0:
        return []
    if count >= length:
        return list(range(length))
    return [int(idx) for idx in np.linspace(0, length - 1, int(count)).round().astype(int)]


def _pair_sort_key(item: dict[str, Any]) -> tuple[int, int, int]:
    return int(item["frame_i"]), int(item["frame_j"]), int(item["gap"])


def _pair_priority_key(item: dict[str, Any], has_baseline: bool) -> tuple[float, int, int]:
    baseline_score = -float(item["baseline"]) if has_baseline and item.get("baseline") is not None else 0.0
    return baseline_score, -int(item["gap"]), int(item["frame_i"])


def _summarize_pair_selection(pair_reports: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    gaps = [int(item.get("gap", int(item["frame_j"]) - int(item["frame_i"]))) for item in pair_reports]
    used_gaps = [
        int(item.get("gap", int(item["frame_j"]) - int(item["frame_i"])))
        for item in pair_reports
        if item.get("used")
    ]
    baselines = [float(item["baseline"]) for item in pair_reports if item.get("baseline") is not None]
    used_baselines = [float(item["baseline"]) for item in pair_reports if item.get("used") and item.get("baseline") is not None]
    summary: dict[str, Any] = {
        "strategy": _pair_strategy(config),
        "pair_gaps": _resolve_pair_gaps(max((int(item["frame_j"]) for item in pair_reports), default=1) + 1, config),
        "selected_pairs": len(pair_reports),
        "used_pairs": int(sum(1 for item in pair_reports if item.get("used"))),
        "gap_counts": {str(key): int(value) for key, value in sorted(Counter(gaps).items())},
        "used_gap_counts": {str(key): int(value) for key, value in sorted(Counter(used_gaps).items())},
    }
    selection_min_baseline = _optional_float(config.get("selection_min_baseline", config.get("min_baseline")))
    if selection_min_baseline is not None:
        summary["selection_min_baseline"] = selection_min_baseline
    if baselines:
        summary["baseline_min"] = float(np.min(baselines))
        summary["baseline_median"] = float(np.median(baselines))
        summary["baseline_max"] = float(np.max(baselines))
    if used_baselines:
        summary["used_baseline_min"] = float(np.min(used_baselines))
        summary["used_baseline_median"] = float(np.median(used_baselines))
        summary["used_baseline_max"] = float(np.max(used_baselines))
    return summary


def _pair_scale_statistics(observations: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[int, int], list[float]] = {}
    for obs in observations:
        if "frame_i" not in obs or "frame_j" not in obs:
            continue
        scale = float(obs.get("scale_candidate", float("nan")))
        if not math.isfinite(scale) or scale <= 0.0:
            continue
        key = (int(obs["frame_i"]), int(obs["frame_j"]))
        grouped.setdefault(key, []).append(scale)
    if not grouped:
        return {}
    pair_scales = np.asarray([float(np.median(values)) for values in grouped.values() if values], dtype=np.float64)
    if pair_scales.size == 0:
        return {}
    p10 = float(np.percentile(pair_scales, 10))
    p90 = float(np.percentile(pair_scales, 90))
    return {
        "num_pair_scale_candidates": int(pair_scales.size),
        "pair_scale_median": float(np.median(pair_scales)),
        "pair_scale_p10": p10,
        "pair_scale_p90": p90,
        "pair_scale_p90_p10_ratio": float(p90 / p10) if p10 > 0.0 else None,
    }


def _pair_scale_spread_failure(estimate: dict[str, Any], config: dict[str, Any]) -> dict[str, Any] | None:
    if _pair_strategy(config) == "single_gap":
        return None
    threshold_value = config.get("max_pair_scale_spread_ratio", 3.0)
    if threshold_value in (None, "", 0, "0"):
        return None
    threshold = float(threshold_value)
    ratio = estimate.get("pair_scale_p90_p10_ratio")
    num_pair_scales = int(estimate.get("num_pair_scale_candidates", 0) or 0)
    if ratio is None or num_pair_scales < 3:
        return None
    if float(ratio) <= threshold:
        return None
    return {
        "max_pair_scale_spread_ratio": threshold,
        "pair_scale_p90_p10_ratio": float(ratio),
        "num_pair_scale_candidates": num_pair_scales,
    }


def _validate_transforms(transforms: dict[str, Any], frame_count: int) -> list[dict[str, Any]]:
    frames = transforms.get("frames")
    if not isinstance(frames, list):
        raise ValueError("transforms.frames must be a list")
    if len(frames) != int(frame_count):
        raise ValueError(f"transforms frame count {len(frames)} does not match image count {frame_count}")
    for key in ["fl_x", "fl_y", "cx", "cy"]:
        value = float(transforms[key])
        if not math.isfinite(value):
            raise ValueError(f"transforms.{key} is not finite")
    for idx, frame in enumerate(frames):
        _pose_matrix(frame.get("transform_matrix"), f"transform_matrix[{idx}]")
        if not frame.get("file_path"):
            raise ValueError(f"frame {idx} missing file_path")
    return frames


def _pose_matrix(value: Any, name: str) -> np.ndarray:
    mat = np.asarray(value, dtype=np.float64)
    if mat.shape != (4, 4):
        raise ValueError(f"{name} must be 4x4, got {mat.shape}")
    if not np.isfinite(mat).all():
        raise ValueError(f"{name} contains NaN/Inf")
    return mat.copy()


def _resolve_anchor_center(centers: np.ndarray, anchor: Any) -> np.ndarray:
    if isinstance(anchor, str):
        key = anchor.strip().lower()
        if key in {"first", "start"}:
            return centers[0].copy()
        if key in {"mean", "center"}:
            return np.mean(centers, axis=0)
        if key in {"origin", "zero"}:
            return np.zeros(3, dtype=np.float64)
        try:
            idx = int(key)
            return centers[max(0, min(idx, len(centers) - 1))].copy()
        except ValueError:
            raise ValueError(f"Unsupported camera_scale.anchor: {anchor}") from None
    idx = int(anchor)
    return centers[max(0, min(idx, len(centers) - 1))].copy()


def _sample_depth(depth: np.ndarray, u: float, v: float) -> float:
    arr = np.asarray(depth, dtype=np.float32)
    if arr.ndim != 2:
        return float("nan")
    h, w = arr.shape
    x = int(round(float(u)))
    y = int(round(float(v)))
    if x < 0 or y < 0 or x >= w or y >= h:
        return float("nan")
    return float(arr[y, x])


def _valid_depth(value: float, min_depth: float, max_depth: float) -> bool:
    return math.isfinite(float(value)) and float(min_depth) <= float(value) <= float(max_depth)


def _backproject_point(u: float, v: float, z: float, fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    return np.asarray([(u - cx) * z / fx, (v - cy) * z / fy, z], dtype=np.float64)


def _residuals_for_scale(observations: list[dict[str, Any]], scale: float) -> np.ndarray:
    values = []
    for obs in observations:
        t = np.asarray(obs["t"], dtype=np.float64)
        rotated = np.asarray(obs["rotated_point"], dtype=np.float64)
        target = np.asarray(obs["target_point"], dtype=np.float64)
        values.append(float(np.linalg.norm(rotated + float(scale) * t - target)))
    return np.asarray(values, dtype=np.float64)
