from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from holoprep.camera_scale import (
    apply_camera_scale_to_transforms,
    estimate_scale_from_observations,
    _pair_scale_spread_failure,
    _select_frame_pairs,
)
from holoprep.pipeline import parse_stages


class CameraScaleTests(unittest.TestCase):
    def test_estimate_scale_from_observations_rejects_outliers(self) -> None:
        rng = np.random.default_rng(3)
        observations = []
        true_scale = 2.5
        for _ in range(80):
            t = rng.normal(size=3)
            t = t / max(np.linalg.norm(t), 1e-8)
            rotated = rng.normal(size=3)
            target = rotated + true_scale * t + rng.normal(scale=0.005, size=3)
            scale_candidate = float(np.dot(t, target - rotated) / np.dot(t, t))
            observations.append(
                {
                    "scale_candidate": scale_candidate,
                    "t": t.tolist(),
                    "rotated_point": rotated.tolist(),
                    "target_point": target.tolist(),
                }
            )
        for _ in range(15):
            t = np.asarray([1.0, 0.0, 0.0])
            rotated = rng.normal(size=3)
            target = rotated + 10.0 * t
            observations.append(
                {
                    "scale_candidate": 10.0,
                    "t": t.tolist(),
                    "rotated_point": rotated.tolist(),
                    "target_point": target.tolist(),
                }
            )

        report = estimate_scale_from_observations(
            observations,
            {"min_observations": 20, "ransac_threshold": 0.05, "robust_percentile": [0, 100]},
        )
        self.assertTrue(report["ok"])
        self.assertAlmostEqual(report["scale_factor"], true_scale, delta=0.02)
        self.assertLess(report["residual_median"], 0.02)

    def test_apply_camera_scale_preserves_schema_and_scales_centers(self) -> None:
        pose0 = np.eye(4)
        pose1 = np.eye(4)
        pose1[:3, 3] = [1.0, 2.0, 0.0]
        transforms = {
            "camera_model": "OPENCV",
            "fl_x": 100.0,
            "fl_y": 100.0,
            "cx": 32.0,
            "cy": 24.0,
            "w": 64,
            "h": 48,
            "custom_top_level": "kept",
            "frames": [
                {"file_path": "images/frame000000.jpg", "transform_matrix": pose0.tolist(), "custom_frame": 1},
                {"file_path": "images/frame000001.jpg", "transform_matrix": pose1.tolist(), "custom_frame": 2},
            ],
        }
        out = apply_camera_scale_to_transforms(transforms, 3.0, anchor="first")
        self.assertEqual(out["custom_top_level"], "kept")
        self.assertEqual(out["frames"][0]["custom_frame"], 1)
        self.assertEqual(out["frames"][1]["custom_frame"], 2)
        self.assertEqual(np.asarray(out["frames"][0]["transform_matrix"])[:3, 3].tolist(), [0.0, 0.0, 0.0])
        self.assertEqual(np.asarray(out["frames"][1]["transform_matrix"])[:3, 3].tolist(), [3.0, 6.0, 0.0])

    def test_select_frame_pairs_keeps_legacy_single_gap_behavior(self) -> None:
        pairs = _select_frame_pairs(10, pair_gap=2, max_pairs=3)
        self.assertEqual(pairs, [(0, 2), (4, 6), (7, 9)])

    def test_select_frame_pairs_multi_gap_prefers_usable_baselines(self) -> None:
        frames = []
        for idx in range(100):
            pose = np.eye(4)
            pose[:3, 3] = [float(idx) * 0.01, 0.0, 0.0]
            frames.append({"file_path": f"images/frame{idx:06d}.jpg", "transform_matrix": pose.tolist()})

        pairs = _select_frame_pairs(
            100,
            config={
                "pair_stride": 5,
                "max_pairs": 12,
                "min_baseline": 0.2,
            },
            frames=frames,
        )

        self.assertLessEqual(len(pairs), 12)
        self.assertTrue(pairs)
        self.assertTrue(all(j - i >= 20 for i, j in pairs))
        self.assertGreater(len({j - i for i, j in pairs}), 1)

    def test_pair_scale_spread_gate_rejects_inconsistent_multi_gap_candidates(self) -> None:
        failure = _pair_scale_spread_failure(
            {"num_pair_scale_candidates": 12, "pair_scale_p90_p10_ratio": 4.5},
            {"pair_strategy": "multi_gap", "max_pair_scale_spread_ratio": 3.0},
        )
        self.assertIsNotNone(failure)
        self.assertEqual(failure["max_pair_scale_spread_ratio"], 3.0)
        self.assertIsNone(
            _pair_scale_spread_failure(
                {"num_pair_scale_candidates": 12, "pair_scale_p90_p10_ratio": 4.5},
                {"pair_strategy": "single_gap", "max_pair_scale_spread_ratio": 3.0},
            )
        )

    def test_parse_stages_accepts_camera_scale(self) -> None:
        self.assertEqual(parse_stages("frames,camera,depth,camera_scale,normal"), ["frames", "camera", "depth", "camera_scale", "normal"])


if __name__ == "__main__":
    unittest.main()
