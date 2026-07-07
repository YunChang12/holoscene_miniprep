from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from holoprep.wrappers.vggt_wrapper import LongSequenceVGGTWrapper, _convert_long_sequence_transforms


class LongSequenceVGGTWrapperTests(unittest.TestCase):
    def test_convert_long_sequence_transforms_for_miniprep(self) -> None:
        pose0 = np.eye(4).tolist()
        pose1 = np.eye(4)
        pose1[:3, 3] = [1.0, 2.0, 3.0]
        data = {
            "camera_model": "PINHOLE",
            "w": 1000,
            "h": 500,
            "fl_x": 800.0,
            "fl_y": 700.0,
            "cx": 500.0,
            "cy": 250.0,
            "frames": [
                {"file_path": "frames/frame000000.jpg", "transform_matrix": pose0, "pose_confidence": 0.9},
                {"file_path": "frames/frame000001.jpg", "transform_matrix": pose1.tolist(), "pose_confidence": 0.8},
            ],
        }
        out = _convert_long_sequence_transforms(data, frame_count=2, resolution=(500, 250), camera_model="OPENCV")
        self.assertEqual(out["camera_model"], "OPENCV")
        self.assertEqual(out["w"], 500)
        self.assertEqual(out["h"], 250)
        self.assertAlmostEqual(out["fl_x"], 400.0)
        self.assertAlmostEqual(out["fl_y"], 350.0)
        self.assertAlmostEqual(out["cx"], 250.0)
        self.assertAlmostEqual(out["cy"], 125.0)
        self.assertEqual(out["frames"][0]["file_path"], "images/frame000000.jpg")
        self.assertEqual(out["frames"][1]["file_path"], "images/frame000001.jpg")
        self.assertNotIn("pose_confidence", out["frames"][0])
        self.assertEqual(np.asarray(out["frames"][1]["transform_matrix"])[3].tolist(), [0.0, 0.0, 0.0, 1.0])

    def test_build_command_uses_sse_without_direct_http_probe(self) -> None:
        wrapper = LongSequenceVGGTWrapper()
        image_dir = Path("/tmp/miniprep_images")
        cmd = wrapper._build_command(
            script=Path("/tmp/vggt_long_sequence_pose/scripts/run_vggt_long_sequence_pose.py"),
            image_dir=image_dir,
            raw_dir=Path("/tmp/miniprep_scene/raw_outputs/vggt_long_sequence"),
            config={"service_url": "http://127.0.0.1:20008/sse", "python_executable": "/usr/bin/python3"},
            scene_id="scene",
        )
        self.assertEqual(cmd[0], "/usr/bin/python3")
        self.assertEqual(cmd[cmd.index("--frames-dir") + 1], str(image_dir.resolve()))
        self.assertEqual(cmd.count(str(image_dir.resolve())), 1)
        self.assertNotIn("--prefer-direct-http", cmd)
        self.assertIn("--resume", cmd)
        self.assertIn("--disable-depth", cmd)

    def test_run_with_fake_long_sequence_project(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scene = root / "scene"
            image_dir = scene / "images"
            image_dir.mkdir(parents=True)
            project = root / "vggt_long_sequence_pose"
            script_dir = project / "scripts"
            script_dir.mkdir(parents=True)
            runner = script_dir / "run_vggt_long_sequence_pose.py"
            runner.write_text(
                """
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--output-dir", required=True)
parser.add_argument("--frames-dir", required=True)
args, _ = parser.parse_known_args()
out = Path(args.output_dir)
(out / "all_frame_alignment").mkdir(parents=True, exist_ok=True)
pose0 = np.eye(4).tolist()
pose1 = np.eye(4)
pose1[:3, 3] = [1.0, 0.0, 0.0]
final = {
    "camera_model": "PINHOLE",
    "w": 640,
    "h": 480,
    "fl_x": 500.0,
    "fl_y": 500.0,
    "cx": 320.0,
    "cy": 240.0,
    "frames": [
        {"file_path": "frames/frame000000.jpg", "transform_matrix": pose0},
        {"file_path": "frames/frame000001.jpg", "transform_matrix": pose1.tolist()},
    ],
    "missing_pose_filled_frames": [1],
}
(out / "final_transforms.json").write_text(json.dumps(final), encoding="utf-8")
(out / "pose_quality_report.json").write_text(json.dumps({"failed_key_windows": [], "failed_all_windows": [], "suspicious_frames": [1]}), encoding="utf-8")
(out / "all_frame_alignment" / "window_quality_report.json").write_text(json.dumps({"enabled": True, "dropped_windows": [], "downweighted_windows": []}), encoding="utf-8")
""",
                encoding="utf-8",
            )

            transforms = LongSequenceVGGTWrapper().run(
                image_dir=image_dir,
                output_dir=scene,
                config={
                    "service_url": "http://127.0.0.1:20008/sse",
                    "long_sequence_project_dir": str(project),
                    "python_executable": sys.executable,
                    "resume": False,
                },
                frame_count=2,
                resolution=(320, 240),
                scene_id="fake_scene",
            )
            self.assertEqual(transforms["frames"][0]["file_path"], "images/frame000000.jpg")
            self.assertEqual(transforms["w"], 320)
            self.assertEqual(transforms["h"], 240)
            self.assertAlmostEqual(transforms["fl_x"], 250.0)
            report = json.loads((scene / "meta" / "camera_source_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["source"], "long_sequence_vggt")
            self.assertEqual(report["missing_pose_filled_count"], 1)
            self.assertEqual(report["suspicious_frame_count"], 1)
            self.assertTrue(report["window_quality_gating_enabled"])


if __name__ == "__main__":
    unittest.main()
