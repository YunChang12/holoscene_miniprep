#!/usr/bin/env python3
"""Smoke test a MiniPrep scene with a lightweight HoloScene-like loader."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from holoprep.writer import write_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test whether a scene can be read like a HoloScene dataset.")
    parser.add_argument("--scene_dir", help="Path to data_dir/custom/<scene_name>.")
    parser.add_argument("--conf", help="HoloScene conf path. Used to infer scene_dir if --scene_dir is omitted.")
    parser.add_argument("--holoscene_root", help="Optional HoloScene root. Official import is attempted on a best-effort basis.")
    args = parser.parse_args()

    scene = _resolve_scene_dir(args.scene_dir, args.conf, args.holoscene_root)
    try:
        report = lightweight_loader_test(scene)
        if args.holoscene_root:
            report["official_loader_import"] = try_official_loader_import(Path(args.holoscene_root))
        write_json(scene / "meta" / "holoscene_loader_test.json", report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["ok"] else 1
    except Exception as exc:
        report = {"ok": False, "scene_dir": str(scene), "error": str(exc)}
        write_json(scene / "meta" / "holoscene_loader_test.json", report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 1


def lightweight_loader_test(scene: Path) -> dict[str, Any]:
    """Read the first frame and core metadata using HoloScene-like assumptions."""

    scene = scene.expanduser().resolve()
    required = ["images", "instance_mask", "depth", "normal", "transforms.json", "graph.json", "meta/id_mapping.json"]
    missing = [name for name in required if not (scene / name).exists()]
    if missing:
        raise RuntimeError(f"缺少必要文件/目录: {missing}")

    transforms = json.loads((scene / "transforms.json").read_text(encoding="utf-8"))
    frames = transforms.get("frames", [])
    if not frames:
        raise RuntimeError("transforms.json.frames 为空")

    first = frames[0]
    image_path = scene / first.get("file_path", "")
    stem = image_path.stem
    mask_path = scene / "instance_mask" / f"{stem}.png"
    depth_path = scene / "depth" / f"{stem}.npy"
    normal_path = scene / "normal" / f"{stem}.png"
    if not image_path.is_file():
        raise RuntimeError(f"transforms file_path not found: {first.get('file_path')}")
    if not depth_path.is_file():
        raise RuntimeError("depth file is not .npy or is missing")

    with Image.open(image_path) as im:
        image = np.asarray(im.convert("RGB"))
    with Image.open(mask_path) as im:
        mask = np.asarray(im.convert("L"))
    depth = np.load(depth_path)
    with Image.open(normal_path) as im:
        normal = np.asarray(im.convert("RGB"))

    if mask.shape != image.shape[:2]:
        raise RuntimeError(f"mask shape mismatch: {mask.shape} vs image {image.shape[:2]}")
    if depth.shape != image.shape[:2]:
        raise RuntimeError(f"depth shape mismatch: {depth.shape} vs image {image.shape[:2]}")
    if normal.shape[:2] != image.shape[:2] or normal.shape[2] != 3:
        raise RuntimeError(f"normal shape mismatch: {normal.shape} vs image {image.shape}")

    graph = json.loads((scene / "graph.json").read_text(encoding="utf-8"))
    mapping = json.loads((scene / "meta" / "id_mapping.json").read_text(encoding="utf-8"))
    expected_nodes = {0} | {int(obj["holoscene_node_id"]) for obj in mapping.get("objects", [])}
    graph_nodes = {int(item["node_id"]) for item in graph}
    if not expected_nodes.issubset(graph_nodes):
        raise RuntimeError(f"graph node_id does not match id_mapping: missing {sorted(expected_nodes - graph_nodes)}")

    report = {
        "ok": True,
        "mode": "lightweight",
        "scene_dir": str(scene),
        "num_frames": len(frames),
        "image_shape": list(image.shape),
        "mask_shape": list(mask.shape),
        "depth_shape": list(depth.shape),
        "normal_shape": list(normal.shape),
        "camera_intrinsics": {
            "fl_x": transforms.get("fl_x"),
            "fl_y": transforms.get("fl_y"),
            "cx": transforms.get("cx"),
            "cy": transforms.get("cy"),
            "w": transforms.get("w"),
            "h": transforms.get("h"),
        },
        "first_transform_matrix": first.get("transform_matrix"),
        "unique_mask_ids": sorted(int(v) for v in np.unique(mask)),
        "graph_loaded": True,
        "graph_nodes": sorted(graph_nodes),
    }
    return report


def try_official_loader_import(holoscene_root: Path) -> dict[str, Any]:
    """Best-effort import probe for HoloScene's official dataset package."""

    root = holoscene_root.expanduser().resolve()
    sys.path.insert(0, str(root))
    result = {"holoscene_root": str(root), "import_ok": False, "error": None}
    try:
        __import__("datasets.ns_dataset")
        result["import_ok"] = True
    except Exception as exc:
        result["error"] = str(exc)
    return result


def _resolve_scene_dir(scene_dir: str | None, conf: str | None, holoscene_root: str | None) -> Path:
    if scene_dir:
        return Path(scene_dir)
    if not conf:
        raise SystemExit("必须提供 --scene_dir 或 --conf")
    conf_path = Path(conf).expanduser().resolve()
    text = conf_path.read_text(encoding="utf-8")
    root_match = re.search(r"data_root_dir\s*=\s*([^\n]+)", text)
    dir_match = re.search(r"data_dir\s*=\s*([^\n]+)", text)
    if not root_match or not dir_match:
        raise SystemExit("无法从 conf 中解析 dataset.data_root_dir/data_dir")
    data_root = root_match.group(1).strip().strip('"').strip("'")
    data_dir = dir_match.group(1).strip().strip('"').strip("'")
    root = Path(data_root)
    if not root.is_absolute():
        base = Path(holoscene_root).expanduser().resolve() if holoscene_root else conf_path.parent
        root = (base / root).resolve()
    return root / data_dir


if __name__ == "__main__":
    raise SystemExit(main())
