#!/usr/bin/env python3
"""Generate review artifacts for a prepared HoloScene scene directory."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from holoprep.graph_utils import visualize_graph
from holoprep.visualization import visualize_scene
from holoprep.writer import read_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Create mask/depth/normal/graph visualizations.")
    parser.add_argument("--scene_dir", required=True, help="Path to data_dir/custom/<scene_name>.")
    args = parser.parse_args()

    scene = Path(args.scene_dir).expanduser().resolve()
    visualize_scene(scene)
    graph_path = scene / "graph.json"
    if graph_path.is_file():
        visualize_graph(scene, read_json(graph_path))
    print(f"review_dir={scene / 'review'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
