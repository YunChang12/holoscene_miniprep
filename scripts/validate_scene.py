#!/usr/bin/env python3
"""Validate a prepared HoloScene scene directory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from holoprep.validation import validate_scene


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a HoloScene miniprep scene.")
    parser.add_argument("--scene_dir", required=True, help="Path to data_dir/custom/<scene_name>.")
    parser.add_argument("--clip_min", type=float, default=0.05, help="Expected minimum depth value for statistics.")
    parser.add_argument("--clip_max", type=float, default=20.0, help="Expected maximum depth value for statistics.")
    args = parser.parse_args()

    report = validate_scene(args.scene_dir, clip_min=args.clip_min, clip_max=args.clip_max)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("ok", False) else 1


if __name__ == "__main__":
    raise SystemExit(main())
