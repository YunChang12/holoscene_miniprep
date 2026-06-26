#!/usr/bin/env python3
"""Run the minimal HoloScene preprocessing pipeline."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from holoprep.config import load_config
from holoprep.pipeline import parse_stages, run_pipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare a video or image sequence for HoloScene.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument(
        "--stages",
        default=None,
        help="Comma-separated stages. Default: frames,camera,mask,depth,normal,geometry,graph,validate,review",
    )
    parser.add_argument("--resume", action="store_true", help="Reuse existing stage outputs when possible.")
    parser.add_argument("--log-level", default="INFO", help="Python logging level.")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s %(message)s")
    config = load_config(args.config)
    summary = run_pipeline(config, stages=parse_stages(args.stages), resume=args.resume)
    print(f"scene_dir={summary['scene_dir']}")
    print(f"frame_count={summary['frame_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
