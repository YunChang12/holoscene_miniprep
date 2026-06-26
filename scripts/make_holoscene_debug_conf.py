#!/usr/bin/env python3
"""Generate a short HoloScene Stage 1 debug conf for a prepared scene."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from PIL import Image


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a debug HoloScene conf from an existing template.")
    parser.add_argument("--scene_dir", required=True, help="Prepared scene directory, e.g. data_dir/custom/my_scene.")
    parser.add_argument("--template_conf", help="Template .conf path, e.g. HoloScene/confs/replica/room_0/replica_room_0.conf.")
    parser.add_argument("--output_conf", help="Output debug conf path.")
    parser.add_argument("--max_total_iters", type=int, default=50)
    parser.add_argument("--plot_freq", type=int, default=10)
    parser.add_argument("--checkpoint_freq", type=int, default=10)
    args = parser.parse_args()

    scene = Path(args.scene_dir).expanduser().resolve()
    if not scene.is_dir():
        raise SystemExit(f"scene_dir 不存在: {scene}")
    template = Path(args.template_conf).expanduser().resolve() if args.template_conf else _find_default_template()
    if not template or not template.is_file():
        raise SystemExit("找不到 HoloScene 配置模板，请通过 --template_conf 手动提供。")

    scene_name = scene.name
    output = Path(args.output_conf).expanduser() if args.output_conf else Path("confs") / "custom" / scene_name / f"{scene_name}_debug.conf"
    if not output.is_absolute():
        output = (Path.cwd() / output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    data_root_dir = scene.parent
    data_dir = scene.name
    img_res = _read_image_resolution(scene)

    text = template.read_text(encoding="utf-8")
    text = _replace_assign(text, "expname", f"holoscene_custom_{scene_name}_debug")
    text = _replace_assign(text, "data_root_dir", str(data_root_dir))
    text = _replace_assign(text, "data_dir", data_dir)
    text = _replace_assign(text, "img_res", f"[{img_res[0]}, {img_res[1]}]", raw=True)
    text = _replace_assign(text, "max_total_iters", str(args.max_total_iters), raw=True)
    text = _replace_assign(text, "stop_iter", str(args.max_total_iters), raw=True)
    text = _replace_assign(text, "plot_freq", str(args.plot_freq), raw=True)
    text = _replace_assign(text, "checkpoint_freq", str(args.checkpoint_freq), raw=True)
    output.write_text(text, encoding="utf-8")

    print(f"debug_conf={output}")
    print("下一步可在 HoloScene 根目录运行：")
    print(f"python training/exp_runner.py --conf {output} --none_wandb")
    return 0


def _find_default_template() -> Path | None:
    candidates = [
        Path("/root/autodl-fs/Zaiwu/third_party/HoloScene/confs/replica/room_0/replica_room_0.conf"),
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def _read_image_resolution(scene: Path) -> tuple[int, int]:
    first = next(iter(sorted((scene / "images").glob("frame*.jpg"))), None)
    if first is None:
        raise SystemExit(f"找不到 scene images/frame*.jpg: {scene}")
    with Image.open(first) as im:
        return im.size


def _replace_assign(text: str, key: str, value: str, raw: bool = False) -> str:
    replacement = f"{key} = {value if raw else value}"
    pattern = re.compile(rf"(^\s*{re.escape(key)}\s*=\s*)(.+)$", re.MULTILINE)
    if pattern.search(text):
        return pattern.sub(lambda m: m.group(1) + (value if raw else value), text, count=1)
    return text + f"\n{replacement}\n"


if __name__ == "__main__":
    raise SystemExit(main())
