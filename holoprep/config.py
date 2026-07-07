"""Configuration loading for holoscene_miniprep."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class PrepConfig:
    """Small wrapper around the YAML config dictionary."""

    path: Path
    data: dict[str, Any]

    @property
    def scene_name(self) -> str:
        return str(self.data["scene"]["name"])

    @property
    def output_dir(self) -> Path:
        return Path(self.data["scene"]["output_dir"]).expanduser()

    @property
    def resolution(self) -> tuple[int, int]:
        value = self.data.get("frame", {}).get("resolution", [512, 512])
        if len(value) != 2:
            raise ValueError("frame.resolution must be [width, height]")
        return int(value[0]), int(value[1])


def load_config(path: str | Path) -> PrepConfig:
    """Load and minimally validate a YAML configuration file."""

    cfg_path = Path(path).expanduser().resolve()
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML mapping: {cfg_path}")
    for section in ["scene", "frame", "camera", "camera_scale", "mask", "depth", "normal", "graph", "review"]:
        data.setdefault(section, {})
    scene = data["scene"]
    for key in ["name", "input_type", "input_path", "output_dir"]:
        if key not in scene or scene[key] in (None, ""):
            raise ValueError(f"Missing required config field: scene.{key}")
    return PrepConfig(path=cfg_path, data=data)


def section(config: PrepConfig, name: str) -> dict[str, Any]:
    """Return a config section as a mutable dict-like value."""

    value = config.data.get(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"Config section {name!r} must be a mapping")
    return value
