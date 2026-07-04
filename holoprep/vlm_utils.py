"""VLM-assisted object prompt generation for Seg2Track-SAM2."""

from __future__ import annotations

import base64
import json
import re
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import yaml
from PIL import Image

from .writer import ensure_dir, write_json

DEFAULT_IGNORED_LABELS = [
    "floor",
    "wall",
    "ceiling",
    "window",
    "door",
    "carpet",
    "rug",
    "room surface",
    "shadow",
    "reflection",
    "person",
    "hand",
]


def generate_vlm_object_prompt(
    *,
    scene_dir: str | Path,
    scene_name: str,
    config: dict[str, Any],
    frame_count: int,
) -> dict[str, Any]:
    """Sample scene frames and ask a VLM for foreground/background object labels."""

    scene = Path(scene_dir)
    cfg = _load_effective_vlm_config(config)
    if not bool(cfg.get("enabled", True)):
        prompt = str(cfg.get("fallback_text_prompt", "object."))
        report = {
            "enabled": False,
            "scene_id": scene_name,
            "prompt": prompt,
            "foreground_labels": _labels_from_prompt(prompt),
            "background_labels": list(DEFAULT_IGNORED_LABELS),
            "ignored_labels": list(DEFAULT_IGNORED_LABELS),
            "source": "fallback_disabled",
        }
        return _write_vlm_report(scene, report)

    api_base = str(cfg.get("api_base", "")).rstrip("/")
    api_key = str(cfg.get("api_key", ""))
    model = str(cfg.get("model", ""))
    if not api_base or not api_key or not model:
        raise ValueError("vlm requires api_base, api_key, and model")

    image_paths = _sample_images(scene / "images", frame_count, cfg.get("frame_sampling", {}))
    if not image_paths:
        raise RuntimeError(f"No images found for VLM prompt generation: {scene / 'images'}")

    ignored_labels = _string_list(cfg.get("ignored_labels")) or list(DEFAULT_IGNORED_LABELS)
    system_prompt = str(cfg.get("system_prompt") or _default_system_prompt())
    user_prompt = _build_user_prompt(str(cfg.get("user_prompt") or ""), ignored_labels)
    content: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]
    max_side = int((cfg.get("frame_sampling") or {}).get("image_max_side", 768))
    for path in image_paths:
        content.append({"type": "image_url", "image_url": {"url": _image_to_data_url(path, max_side=max_side)}})

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ],
        "temperature": float(cfg.get("temperature", 0.0)),
        "max_tokens": int(cfg.get("max_tokens", 512)),
    }
    raw_response, latency = _post_chat_completion(
        api_base=api_base,
        api_key=api_key,
        payload=payload,
        timeout=float(cfg.get("timeout_sec", 120)),
        max_retries=int(cfg.get("max_retries", 3)),
    )
    text = _extract_message_content(raw_response)
    parsed = _parse_vlm_text(text)

    foreground = _normalize_labels(parsed.get("foreground_labels") or parsed.get("labels") or _labels_from_prompt(parsed.get("prompt")))
    background = _normalize_labels(parsed.get("background_labels") or [])
    ignored = sorted(set(_normalize_labels(parsed.get("ignored_labels") or [])) | set(_normalize_labels(ignored_labels)) | set(background))
    foreground = _filter_foreground_labels(foreground, ignored)
    prompt = _prompt_from_labels(foreground) or str(cfg.get("fallback_text_prompt", "object."))

    report = {
        "enabled": True,
        "scene_id": scene_name,
        "source": "vlm",
        "model": model,
        "api_base": _redact_api_base(api_base),
        "sampled_frames": [str(path.relative_to(scene)) if path.is_relative_to(scene) else str(path) for path in image_paths],
        "latency_sec": latency,
        "raw_text": text,
        "raw_parsed": parsed,
        "foreground_labels": foreground,
        "background_labels": sorted(set(background) | set(_normalize_labels(DEFAULT_IGNORED_LABELS[:7]))),
        "ignored_labels": ignored,
        "prompt": prompt,
        "notes": [
            "Only foreground labels should be sent to Seg2Track-SAM2.",
            f"Ignored/background labels are config-driven: {', '.join(ignored)}.",
        ],
    }
    return _write_vlm_report(scene, report)


def load_vlm_prompt_for_mask(scene_dir: str | Path) -> dict[str, Any]:
    """Load the VLM prompt report produced by the vlm stage."""

    path = Path(scene_dir) / "meta" / "vlm_object_prompt.json"
    if not path.is_file():
        raise FileNotFoundError(f"VLM prompt report not found: {path}. Run --stages vlm first.")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_effective_vlm_config(config: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(config or {})
    config_path = cfg.get("config_path")
    if config_path:
        path = Path(str(config_path)).expanduser()
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        base = dict(loaded.get("vlm", loaded))
        base.update({k: v for k, v in cfg.items() if k != "config_path"})
        return base
    default_path = Path(__file__).resolve().parents[1] / "configs" / "vlm_local.yaml"
    if default_path.is_file() and not cfg.get("api_key"):
        loaded = yaml.safe_load(default_path.read_text(encoding="utf-8")) or {}
        base = dict(loaded.get("vlm", loaded))
        base.update(cfg)
        return base
    return cfg


def _sample_images(image_dir: Path, frame_count: int, sampling_cfg: Any) -> list[Path]:
    paths = sorted(image_dir.glob("frame*.jpg"))
    if not paths:
        return []
    cfg = sampling_cfg if isinstance(sampling_cfg, dict) else {}
    max_images = max(1, int(cfg.get("max_images", 8)))
    n = min(max_images, len(paths), max(1, int(frame_count or len(paths))))
    if n >= len(paths):
        return paths
    indices = sorted(set(round(i * (len(paths) - 1) / max(n - 1, 1)) for i in range(n)))
    return [paths[int(idx)] for idx in indices]


def _image_to_data_url(path: Path, *, max_side: int) -> str:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        scale = min(1.0, float(max_side) / max(rgb.size))
        if scale < 1.0:
            rgb = rgb.resize((max(1, round(rgb.width * scale)), max(1, round(rgb.height * scale))), Image.Resampling.BILINEAR)
        from io import BytesIO

        buffer = BytesIO()
        rgb.save(buffer, format="JPEG", quality=85)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _post_chat_completion(
    *,
    api_base: str,
    api_key: str,
    payload: dict[str, Any],
    timeout: float,
    max_retries: int,
) -> tuple[dict[str, Any], float]:
    url = f"{api_base}/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    last_error: Exception | None = None
    start = time.monotonic()
    for attempt in range(max(1, max_retries)):
        request = Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
        try:
            with urlopen(request, timeout=timeout) as response:  # noqa: S310 - user-configured API endpoint
                data = json.loads(response.read().decode("utf-8"))
            return data, float(time.monotonic() - start)
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            last_error = RuntimeError(f"HTTP {exc.code} from VLM API: {detail[:1000]}")
        except (URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
        if attempt + 1 < max_retries:
            time.sleep(min(2.0 * (attempt + 1), 8.0))
    raise RuntimeError(f"VLM request failed: {last_error}") from last_error


def _extract_message_content(response: dict[str, Any]) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError(f"VLM response missing choices: {response}")
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("text"):
                parts.append(str(item["text"]))
        return "\n".join(parts)
    if not isinstance(content, str):
        raise RuntimeError(f"VLM response content is not text: {content}")
    return content


def _parse_vlm_text(text: str) -> dict[str, Any]:
    stripped = text.strip()
    json_text = stripped
    fence = re.search(r"```(?:json)?\s*(.*?)```", stripped, flags=re.IGNORECASE | re.DOTALL)
    if fence:
        json_text = fence.group(1).strip()
    try:
        data = json.loads(json_text)
        if isinstance(data, dict):
            return data
        if isinstance(data, list):
            return {"foreground_labels": data}
    except Exception:
        pass
    return {"foreground_labels": _labels_from_prompt(stripped), "prompt": stripped}


def _build_user_prompt(base_prompt: str, ignored_labels: list[str]) -> str:
    ignored = ", ".join(ignored_labels)
    return (
        f"{base_prompt.strip()}\n\n"
        "Inspect the sampled frames and separate foreground objects from background structures.\n"
        "Foreground labels must be physical object categories with clear instance boundaries, useful for instance masks.\n"
        "Background labels must remain unsegmented background in HoloScene.\n"
        f"Always treat these as background/ignored, even if visible: {ignored}.\n"
        "Do not include ignored/background labels in foreground_labels.\n"
        "Return JSON only with this schema:\n"
        "{\n"
        '  "foreground_labels": ["chair", "sofa"],\n'
        '  "background_labels": ["floor", "wall", "ceiling"],\n'
        '  "ignored_labels": ["floor", "wall", "ceiling", "window", "carpet", "rug"],\n'
        '  "prompt": "chair. sofa."\n'
        "}\n"
    )


def _default_system_prompt() -> str:
    return (
        "You generate conservative open-vocabulary prompts for indoor instance segmentation. "
        "You must distinguish movable/support foreground objects from background room structures. "
        "Return compact JSON only."
    )


def _labels_from_prompt(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    text = str(value)
    return [part.strip() for part in re.split(r"[.,;\n]+", text) if part.strip()]


def _normalize_labels(labels: Any) -> list[str]:
    values = _labels_from_prompt(labels)
    normalized = []
    seen = set()
    for value in values:
        label = " ".join(str(value).lower().replace("_", " ").replace("-", " ").split())
        label = re.sub(r"^(a|an|the)\s+", "", label)
        if not label or label in seen:
            continue
        seen.add(label)
        normalized.append(label)
    return normalized


def _filter_foreground_labels(labels: list[str], ignored_labels: list[str]) -> list[str]:
    ignored = set(_normalize_labels(ignored_labels))
    out = []
    for label in labels:
        if any(token and token in label for token in ignored):
            continue
        out.append(label)
    return out


def _prompt_from_labels(labels: list[str]) -> str:
    return ". ".join(labels) + "." if labels else ""


def _string_list(value: Any) -> list[str]:
    return _normalize_labels(value)


def _redact_api_base(api_base: str) -> str:
    return api_base.rstrip("/")


def _write_vlm_report(scene: Path, report: dict[str, Any]) -> dict[str, Any]:
    ensure_dir(scene / "meta")
    write_json(scene / "meta" / "vlm_object_prompt.json", report)
    lines = [
        "# VLM Object Prompt",
        "",
        f"source: {report.get('source')}",
        f"model: {report.get('model')}",
        f"prompt: {report.get('prompt')}",
        "",
        "## Foreground",
        ", ".join(report.get("foreground_labels", [])),
        "",
        "## Background/Ignored",
        ", ".join(report.get("ignored_labels", [])),
    ]
    (scene / "meta" / "vlm_object_prompt.md").write_text("\n".join(lines), encoding="utf-8")
    return report
