"""Depth model integrations."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import numpy as np

from ..writer import ensure_dir, write_json
from .zaiwu_client import ZaiwuClient


class DepthWrapper:
    """External depth model wrapper placeholder."""

    def run(self, *args, **kwargs):
        """Generate depth maps with an external model."""

        raise RuntimeError(
            "Depth model is not integrated in holoscene_miniprep. Provide depth.provided_dir, "
            "use depth.mode=dummy, or implement DepthWrapper.run()."
        )


class ZaiwuDepthAnythingWrapper:
    """Call Zaiwu ``services.depth_anything3.estimate_from_dir`` and load depth maps."""

    handler = "services.depth_anything3.estimate_from_dir"

    def run(
        self,
        image_dir: str | Path,
        output_dir: str | Path,
        config: dict[str, Any],
        *,
        frame_count: int,
        resolution: tuple[int, int],
    ) -> list[np.ndarray]:
        raw_dir = Path(output_dir) / "raw_outputs" / "depth_zaiwu_da3"
        if raw_dir.exists():
            shutil.rmtree(raw_dir)
        ensure_dir(raw_dir)

        client = ZaiwuClient(
            str(config.get("service_url", "")),
            timeout=float(config.get("request_timeout_sec", 30.0)),
        )
        try:
            health = client.health_check()
            write_json(raw_dir / "health.json", health)
        except Exception as exc:
            write_json(raw_dir / "health.json", {"ok": False, "error": str(exc)})

        frames_dir = str(Path(image_dir).expanduser().resolve())
        result: dict[str, Any]
        try:
            record = client.submit_job(
                self.handler,
                {"frames_dir": frames_dir},
                labels={"service_id": "services.depth_anything3"},
            )
            write_json(raw_dir / "job_submit.json", record)
            job_id = _extract_job_id(record)
            final = client.poll_job(
                job_id,
                timeout_sec=float(config.get("job_timeout_sec", 1800.0)),
                poll_interval=float(config.get("poll_interval_sec", 2.0)),
            )
            write_json(raw_dir / "job_record.json", final)
            job_result = final.get("result")
            if not isinstance(job_result, dict):
                raise RuntimeError(f"Depth job {job_id} succeeded but result is not an object: {job_result}")
            result = job_result
        except Exception as exc:
            write_json(raw_dir / "gateway_job_error.json", {"error": str(exc)})
            result = client.call_mcp_tool(
                "estimate_from_dir",
                {"frames_dir": frames_dir},
                sse_read_timeout=float(config.get("job_timeout_sec", 1800.0)),
            )

        if not isinstance(result, dict):
            raise RuntimeError(f"Depth result is not an object: {result}")
        write_json(raw_dir / "result.json", result)

        file_id = result.get("output_file_id") or result.get("depth_file_id")
        if not file_id:
            raise RuntimeError(f"Depth job result missing output_file_id: {result}")
        stacked_path = client.download_file(str(file_id), raw_dir / "depth_stacked.npy")
        depth_3d = np.load(stacked_path)
        if depth_3d.ndim == 2:
            depth_3d = depth_3d[None, ...]
        if depth_3d.ndim != 3:
            raise RuntimeError(f"Expected stacked depth [N,H,W], got shape {depth_3d.shape}")
        if int(depth_3d.shape[0]) != int(frame_count):
            raise RuntimeError(f"Depth frame count {depth_3d.shape[0]} does not match images {frame_count}")

        width, height = resolution
        depths: list[np.ndarray] = []
        for idx, depth in enumerate(depth_3d):
            arr = np.asarray(depth, dtype=np.float32)
            if arr.shape != (height, width):
                arr = _resize_float(arr, width=width, height=height)
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            depths.append(arr)
            np.save(raw_dir / f"frame{idx:06d}.npy", arr)

        write_json(
            raw_dir / "metadata.json",
            {
                "handler": self.handler,
                "job_id": locals().get("job_id"),
                "frame_count": int(frame_count),
                "raw_shape": list(depth_3d.shape),
                "model_name": result.get("model_name"),
                "depth_type": result.get("depth_type"),
                "depth_unit": result.get("depth_unit"),
            },
        )
        return depths


def _extract_job_id(record: dict[str, Any]) -> str:
    spec = record.get("spec")
    if isinstance(spec, dict) and spec.get("job_id"):
        return str(spec["job_id"])
    if record.get("job_id"):
        return str(record["job_id"])
    raise RuntimeError(f"Could not find job_id in Zaiwu job record: {record}")


def _resize_float(arr: np.ndarray, *, width: int, height: int) -> np.ndarray:
    from PIL import Image

    image = Image.fromarray(np.asarray(arr, dtype=np.float32), mode="F")
    image = image.resize((int(width), int(height)), Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.float32)
