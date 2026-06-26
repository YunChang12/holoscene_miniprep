"""Small REST client for Zaiwu gateway style jobs and artifacts."""

from __future__ import annotations

import json
import asyncio
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


class ZaiwuClient:
    """Minimal stdlib-only client for Zaiwu gateway endpoints.

    The MiniPrep wrappers intentionally use the gateway job API as the stable
    integration surface:

    - POST /api/v1/jobs
    - GET /api/v1/jobs/{job_id}
    - POST /upload?filename=...
    - GET /download/{file_id}
    """

    def __init__(self, service_url: str, timeout: float = 30.0) -> None:
        if not service_url:
            raise ValueError("Missing service_url for Zaiwu wrapper")
        self.base_url = service_url.rstrip("/")
        self.timeout = float(timeout)

    def health_check(self) -> dict[str, Any]:
        """Probe common Zaiwu health endpoints and return the first success."""

        errors: list[str] = []
        for path in ("/readyz", "/healthz", "/health"):
            try:
                return self.get_json(path)
            except Exception as exc:
                errors.append(f"{path}: {exc}")
        raise RuntimeError(f"Zaiwu service is not healthy at {self.base_url}. Tried: {'; '.join(errors)}")

    def upload_file(self, path: str | Path) -> str:
        """Upload a local file through the gateway and return file_id."""

        src = Path(path).expanduser().resolve()
        if not src.is_file():
            raise FileNotFoundError(f"Upload file not found: {src}")
        query = urlencode({"filename": src.name})
        payload = self.request_bytes("POST", f"/upload?{query}", body=src.read_bytes())
        data = _decode_json(payload)
        file_id = data.get("file_id")
        if not file_id:
            raise RuntimeError(f"Upload response missing file_id: {data}")
        return str(file_id)

    def submit_job(
        self,
        handler: str,
        payload: dict[str, Any],
        *,
        labels: dict[str, str] | None = None,
        requested_by: str = "holoscene_miniprep",
    ) -> dict[str, Any]:
        """Submit a Zaiwu job and return the created JobRecord dict."""

        body = {
            "handler": handler,
            "payload": payload,
            "labels": labels or {},
            "requested_by": requested_by,
        }
        return self.post_json("/api/v1/jobs", body)

    def poll_job(
        self,
        job_id: str,
        *,
        timeout_sec: float = 1800.0,
        poll_interval: float = 2.0,
    ) -> dict[str, Any]:
        """Poll a submitted job until a terminal status is reached."""

        deadline = time.monotonic() + float(timeout_sec)
        last: dict[str, Any] | None = None
        while time.monotonic() < deadline:
            last = self.get_json(f"/api/v1/jobs/{job_id}")
            status = str(last.get("status", "")).lower()
            if status == "succeeded":
                return last
            if status in {"failed", "cancelled"}:
                raise RuntimeError(f"Zaiwu job {job_id} {status}: {last.get('error') or last}")
            time.sleep(float(poll_interval))
        raise TimeoutError(f"Timed out waiting for Zaiwu job {job_id}. Last status: {last}")

    def download_file(self, file_id: str, dst: str | Path) -> Path:
        """Download a Zaiwu artifact/file_id to a local path."""

        out = Path(dst)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(self.request_bytes("GET", f"/download/{file_id}"))
        return out

    def call_mcp_tool(
        self,
        tool_name: str,
        args: dict[str, Any],
        *,
        sse_read_timeout: float = 1800.0,
    ) -> dict[str, Any]:
        """Call a direct FastMCP SSE worker tool.

        This is used as a compatibility fallback when ``service_url`` points to
        a direct worker port instead of the unified Zaiwu gateway.
        """

        sse_url = self.base_url if self.base_url.rstrip("/").endswith("/sse") else f"{self.base_url}/sse"

        async def _call() -> dict[str, Any]:
            from mcp.client.session import ClientSession
            from mcp.client.sse import sse_client

            async with sse_client(sse_url, sse_read_timeout=float(sse_read_timeout)) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool(tool_name, arguments=args)
                    if result.isError:
                        err = ""
                        if result.content:
                            err = getattr(result.content[0], "text", str(result.content))
                        raise RuntimeError(f"MCP tool {tool_name!r} returned error: {err}")
                    text = result.content[0].text if result.content else "{}"
                    data = json.loads(text)
                    if not isinstance(data, dict):
                        raise RuntimeError(f"MCP tool {tool_name!r} returned {type(data).__name__}, expected object")
                    return data

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is not None and loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, _call()).result()
        return asyncio.run(_call())

    def get_json(self, path: str) -> dict[str, Any]:
        return _decode_json(self.request_bytes("GET", path))

    def post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        return _decode_json(self.request_bytes("POST", path, body=body, content_type="application/json"))

    def request_bytes(
        self,
        method: str,
        path: str,
        *,
        body: bytes | None = None,
        content_type: str | None = None,
    ) -> bytes:
        url = f"{self.base_url}{path if path.startswith('/') else '/' + path}"
        headers: dict[str, str] = {}
        if content_type:
            headers["Content-Type"] = content_type
        request = Request(url, data=body, headers=headers, method=method.upper())
        try:
            with urlopen(request, timeout=self.timeout) as response:  # noqa: S310 - user-provided local service URL
                return response.read()
        except HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code} {method.upper()} {url}: {details}") from exc
        except URLError as exc:
            raise RuntimeError(f"Could not connect to {url}: {exc.reason}") from exc


def _decode_json(payload: bytes) -> dict[str, Any]:
    try:
        data = json.loads(payload.decode("utf-8"))
    except Exception as exc:
        preview = payload[:500].decode("utf-8", errors="replace")
        raise RuntimeError(f"Expected JSON response, got: {preview}") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"Expected JSON object response, got: {type(data).__name__}")
    return data
