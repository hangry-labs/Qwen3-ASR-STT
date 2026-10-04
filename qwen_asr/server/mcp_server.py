from __future__ import annotations

import asyncio
import mimetypes
import os
import platform
import shutil
import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import ToolAnnotations
from pydantic import BaseModel, Field

from qwen_asr import __version__
from qwen_asr.inference.utils import SUPPORTED_LANGUAGES
from qwen_asr.server.openai_api import (
    TranscriptionService,
    _duration,
    _segments_payload,
    _words_payload,
)
from qwen_asr.standalone_ui.gpu import GPU_MONITOR

DEFAULT_MCP_INPUT_DIR = "/app/persistent/mcp-input"
DEFAULT_ALLOWED_HOSTS = [
    "127.0.0.1",
    "127.0.0.1:*",
    "localhost",
    "localhost:*",
    "[::1]",
    "[::1]:*",
    "host.docker.internal",
    "host.docker.internal:*",
]
DEFAULT_ALLOWED_ORIGINS = [
    "http://127.0.0.1",
    "http://127.0.0.1:*",
    "http://localhost",
    "http://localhost:*",
    "https://127.0.0.1",
    "https://127.0.0.1:*",
    "https://localhost",
    "https://localhost:*",
]
TimestampGranularity = Literal["word", "segment"]


def _product_version() -> str:
    try:
        return (
            Path(__file__).resolve().parents[2] / "VERSION"
        ).read_text(encoding="utf-8").strip() or __version__
    except OSError:
        return __version__


class MCPWord(BaseModel):
    word: str
    start: float
    end: float


class MCPSegment(BaseModel):
    id: int
    start: float
    end: float
    text: str


class MCPTranscriptionResult(BaseModel):
    text: str
    language: str
    duration: float
    model: str
    words: list[MCPWord] = Field(default_factory=list)
    segments: list[MCPSegment] = Field(default_factory=list)


class MCPHealthOverview(BaseModel):
    status: str
    product: dict[str, Any]
    inference: dict[str, Any]
    aligner: dict[str, Any]
    realtime_defaults: dict[str, float | int]
    mcp: dict[str, Any]
    gpu: list[dict[str, Any]]
    persistent_storage: dict[str, Any]


class MCPAccessGate:
    """Keep the mounted protocol endpoint unavailable until the operator opts in."""

    def __init__(self, app: Any, enabled: Callable[[], bool]) -> None:
        self.app = app
        self.enabled = enabled

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") == "http" and not self.enabled():
            response = JSONResponse(
                status_code=403,
                content={
                    "error": {
                        "message": (
                            "MCP access has not been authorized by this deployment owner. "
                            "Enable MCP connectivity from the System tab on a trusted deployment."
                        ),
                        "type": "mcp_access_forbidden",
                    }
                },
            )
            await response(scope, receive, send)
            return
        await self.app(scope, receive, send)


def _csv_setting(name: str, default: list[str]) -> list[str]:
    value = os.getenv(name)
    if value is None:
        return default
    return [item.strip() for item in value.split(",") if item.strip()]


def _enabled(name: str, default: bool = True) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _configured_input_directory(value: str | Path | None) -> Path | None:
    if value is not None:
        raw = str(value).strip()
    else:
        raw = os.getenv("QWEN_ASR_MCP_INPUT_DIR", DEFAULT_MCP_INPUT_DIR).strip()
    return Path(raw).resolve() if raw else None


def _tool_error(exc: HTTPException) -> ToolError:
    detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
    return ToolError(detail)


def _result_payload(
    item: Any,
    *,
    model_name: str,
    timestamp_granularities: list[TimestampGranularity],
) -> MCPTranscriptionResult:
    words = _words_payload(item) if "word" in timestamp_granularities else []
    raw_segments = (
        _segments_payload(item) if "segment" in timestamp_granularities else []
    )
    return MCPTranscriptionResult(
        text=str(getattr(item, "text", "") or ""),
        language=str(getattr(item, "language", "") or ""),
        duration=float(_duration(item) or 0.0),
        model=model_name,
        words=[MCPWord.model_validate(word) for word in words],
        segments=[
            MCPSegment(
                id=int(segment["id"]),
                start=float(segment["start"]),
                end=float(segment["end"]),
                text=str(segment["text"]),
            )
            for segment in raw_segments
        ],
    )


def _storage_overview(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"path": None, "available": False}
    try:
        usage = shutil.disk_usage(path)
    except OSError as exc:
        return {"path": str(path), "available": False, "error": str(exc)}
    return {
        "path": str(path),
        "available": True,
        "total_bytes": usage.total,
        "used_bytes": usage.used,
        "free_bytes": usage.free,
    }


def create_mcp_server(
    *,
    api_app: FastAPI,
    input_directory: str | Path | None = None,
    gpu_snapshot_provider: Callable[[], dict[str, object]] | None = None,
) -> tuple[MCPServer[Any], Path | None]:
    """Build the path-only MCP surface around the product's shared runtime."""
    service: TranscriptionService = api_app.state.transcription_service
    model_name = str(api_app.state.model_name)
    coordinator = api_app.state.inference_coordinator
    runtime_settings = api_app.state.runtime_settings
    product_version = _product_version()
    started_at = time.monotonic()
    gpu_snapshot = gpu_snapshot_provider or GPU_MONITOR.request_snapshot
    mounted_input = _configured_input_directory(input_directory)
    if mounted_input is not None:
        try:
            mounted_input.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise RuntimeError(
                f"The configured MCP input directory could not be prepared: {mounted_input}. "
                "Create it or mount a writable directory before starting the service."
            ) from exc

    mcp: MCPServer[Any] = MCPServer(
        "qwen3-asr-stt",
        title="Qwen3-ASR STT by Hangry Labs",
        description=(
            "Private local speech-to-text, health visibility, and trusted operator controls using the runtime "
            "already loaded by Qwen3-ASR STT."
        ),
        instructions=(
            "Call get_health to inspect readiness and capabilities. Audio must already be available inside the "
            "configured MCP input directory; call transcribe_audio_file with its relative path. Never place "
            "audio bytes or base64 in tool arguments. Language may be omitted for automatic detection. Runtime "
            "control tools change this deployment and should only be used when the user asks."
        ),
        website_url="https://hangrylabs.app/software/qwen3-asr-stt",
        version=product_version,
    )

    read_only = ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    )
    runtime_control = ToolAnnotations(
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    )

    @mcp.tool(
        title="Get deployment health",
        description=(
            "Inspect the complete local deployment overview: inference readiness and metrics, model/version, "
            "aligner state, saved realtime defaults, current GPU telemetry, MCP file location, uptime, and "
            "persistent-storage capacity. This does not change runtime state."
        ),
        annotations=read_only,
        structured_output=True,
    )
    async def get_health() -> MCPHealthOverview:
        inference = await coordinator.metrics()
        aligner = api_app.state.aligner_snapshot()
        gpu_payload = await asyncio.to_thread(gpu_snapshot)
        storage = await asyncio.to_thread(_storage_overview, mounted_input)
        realtime_sessions = api_app.state.realtime_sessions
        return MCPHealthOverview(
            status=str(inference.get("status") or "unknown"),
            product={
                "name": "Qwen3-ASR STT",
                "version": product_version,
                "model": model_name,
                "python_version": platform.python_version(),
                "uptime_seconds": round(time.monotonic() - started_at, 3),
                "max_audio_file_mb": service.max_upload_bytes / (1024 * 1024),
                "supported_languages": list(SUPPORTED_LANGUAGES),
                "active_realtime_sessions": len(realtime_sessions),
            },
            inference=inference,
            aligner=aligner,
            realtime_defaults=runtime_settings.realtime_defaults(),
            mcp={
                "enabled": True,
                "transport": "streamable-http",
                "endpoint": "/mcp",
                "input_directory": str(mounted_input) if mounted_input is not None else None,
                "path_transcription_available": mounted_input is not None,
            },
            gpu=list(gpu_payload.get("gpus") or []),
            persistent_storage=storage,
        )

    if mounted_input is not None:

        @mcp.tool(
            title="Transcribe a shared audio file",
            description=(
                "Transcribe an audio file that already exists inside the configured MCP input directory. Pass "
                "a relative path such as 'exports/video-audio.mp3'. Paths outside the directory are rejected. "
                "Optional word or segment timestamps load the forced aligner when supported."
            ),
            annotations=read_only,
            structured_output=True,
        )
        async def transcribe_audio_file(
            file_path: str,
            language: str | None = None,
            prompt: str = "",
            timestamp_granularities: list[TimestampGranularity] | None = None,
        ) -> MCPTranscriptionResult:
            requested = Path(file_path)
            resolved = (
                requested if requested.is_absolute() else mounted_input / requested
            ).resolve()
            try:
                resolved.relative_to(mounted_input)
            except ValueError as exc:
                raise ToolError(
                    "file_path must stay inside the configured MCP input directory. Use a relative path such as "
                    "'sample.mp3' for a file already placed or exported there."
                ) from exc
            if not resolved.is_file():
                raise ToolError(
                    "The requested audio file does not exist in the configured MCP input directory. Export, "
                    "copy, or mount the file there, then retry with its relative path."
                )
            if resolved.stat().st_size > service.max_upload_bytes:
                limit_mb = service.max_upload_bytes / (1024 * 1024)
                raise ToolError(f"Audio exceeds the configured {limit_mb:g} MB limit")

            payload = await asyncio.to_thread(resolved.read_bytes)
            granularities = list(dict.fromkeys(timestamp_granularities or []))
            try:
                item = await service.transcribe_bytes(
                    payload=payload,
                    filename=resolved.name,
                    content_type=mimetypes.guess_type(resolved.name)[0],
                    language=language,
                    prompt=prompt,
                    timestamp_granularities=granularities,
                )
            except HTTPException as exc:
                raise _tool_error(exc) from exc
            return _result_payload(
                item, model_name=model_name, timestamp_granularities=granularities
            )

    @mcp.tool(
        title="Configure forced-aligner residency",
        description=(
            "Persist whether the forced aligner should always remain loaded. true loads and warms it now and on "
            "future startups. false disables persistent residency and releases its VRAM immediately."
        ),
        annotations=runtime_control,
        structured_output=True,
    )
    async def configure_aligner_residency(
        load_aligner_always: bool,
    ) -> dict[str, Any]:
        try:
            return await api_app.state.configure_aligner(load_aligner_always)
        except HTTPException as exc:
            raise _tool_error(exc) from exc

    @mcp.tool(
        title="Load the forced aligner",
        description=(
            "Load and fully warm the forced aligner now for word or segment timestamps. This does not change the "
            "saved always-loaded preference. Loading can use several GiB of additional VRAM."
        ),
        annotations=runtime_control,
        structured_output=True,
    )
    async def load_forced_aligner() -> dict[str, Any]:
        try:
            return await api_app.state.ensure_aligner_loaded()
        except HTTPException as exc:
            raise _tool_error(exc) from exc

    @mcp.tool(
        title="Release forced-aligner VRAM",
        description=(
            "Unload the optional forced aligner and release its VRAM. If always-loaded residency is enabled, "
            "disable it first with configure_aligner_residency. The main transcription model remains loaded."
        ),
        annotations=runtime_control,
        structured_output=True,
    )
    async def release_forced_aligner_vram() -> dict[str, Any]:
        try:
            return await api_app.state.release_aligner()
        except HTTPException as exc:
            raise _tool_error(exc) from exc

    @mcp.tool(
        title="Save realtime transcription defaults",
        description=(
            "Persist defaults for future browser and API realtime sessions. Required ranges: chunk_size_sec "
            "0.5-5, max_window_sec 10-60 and not below chunk_size_sec, unfixed_chunk_num 0-6, and "
            "unfixed_token_num 0-20. Individual sessions can still override them."
        ),
        annotations=runtime_control,
        structured_output=True,
    )
    async def configure_realtime_defaults(
        chunk_size_sec: float,
        max_window_sec: float,
        unfixed_chunk_num: int,
        unfixed_token_num: int,
    ) -> dict[str, Any]:
        try:
            defaults = await asyncio.to_thread(
                runtime_settings.set_realtime_defaults,
                {
                    "chunk_size_sec": chunk_size_sec,
                    "max_window_sec": max_window_sec,
                    "unfixed_chunk_num": unfixed_chunk_num,
                    "unfixed_token_num": unfixed_token_num,
                },
            )
        except (OSError, ValueError) as exc:
            raise ToolError(str(exc)) from exc
        return {"realtime_defaults": defaults}

    return mcp, mounted_input


def attach_mcp(
    *,
    api_app: FastAPI,
    input_directory: str | Path | None = None,
) -> FastAPI:
    """Mount opt-in stateless Streamable HTTP MCP at /mcp."""
    mcp, mounted_input = create_mcp_server(
        api_app=api_app, input_directory=input_directory
    )
    transport_security = TransportSecuritySettings(
        enable_dns_rebinding_protection=_enabled(
            "QWEN_ASR_MCP_DNS_REBINDING_PROTECTION"
        ),
        allowed_hosts=_csv_setting("QWEN_ASR_MCP_ALLOWED_HOSTS", DEFAULT_ALLOWED_HOSTS),
        allowed_origins=_csv_setting(
            "QWEN_ASR_MCP_ALLOWED_ORIGINS", DEFAULT_ALLOWED_ORIGINS
        ),
    )
    mcp_app = mcp.streamable_http_app(
        streamable_http_path="/",
        json_response=True,
        stateless_http=True,
        max_request_body_size=4 * 1024 * 1024,
        transport_security=transport_security,
    )

    original_lifespan = api_app.router.lifespan_context

    @asynccontextmanager
    async def combined_lifespan(app: FastAPI):
        async with original_lifespan(app), mcp.session_manager.run():
            yield

    def mcp_enabled() -> bool:
        return api_app.state.runtime_settings.mcp_enabled(
            default=bool(api_app.state.mcp_default_enabled)
        )

    api_app.router.lifespan_context = combined_lifespan
    api_app.mount("/mcp", MCPAccessGate(mcp_app, mcp_enabled), name="mcp")
    api_app.state.mcp_server = mcp
    api_app.state.mcp_input_directory = mounted_input
    return api_app
