from __future__ import annotations

import html
import json
import os
from pathlib import Path
from typing import Any, Dict

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from qwen_asr.server.openai_api import create_app as create_openai_app
from qwen_asr.server.startup_warmup import run_startup_warmup
from qwen_asr.startup_logging import StartupTimer, log_startup
from qwen_asr.web.gpu import gpu_monitor_html


PACKAGE_DIR = Path(__file__).resolve().parent
STATIC_DIR = PACKAGE_DIR / "static"
REPO_ROOT = PACKAGE_DIR.parents[1]
BRAND_DIR = REPO_ROOT / "hangrylabs"
TESTBENCH_DIR = REPO_ROOT / "testbench"
MANIFEST_PATH = TESTBENCH_DIR / "manifest.json"


def _read_version_file() -> str:
    try:
        return (REPO_ROOT / "VERSION").read_text(encoding="utf-8").strip() or "0.0.0"
    except OSError:
        return "0.0.0"


def _example_catalog() -> list[dict[str, str]]:
    if not MANIFEST_PATH.is_file():
        return []
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    examples: list[dict[str, str]] = []
    seen: set[str] = set()
    for case in manifest.get("cases", []):
        language = str(case.get("language") or "").strip()
        relative_audio = str(case.get("audio") or "").replace("\\", "/").lstrip("/")
        if not language or language in seen or not relative_audio:
            continue
        audio_path = (TESTBENCH_DIR / relative_audio).resolve()
        try:
            audio_path.relative_to(TESTBENCH_DIR.resolve())
        except ValueError:
            continue
        if not audio_path.is_file():
            continue
        seen.add(language)
        examples.append(
            {
                "label": f"{language} - {audio_path.name}",
                "language": language,
                "name": audio_path.name,
                "url": f"/example-audio/{relative_audio}",
            }
        )
    return examples


def create_app(*, api_app: FastAPI) -> FastAPI:
    """Add the browser UI to the OpenAI-compatible API application."""
    development_assets = os.getenv("QWEN_ASR_UI_DEV", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    @api_app.middleware("http")
    async def disable_development_asset_cache(request, call_next):
        response = await call_next(request)
        if development_assets and (
            request.url.path == "/"
            or request.url.path.startswith("/static/")
            or request.url.path.startswith("/brand/")
        ):
            response.headers["Cache-Control"] = "no-store"
        return response

    api_app.mount("/static", StaticFiles(directory=STATIC_DIR), name="ui-static")
    if BRAND_DIR.is_dir():
        api_app.mount("/brand", StaticFiles(directory=BRAND_DIR), name="ui-brand")
    if TESTBENCH_DIR.is_dir():
        api_app.mount("/example-audio", StaticFiles(directory=TESTBENCH_DIR), name="ui-examples")

    @api_app.get("/", include_in_schema=False)
    async def index() -> HTMLResponse:
        index_html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
        rendered_html = index_html.replace("{{UI_VERSION}}", html.escape(_read_version_file()))
        return HTMLResponse(rendered_html, headers={"Cache-Control": "no-cache"})

    @api_app.get("/examples", include_in_schema=False)
    async def examples() -> dict[str, list[dict[str, str]]]:
        return {"examples": _example_catalog()}

    @api_app.get("/system/gpu", include_in_schema=False)
    async def gpu() -> HTMLResponse:
        return HTMLResponse(gpu_monitor_html())

    return api_app


def _coerce_special_types(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    coerced = dict(kwargs)
    dtype = coerced.get("dtype")
    if isinstance(dtype, str):
        import torch

        if not hasattr(torch, dtype):
            raise ValueError(f"Unknown torch dtype: {dtype}")
        coerced["dtype"] = getattr(torch, dtype)
    return coerced


def run_server(
    *,
    asr_checkpoint: str,
    aligner_checkpoint: str | None,
    backend: str,
    model_kwargs: Dict[str, Any] | None,
    aligner_kwargs: Dict[str, Any] | None,
    cuda_visible_devices: str,
    host: str,
    port: int,
    concurrency: int,
    share: bool = False,
    ssl_certfile: str | None = None,
    ssl_keyfile: str | None = None,
    ssl_verify: bool = True,
) -> None:
    del share, ssl_verify

    if bool(ssl_certfile) != bool(ssl_keyfile):
        raise ValueError("Both SSL certfile and SSL keyfile must be provided to enable HTTPS.")

    log_startup("browser UI and OpenAI-compatible API startup entered")
    if cuda_visible_devices.strip():
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices.strip()
        log_startup(f"CUDA_VISIBLE_DEVICES set to {cuda_visible_devices.strip()}")

    resolved_model_kwargs = _coerce_special_types(model_kwargs or {})
    resolved_aligner_kwargs = _coerce_special_types(aligner_kwargs or {})
    forced_aligner = aligner_checkpoint if aligner_checkpoint else None

    with StartupTimer("import Qwen3ASRModel"):
        from qwen_asr.inference.qwen3_asr import Qwen3ASRModel

    with StartupTimer(f"load ASR model via {backend} backend"):
        if backend == "vllm":
            loader = Qwen3ASRModel.LLM
        elif backend == "transformers":
            loader = Qwen3ASRModel.from_pretrained
        else:
            raise ValueError(f"Unsupported backend: {backend}")
        asr = loader(
            asr_checkpoint,
            forced_aligner=forced_aligner,
            forced_aligner_kwargs=resolved_aligner_kwargs if forced_aligner else None,
            **resolved_model_kwargs,
        )

    trace_requests = os.getenv("QWEN_ASR_TRACE_REQUESTS", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
    }
    with StartupTimer("create UI and OpenAI API app"):
        api_app = create_openai_app(
            asr=asr,
            model_name=asr_checkpoint,
            concurrency=concurrency,
            trace_requests=trace_requests,
            startup_warmup=lambda: run_startup_warmup(asr),
        )
        app = create_app(api_app=api_app)

    uvicorn_kwargs: Dict[str, Any] = {}
    scheme = "http"
    if ssl_certfile and ssl_keyfile:
        scheme = "https"
        uvicorn_kwargs["ssl_certfile"] = ssl_certfile
        uvicorn_kwargs["ssl_keyfile"] = ssl_keyfile

    log_startup(f"starting UI/API uvicorn on {scheme}://{host}:{port}")
    uvicorn.run(app, host=host, port=port, **uvicorn_kwargs)
