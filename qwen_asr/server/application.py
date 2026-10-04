from __future__ import annotations

import os
from typing import Any

import uvicorn
from fastapi import FastAPI

from qwen_asr.server.aligner_runtime import AlignerRuntime, RuntimeSettingsStore
from qwen_asr.server.contracts import ASRRuntime
from qwen_asr.server.mcp_server import attach_mcp
from qwen_asr.server.openai_api import create_app as create_openai_app
from qwen_asr.server.startup_warmup import run_aligner_warmup, run_startup_warmup
from qwen_asr.standalone_ui.server import attach_ui
from qwen_asr.startup_logging import StartupTimer, log_startup


def _coerce_special_types(kwargs: dict[str, Any]) -> dict[str, Any]:
    coerced = dict(kwargs)
    dtype = coerced.get("dtype")
    if isinstance(dtype, str):
        import torch

        if not hasattr(torch, dtype):
            raise ValueError(f"Unknown torch dtype: {dtype}")
        coerced["dtype"] = getattr(torch, dtype)
    return coerced


def _load_asr_model(
    *,
    checkpoint: str,
    backend: str,
    model_kwargs: dict[str, Any],
) -> ASRRuntime:
    with StartupTimer("import Qwen3ASRModel"):
        from qwen_asr.inference.qwen3_asr import Qwen3ASRModel

    with StartupTimer(f"load ASR model via {backend} backend"):
        if backend == "vllm":
            loader = Qwen3ASRModel.LLM
        elif backend == "transformers":
            loader = Qwen3ASRModel.from_pretrained
        else:
            raise ValueError(f"Unsupported backend: {backend}")
        return loader(checkpoint, **model_kwargs)


def create_product_app(
    *,
    asr: ASRRuntime,
    model_name: str,
    concurrency: int,
    aligner_runtime: AlignerRuntime,
    trace_requests: bool,
) -> FastAPI:
    """Compose the supported API and browser UI into one product application."""
    api_app = create_openai_app(
        asr=asr,
        model_name=model_name,
        concurrency=concurrency,
        trace_requests=trace_requests,
        startup_warmup=lambda: run_startup_warmup(asr),
        aligner_runtime=aligner_runtime,
    )
    attach_mcp(api_app=api_app)
    return attach_ui(api_app=api_app)


def run_server(
    *,
    asr_checkpoint: str,
    aligner_checkpoint: str | None,
    load_aligner_at_startup: bool,
    settings_path: str,
    backend: str,
    model_kwargs: dict[str, Any] | None,
    aligner_kwargs: dict[str, Any] | None,
    cuda_visible_devices: str,
    host: str,
    port: int,
    concurrency: int,
    trace_requests: bool,
    ssl_certfile: str | None = None,
    ssl_keyfile: str | None = None,
) -> None:
    """Load the configured runtime and serve the Docker UI/API product."""
    if bool(ssl_certfile) != bool(ssl_keyfile):
        raise ValueError("Both SSL certfile and SSL keyfile must be provided to enable HTTPS.")

    log_startup("browser UI and OpenAI-compatible API startup entered")
    if cuda_visible_devices.strip():
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices.strip()
        log_startup(f"CUDA_VISIBLE_DEVICES set to {cuda_visible_devices.strip()}")

    resolved_model_kwargs = _coerce_special_types(model_kwargs or {})
    resolved_aligner_kwargs = _coerce_special_types(aligner_kwargs or {})
    asr = _load_asr_model(
        checkpoint=asr_checkpoint,
        backend=backend,
        model_kwargs=resolved_model_kwargs,
    )

    settings = RuntimeSettingsStore(settings_path)
    aligner_runtime = AlignerRuntime(
        asr=asr,
        checkpoint=aligner_checkpoint,
        model_kwargs=resolved_aligner_kwargs,
        settings=settings,
        default_load_always=load_aligner_at_startup,
        warmup=run_aligner_warmup,
    )
    if aligner_runtime.load_always:
        with StartupTimer("load and warm persistent forced aligner"):
            aligner_runtime.load()

    with StartupTimer("create UI and OpenAI API app"):
        app = create_product_app(
            asr=asr,
            model_name=asr_checkpoint,
            concurrency=concurrency,
            aligner_runtime=aligner_runtime,
            trace_requests=trace_requests,
        )

    uvicorn_kwargs: dict[str, Any] = {}
    scheme = "http"
    if ssl_certfile and ssl_keyfile:
        scheme = "https"
        uvicorn_kwargs["ssl_certfile"] = ssl_certfile
        uvicorn_kwargs["ssl_keyfile"] = ssl_keyfile

    log_startup(f"starting UI/API uvicorn on {scheme}://{host}:{port}")
    uvicorn.run(app, host=host, port=port, **uvicorn_kwargs)
