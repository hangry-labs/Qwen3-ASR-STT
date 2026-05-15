# coding=utf-8
from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse

from qwen_asr.inference.utils import SUPPORTED_LANGUAGES, normalize_language_name, validate_language
from qwen_asr.startup_logging import StartupTimer, log_startup, optional_timer

if TYPE_CHECKING:
    from qwen_asr.inference.qwen3_asr import Qwen3ASRModel


def _coerce_special_types(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    coerced = dict(kwargs)
    dtype = coerced.get("dtype")
    if isinstance(dtype, str):
        import torch

        if not hasattr(torch, dtype):
            raise ValueError(f"Unknown torch dtype: {dtype}")
        coerced["dtype"] = getattr(torch, dtype)
    return coerced


def create_app(
    *,
    asr: "Qwen3ASRModel",
    model_name: str,
    concurrency: int,
    trace_requests: bool = False,
) -> FastAPI:
    app = FastAPI(title="Qwen3-ASR OpenAI-compatible API")
    semaphore = asyncio.Semaphore(max(1, int(concurrency)))

    @app.get("/health")
    def health() -> Dict[str, Any]:
        return {"status": "ok", "model": model_name}

    @app.get("/v1/models")
    def models() -> Dict[str, Any]:
        return {
            "object": "list",
            "data": [
                {
                    "id": model_name,
                    "object": "model",
                    "owned_by": "hangry-labs",
                }
            ],
        }

    @app.post("/v1/audio/transcriptions")
    async def transcriptions(
        file: UploadFile = File(...),
        model: str = Form(default=""),
        language: str = Form(default=""),
        response_format: str = Form(default="json"),
    ):
        with optional_timer("transcription request", trace_requests):
            if model and model not in {model_name, "qwen3-asr", "qwen3-asr-stt"}:
                raise HTTPException(status_code=400, detail=f"Unsupported model: {model}")

            forced_language = None
            if language.strip():
                forced_language = normalize_language_name(language)
                try:
                    validate_language(forced_language)
                except ValueError as exc:
                    raise HTTPException(status_code=400, detail=str(exc)) from exc

            suffix = Path(file.filename or "audio.wav").suffix or ".wav"
            with optional_timer("read uploaded audio", trace_requests):
                payload = await file.read()
            if not payload:
                raise HTTPException(status_code=400, detail="Uploaded audio file is empty")

            with optional_timer("write uploaded audio temp file", trace_requests):
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(payload)
                    tmp_path = tmp.name

            try:
                async with semaphore:
                    with optional_timer("run ASR transcription", trace_requests):
                        result = await asyncio.to_thread(
                            asr.transcribe,
                            audio=tmp_path,
                            language=forced_language,
                            return_time_stamps=False,
                        )
            finally:
                Path(tmp_path).unlink(missing_ok=True)

            item = result[0]
            if response_format == "text":
                return PlainTextResponse(item.text)
            if response_format in {"json", ""}:
                return {"text": item.text}
            if response_format == "verbose_json":
                return {
                    "text": item.text,
                    "language": item.language,
                    "duration": None,
                    "segments": [],
                }
            raise HTTPException(status_code=400, detail=f"Unsupported response_format: {response_format}")

    @app.get("/v1/audio/supported_languages")
    def supported_languages() -> Dict[str, Any]:
        return {"languages": list(SUPPORTED_LANGUAGES)}

    return app


def run_server(
    *,
    asr_checkpoint: str,
    backend: str,
    backend_kwargs: Dict[str, Any] | None,
    cuda_visible_devices: str,
    host: str,
    port: int,
    concurrency: int,
) -> None:
    log_startup("OpenAI-compatible API startup entered")
    if cuda_visible_devices.strip():
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices.strip()
        log_startup(f"CUDA_VISIBLE_DEVICES set to {cuda_visible_devices.strip()}")

    with StartupTimer("coerce backend kwargs"):
        resolved_backend_kwargs = _coerce_special_types(backend_kwargs or {})

    with StartupTimer("import Qwen3ASRModel"):
        from qwen_asr.inference.qwen3_asr import Qwen3ASRModel

    with StartupTimer(f"load ASR model via {backend} backend"):
        if backend == "vllm":
            asr = Qwen3ASRModel.LLM(asr_checkpoint, **resolved_backend_kwargs)
        elif backend == "transformers":
            asr = Qwen3ASRModel.from_pretrained(asr_checkpoint, **resolved_backend_kwargs)
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    warmup_enabled = os.getenv("QWEN_ASR_STARTUP_WARMUP", "0").strip().lower() in {"1", "true", "yes", "y"}
    if warmup_enabled:
        warmup_tokens = int(os.getenv("QWEN_ASR_STARTUP_WARMUP_TOKENS", "1"))
        with StartupTimer(f"startup ASR warmup max_new_tokens={warmup_tokens}"):
            asr.warm_up(max_new_tokens=warmup_tokens)

    with StartupTimer("create FastAPI app"):
        trace_requests = os.getenv("QWEN_ASR_TRACE_REQUESTS", "0").strip().lower() in {"1", "true", "yes", "y"}
        app = create_app(
            asr=asr,
            model_name=asr_checkpoint,
            concurrency=concurrency,
            trace_requests=trace_requests,
        )

    log_startup(f"starting uvicorn on {host}:{port}")
    uvicorn.run(app, host=host, port=port)
