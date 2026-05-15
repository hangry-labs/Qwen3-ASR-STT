# coding=utf-8
from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path
from typing import Any, Dict

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse

from qwen_asr.inference.qwen3_asr import Qwen3ASRModel
from qwen_asr.inference.utils import SUPPORTED_LANGUAGES, normalize_language_name, validate_language


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
    asr: Qwen3ASRModel,
    model_name: str,
    concurrency: int,
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
        payload = await file.read()
        if not payload:
            raise HTTPException(status_code=400, detail="Uploaded audio file is empty")

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(payload)
            tmp_path = tmp.name

        try:
            async with semaphore:
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
    if cuda_visible_devices.strip():
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices.strip()

    resolved_backend_kwargs = _coerce_special_types(backend_kwargs or {})

    if backend == "vllm":
        asr = Qwen3ASRModel.LLM(asr_checkpoint, **resolved_backend_kwargs)
    elif backend == "transformers":
        asr = Qwen3ASRModel.from_pretrained(asr_checkpoint, **resolved_backend_kwargs)
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    app = create_app(asr=asr, model_name=asr_checkpoint, concurrency=concurrency)
    uvicorn.run(app, host=host, port=port)
