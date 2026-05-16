# coding=utf-8
from __future__ import annotations

import asyncio
import json
import os
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable

import uvicorn
from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse
from starlette.datastructures import UploadFile as StarletteUploadFile

from qwen_asr.inference.utils import SUPPORTED_LANGUAGES, normalize_audios, normalize_language_name, validate_language
from qwen_asr.startup_logging import StartupTimer, log_startup, optional_timer

if TYPE_CHECKING:
    from qwen_asr.inference.qwen3_asr import Qwen3ASRModel


MODEL_ALIASES = {"qwen3-asr", "qwen3-asr-stt"}
RESPONSE_FORMATS = {"json", "text", "verbose_json", "srt", "vtt"}
TIMESTAMP_GRANULARITIES = {"segment", "word"}
LANGUAGE_ALIASES = {
    "zh": "Chinese",
    "chinese": "Chinese",
    "en": "English",
    "english": "English",
    "yue": "Cantonese",
    "cantonese": "Cantonese",
    "ar": "Arabic",
    "arabic": "Arabic",
    "de": "German",
    "german": "German",
    "fr": "French",
    "french": "French",
    "es": "Spanish",
    "spanish": "Spanish",
    "pt": "Portuguese",
    "portuguese": "Portuguese",
    "id": "Indonesian",
    "indonesian": "Indonesian",
    "it": "Italian",
    "italian": "Italian",
    "ko": "Korean",
    "korean": "Korean",
    "ru": "Russian",
    "russian": "Russian",
    "th": "Thai",
    "thai": "Thai",
    "vi": "Vietnamese",
    "vietnamese": "Vietnamese",
    "ja": "Japanese",
    "japanese": "Japanese",
    "tr": "Turkish",
    "turkish": "Turkish",
    "hi": "Hindi",
    "hindi": "Hindi",
    "ms": "Malay",
    "malay": "Malay",
    "nl": "Dutch",
    "dutch": "Dutch",
    "sv": "Swedish",
    "swedish": "Swedish",
    "da": "Danish",
    "danish": "Danish",
    "fi": "Finnish",
    "finnish": "Finnish",
    "pl": "Polish",
    "polish": "Polish",
    "cs": "Czech",
    "czech": "Czech",
    "fil": "Filipino",
    "tl": "Filipino",
    "filipino": "Filipino",
    "fa": "Persian",
    "persian": "Persian",
    "el": "Greek",
    "greek": "Greek",
    "ro": "Romanian",
    "romanian": "Romanian",
    "hu": "Hungarian",
    "hungarian": "Hungarian",
    "mk": "Macedonian",
    "macedonian": "Macedonian",
}


def _openai_error_response(
    *,
    message: str,
    status_code: int = 400,
    error_type: str = "invalid_request_error",
    param: str | None = None,
    code: str | None = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "message": message,
                "type": error_type,
                "param": param,
                "code": code,
            }
        },
    )


def _openai_http_exception(exc: HTTPException) -> JSONResponse:
    detail = exc.detail if isinstance(exc.detail, str) else json.dumps(exc.detail)
    return _openai_error_response(message=detail, status_code=exc.status_code)


def _form_string(form: Any, name: str, default: str = "") -> str:
    value = form.get(name)
    if value is None:
        return default
    return str(value)


def _form_bool(form: Any, name: str, default: bool = False) -> bool:
    value = form.get(name)
    if value is None or str(value).strip() == "":
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _form_list(form: Any, name: str) -> list[str]:
    values = []
    for key in (name, f"{name}[]"):
        if hasattr(form, "getlist"):
            values.extend(form.getlist(key))
        elif form.get(key) is not None:
            values.append(form.get(key))
    out: list[str] = []
    for value in values:
        if value is None:
            continue
        for item in str(value).split(","):
            item = item.strip()
            if item:
                out.append(item)
    return out


def _normalize_openai_language(language: str) -> str:
    value = (language or "").strip()
    if not value:
        raise ValueError("language is empty")
    resolved = LANGUAGE_ALIASES.get(value.lower(), normalize_language_name(value))
    validate_language(resolved)
    return resolved


def _validate_temperature(value: str) -> None:
    if value.strip() == "":
        return
    try:
        temperature = float(value)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="temperature must be a number") from exc
    if temperature != 0.0:
        raise HTTPException(
            status_code=400,
            detail="Only temperature=0 is supported because deterministic transcription is required.",
        )


def _validate_model(model: str, model_name: str) -> None:
    if model and model not in {model_name, *MODEL_ALIASES}:
        raise HTTPException(status_code=400, detail=f"Unsupported model: {model}")


def _validate_response_format(response_format: str) -> str:
    resolved = response_format or "json"
    if resolved not in RESPONSE_FORMATS:
        raise HTTPException(status_code=400, detail=f"Unsupported response_format: {response_format}")
    return resolved


def _validate_timestamp_granularities(values: Iterable[str]) -> list[str]:
    out = []
    for value in values:
        normalized = value.strip().lower()
        if normalized not in TIMESTAMP_GRANULARITIES:
            raise HTTPException(status_code=400, detail=f"Unsupported timestamp granularity: {value}")
        if normalized not in out:
            out.append(normalized)
    return out


def _align_items(item: Any) -> list[Any]:
    timestamps = getattr(item, "time_stamps", None)
    if timestamps is None:
        return []
    return list(getattr(timestamps, "items", timestamps) or [])


def _words_payload(item: Any) -> list[dict[str, Any]]:
    words = []
    for span in _align_items(item):
        words.append(
            {
                "word": getattr(span, "text", ""),
                "start": float(getattr(span, "start_time", 0.0)),
                "end": float(getattr(span, "end_time", 0.0)),
            }
        )
    return words


def _segments_payload(item: Any) -> list[dict[str, Any]]:
    words = _words_payload(item)
    if not words:
        return []
    return [
        {
            "id": 0,
            "seek": 0,
            "start": words[0]["start"],
            "end": words[-1]["end"],
            "text": item.text,
            "tokens": [],
            "temperature": 0.0,
            "avg_logprob": None,
            "compression_ratio": None,
            "no_speech_prob": None,
        }
    ]


def _duration(item: Any) -> float | None:
    words = _words_payload(item)
    if words:
        return words[-1]["end"]
    return None


def _timestamp(seconds: float, *, decimal: str) -> str:
    seconds = max(0.0, float(seconds))
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    return f"{hours:02}:{minutes:02}:{secs:02}{decimal}{millis:03}"


def _srt_response(item: Any) -> PlainTextResponse:
    end = _duration(item) or 0.001
    body = f"1\n{_timestamp(0, decimal=',')} --> {_timestamp(end, decimal=',')}\n{item.text}\n"
    return PlainTextResponse(body, media_type="application/x-subrip")


def _vtt_response(item: Any) -> PlainTextResponse:
    end = _duration(item) or 0.001
    body = f"WEBVTT\n\n{_timestamp(0, decimal='.')} --> {_timestamp(end, decimal='.')}\n{item.text}\n"
    return PlainTextResponse(body, media_type="text/vtt")


def _json_response(item: Any, *, response_format: str, timestamp_granularities: list[str]) -> Any:
    if response_format == "text":
        return PlainTextResponse(item.text)
    if response_format == "srt":
        return _srt_response(item)
    if response_format == "vtt":
        return _vtt_response(item)
    if response_format == "verbose_json":
        payload: dict[str, Any] = {
            "text": item.text,
            "language": item.language,
            "duration": _duration(item),
            "segments": _segments_payload(item) if "segment" in timestamp_granularities else [],
        }
        if "word" in timestamp_granularities:
            payload["words"] = _words_payload(item)
        return payload
    return {"text": item.text}


def _sse_event(event_type: str, payload: dict[str, Any]) -> str:
    payload = {"type": event_type, **payload}
    return f"event: {event_type}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _stream_response(item: Any) -> PlainTextResponse:
    body = ""
    if item.text:
        body += _sse_event("transcript.text.delta", {"delta": item.text})
    body += _sse_event("transcript.text.done", {"text": item.text})
    body += "data: [DONE]\n\n"
    return PlainTextResponse(body, media_type="text/event-stream")


@dataclass
class _RealtimeSession:
    state: Any
    lock: asyncio.Lock
    model: str
    language: str | None
    prompt: str
    chunk_size_sec: float


def _realtime_payload(session_id: str, state: Any, *, final: bool) -> dict[str, Any]:
    return {
        "id": session_id,
        "object": "realtime.transcription",
        "type": "transcript.text.done" if final else "transcript.text.delta",
        "text": getattr(state, "text", ""),
        "language": getattr(state, "language", ""),
        "chunk_id": int(getattr(state, "chunk_id", 0)),
        "final": final,
    }


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
    realtime_sessions: dict[str, _RealtimeSession] = {}

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        return _openai_http_exception(exc)

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        return _openai_error_response(
            message=str(exc),
            status_code=422,
            error_type="invalid_request_error",
        )

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

    @app.get("/v1/models/{requested_model:path}")
    def retrieve_model(requested_model: str) -> Dict[str, Any]:
        _validate_model(requested_model, model_name)
        return {
            "id": model_name if requested_model in MODEL_ALIASES else requested_model,
            "object": "model",
            "owned_by": "hangry-labs",
        }

    @app.post("/v1/audio/transcriptions")
    async def transcriptions(request: Request):
        with optional_timer("transcription request", trace_requests):
            form = await request.form()

            upload = form.get("file")
            if not isinstance(upload, (UploadFile, StarletteUploadFile)):
                raise HTTPException(status_code=400, detail="Missing required multipart file field: file")

            try:
                model = _form_string(form, "model")
                _validate_model(model, model_name)

                forced_language = None
                language = _form_string(form, "language")
                if language.strip():
                    try:
                        forced_language = _normalize_openai_language(language)
                    except ValueError as exc:
                        raise HTTPException(status_code=400, detail=str(exc)) from exc

                response_format = _validate_response_format(_form_string(form, "response_format", "json"))
                timestamp_granularities = _validate_timestamp_granularities(_form_list(form, "timestamp_granularities"))
                include = _form_list(form, "include")
                if include:
                    raise HTTPException(status_code=400, detail=f"Unsupported include values: {include}")
                _validate_temperature(_form_string(form, "temperature"))
                stream = _form_bool(form, "stream")
                prompt = _form_string(form, "prompt")

                return_time_stamps = bool(timestamp_granularities)
                if return_time_stamps and getattr(asr, "forced_aligner", None) is None:
                    raise HTTPException(
                        status_code=400,
                        detail="timestamp_granularities requires QWEN_ASR_ENABLE_ALIGNER=1",
                    )
            except HTTPException:
                await upload.close()
                raise

            suffix = Path(upload.filename or "audio.wav").suffix or ".wav"
            with optional_timer("read uploaded audio", trace_requests):
                payload = await upload.read()
                await upload.close()
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
                            context=prompt,
                            language=forced_language,
                            return_time_stamps=return_time_stamps,
                        )
            finally:
                Path(tmp_path).unlink(missing_ok=True)

            item = result[0]
            if stream:
                if response_format not in {"json", "text"}:
                    raise HTTPException(status_code=400, detail="stream=true supports response_format json or text")
                return _stream_response(item)
            return _json_response(
                item,
                response_format=response_format,
                timestamp_granularities=timestamp_granularities,
            )

    @app.post("/v1/audio/translations")
    async def translations(request: Request):
        raise HTTPException(
            status_code=501,
            detail=(
                "/v1/audio/translations is not implemented yet. "
                "Qwen3-ASR translation mode needs a dedicated compatibility pass before it is advertised."
            ),
        )

    @app.post("/v1/realtime/transcriptions/sessions")
    async def create_realtime_session(request: Request) -> Dict[str, Any]:
        try:
            payload = await request.json()
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="Request body must be JSON") from exc

        model = str(payload.get("model") or "")
        _validate_model(model, model_name)
        _validate_temperature(str(payload.get("temperature", "")))

        forced_language = None
        language = str(payload.get("language") or "")
        if language.strip():
            try:
                forced_language = _normalize_openai_language(language)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

        chunk_size_sec = float(payload.get("chunk_size_sec", 2.0))
        unfixed_chunk_num = int(payload.get("unfixed_chunk_num", 2))
        unfixed_token_num = int(payload.get("unfixed_token_num", 5))
        prompt = str(payload.get("prompt") or "")

        if not hasattr(asr, "init_streaming_state"):
            raise HTTPException(status_code=501, detail="Realtime streaming is not available for this backend")

        try:
            state = asr.init_streaming_state(
                context=prompt,
                language=forced_language,
                unfixed_chunk_num=unfixed_chunk_num,
                unfixed_token_num=unfixed_token_num,
                chunk_size_sec=chunk_size_sec,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        session_id = f"rt_{uuid.uuid4().hex}"
        realtime_sessions[session_id] = _RealtimeSession(
            state=state,
            lock=asyncio.Lock(),
            model=model_name if model in MODEL_ALIASES else model_name,
            language=forced_language,
            prompt=prompt,
            chunk_size_sec=chunk_size_sec,
        )
        return {
            "id": session_id,
            "object": "realtime.transcription_session",
            "model": model_name,
            "language": forced_language,
            "chunk_size_sec": chunk_size_sec,
        }

    @app.post("/v1/realtime/transcriptions/sessions/{session_id}/audio")
    async def append_realtime_audio(session_id: str, request: Request) -> Dict[str, Any]:
        session = realtime_sessions.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail=f"Unknown realtime transcription session: {session_id}")

        form = await request.form()
        upload = form.get("file")
        if not isinstance(upload, (UploadFile, StarletteUploadFile)):
            raise HTTPException(status_code=400, detail="Missing required multipart file field: file")

        suffix = Path(upload.filename or "audio.wav").suffix or ".wav"
        with optional_timer("read realtime audio chunk", trace_requests):
            payload = await upload.read()
            await upload.close()
        if not payload:
            return _realtime_payload(session_id, session.state, final=False)

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(payload)
            tmp_path = tmp.name
        try:
            chunk = normalize_audios(tmp_path)[0]
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        async with session.lock:
            async with semaphore:
                try:
                    await asyncio.to_thread(asr.streaming_transcribe, chunk, session.state)
                except ValueError as exc:
                    raise HTTPException(status_code=400, detail=str(exc)) from exc

        return _realtime_payload(session_id, session.state, final=False)

    @app.post("/v1/realtime/transcriptions/sessions/{session_id}/finish")
    async def finish_realtime_session(session_id: str) -> Dict[str, Any]:
        session = realtime_sessions.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail=f"Unknown realtime transcription session: {session_id}")

        async with session.lock:
            async with semaphore:
                try:
                    await asyncio.to_thread(asr.finish_streaming_transcribe, session.state)
                except ValueError as exc:
                    raise HTTPException(status_code=400, detail=str(exc)) from exc
        realtime_sessions.pop(session_id, None)
        return _realtime_payload(session_id, session.state, final=True)

    @app.delete("/v1/realtime/transcriptions/sessions/{session_id}")
    async def delete_realtime_session(session_id: str) -> Dict[str, Any]:
        removed = realtime_sessions.pop(session_id, None)
        return {"id": session_id, "deleted": removed is not None}

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
