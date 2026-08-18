# coding=utf-8
from __future__ import annotations

import asyncio
import json
import os
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable

import uvicorn
from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse
from starlette.datastructures import UploadFile as StarletteUploadFile

from qwen_asr.inference.utils import SUPPORTED_LANGUAGES, normalize_audios, normalize_language_name, validate_language
from qwen_asr.server.inference_runtime import (
    DEFAULT_INFERENCE_TIMEOUT_SECONDS,
    DEFAULT_QUEUE_TIMEOUT_SECONDS,
    InferenceCoordinator,
    InferenceTimeoutError,
    InferenceUnavailableError,
    schedule_process_recycle,
)
from qwen_asr.startup_logging import StartupTimer, log_startup, optional_timer

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
    response = _openai_error_response(message=detail, status_code=exc.status_code)
    if exc.headers:
        response.headers.update(exc.headers)
    return response


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
    created_monotonic: float
    updated_monotonic: float
    status: str = "active"


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


def _streaming_append_requires_inference(state: Any, chunk: Any) -> bool:
    buffer = getattr(state, "buffer", None)
    chunk_size_samples = getattr(state, "chunk_size_samples", None)
    if buffer is None or chunk_size_samples is None:
        return True
    return len(buffer) + len(chunk) >= int(chunk_size_samples)


def _streaming_finish_requires_inference(state: Any) -> bool:
    buffer = getattr(state, "buffer", None)
    return buffer is None or len(buffer) > 0


def _inference_http_exception(exc: InferenceUnavailableError) -> HTTPException:
    return HTTPException(
        status_code=503,
        detail=f"Inference service is unavailable: {exc}",
        headers={"Retry-After": "5"},
    )


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
    inference_timeout_seconds: float | None = None,
    queue_timeout_seconds: float | None = None,
    realtime_session_ttl_seconds: float | None = None,
    recycle_process: Callable[[str], None] | None = None,
    enable_watchdog: bool | None = None,
) -> FastAPI:
    realtime_sessions: dict[str, _RealtimeSession] = {}
    inference_timeout = float(
        inference_timeout_seconds
        if inference_timeout_seconds is not None
        else os.getenv("QWEN_ASR_INFERENCE_TIMEOUT_SECONDS", DEFAULT_INFERENCE_TIMEOUT_SECONDS)
    )
    queue_timeout = float(
        queue_timeout_seconds
        if queue_timeout_seconds is not None
        else os.getenv("QWEN_ASR_INFERENCE_QUEUE_TIMEOUT_SECONDS", DEFAULT_QUEUE_TIMEOUT_SECONDS)
    )
    session_ttl = float(
        realtime_session_ttl_seconds
        if realtime_session_ttl_seconds is not None
        else os.getenv("QWEN_ASR_REALTIME_SESSION_TTL_SECONDS", "900")
    )
    recycle_delay = float(os.getenv("QWEN_ASR_RECYCLE_DELAY_SECONDS", "2"))
    watchdog_enabled = (
        enable_watchdog
        if enable_watchdog is not None
        else os.getenv("QWEN_ASR_WATCHDOG_ENABLED", "1").strip().lower() in {"1", "true", "yes", "y"}
    )
    watchdog_interval = max(1.0, float(os.getenv("QWEN_ASR_WATCHDOG_INTERVAL_SECONDS", "300")))
    watchdog_timeout = max(0.001, float(os.getenv("QWEN_ASR_WATCHDOG_TIMEOUT_SECONDS", "60")))
    default_watchdog_audio = Path(__file__).resolve().parents[2] / "testbench/assets/english/random/01.mp3"
    watchdog_audio = Path(os.getenv("QWEN_ASR_WATCHDOG_AUDIO", str(default_watchdog_audio)))

    def session_diagnostics() -> dict[str, Any]:
        now = time.monotonic()
        return {
            "realtime_sessions": [
                {
                    "session_id": session_id,
                    "status": session.status,
                    "age_seconds": round(now - session.created_monotonic, 3),
                    "idle_seconds": round(now - session.updated_monotonic, 3),
                    "lock_active": session.lock.locked(),
                }
                for session_id, session in realtime_sessions.items()
            ]
        }

    recycler = recycle_process or (lambda reason: schedule_process_recycle(reason, recycle_delay))
    coordinator = InferenceCoordinator(
        model_name=model_name,
        capacity=concurrency,
        inference_timeout_seconds=inference_timeout,
        queue_timeout_seconds=queue_timeout,
        recycle_process=recycler,
        diagnostics_provider=session_diagnostics,
    )
    cleanup_task: asyncio.Task[Any] | None = None
    watchdog_task: asyncio.Task[Any] | None = None

    def purge_expired_sessions() -> None:
        now = time.monotonic()
        expired = [
            session_id
            for session_id, session in realtime_sessions.items()
            if not session.lock.locked() and now - session.updated_monotonic >= session_ttl
        ]
        for session_id in expired:
            session = realtime_sessions.pop(session_id, None)
            if session is not None:
                session.status = "expired"

    async def cleanup_sessions() -> None:
        interval = max(1.0, min(60.0, session_ttl))
        while True:
            await asyncio.sleep(interval)
            purge_expired_sessions()

    async def run_watchdog_probe() -> None:
        await coordinator.run(
            "watchdog",
            asr.transcribe,
            audio=str(watchdog_audio),
            context="",
            language=None,
            return_time_stamps=False,
            language_mode="auto",
            deadline_seconds=watchdog_timeout,
        )

    async def inference_watchdog() -> None:
        await asyncio.sleep(watchdog_interval)
        while True:
            readiness = await coordinator.snapshot()
            if readiness["status"] != "ok":
                return
            if readiness["active_inference"] == 0 and readiness["queued_inference"] == 0:
                try:
                    await run_watchdog_probe()
                except InferenceUnavailableError as exc:
                    if str(exc) != "inference_capacity_exhausted":
                        return
                except Exception:
                    await coordinator.mark_degraded("watchdog_failure")
                    return
            await asyncio.sleep(watchdog_interval)

    @asynccontextmanager
    async def lifespan(_application: FastAPI):
        nonlocal cleanup_task, watchdog_task
        try:
            cleanup_task = asyncio.create_task(cleanup_sessions(), name="qwen-asr-session-cleanup")
            if watchdog_enabled:
                if watchdog_audio.is_file():
                    await run_watchdog_probe()
                    watchdog_task = asyncio.create_task(inference_watchdog(), name="qwen-asr-inference-watchdog")
                else:
                    raise RuntimeError(f"Inference watchdog fixture not found: {watchdog_audio}")
            yield
        except InferenceUnavailableError as exc:
            raise RuntimeError(f"Startup inference readiness probe failed: {exc}") from exc
        except Exception as exc:
            readiness = await coordinator.snapshot()
            if readiness["status"] == "ok":
                await coordinator.mark_degraded("watchdog_failure")
            raise RuntimeError("Startup inference readiness probe failed") from exc
        finally:
            if cleanup_task is not None:
                cleanup_task.cancel()
                try:
                    await cleanup_task
                except asyncio.CancelledError:
                    # Expected when the application lifespan shuts down.
                    pass
            if watchdog_task is not None:
                watchdog_task.cancel()
                try:
                    await watchdog_task
                except asyncio.CancelledError:
                    # Expected when the application lifespan shuts down.
                    pass
            coordinator.shutdown()

    app = FastAPI(title="Qwen3-ASR OpenAI-compatible API", lifespan=lifespan)
    app.state.inference_coordinator = coordinator
    app.state.realtime_sessions = realtime_sessions

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

    @app.get("/health/live")
    def health_live() -> Dict[str, Any]:
        return {"status": "ok", "model": model_name}

    async def readiness_response() -> JSONResponse:
        payload = await coordinator.snapshot()
        return JSONResponse(status_code=200 if payload["status"] == "ok" else 503, content=payload)

    @app.get("/health/ready")
    async def health_ready() -> JSONResponse:
        return await readiness_response()

    @app.get("/health")
    async def health() -> JSONResponse:
        return await readiness_response()

    @app.get("/metrics/inference")
    async def inference_metrics() -> Dict[str, Any]:
        return await coordinator.metrics()

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
        readiness = await coordinator.snapshot()
        if readiness["status"] != "ok":
            raise _inference_http_exception(InferenceUnavailableError(readiness["reason"] or "degraded"))
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
                try:
                    with optional_timer("run ASR transcription", trace_requests):
                        result = await coordinator.run(
                            "ordinary",
                            asr.transcribe,
                            audio=tmp_path,
                            context=prompt,
                            language=forced_language,
                            return_time_stamps=return_time_stamps,
                            language_mode="forced" if forced_language else "auto",
                        )
                except InferenceUnavailableError as exc:
                    raise _inference_http_exception(exc) from exc
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
        purge_expired_sessions()
        readiness = await coordinator.snapshot()
        if readiness["status"] != "ok":
            raise _inference_http_exception(InferenceUnavailableError(readiness["reason"] or "degraded"))
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
        now = time.monotonic()
        realtime_sessions[session_id] = _RealtimeSession(
            state=state,
            lock=asyncio.Lock(),
            model=model_name if model in MODEL_ALIASES else model_name,
            language=forced_language,
            prompt=prompt,
            chunk_size_sec=chunk_size_sec,
            created_monotonic=now,
            updated_monotonic=now,
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
        purge_expired_sessions()
        session = realtime_sessions.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail=f"Unknown realtime transcription session: {session_id}")
        if session.status != "active":
            raise HTTPException(status_code=409, detail=f"Realtime session is {session.status}")
        readiness = await coordinator.snapshot()
        if readiness["status"] != "ok":
            raise _inference_http_exception(InferenceUnavailableError(readiness["reason"] or "degraded"))

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

        if session.status != "active":
            raise HTTPException(status_code=409, detail=f"Realtime session is {session.status}")

        async with session.lock:
            if session.status != "active":
                raise HTTPException(status_code=409, detail=f"Realtime session is {session.status}")
            try:
                if _streaming_append_requires_inference(session.state, chunk):
                    await coordinator.run(
                        "realtime_append",
                        asr.streaming_transcribe,
                        chunk,
                        session.state,
                        session_id=session_id,
                        language_mode="forced" if session.language else "auto",
                        audio_samples=len(chunk),
                    )
                else:
                    asr.streaming_transcribe(chunk, session.state)
            except InferenceTimeoutError as exc:
                session.status = "failed"
                raise _inference_http_exception(exc) from exc
            except InferenceUnavailableError as exc:
                session.status = "failed"
                raise _inference_http_exception(exc) from exc
            except ValueError as exc:
                session.status = "failed"
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            finally:
                session.updated_monotonic = time.monotonic()

        return _realtime_payload(session_id, session.state, final=False)

    @app.post("/v1/realtime/transcriptions/sessions/{session_id}/finish")
    async def finish_realtime_session(session_id: str) -> Dict[str, Any]:
        purge_expired_sessions()
        session = realtime_sessions.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail=f"Unknown realtime transcription session: {session_id}")

        if session.status != "active":
            raise HTTPException(status_code=409, detail=f"Realtime session is {session.status}")
        readiness = await coordinator.snapshot()
        if readiness["status"] != "ok":
            raise _inference_http_exception(InferenceUnavailableError(readiness["reason"] or "degraded"))
        async with session.lock:
            if session.status != "active":
                raise HTTPException(status_code=409, detail=f"Realtime session is {session.status}")
            session.status = "finishing"
            try:
                if _streaming_finish_requires_inference(session.state):
                    await coordinator.run(
                        "realtime_finish",
                        asr.finish_streaming_transcribe,
                        session.state,
                        session_id=session_id,
                        language_mode="forced" if session.language else "auto",
                        audio_samples=len(getattr(session.state, "buffer", [])),
                    )
                else:
                    asr.finish_streaming_transcribe(session.state)
            except InferenceTimeoutError as exc:
                session.status = "failed"
                raise _inference_http_exception(exc) from exc
            except InferenceUnavailableError as exc:
                session.status = "failed"
                raise _inference_http_exception(exc) from exc
            except ValueError as exc:
                session.status = "failed"
                raise HTTPException(status_code=400, detail=str(exc)) from exc
        session.status = "finished"
        realtime_sessions.pop(session_id, None)
        return _realtime_payload(session_id, session.state, final=True)

    @app.delete("/v1/realtime/transcriptions/sessions/{session_id}")
    async def delete_realtime_session(session_id: str) -> Dict[str, Any]:
        purge_expired_sessions()
        session = realtime_sessions.get(session_id)
        if session is None:
            return {"id": session_id, "deleted": False}
        if session.lock.locked() or session.status in {"finishing", "failed"}:
            raise HTTPException(status_code=409, detail=f"Realtime session is {session.status}; deletion is unsafe")
        session.status = "deleted"
        realtime_sessions.pop(session_id, None)
        return {"id": session_id, "deleted": True}

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
        warmup_tokens = int(os.getenv("QWEN_ASR_STARTUP_WARMUP_TOKENS", os.getenv("QWEN_ASR_MAX_NEW_TOKENS", "512")))
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
