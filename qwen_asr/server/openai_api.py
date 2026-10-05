# coding=utf-8
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import tempfile
import time
import uuid
from collections.abc import Awaitable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse
from starlette.datastructures import UploadFile as StarletteUploadFile

from qwen_asr.inference.utils import (
    SAMPLE_RATE,
    SUPPORTED_LANGUAGES,
    normalize_audios,
    normalize_language_name,
    validate_language,
)
from qwen_asr.server.aligner_runtime import (
    ALIGNER_MODEL_SUPPORTED_LANGUAGES,
    DEFAULT_SETTINGS_PATH,
    AlignerRuntime,
    AlignerUnavailableError,
    RuntimeSettingsStore,
    aligner_language_capabilities,
)
from qwen_asr.server.api_contracts import (
    ERROR_RESPONSES,
    REALTIME_AUDIO_OPENAPI_EXTRA,
    TRANSCRIPTION_OPENAPI_EXTRA,
    TRANSCRIPTION_RESPONSES,
    AlignerSettingsUpdate,
    MCPSettingsUpdate,
    ModelListResponse,
    ModelObject,
    RealtimeSessionCreateRequest,
    RealtimeSessionDeleteResponse,
    RealtimeSessionResponse,
    RealtimeSettingsUpdate,
    RealtimeTranscriptionEvent,
    SupportedLanguagesResponse,
)
from qwen_asr.server.contracts import ASRRuntime
from qwen_asr.server.inference_runtime import (
    DEFAULT_INFERENCE_TIMEOUT_SECONDS,
    DEFAULT_QUEUE_TIMEOUT_SECONDS,
    InferenceCoordinator,
    InferenceTimeoutError,
    InferenceUnavailableError,
    schedule_process_recycle,
)
from qwen_asr.startup_logging import optional_timer

MODEL_ALIASES = {"qwen3-asr", "qwen3-asr-stt"}
RESPONSE_FORMATS = {"json", "text", "verbose_json", "srt", "vtt"}
TIMESTAMP_GRANULARITIES = {"segment", "word"}
SUPPORTED_AUDIO_CONTENT_TYPES = {
    "audio/aac": ".aac",
    "audio/flac": ".flac",
    "audio/mp4": ".mp4",
    "audio/mpeg": ".mp3",
    "audio/ogg": ".ogg",
    "audio/wav": ".wav",
    "audio/wave": ".wav",
    "audio/webm": ".webm",
    "audio/x-aac": ".aac",
    "video/mp4": ".mp4",
    "video/webm": ".webm",
}
DEFAULT_MAX_UPLOAD_MB = 100.0
SEGMENT_SILENCE_GAP_SECONDS = 0.75
SEGMENT_MAX_SECONDS = 12.0
SEGMENT_TERMINATORS = (".", "!", "?", "。", "！", "？")
NO_SPACE_ALIGNMENT_LANGUAGES = {"Chinese", "Cantonese", "Japanese"}
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

logger = logging.getLogger(__name__)


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


def _audio_suffix(filename: str | None, content_type: str | None) -> str:
    suffix = Path(filename or "").suffix.lower()
    if re.fullmatch(r"\.[a-z0-9]{1,10}", suffix):
        return suffix
    normalized_content_type = str(content_type or "").split(";", 1)[0].strip().lower()
    inferred = SUPPORTED_AUDIO_CONTENT_TYPES.get(normalized_content_type)
    if inferred:
        return inferred
    return ".audio"


def _upload_suffix(upload: UploadFile | StarletteUploadFile) -> str:
    return _audio_suffix(upload.filename, upload.content_type)


async def _read_upload(
    upload: UploadFile | StarletteUploadFile,
    *,
    max_upload_bytes: int,
    allow_empty: bool = False,
) -> bytes:
    try:
        payload = await upload.read(max_upload_bytes + 1)
    finally:
        await upload.close()
    if len(payload) > max_upload_bytes:
        limit_mb = max_upload_bytes / (1024 * 1024)
        raise HTTPException(
            status_code=413,
            detail=f"Uploaded audio exceeds the configured {limit_mb:g} MB limit.",
        )
    if not payload and not allow_empty:
        raise HTTPException(status_code=400, detail="Uploaded audio file is empty")
    return payload


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
    if not model.strip():
        return
    if model not in {model_name, *MODEL_ALIASES}:
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


def _alignment_dependency_message(exc: ImportError) -> str:
    message = str(exc).strip()
    if "soynlp" in message.lower():
        return (
            "Korean timestamp alignment is not available in this image because its optional "
            "soynlp tokenizer is not bundled. Korean transcription without timestamps remains available."
        )
    return f"Timestamp alignment is unavailable for this language: {message}"


def _segment_text(words: list[dict[str, Any]], language: str) -> str:
    parts = [str(word.get("word") or "") for word in words]
    if language in NO_SPACE_ALIGNMENT_LANGUAGES:
        return "".join(parts).strip()
    text = " ".join(part.strip() for part in parts if part.strip())
    return re.sub(r"\s+([,.;:!?%\)\]\}])", r"\1", text).strip()


def _project_terminal_punctuation(text: str, words: list[dict[str, Any]]) -> dict[int, str]:
    """Map transcript sentence endings back onto punctuation-free aligner words."""
    transcript_length = sum(character.isalnum() for character in text)
    aligned_lengths = [sum(character.isalnum() for character in str(word.get("word") or "")) for word in words]
    aligned_length = sum(aligned_lengths)
    if transcript_length <= 0 or aligned_length <= 0:
        return {}

    transcript_boundaries: dict[int, str] = {}
    position = 0
    for character in text:
        if character.isalnum():
            position += 1
        elif character in SEGMENT_TERMINATORS and position:
            transcript_boundaries[position] = transcript_boundaries.get(position, "") + character

    projected: dict[int, str] = {}
    cumulative = 0
    word_index = 0
    for transcript_position, punctuation in transcript_boundaries.items():
        target = transcript_position / transcript_length * aligned_length
        while word_index < len(words) - 1 and cumulative + aligned_lengths[word_index] < target:
            cumulative += aligned_lengths[word_index]
            word_index += 1
        projected[word_index] = projected.get(word_index, "") + punctuation
    return projected


def _segments_payload(item: Any) -> list[dict[str, Any]]:
    words = _words_payload(item)
    if not words:
        return []
    punctuation = _project_terminal_punctuation(str(getattr(item, "text", "") or ""), words)
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for index, word in enumerate(words):
        suffix = punctuation.get(index, "")
        if suffix and not str(word.get("word") or "").rstrip().endswith(SEGMENT_TERMINATORS):
            word = {**word, "word": f"{word.get('word') or ''}{suffix}"}
        current.append(word)
        next_word = words[index + 1] if index + 1 < len(words) else None
        gap_after = max(0.0, float(next_word["start"]) - float(word["end"])) if next_word else 0.0
        duration = float(word["end"]) - float(current[0]["start"])
        terminal = bool(suffix) or str(word.get("word") or "").rstrip().endswith(SEGMENT_TERMINATORS)
        if next_word is None or terminal or gap_after >= SEGMENT_SILENCE_GAP_SECONDS or duration >= SEGMENT_MAX_SECONDS:
            groups.append(current)
            current = []

    language = str(getattr(item, "language", "") or "")
    return [
        {
            "id": index,
            "seek": round(float(group[0]["start"]) * 100),
            "start": float(group[0]["start"]),
            "end": float(group[-1]["end"]),
            "text": _segment_text(group, language),
            "tokens": [],
            "temperature": 0.0,
            "avg_logprob": None,
            "compression_ratio": None,
            "no_speech_prob": None,
        }
        for index, group in enumerate(groups)
    ]


def _duration(item: Any) -> float | None:
    duration = getattr(item, "duration", None)
    if isinstance(duration, (int, float)) and not isinstance(duration, bool):
        return max(0.0, float(duration))
    words = _words_payload(item)
    if words:
        return words[-1]["end"]
    return None


def _coarse_segments_payload(item: Any) -> list[dict[str, Any]]:
    text = str(getattr(item, "text", "") or "")
    duration = _duration(item)
    if not text or duration is None:
        return []
    return [
        {
            "id": 0,
            "seek": 0,
            "start": 0.0,
            "end": duration,
            "text": text,
            "tokens": [],
            "temperature": 0.0,
            "avg_logprob": None,
            "compression_ratio": None,
            "no_speech_prob": None,
        }
    ]


def _timestamp(seconds: float, *, decimal: str) -> str:
    seconds = max(0.0, float(seconds))
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))
    return f"{hours:02}:{minutes:02}:{secs:02}{decimal}{millis:03}"


def _srt_response(item: Any) -> PlainTextResponse:
    segments = _segments_payload(item)
    body = "\n".join(
        f"{index}\n{_timestamp(segment['start'], decimal=',')} --> {_timestamp(segment['end'], decimal=',')}\n{segment['text']}\n"
        for index, segment in enumerate(segments, start=1)
    )
    return PlainTextResponse(body, media_type="application/x-subrip")


def _vtt_response(item: Any) -> PlainTextResponse:
    segments = _segments_payload(item)
    cues = "\n".join(
        f"{_timestamp(segment['start'], decimal='.')} --> {_timestamp(segment['end'], decimal='.')}\n{segment['text']}\n"
        for segment in segments
    )
    body = f"WEBVTT\n\n{cues}"
    return PlainTextResponse(body, media_type="text/vtt")


def _json_response(item: Any, *, response_format: str, timestamp_granularities: list[str]) -> Any:
    if response_format == "text":
        return PlainTextResponse(item.text)
    if response_format == "srt":
        return _srt_response(item)
    if response_format == "vtt":
        return _vtt_response(item)
    if response_format == "verbose_json":
        segments = _segments_payload(item) if "segment" in timestamp_granularities else _coarse_segments_payload(item)
        payload: dict[str, Any] = {
            "task": "transcribe",
            "text": item.text,
            "language": item.language,
            "duration": _duration(item) or 0.0,
            "segments": segments,
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
    max_window_sec: float
    created_monotonic: float
    updated_monotonic: float
    status: str = "active"


def _realtime_payload(session_id: str, state: Any, *, final: bool) -> dict[str, Any]:
    audio_samples = int(getattr(state, "audio_samples_seen", 0)) + len(getattr(state, "buffer", []))
    inference_samples = len(getattr(state, "audio_accum", []))
    return {
        "id": session_id,
        "object": "realtime.transcription",
        "type": "transcript.text.done" if final else "transcript.text.delta",
        "text": getattr(state, "text", ""),
        "language": getattr(state, "language", ""),
        "chunk_id": int(getattr(state, "chunk_id", 0)),
        "audio_seconds": round(audio_samples / SAMPLE_RATE, 3),
        "inference_window_seconds": round(inference_samples / SAMPLE_RATE, 3),
        "max_window_sec": float(getattr(state, "max_window_sec", 0)),
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


@dataclass
class TranscriptionService:
    """One transcription boundary shared by HTTP, MCP, and future adapters."""

    asr: ASRRuntime
    coordinator: InferenceCoordinator
    max_upload_bytes: int
    ensure_aligner_loaded: Callable[[], Awaitable[dict[str, Any]]]
    timestamp_language_error: Callable[[str | None], str | None]
    trace_requests: bool = False

    async def transcribe_bytes(
        self,
        *,
        payload: bytes,
        filename: str | None,
        content_type: str | None = None,
        language: str | None = None,
        prompt: str = "",
        timestamp_granularities: Iterable[str] = (),
    ) -> Any:
        readiness = await self.coordinator.snapshot()
        if readiness["status"] != "ok":
            raise _inference_http_exception(InferenceUnavailableError(readiness["reason"] or "degraded"))

        if not payload:
            raise HTTPException(status_code=400, detail="Uploaded audio file is empty")
        if len(payload) > self.max_upload_bytes:
            limit_mb = self.max_upload_bytes / (1024 * 1024)
            raise HTTPException(
                status_code=413,
                detail=f"Uploaded audio exceeds the configured {limit_mb:g} MB limit.",
            )

        forced_language = None
        if language and language.strip():
            try:
                forced_language = _normalize_openai_language(language)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

        granularities = _validate_timestamp_granularities(timestamp_granularities)
        return_time_stamps = bool(granularities)
        if return_time_stamps:
            language_error = self.timestamp_language_error(forced_language)
            if language_error:
                raise HTTPException(status_code=422, detail=language_error)
            await self.ensure_aligner_loaded()

        suffix = _audio_suffix(filename, content_type)
        with (
            optional_timer("write uploaded audio temp file", self.trace_requests),
            tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp,
        ):
            tmp.write(payload)
            tmp_path = tmp.name

        try:
            try:
                with optional_timer("run ASR transcription", self.trace_requests):
                    result = await self.coordinator.run(
                        "ordinary",
                        self.asr.transcribe,
                        audio=tmp_path,
                        context=prompt,
                        language=forced_language,
                        return_time_stamps=return_time_stamps,
                        language_mode="forced" if forced_language else "auto",
                    )
            except InferenceUnavailableError as exc:
                raise _inference_http_exception(exc) from exc
            except ImportError as exc:
                raise HTTPException(status_code=422, detail=_alignment_dependency_message(exc)) from exc
            except ValueError as exc:
                if return_time_stamps and "not supported by the forced aligner" in str(exc):
                    raise HTTPException(status_code=422, detail=str(exc)) from exc
                raise HTTPException(status_code=400, detail=f"Audio could not be transcribed: {exc}") from exc
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        return result[0]


def create_app(
    *,
    asr: ASRRuntime,
    model_name: str,
    concurrency: int,
    trace_requests: bool = False,
    inference_timeout_seconds: float | None = None,
    queue_timeout_seconds: float | None = None,
    realtime_session_ttl_seconds: float | None = None,
    recycle_process: Callable[[str], None] | None = None,
    enable_watchdog: bool | None = None,
    startup_warmup: Callable[[], Any] | None = None,
    aligner_runtime: AlignerRuntime | None = None,
    max_upload_bytes: int | None = None,
) -> FastAPI:
    realtime_sessions: dict[str, _RealtimeSession] = {}
    model_registered_at = int(time.time())
    upload_limit = int(
        max_upload_bytes
        if max_upload_bytes is not None
        else float(os.getenv("QWEN_ASR_MAX_UPLOAD_MB", str(DEFAULT_MAX_UPLOAD_MB))) * 1024 * 1024
    )
    if upload_limit <= 0:
        raise ValueError("The configured upload limit must be greater than zero.")
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
    startup_warmup_timeout = max(
        0.001,
        float(os.getenv("QWEN_ASR_STARTUP_WARMUP_TIMEOUT_SECONDS", "600")),
    )
    aligner_load_timeout = max(
        0.001,
        float(os.getenv("QWEN_ASR_ALIGNER_LOAD_TIMEOUT_SECONDS", "600")),
    )
    default_watchdog_audio = Path(__file__).resolve().parents[2] / "testbench/assets/english/random/01.mp3"
    watchdog_audio = Path(os.getenv("QWEN_ASR_WATCHDOG_AUDIO", str(default_watchdog_audio)))
    runtime_settings = getattr(aligner_runtime, "settings", None) or RuntimeSettingsStore(
        os.getenv("QWEN_ASR_SETTINGS_PATH", DEFAULT_SETTINGS_PATH)
    )
    mcp_default_enabled = os.getenv("QWEN_ASR_ENABLE_MCP", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }

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
            if startup_warmup is not None:
                await coordinator.run(
                    "warmup",
                    startup_warmup,
                    language_mode="mixed",
                    deadline_seconds=startup_warmup_timeout,
                )
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

    app = FastAPI(
        title="Qwen3-ASR STT API",
        description=(
            "OpenAI-compatible file transcription plus Hangry Labs realtime, system, and operations APIs. "
            "This local-first service does not require authentication; do not expose it to an untrusted network."
        ),
        version="1.0",
        openapi_tags=[
            {
                "name": "OpenAI compatibility",
                "description": "Drop-in routes for OpenAI transcription clients.",
            },
            {
                "name": "Hangry Labs realtime",
                "description": (
                    "Progressive bounded-window HTTP sessions used by the browser UI. "
                    "These are extensions, not OpenAI Realtime WebSocket endpoints."
                ),
            },
            {"name": "Discovery", "description": "Model, language, and capability discovery."},
            {"name": "Operations", "description": "Liveness, readiness, and inference telemetry."},
            {"name": "System", "description": "Persistent runtime settings and aligner lifecycle."},
        ],
        lifespan=lifespan,
    )
    app.state.inference_coordinator = coordinator
    app.state.realtime_sessions = realtime_sessions
    app.state.aligner_runtime = aligner_runtime
    app.state.runtime_settings = runtime_settings
    app.state.mcp_default_enabled = mcp_default_enabled

    def aligner_snapshot() -> dict[str, Any]:
        if aligner_runtime is not None:
            return aligner_runtime.snapshot()
        loaded = getattr(asr, "forced_aligner", None) is not None
        available_languages, unavailable_languages = aligner_language_capabilities()
        return {
            "configured": loaded,
            "loaded": loaded,
            "status": "loaded" if loaded else "unavailable",
            "load_always": loaded,
            "model": None,
            "last_error": None,
            "model_supported_languages": list(ALIGNER_MODEL_SUPPORTED_LANGUAGES),
            "available_languages": available_languages,
            "unavailable_languages": unavailable_languages,
        }

    def timestamp_language_error(language: str | None) -> str | None:
        if language is None:
            return None
        aligner = aligner_snapshot()
        supported = aligner.get("model_supported_languages") or []
        if supported and language not in supported:
            return (
                f"{language} transcription is supported, but the configured forced aligner does not support "
                f"{language} timestamps. Supported timestamp languages: {', '.join(supported)}."
            )
        unavailable = aligner.get("unavailable_languages") or {}
        return unavailable.get(language)

    async def ensure_aligner_loaded() -> dict[str, Any]:
        snapshot = aligner_snapshot()
        if snapshot["loaded"]:
            return snapshot
        if aligner_runtime is None or not snapshot["configured"]:
            raise HTTPException(
                status_code=400,
                detail="Timestamp output is unavailable because no forced-aligner model is configured.",
            )
        try:
            return await coordinator.run(
                "aligner_load",
                aligner_runtime.load,
                warm_up=True,
                language_mode="alignment",
                deadline_seconds=aligner_load_timeout,
            )
        except AlignerUnavailableError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except InferenceUnavailableError as exc:
            raise _inference_http_exception(exc) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail=f"Forced aligner could not be loaded: {type(exc).__name__}: {exc}",
            ) from exc

    async def release_aligner_runtime() -> dict[str, Any]:
        if aligner_runtime is None:
            raise HTTPException(status_code=400, detail="Dynamic forced-aligner management is unavailable.")
        if aligner_runtime.load_always:
            raise HTTPException(
                status_code=409,
                detail="Disable 'Always keep aligner loaded' before unloading it.",
            )
        try:
            return await coordinator.run(
                "aligner_unload",
                aligner_runtime.unload,
                language_mode="alignment",
                deadline_seconds=aligner_load_timeout,
            )
        except InferenceUnavailableError as exc:
            raise _inference_http_exception(exc) from exc

    async def set_aligner_loaded(enabled: bool) -> dict[str, Any]:
        """Set only the aligner's current in-memory state, preserving startup policy."""
        snapshot = aligner_snapshot()
        if snapshot["loaded"] == enabled:
            return snapshot
        if enabled:
            return await ensure_aligner_loaded()
        if aligner_runtime is None:
            raise HTTPException(status_code=400, detail="Dynamic forced-aligner management is unavailable.")
        try:
            return await coordinator.run(
                "aligner_unload",
                aligner_runtime.unload,
                language_mode="alignment",
                deadline_seconds=aligner_load_timeout,
            )
        except InferenceUnavailableError as exc:
            raise _inference_http_exception(exc) from exc

    async def configure_aligner_runtime(enabled: bool) -> dict[str, Any]:
        if aligner_runtime is None:
            raise HTTPException(status_code=400, detail="Dynamic forced-aligner management is unavailable.")
        if enabled:
            await ensure_aligner_loaded()
            try:
                aligner_runtime.set_load_always(True)
            except OSError as exc:
                raise HTTPException(status_code=500, detail=f"Runtime setting could not be saved: {exc}") from exc
            return aligner_runtime.snapshot()

        try:
            aligner_runtime.set_load_always(False)
        except OSError as exc:
            raise HTTPException(status_code=500, detail=f"Runtime setting could not be saved: {exc}") from exc
        return await release_aligner_runtime()

    transcription_service = TranscriptionService(
        asr=asr,
        coordinator=coordinator,
        max_upload_bytes=upload_limit,
        ensure_aligner_loaded=ensure_aligner_loaded,
        timestamp_language_error=timestamp_language_error,
        trace_requests=trace_requests,
    )
    app.state.transcription_service = transcription_service
    app.state.model_name = model_name
    app.state.aligner_snapshot = aligner_snapshot
    app.state.ensure_aligner_loaded = ensure_aligner_loaded
    app.state.release_aligner = release_aligner_runtime
    app.state.set_aligner_loaded = set_aligner_loaded
    app.state.configure_aligner = configure_aligner_runtime

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        return _openai_http_exception(exc)

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        return _openai_error_response(
            message=str(exc),
            status_code=400,
            error_type="invalid_request_error",
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled API error for %s %s", request.method, request.url.path, exc_info=exc)
        return _openai_error_response(
            message="Internal server error.",
            status_code=500,
            error_type="server_error",
        )

    @app.get("/health/live", tags=["Operations"])
    def health_live() -> Dict[str, Any]:
        return {"status": "ok", "model": model_name}

    async def readiness_response() -> JSONResponse:
        payload = await coordinator.snapshot()
        aligner = aligner_snapshot()
        payload["capabilities"] = {
            "timestamps": aligner["configured"],
            "timestamps_loaded": aligner["loaded"],
            "timestamp_granularities": ["word", "segment"] if aligner["configured"] else [],
            "forced_aligner": aligner,
        }
        return JSONResponse(status_code=200 if payload["status"] == "ok" else 503, content=payload)

    @app.get("/health/ready", tags=["Operations"])
    async def health_ready() -> JSONResponse:
        return await readiness_response()

    @app.get("/health", tags=["Operations"])
    async def health() -> JSONResponse:
        return await readiness_response()

    @app.get("/metrics/inference", tags=["Operations"])
    async def inference_metrics() -> Dict[str, Any]:
        return await coordinator.metrics()

    @app.get("/system/aligner", tags=["System"])
    def aligner_status() -> Dict[str, Any]:
        return aligner_snapshot()

    @app.get("/system/settings", tags=["System"])
    def system_settings() -> Dict[str, Any]:
        return {
            "realtime_defaults": runtime_settings.realtime_defaults(),
            "mcp": {
                "enabled": runtime_settings.mcp_enabled(default=mcp_default_enabled),
                "endpoint": "/mcp",
                "input_directory": str(getattr(app.state, "mcp_input_directory", "") or "") or None,
            },
        }

    @app.put("/system/settings/mcp", tags=["System"], responses=ERROR_RESPONSES)
    async def configure_mcp(settings: MCPSettingsUpdate) -> Dict[str, Any]:
        try:
            runtime_settings.set_mcp_enabled(settings.enabled)
        except OSError as exc:
            raise HTTPException(status_code=500, detail=f"Runtime setting could not be saved: {exc}") from exc
        return {
            "mcp": {
                "enabled": runtime_settings.mcp_enabled(default=mcp_default_enabled),
                "endpoint": "/mcp",
                "input_directory": str(getattr(app.state, "mcp_input_directory", "") or "") or None,
            }
        }

    @app.put("/system/settings/realtime", tags=["System"], responses=ERROR_RESPONSES)
    async def configure_realtime_defaults(settings: RealtimeSettingsUpdate) -> Dict[str, Any]:
        try:
            defaults = runtime_settings.set_realtime_defaults(settings.model_dump())
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except OSError as exc:
            raise HTTPException(status_code=500, detail=f"Runtime setting could not be saved: {exc}") from exc
        return {"realtime_defaults": defaults}

    @app.post("/system/aligner/load", tags=["System"], responses=ERROR_RESPONSES)
    async def load_aligner() -> Dict[str, Any]:
        return await ensure_aligner_loaded()

    @app.post("/system/aligner/unload", tags=["System"], responses=ERROR_RESPONSES)
    async def unload_aligner() -> Dict[str, Any]:
        return await release_aligner_runtime()

    @app.put("/system/aligner", tags=["System"], responses=ERROR_RESPONSES)
    async def configure_aligner(settings: AlignerSettingsUpdate) -> Dict[str, Any]:
        return await configure_aligner_runtime(settings.load_aligner_always)

    @app.get(
        "/v1/models",
        tags=["OpenAI compatibility", "Discovery"],
        response_model=ModelListResponse,
    )
    def models() -> ModelListResponse:
        return {
            "object": "list",
            "data": [
                {
                    "id": model_name,
                    "object": "model",
                    "created": model_registered_at,
                    "owned_by": "hangry-labs",
                }
            ],
        }

    @app.get(
        "/v1/models/{requested_model:path}",
        tags=["OpenAI compatibility", "Discovery"],
        response_model=ModelObject,
        responses=ERROR_RESPONSES,
    )
    def retrieve_model(requested_model: str) -> ModelObject:
        _validate_model(requested_model, model_name)
        return {
            "id": model_name if requested_model in MODEL_ALIASES else requested_model,
            "object": "model",
            "created": model_registered_at,
            "owned_by": "hangry-labs",
        }

    @app.post(
        "/v1/audio/transcriptions",
        tags=["OpenAI compatibility"],
        summary="Create transcription",
        description=(
            "Transcribe one completed audio file. The request and ordinary response formats are compatible with "
            "OpenAI transcription clients. `stream=true` returns compatible SSE event shapes after inference; "
            "use the Hangry Labs realtime session endpoints for progressive microphone transcription."
        ),
        responses=TRANSCRIPTION_RESPONSES,
        openapi_extra=TRANSCRIPTION_OPENAPI_EXTRA,
    )
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

                language = _form_string(form, "language")
                if language.strip():
                    try:
                        _normalize_openai_language(language)
                    except ValueError as exc:
                        raise HTTPException(status_code=400, detail=str(exc)) from exc

                response_format = _validate_response_format(_form_string(form, "response_format", "json"))
                timestamp_granularities = _validate_timestamp_granularities(_form_list(form, "timestamp_granularities"))
                if timestamp_granularities and response_format != "verbose_json":
                    raise HTTPException(
                        status_code=400,
                        detail="timestamp_granularities requires response_format=verbose_json",
                    )
                include = _form_list(form, "include")
                if include:
                    raise HTTPException(status_code=400, detail=f"Unsupported include values: {include}")
                _validate_temperature(_form_string(form, "temperature"))
                stream = _form_bool(form, "stream")
                if stream and response_format not in {"json", "text"}:
                    raise HTTPException(status_code=400, detail="stream=true supports response_format json or text")
                prompt = _form_string(form, "prompt")
            except HTTPException:
                await upload.close()
                raise

            with optional_timer("read uploaded audio", trace_requests):
                payload = await _read_upload(upload, max_upload_bytes=upload_limit)

            effective_timestamp_granularities = list(timestamp_granularities)
            if response_format in {"srt", "vtt"}:
                effective_timestamp_granularities = ["segment"]
            item = await transcription_service.transcribe_bytes(
                payload=payload,
                filename=upload.filename,
                content_type=upload.content_type,
                language=language,
                prompt=prompt,
                timestamp_granularities=effective_timestamp_granularities,
            )
            if stream:
                return _stream_response(item)
            return _json_response(
                item,
                response_format=response_format,
                timestamp_granularities=timestamp_granularities,
            )

    @app.post(
        "/v1/audio/translations",
        tags=["OpenAI compatibility"],
        summary="Create translation (not implemented)",
        responses={501: {"description": "Qwen3-ASR translation compatibility has not been validated"}},
    )
    async def translations(request: Request):
        raise HTTPException(
            status_code=501,
            detail=(
                "/v1/audio/translations is not implemented yet. "
                "Qwen3-ASR translation mode needs a dedicated compatibility pass before it is advertised."
            ),
        )

    @app.post(
        "/v1/realtime/transcriptions/sessions",
        tags=["Hangry Labs realtime"],
        response_model=RealtimeSessionResponse,
        responses=ERROR_RESPONSES,
    )
    async def create_realtime_session(request_data: RealtimeSessionCreateRequest) -> RealtimeSessionResponse:
        purge_expired_sessions()
        readiness = await coordinator.snapshot()
        if readiness["status"] != "ok":
            raise _inference_http_exception(InferenceUnavailableError(readiness["reason"] or "degraded"))
        payload = request_data.model_dump(exclude_none=True)

        model = request_data.model
        _validate_model(model, model_name)
        _validate_temperature(str(request_data.temperature))

        forced_language = None
        language = str(request_data.language or "")
        if language.strip():
            try:
                forced_language = _normalize_openai_language(language)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

        realtime_defaults = runtime_settings.realtime_defaults()
        chunk_size_sec = float(payload.get("chunk_size_sec", realtime_defaults["chunk_size_sec"]))
        max_window_sec = float(payload.get("max_window_sec", realtime_defaults["max_window_sec"]))
        unfixed_chunk_num = int(payload.get("unfixed_chunk_num", realtime_defaults["unfixed_chunk_num"]))
        unfixed_token_num = int(payload.get("unfixed_token_num", realtime_defaults["unfixed_token_num"]))
        prompt = request_data.prompt
        if not 10 <= max_window_sec <= 60:
            raise HTTPException(status_code=400, detail="max_window_sec must be between 10 and 60 seconds.")

        if not hasattr(asr, "init_streaming_state"):
            raise HTTPException(status_code=501, detail="Realtime streaming is not available for this model")

        try:
            state = asr.init_streaming_state(
                context=prompt,
                language=forced_language,
                unfixed_chunk_num=unfixed_chunk_num,
                unfixed_token_num=unfixed_token_num,
                chunk_size_sec=chunk_size_sec,
                max_window_sec=max_window_sec,
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
            max_window_sec=max_window_sec,
            created_monotonic=now,
            updated_monotonic=now,
        )
        return RealtimeSessionResponse(
            id=session_id,
            model=model_name,
            language=forced_language,
            chunk_size_sec=chunk_size_sec,
            max_window_sec=max_window_sec,
        )

    @app.post(
        "/v1/realtime/transcriptions/sessions/{session_id}/audio",
        tags=["Hangry Labs realtime"],
        response_model=RealtimeTranscriptionEvent,
        responses=ERROR_RESPONSES,
        openapi_extra=REALTIME_AUDIO_OPENAPI_EXTRA,
    )
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

        try:
            suffix = _upload_suffix(upload)
        except HTTPException:
            await upload.close()
            raise
        with optional_timer("read realtime audio chunk", trace_requests):
            payload = await _read_upload(upload, max_upload_bytes=upload_limit, allow_empty=True)
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

    @app.post(
        "/v1/realtime/transcriptions/sessions/{session_id}/finish",
        tags=["Hangry Labs realtime"],
        response_model=RealtimeTranscriptionEvent,
        responses=ERROR_RESPONSES,
    )
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

    @app.delete(
        "/v1/realtime/transcriptions/sessions/{session_id}",
        tags=["Hangry Labs realtime"],
        response_model=RealtimeSessionDeleteResponse,
        responses=ERROR_RESPONSES,
    )
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

    @app.get(
        "/v1/audio/supported_languages",
        tags=["Discovery"],
        response_model=SupportedLanguagesResponse,
    )
    def supported_languages() -> SupportedLanguagesResponse:
        return {"languages": list(SUPPORTED_LANGUAGES)}

    return app
