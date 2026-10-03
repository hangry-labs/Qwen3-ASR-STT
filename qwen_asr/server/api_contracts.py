from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class PermissiveRequestModel(BaseModel):
    """Validate fields we use while ignoring harmless client extensions."""

    model_config = ConfigDict(extra="ignore")


class OpenAIErrorDetail(BaseModel):
    message: str
    type: str = "invalid_request_error"
    param: str | None = None
    code: str | None = None


class OpenAIErrorResponse(BaseModel):
    error: OpenAIErrorDetail


class ModelObject(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int = Field(description="Unix timestamp when this model was registered by the running server.")
    owned_by: str = "hangry-labs"


class ModelListResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelObject]


class TranscriptionResponse(BaseModel):
    text: str


class TranscriptionWord(BaseModel):
    word: str
    start: float
    end: float


class TranscriptionSegment(BaseModel):
    id: int
    seek: int
    start: float
    end: float
    text: str
    tokens: list[int] = Field(default_factory=list)
    temperature: float = 0.0
    avg_logprob: float | None = None
    compression_ratio: float | None = None
    no_speech_prob: float | None = None


class TranscriptionVerboseResponse(BaseModel):
    task: Literal["transcribe"] = "transcribe"
    language: str
    duration: float
    text: str
    segments: list[TranscriptionSegment]
    words: list[TranscriptionWord] | None = None


class SupportedLanguagesResponse(BaseModel):
    languages: list[str]


class RealtimeSessionCreateRequest(PermissiveRequestModel):
    model: str = Field(
        default="",
        description="Optional active model ID or supported local alias. Omit to use the single loaded model.",
    )
    language: str | None = Field(
        default=None,
        description="Optional ISO language code or language name. Omit for automatic detection.",
    )
    prompt: str = Field(default="", description="Optional transcription context.")
    temperature: float = Field(
        default=0,
        description="Qwen3-ASR transcription is intentionally deterministic; only 0 is supported.",
    )
    chunk_size_sec: float | None = None
    max_window_sec: float | None = None
    unfixed_chunk_num: int | None = None
    unfixed_token_num: int | None = None


class RealtimeSessionResponse(BaseModel):
    id: str
    object: Literal["realtime.transcription_session"] = "realtime.transcription_session"
    model: str
    language: str | None
    chunk_size_sec: float
    max_window_sec: float


class RealtimeTranscriptionEvent(BaseModel):
    id: str
    object: Literal["realtime.transcription"] = "realtime.transcription"
    type: Literal["transcript.text.delta", "transcript.text.done"]
    text: str
    language: str
    chunk_id: int
    audio_seconds: float
    inference_window_seconds: float
    max_window_sec: float
    final: bool


class RealtimeSessionDeleteResponse(BaseModel):
    id: str
    deleted: bool


class RealtimeSettingsUpdate(PermissiveRequestModel):
    chunk_size_sec: float = Field(ge=0.5, le=5)
    max_window_sec: float = Field(ge=10, le=60)
    unfixed_chunk_num: int = Field(ge=0, le=6)
    unfixed_token_num: int = Field(ge=0, le=20)


class AlignerSettingsUpdate(PermissiveRequestModel):
    load_aligner_always: bool


ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    400: {"model": OpenAIErrorResponse, "description": "Invalid request"},
    413: {"model": OpenAIErrorResponse, "description": "Uploaded audio exceeds the configured limit"},
    422: {"model": OpenAIErrorResponse, "description": "Audio or alignment request cannot be processed"},
    503: {"model": OpenAIErrorResponse, "description": "Inference runtime unavailable"},
}


_WORD_SCHEMA = {
    "type": "object",
    "required": ["word", "start", "end"],
    "properties": {
        "word": {"type": "string"},
        "start": {"type": "number"},
        "end": {"type": "number"},
    },
}

_SEGMENT_SCHEMA = {
    "type": "object",
    "required": ["id", "seek", "start", "end", "text", "tokens", "temperature"],
    "properties": {
        "id": {"type": "integer"},
        "seek": {"type": "integer"},
        "start": {"type": "number"},
        "end": {"type": "number"},
        "text": {"type": "string"},
        "tokens": {"type": "array", "items": {"type": "integer"}},
        "temperature": {"type": "number", "enum": [0]},
        "avg_logprob": {"type": ["number", "null"]},
        "compression_ratio": {"type": ["number", "null"]},
        "no_speech_prob": {"type": ["number", "null"]},
    },
}

TRANSCRIPTION_RESPONSES: dict[int | str, dict[str, Any]] = {
    200: {
        "description": "Transcription response in the requested format",
        "content": {
            "application/json": {
                "schema": {
                    "oneOf": [
                        {
                            "type": "object",
                            "required": ["text"],
                            "properties": {"text": {"type": "string"}},
                        },
                        {
                            "type": "object",
                            "required": ["task", "language", "duration", "text", "segments"],
                            "properties": {
                                "task": {"type": "string", "enum": ["transcribe"]},
                                "language": {"type": "string"},
                                "duration": {"type": "number"},
                                "text": {"type": "string"},
                                "segments": {"type": "array", "items": _SEGMENT_SCHEMA},
                                "words": {"type": "array", "items": _WORD_SCHEMA},
                            },
                        },
                    ]
                }
            },
            "text/plain": {"schema": {"type": "string"}},
            "application/x-subrip": {"schema": {"type": "string"}},
            "text/vtt": {"schema": {"type": "string"}},
            "text/event-stream": {"schema": {"type": "string"}},
        },
    },
    **ERROR_RESPONSES,
}


TRANSCRIPTION_OPENAPI_EXTRA: dict[str, Any] = {
    "requestBody": {
        "required": True,
        "content": {
            "multipart/form-data": {
                "schema": {
                    "type": "object",
                    "required": ["file"],
                    "properties": {
                        "file": {
                            "type": "string",
                            "format": "binary",
                            "description": (
                                "Audio file. Tested formats: aac, flac, mp3, mp4, mpeg, mpga, m4a, ogg, wav, "
                                "and webm. Other formats are attempted by the bundled decoder."
                            ),
                        },
                        "model": {
                            "type": "string",
                            "description": (
                                "Optional active model ID, qwen3-asr, or qwen3-asr-stt. "
                                "Omit to use the single loaded model."
                            ),
                            "examples": ["qwen3-asr"],
                        },
                        "language": {
                            "type": "string",
                            "description": "Optional ISO code or language name. Omit for automatic detection.",
                            "examples": ["en"],
                        },
                        "prompt": {
                            "type": "string",
                            "description": "Optional context to guide transcription vocabulary and style.",
                        },
                        "response_format": {
                            "type": "string",
                            "enum": ["json", "text", "verbose_json", "srt", "vtt"],
                            "default": "json",
                        },
                        "temperature": {
                            "type": "number",
                            "enum": [0],
                            "default": 0,
                            "description": "Only deterministic temperature=0 is supported.",
                        },
                        "timestamp_granularities": {
                            "type": "array",
                            "items": {"type": "string", "enum": ["word", "segment"]},
                            "description": "Requires response_format=verbose_json and loads the forced aligner on demand.",
                        },
                        "stream": {
                            "type": "boolean",
                            "default": False,
                            "description": (
                                "Return OpenAI-compatible SSE events after file inference completes. "
                                "For progressive microphone transcription, use the Hangry Labs realtime session API."
                            ),
                        },
                    },
                }
            }
        },
    }
}


REALTIME_AUDIO_OPENAPI_EXTRA: dict[str, Any] = {
    "requestBody": {
        "required": True,
        "content": {
            "multipart/form-data": {
                "schema": {
                    "type": "object",
                    "required": ["file"],
                    "properties": {
                        "file": {
                            "type": "string",
                            "format": "binary",
                            "description": "The next audio chunk for this realtime session.",
                        }
                    },
                }
            }
        },
    }
}
