from __future__ import annotations

import gc
import importlib.util
import json
import os
import tempfile
import threading
import traceback
from pathlib import Path
from typing import Any, Callable

from qwen_asr.server.contracts import ASRRuntime
from qwen_asr.startup_logging import log_startup


DEFAULT_SETTINGS_PATH = "/app/persistent/app/settings.json"
LOAD_ALWAYS_KEY = "load_aligner_always"
REALTIME_DEFAULTS_KEY = "realtime_defaults"
DEFAULT_REALTIME_SETTINGS = {
    "chunk_size_sec": 2.0,
    "unfixed_chunk_num": 2,
    "unfixed_token_num": 5,
}
ALIGNER_MODEL_SUPPORTED_LANGUAGES = (
    "Chinese",
    "English",
    "Cantonese",
    "French",
    "German",
    "Italian",
    "Japanese",
    "Korean",
    "Portuguese",
    "Russian",
    "Spanish",
)
ALIGNER_LANGUAGE_DEPENDENCIES = {
    "Japanese": ("nagisa", "Japanese timestamps require the nagisa tokenizer."),
    "Korean": (
        "soynlp",
        "Korean timestamps require the optional soynlp tokenizer, which is not bundled in this image.",
    ),
}


def aligner_language_capabilities() -> tuple[list[str], dict[str, str]]:
    unavailable: dict[str, str] = {}
    for language, (module, reason) in ALIGNER_LANGUAGE_DEPENDENCIES.items():
        try:
            installed = importlib.util.find_spec(module) is not None
        except (ImportError, ValueError):
            installed = False
        if not installed:
            unavailable[language] = reason
    available = [language for language in ALIGNER_MODEL_SUPPORTED_LANGUAGES if language not in unavailable]
    return available, unavailable


class AlignerUnavailableError(RuntimeError):
    pass


class RuntimeSettingsStore:
    """Persist the small set of operator-controlled runtime preferences."""

    def __init__(self, path: str | Path = DEFAULT_SETTINGS_PATH) -> None:
        self.path = Path(path)
        self._lock = threading.RLock()

    def _read_unlocked(self) -> dict[str, Any]:
        if not self.path.is_file():
            return {}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            log_startup(f"runtime settings ignored because {self.path} could not be read: {exc}")
            return {}
        if not isinstance(payload, dict):
            log_startup(f"runtime settings ignored because {self.path} is not a JSON object")
            return {}
        return payload

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._read_unlocked())

    def load_aligner_always(self, *, default: bool = False) -> bool:
        value = self.snapshot().get(LOAD_ALWAYS_KEY, default)
        return value if isinstance(value, bool) else default

    def realtime_defaults(self) -> dict[str, float | int]:
        values = self.snapshot().get(REALTIME_DEFAULTS_KEY, {})
        if not isinstance(values, dict):
            return dict(DEFAULT_REALTIME_SETTINGS)
        defaults = dict(DEFAULT_REALTIME_SETTINGS)
        chunk_size = values.get("chunk_size_sec")
        unfixed_chunks = values.get("unfixed_chunk_num")
        unfixed_tokens = values.get("unfixed_token_num")
        if isinstance(chunk_size, (int, float)) and not isinstance(chunk_size, bool) and 0.5 <= chunk_size <= 5:
            defaults["chunk_size_sec"] = float(chunk_size)
        if isinstance(unfixed_chunks, int) and not isinstance(unfixed_chunks, bool) and 0 <= unfixed_chunks <= 6:
            defaults["unfixed_chunk_num"] = unfixed_chunks
        if isinstance(unfixed_tokens, int) and not isinstance(unfixed_tokens, bool) and 0 <= unfixed_tokens <= 20:
            defaults["unfixed_token_num"] = unfixed_tokens
        return defaults

    def set_realtime_defaults(self, values: dict[str, Any]) -> dict[str, float | int]:
        chunk_size = values.get("chunk_size_sec")
        unfixed_chunks = values.get("unfixed_chunk_num")
        unfixed_tokens = values.get("unfixed_token_num")
        if not isinstance(chunk_size, (int, float)) or isinstance(chunk_size, bool) or not 0.5 <= chunk_size <= 5:
            raise ValueError("chunk_size_sec must be between 0.5 and 5 seconds.")
        if not isinstance(unfixed_chunks, int) or isinstance(unfixed_chunks, bool) or not 0 <= unfixed_chunks <= 6:
            raise ValueError("unfixed_chunk_num must be an integer between 0 and 6.")
        if not isinstance(unfixed_tokens, int) or isinstance(unfixed_tokens, bool) or not 0 <= unfixed_tokens <= 20:
            raise ValueError("unfixed_token_num must be an integer between 0 and 20.")
        normalized = {
            "chunk_size_sec": float(chunk_size),
            "unfixed_chunk_num": unfixed_chunks,
            "unfixed_token_num": unfixed_tokens,
        }
        self._update(REALTIME_DEFAULTS_KEY, normalized)
        return normalized

    def _update(self, key: str, value: Any) -> None:
        with self._lock:
            payload = self._read_unlocked()
            payload[key] = value
            self.path.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{self.path.name}.",
                suffix=".tmp",
                dir=self.path.parent,
            )
            temporary_path = Path(temporary_name)
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
                    json.dump(payload, handle, indent=2, sort_keys=True)
                    handle.write("\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary_path, self.path)
            finally:
                temporary_path.unlink(missing_ok=True)

    def set_load_aligner_always(self, enabled: bool) -> None:
        self._update(LOAD_ALWAYS_KEY, bool(enabled))


class AlignerRuntime:
    """Own lazy loading, residency, and persisted startup policy for one aligner."""

    def __init__(
        self,
        *,
        asr: ASRRuntime,
        checkpoint: str | None,
        model_kwargs: dict[str, Any] | None,
        settings: RuntimeSettingsStore,
        default_load_always: bool = False,
        loader: Callable[..., Any] | None = None,
        warmup: Callable[[Any], None] | None = None,
    ) -> None:
        self.asr = asr
        self.checkpoint = (checkpoint or "").strip() or None
        self.model_kwargs = dict(model_kwargs or {})
        self.settings = settings
        self._loader = loader
        self._warmup = warmup
        self._state_lock = threading.RLock()
        self._operation_lock = threading.Lock()
        self._load_always = settings.load_aligner_always(default=default_load_always)
        self._status = "loaded" if getattr(asr, "forced_aligner", None) is not None else "unloaded"
        if self.checkpoint is None:
            self._status = "unavailable"
        self._last_error: str | None = None

    @property
    def load_always(self) -> bool:
        with self._state_lock:
            return self._load_always

    @property
    def loaded(self) -> bool:
        return getattr(self.asr, "forced_aligner", None) is not None

    @property
    def configured(self) -> bool:
        return self.checkpoint is not None

    def snapshot(self) -> dict[str, Any]:
        available_languages, unavailable_languages = aligner_language_capabilities()
        with self._state_lock:
            return {
                "configured": self.configured,
                "loaded": self.loaded,
                "status": self._status,
                "load_always": self._load_always,
                "model": self.checkpoint,
                "last_error": self._last_error,
                "model_supported_languages": list(ALIGNER_MODEL_SUPPORTED_LANGUAGES),
                "available_languages": available_languages,
                "unavailable_languages": unavailable_languages,
            }

    def set_load_always(self, enabled: bool) -> None:
        if enabled and not self.configured:
            raise AlignerUnavailableError("No forced-aligner model is configured for this deployment.")
        self.settings.set_load_aligner_always(enabled)
        with self._state_lock:
            self._load_always = bool(enabled)

    def _resolve_loader(self) -> Callable[..., Any]:
        if self._loader is not None:
            return self._loader
        from qwen_asr.inference.qwen3_forced_aligner import Qwen3ForcedAligner

        return Qwen3ForcedAligner.from_pretrained

    def load(self, *, warm_up: bool = True) -> dict[str, Any]:
        if not self.configured:
            raise AlignerUnavailableError("No forced-aligner model is configured for this deployment.")
        with self._operation_lock:
            if self.loaded:
                with self._state_lock:
                    self._status = "loaded"
                    self._last_error = None
                return self.snapshot()

            with self._state_lock:
                self._status = "loading"
                self._last_error = None
            try:
                log_startup(f"load forced aligner on demand: {self.checkpoint}")
                aligner = self._resolve_loader()(self.checkpoint, **self.model_kwargs)
                if warm_up and self._warmup is not None:
                    self._warmup(aligner)
                self.asr.forced_aligner = aligner
            except Exception as exc:
                log_startup(
                    "forced aligner load failed:\n"
                    f"{traceback.format_exc().rstrip()}"
                )
                with self._state_lock:
                    self._status = "error"
                    self._last_error = f"{type(exc).__name__}: {exc}"
                gc.collect()
                self._empty_cuda_cache()
                raise

            with self._state_lock:
                self._status = "loaded"
                self._last_error = None
            return self.snapshot()

    def unload(self) -> dict[str, Any]:
        with self._operation_lock:
            with self._state_lock:
                self._status = "unloading" if self.loaded else "unloaded"
                self._last_error = None
            aligner = getattr(self.asr, "forced_aligner", None)
            self.asr.forced_aligner = None
            if aligner is not None:
                del aligner
                gc.collect()
                self._empty_cuda_cache()
            with self._state_lock:
                self._status = "unloaded" if self.configured else "unavailable"
            return self.snapshot()

    @staticmethod
    def _empty_cuda_cache() -> None:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as exc:
            log_startup(f"CUDA cache cleanup after aligner release was skipped: {exc}")
