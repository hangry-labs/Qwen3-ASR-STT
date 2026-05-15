from __future__ import annotations

import json
import os
from typing import Any

from qwen_asr.server.openai_api import run_server as run_openai_api
from qwen_asr.server.ui import run_server as run_ui


def _enabled(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y"}


def _json_arg(value: str | None) -> list[str]:
    if not value or not value.strip():
        return []
    json.loads(value)
    return [value]


def _int_env(name: str) -> int | None:
    value = os.getenv(name)
    if not value or not value.strip():
        return None
    return int(value)


def _float_env(name: str) -> float | None:
    value = os.getenv(name)
    if not value or not value.strip():
        return None
    return float(value)


def _bool_env(name: str) -> bool | None:
    value = os.getenv(name)
    if value is None or not value.strip():
        return None
    return _enabled(value)


def _str_env(name: str) -> str | None:
    value = os.getenv(name)
    if not value or not value.strip():
        return None
    return value.strip()


def _backend_kwargs() -> dict[str, Any]:
    kwargs: dict[str, Any] = {}

    max_model_len = _int_env("QWEN_ASR_MAX_MODEL_LEN")
    if max_model_len is not None:
        kwargs["max_model_len"] = max_model_len

    gpu_memory_utilization = _float_env("QWEN_ASR_GPU_MEMORY_UTILIZATION")
    if gpu_memory_utilization is not None:
        kwargs["gpu_memory_utilization"] = gpu_memory_utilization

    max_inference_batch_size = _int_env("QWEN_ASR_MAX_INFERENCE_BATCH_SIZE")
    if max_inference_batch_size is not None:
        kwargs["max_inference_batch_size"] = max_inference_batch_size

    max_new_tokens = _int_env("QWEN_ASR_MAX_NEW_TOKENS")
    if max_new_tokens is not None:
        kwargs["max_new_tokens"] = max_new_tokens

    generation_config = _str_env("QWEN_ASR_GENERATION_CONFIG")
    if generation_config is not None:
        kwargs["generation_config"] = generation_config

    load_format = _str_env("QWEN_ASR_LOAD_FORMAT")
    if load_format is not None:
        kwargs["load_format"] = load_format

    kv_cache_dtype = _str_env("QWEN_ASR_KV_CACHE_DTYPE")
    if kv_cache_dtype is not None:
        kwargs["kv_cache_dtype"] = kv_cache_dtype

    calculate_kv_scales = _bool_env("QWEN_ASR_CALCULATE_KV_SCALES")
    if calculate_kv_scales is not None:
        kwargs["calculate_kv_scales"] = calculate_kv_scales

    raw_kwargs = os.getenv("QWEN_ASR_BACKEND_KWARGS")
    if raw_kwargs and raw_kwargs.strip():
        extra = json.loads(raw_kwargs)
        if not isinstance(extra, dict):
            raise ValueError("QWEN_ASR_BACKEND_KWARGS must be a JSON object")
        kwargs.update(extra)

    return kwargs


def main() -> int:
    host = os.getenv("HOST", "0.0.0.0")
    port = os.getenv("PORT", "8000")
    app_mode = os.getenv("QWEN_ASR_APP", "demo").strip().lower()
    backend = os.getenv("QWEN_ASR_BACKEND", "vllm")
    asr_model = os.getenv("QWEN_ASR_MODEL", "Qwen/Qwen3-ASR-1.7B")
    aligner_model = os.getenv("QWEN_ASR_ALIGNER_MODEL", "")
    cuda_visible_devices = os.getenv("CUDA_VISIBLE_DEVICES", "0")
    concurrency = os.getenv("QWEN_ASR_CONCURRENCY", "16")
    aligner_kwargs = os.getenv("QWEN_ASR_ALIGNER_KWARGS")

    backend_kwargs = _backend_kwargs()
    aligner_kwargs_value = _json_arg(aligner_kwargs)
    resolved_aligner_kwargs = json.loads(aligner_kwargs_value[0]) if aligner_kwargs_value else None

    print(
        f"Starting Qwen3-ASR {app_mode}: model={asr_model} backend={backend} "
        f"host={host} port={port} concurrency={concurrency} backend_kwargs={backend_kwargs}",
        flush=True,
    )
    if app_mode in {"api", "openai", "server"}:
        run_openai_api(
            asr_checkpoint=asr_model,
            backend=backend,
            backend_kwargs=backend_kwargs,
            cuda_visible_devices=cuda_visible_devices,
            host=host,
            port=int(port),
            concurrency=int(concurrency),
        )
    else:
        run_ui(
            asr_checkpoint=asr_model,
            aligner_checkpoint=aligner_model
            if _enabled(os.getenv("QWEN_ASR_ENABLE_ALIGNER"), default=False) and aligner_model
            else None,
            backend=backend,
            backend_kwargs=backend_kwargs,
            aligner_kwargs=resolved_aligner_kwargs,
            cuda_visible_devices=cuda_visible_devices,
            host=host,
            port=int(port),
            concurrency=int(concurrency),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
