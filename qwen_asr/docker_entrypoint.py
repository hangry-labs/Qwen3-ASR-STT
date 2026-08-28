from __future__ import annotations

import json
import os
from typing import Any

from qwen_asr.startup_logging import StartupTimer, log_startup


DEFAULT_ASR_MODEL = "Qwen/Qwen3-ASR-0.6B-hf"
DEFAULT_CONCURRENCY = "2"
DEFAULT_MAX_MODEL_LEN = 2048
DEFAULT_GPU_MEMORY_UTILIZATION = 0.25
DEFAULT_MAX_INFERENCE_BATCH_SIZE = 2
DEFAULT_MAX_NEW_TOKENS = 512
DEFAULT_MAX_NUM_BATCHED_TOKENS = 2048
DEFAULT_MAX_NUM_SEQS = 2


def _enabled(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y"}


def _json_arg(value: str | None) -> list[str]:
    if not value or not value.strip():
        return []
    json.loads(value)
    return [value]


def _int_env(name: str, default: int | None = None) -> int | None:
    value = os.getenv(name)
    if not value or not value.strip():
        return default
    return int(value)


def _float_env(name: str, default: float | None = None) -> float | None:
    value = os.getenv(name)
    if not value or not value.strip():
        return default
    return float(value)


def _str_env(name: str) -> str | None:
    value = os.getenv(name)
    if not value or not value.strip():
        return None
    return value.strip()


def _optional_bool_env(name: str) -> bool | None:
    value = _str_env(name)
    if value is None:
        return None
    return _enabled(value)


def _transformers_kwargs() -> dict[str, Any]:
    kwargs: dict[str, Any] = {}

    max_inference_batch_size = _int_env("QWEN_ASR_MAX_INFERENCE_BATCH_SIZE", DEFAULT_MAX_INFERENCE_BATCH_SIZE)
    if max_inference_batch_size is not None:
        kwargs["max_inference_batch_size"] = max_inference_batch_size

    max_new_tokens = _int_env("QWEN_ASR_MAX_NEW_TOKENS", DEFAULT_MAX_NEW_TOKENS)
    if max_new_tokens is not None:
        kwargs["max_new_tokens"] = max_new_tokens

    kwargs["dtype"] = _str_env("QWEN_ASR_TRANSFORMERS_DTYPE") or "bfloat16"
    kwargs["device_map"] = _str_env("QWEN_ASR_TRANSFORMERS_DEVICE_MAP") or "cuda:0"
    kwargs["torch_compile"] = _enabled(os.getenv("QWEN_ASR_TORCH_COMPILE"), default=True)
    kwargs["torch_compile_backend"] = _str_env("QWEN_ASR_TORCH_COMPILE_BACKEND") or "inductor"
    kwargs["torch_compile_mode"] = _str_env("QWEN_ASR_TORCH_COMPILE_MODE") or "default"
    kwargs["torch_compile_fullgraph"] = _enabled(
        os.getenv("QWEN_ASR_TORCH_COMPILE_FULLGRAPH"), default=False
    )
    generation_cache = _str_env("QWEN_ASR_GENERATION_CACHE_IMPLEMENTATION")
    if generation_cache is not None:
        kwargs["generation_cache_implementation"] = generation_cache
    torch_compile_dynamic = _optional_bool_env("QWEN_ASR_TORCH_COMPILE_DYNAMIC")
    if torch_compile_dynamic is not None:
        kwargs["torch_compile_dynamic"] = torch_compile_dynamic

    raw_kwargs = os.getenv("QWEN_ASR_MODEL_KWARGS")
    if raw_kwargs and raw_kwargs.strip():
        extra = json.loads(raw_kwargs)
        if not isinstance(extra, dict):
            raise ValueError("QWEN_ASR_MODEL_KWARGS must be a JSON object")
        kwargs.update(extra)

    return kwargs


def _vllm_kwargs() -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    values = {
        "max_model_len": _int_env("QWEN_ASR_MAX_MODEL_LEN", DEFAULT_MAX_MODEL_LEN),
        "gpu_memory_utilization": _float_env(
            "QWEN_ASR_GPU_MEMORY_UTILIZATION", DEFAULT_GPU_MEMORY_UTILIZATION
        ),
        "max_inference_batch_size": _int_env(
            "QWEN_ASR_MAX_INFERENCE_BATCH_SIZE", DEFAULT_MAX_INFERENCE_BATCH_SIZE
        ),
        "max_new_tokens": _int_env("QWEN_ASR_MAX_NEW_TOKENS", DEFAULT_MAX_NEW_TOKENS),
        "max_num_batched_tokens": _int_env(
            "QWEN_ASR_MAX_NUM_BATCHED_TOKENS", DEFAULT_MAX_NUM_BATCHED_TOKENS
        ),
        "max_num_seqs": _int_env("QWEN_ASR_MAX_NUM_SEQS", DEFAULT_MAX_NUM_SEQS),
    }
    kwargs.update({key: value for key, value in values.items() if value is not None})
    kwargs["dtype"] = _str_env("QWEN_ASR_VLLM_DTYPE") or "bfloat16"
    kwargs["generation_config"] = _str_env("QWEN_ASR_GENERATION_CONFIG") or "vllm"

    optional_strings = {
        "load_format": "QWEN_ASR_LOAD_FORMAT",
        "kv_cache_dtype": "QWEN_ASR_KV_CACHE_DTYPE",
    }
    for key, env_name in optional_strings.items():
        value = _str_env(env_name)
        if value is not None:
            kwargs[key] = value

    enforce_eager = _optional_bool_env("QWEN_ASR_ENFORCE_EAGER")
    if enforce_eager is not None:
        kwargs["enforce_eager"] = enforce_eager

    raw_kwargs = os.getenv("QWEN_ASR_BACKEND_KWARGS") or os.getenv("QWEN_ASR_MODEL_KWARGS")
    if raw_kwargs and raw_kwargs.strip():
        extra = json.loads(raw_kwargs)
        if not isinstance(extra, dict):
            raise ValueError("QWEN_ASR_BACKEND_KWARGS must be a JSON object")
        kwargs.update(extra)
    return kwargs


def _backend_kwargs(backend: str) -> dict[str, Any]:
    if backend == "vllm":
        return _vllm_kwargs()
    if backend == "transformers":
        return _transformers_kwargs()
    raise ValueError("QWEN_ASR_BACKEND must be one of: vllm, transformers")


def main() -> int:
    log_startup("docker entrypoint started")
    host = os.getenv("HOST", "0.0.0.0")
    port = os.getenv("PORT", "8000")
    ssl_certfile = _str_env("QWEN_ASR_SSL_CERTFILE") or _str_env("SSL_CERTFILE")
    ssl_keyfile = _str_env("QWEN_ASR_SSL_KEYFILE") or _str_env("SSL_KEYFILE")
    ssl_verify = _enabled(os.getenv("QWEN_ASR_SSL_VERIFY"), default=True)
    backend = (os.getenv("QWEN_ASR_BACKEND", "vllm") or "vllm").strip().lower()
    asr_model = os.getenv("QWEN_ASR_MODEL", DEFAULT_ASR_MODEL)
    aligner_model = os.getenv("QWEN_ASR_ALIGNER_MODEL", "")
    cuda_visible_devices = os.getenv("CUDA_VISIBLE_DEVICES", "0")
    concurrency = os.getenv("QWEN_ASR_CONCURRENCY", DEFAULT_CONCURRENCY)
    aligner_kwargs = os.getenv("QWEN_ASR_ALIGNER_KWARGS")

    model_kwargs = _backend_kwargs(backend)
    aligner_kwargs_value = _json_arg(aligner_kwargs)
    resolved_aligner_kwargs = {
        "dtype": _str_env("QWEN_ASR_TRANSFORMERS_DTYPE") or "bfloat16",
        "device_map": _str_env("QWEN_ASR_TRANSFORMERS_DEVICE_MAP") or "cuda:0",
        "torch_compile": _enabled(os.getenv("QWEN_ASR_TORCH_COMPILE"), default=True),
        "torch_compile_backend": _str_env("QWEN_ASR_TORCH_COMPILE_BACKEND") or "inductor",
        "torch_compile_mode": _str_env("QWEN_ASR_TORCH_COMPILE_MODE") or "default",
        "torch_compile_fullgraph": _enabled(
            os.getenv("QWEN_ASR_TORCH_COMPILE_FULLGRAPH"), default=False
        ),
    }
    torch_compile_dynamic = _optional_bool_env("QWEN_ASR_TORCH_COMPILE_DYNAMIC")
    if torch_compile_dynamic is not None:
        resolved_aligner_kwargs["torch_compile_dynamic"] = torch_compile_dynamic
    if aligner_kwargs_value:
        resolved_aligner_kwargs.update(json.loads(aligner_kwargs_value[0]))

    log_startup(
        f"configuration resolved: model={asr_model} backend={backend} "
        f"host={host} port={port} concurrency={concurrency} backend_kwargs={model_kwargs}"
    )
    with StartupTimer("import combined browser UI/OpenAI API server"):
        from qwen_asr.standalone_ui.server import run_server as run_ui

    run_ui(
        asr_checkpoint=asr_model,
        aligner_checkpoint=aligner_model
        if _enabled(os.getenv("QWEN_ASR_ENABLE_ALIGNER"), default=False) and aligner_model
        else None,
        backend=backend,
        model_kwargs=model_kwargs,
        aligner_kwargs=resolved_aligner_kwargs,
        cuda_visible_devices=cuda_visible_devices,
        host=host,
        port=int(port),
        ssl_certfile=ssl_certfile,
        ssl_keyfile=ssl_keyfile,
        ssl_verify=ssl_verify,
        concurrency=int(concurrency),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
