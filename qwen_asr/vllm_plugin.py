"""vLLM plugin hooks for Qwen3-ASR."""

from __future__ import annotations

from qwen_asr.startup_logging import log_startup


def register() -> None:
    """Register Qwen3-ASR's vLLM model in every vLLM process."""
    from vllm import ModelRegistry

    ModelRegistry.register_model(
        "Qwen3ASRForConditionalGeneration",
        "qwen_asr.core.vllm_backend:Qwen3ASRForConditionalGeneration",
    )
    log_startup("registered custom vLLM Qwen3-ASR model plugin")
