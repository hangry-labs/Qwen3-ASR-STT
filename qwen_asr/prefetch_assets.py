from __future__ import annotations

import os

from huggingface_hub import snapshot_download


DEFAULT_ASR_MODEL = "Qwen/Qwen3-ASR-0.6B"
DEFAULT_ALIGNER_MODEL = "Qwen/Qwen3-ForcedAligner-0.6B"
KNOWN_MODEL_CHOICES = {
    "qwen3-asr-1.7b": "Qwen/Qwen3-ASR-1.7B",
    "qwen3-asr-0.6b": "Qwen/Qwen3-ASR-0.6B",
    "qwen3-asr-1.7b-gguf": "ggml-org/Qwen3-ASR-1.7B-GGUF",
    "qwen3-asr-0.6b-q4-k-m": "OpenVoiceOS/qwen3-asr-0.6b-q4-k-m",
}


def _enabled(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y"}


def _split_models(value: str | None) -> list[str]:
    if not value:
        return []
    models = []
    for item in value.replace(";", ",").split(","):
        model = item.strip()
        if model:
            models.append(KNOWN_MODEL_CHOICES.get(model, model))
    return models


def _split_patterns(value: str | None) -> list[str] | None:
    if not value:
        return None
    patterns = []
    for item in value.replace(";", ",").split(","):
        pattern = item.strip()
        if pattern:
            patterns.append(pattern)
    return patterns or None


def main() -> None:
    models = _split_models(os.getenv("QWEN_ASR_PREFETCH_MODELS"))
    if not models:
        models = [os.getenv("QWEN_ASR_MODEL", DEFAULT_ASR_MODEL)]
    allow_patterns = _split_patterns(os.getenv("QWEN_ASR_PREFETCH_ALLOW_PATTERNS"))

    seen = set()
    for model in models:
        if model in seen:
            continue
        seen.add(model)
        print(f"Prefetching Qwen3-ASR model asset: {model}")
        snapshot_download(repo_id=model, allow_patterns=allow_patterns)

    if _enabled(os.getenv("QWEN_ASR_PREFETCH_ALIGNER"), default=True):
        aligner_model = os.getenv("QWEN_ASR_ALIGNER_MODEL", DEFAULT_ALIGNER_MODEL)
        print(f"Prefetching Qwen3 forced aligner: {aligner_model}")
        snapshot_download(repo_id=aligner_model)


if __name__ == "__main__":
    main()
