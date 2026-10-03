from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from qwen_asr.server.contracts import ASRRuntime
from qwen_asr.startup_logging import StartupTimer, log_startup


DEFAULT_WARMUP_TEXT = (
    "I tried to make a cup of tea, but the kettle said, 'I'll put you on the list.' "
    "Very British, very serious."
)
ALIGNER_TOKENIZER_WARMUP_TEXTS = {
    "Chinese": "这是一次对齐预热。",
    "English": "This is an alignment warmup.",
    "Cantonese": "呢個係對齊預熱。",
    "French": "Ceci est un préchauffage d'alignement.",
    "German": "Dies ist ein Aufwärmlauf für die Ausrichtung.",
    "Italian": "Questo è un riscaldamento di allineamento.",
    "Japanese": "これはアラインメントのウォームアップです。",
    "Portuguese": "Este é um aquecimento de alinhamento.",
    "Russian": "Это прогрев выравнивания.",
    "Spanish": "Este es un calentamiento de alineación.",
}


def _enabled(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y"}


def run_startup_warmup(asr: ASRRuntime) -> None:
    if not _enabled(os.getenv("QWEN_ASR_STARTUP_WARMUP"), default=False):
        return

    warmup_tokens = int(
        os.getenv(
            "QWEN_ASR_STARTUP_WARMUP_TOKENS",
            os.getenv("QWEN_ASR_MAX_NEW_TOKENS", "512"),
        )
    )
    warmup_iterations = max(1, int(os.getenv("QWEN_ASR_STARTUP_WARMUP_ITERATIONS", "3")))
    default_audio = Path(__file__).resolve().parents[2] / "testbench/assets/english/random/01.mp3"
    configured_audio = os.getenv("QWEN_ASR_STARTUP_WARMUP_AUDIO")
    warmup_audio = configured_audio or (str(default_audio) if default_audio.is_file() else None)
    warmup_text = os.getenv("QWEN_ASR_STARTUP_WARMUP_TEXT", DEFAULT_WARMUP_TEXT)

    if warmup_audio is None:
        log_startup("startup warmup fixture unavailable; using synthetic audio")

    with StartupTimer(
        f"startup ASR warmup iterations={warmup_iterations} max_new_tokens={warmup_tokens}"
    ):
        asr.warm_up(
            max_new_tokens=warmup_tokens,
            iterations=warmup_iterations,
            audio=warmup_audio,
            aligner_text=warmup_text,
        )


def run_aligner_warmup(aligner: Any) -> None:
    """Compile and initialize a lazily loaded aligner before it is advertised ready."""
    default_audio = Path(__file__).resolve().parents[2] / "testbench/assets/english/random/01.mp3"
    configured_audio = os.getenv("QWEN_ASR_STARTUP_WARMUP_AUDIO")
    warmup_audio = configured_audio or (str(default_audio) if default_audio.is_file() else None)
    warmup_text = os.getenv("QWEN_ASR_STARTUP_WARMUP_TEXT", DEFAULT_WARMUP_TEXT)
    if warmup_audio is None:
        import numpy as np

        warmup_audio = (np.zeros((8_000,), dtype=np.float32), 16_000)

    split_words = getattr(getattr(aligner, "processor", None), "split_words_for_alignment", None)
    if callable(split_words):
        with StartupTimer("forced aligner language tokenizer warmup"):
            for language, text in ALIGNER_TOKENIZER_WARMUP_TEXTS.items():
                split_words(text, language)

    with StartupTimer("forced aligner on-demand warmup"):
        aligner.warm_up(audio=warmup_audio, text=warmup_text, language="English")
