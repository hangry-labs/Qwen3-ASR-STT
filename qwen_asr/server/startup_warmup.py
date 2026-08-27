from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from qwen_asr.startup_logging import StartupTimer, log_startup


DEFAULT_WARMUP_TEXT = (
    "I tried to make a cup of tea, but the kettle said, 'I'll put you on the list.' "
    "Very British, very serious."
)


def _enabled(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y"}


def run_startup_warmup(asr: Any) -> None:
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
