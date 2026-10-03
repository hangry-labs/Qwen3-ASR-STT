from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from qwen_asr.server.startup_warmup import ALIGNER_TOKENIZER_WARMUP_TEXTS, run_aligner_warmup


class _Processor:
    def __init__(self) -> None:
        self.languages: list[str] = []

    def split_words_for_alignment(self, text: str, language: str) -> list[str]:
        self.languages.append(language)
        return [text]


class _Aligner:
    def __init__(self) -> None:
        self.processor = _Processor()
        self.warmups: list[dict] = []

    def warm_up(self, **kwargs) -> None:
        self.warmups.append(kwargs)


class StartupWarmupTests(unittest.TestCase):
    def test_aligner_warmup_runs_even_when_asr_startup_warmup_is_disabled(self) -> None:
        aligner = _Aligner()

        with patch.dict(os.environ, {"QWEN_ASR_STARTUP_WARMUP": "0"}, clear=True):
            run_aligner_warmup(aligner)

        self.assertEqual(aligner.processor.languages, list(ALIGNER_TOKENIZER_WARMUP_TEXTS))
        self.assertEqual(len(aligner.warmups), 1)
        self.assertEqual(aligner.warmups[0]["language"], "English")


if __name__ == "__main__":
    unittest.main()
