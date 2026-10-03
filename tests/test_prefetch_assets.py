from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from qwen_asr import prefetch_assets


class PrefetchAssetsTests(unittest.TestCase):
    def test_requested_model_ids_follow_actual_build_configuration(self) -> None:
        environment = {
            "QWEN_ASR_PREFETCH_MODELS": (
                "qwen3-asr-0.6b-hf;Qwen/Qwen3-ASR-1.7B-hf;qwen3-asr-0.6b-hf"
            ),
            "QWEN_ASR_PREFETCH_ALIGNER": "0",
        }
        with patch.dict(os.environ, environment, clear=False):
            self.assertEqual(
                prefetch_assets.requested_model_ids(),
                ["Qwen/Qwen3-ASR-0.6B-hf", "Qwen/Qwen3-ASR-1.7B-hf"],
            )

    def test_aligner_prefetch_is_not_filtered_by_asr_allow_patterns(self) -> None:
        calls: list[dict[str, object]] = []

        def fake_snapshot_download(**kwargs: object) -> str:
            calls.append(kwargs)
            return "/tmp/model"

        environment = {
            "QWEN_ASR_PREFETCH_MODELS": "Qwen/Qwen3-ASR-0.6B-hf",
            "QWEN_ASR_PREFETCH_ALLOW_PATTERNS": "*.json;*.safetensors",
            "QWEN_ASR_PREFETCH_ALIGNER": "1",
            "QWEN_ASR_ALIGNER_MODEL": "Qwen/Test-Aligner",
        }
        with (
            patch.dict(os.environ, environment, clear=False),
            patch.object(prefetch_assets, "snapshot_download", side_effect=fake_snapshot_download),
        ):
            prefetch_assets.main()

        self.assertEqual(
            calls,
            [
                {
                    "repo_id": "Qwen/Qwen3-ASR-0.6B-hf",
                    "allow_patterns": ["*.json", "*.safetensors"],
                },
                {"repo_id": "Qwen/Test-Aligner"},
            ],
        )


if __name__ == "__main__":
    unittest.main()
