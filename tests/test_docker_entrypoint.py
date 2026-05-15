from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from qwen_asr.docker_entrypoint import _backend_kwargs


BASE_ENV = {
    "QWEN_ASR_MAX_MODEL_LEN": "4096",
    "QWEN_ASR_GPU_MEMORY_UTILIZATION": "0.38",
    "QWEN_ASR_MAX_INFERENCE_BATCH_SIZE": "2",
    "QWEN_ASR_MAX_NEW_TOKENS": "512",
    "QWEN_ASR_GENERATION_CONFIG": "vllm",
    "QWEN_ASR_LOAD_FORMAT": "safetensors",
    "QWEN_ASR_KV_CACHE_DTYPE": "auto",
    "QWEN_ASR_CALCULATE_KV_SCALES": "0",
    "QWEN_ASR_ENFORCE_EAGER": "0",
}


class DockerEntrypointProfileTests(unittest.TestCase):
    def test_balanced_profile_supplies_bounded_piecewise_defaults(self):
        with patch.dict(os.environ, {**BASE_ENV, "QWEN_ASR_PERFORMANCE_PROFILE": "balanced"}, clear=True):
            kwargs = _backend_kwargs()

        self.assertEqual(kwargs["compilation_config"]["cudagraph_mode"], "PIECEWISE")
        self.assertEqual(kwargs["compilation_config"]["cudagraph_capture_sizes"], [1, 2])
        self.assertEqual(kwargs["compilation_config"]["max_cudagraph_capture_size"], 2)

    def test_throughput_profile_omits_graph_overrides_even_when_low_level_env_is_set(self):
        env = {
            **BASE_ENV,
            "QWEN_ASR_PERFORMANCE_PROFILE": "throughput",
            "QWEN_ASR_CUDAGRAPH_MODE": "PIECEWISE",
            "QWEN_ASR_CUDAGRAPH_CAPTURE_SIZES": "1,2",
            "QWEN_ASR_MAX_CUDAGRAPH_CAPTURE_SIZE": "2",
        }
        with patch.dict(os.environ, env, clear=True):
            kwargs = _backend_kwargs()

        self.assertNotIn("compilation_config", kwargs)

    def test_custom_profile_respects_low_level_graph_env(self):
        env = {
            **BASE_ENV,
            "QWEN_ASR_PERFORMANCE_PROFILE": "custom",
            "QWEN_ASR_CUDAGRAPH_MODE": "PIECEWISE",
            "QWEN_ASR_CUDAGRAPH_CAPTURE_SIZES": "1,2,4",
            "QWEN_ASR_MAX_CUDAGRAPH_CAPTURE_SIZE": "4",
        }
        with patch.dict(os.environ, env, clear=True):
            kwargs = _backend_kwargs()

        self.assertEqual(kwargs["compilation_config"]["cudagraph_capture_sizes"], [1, 2, 4])
        self.assertEqual(kwargs["compilation_config"]["max_cudagraph_capture_size"], 4)


if __name__ == "__main__":
    unittest.main()
