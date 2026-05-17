from __future__ import annotations

import importlib.util
import os
import sys
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import patch


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


def _load_entrypoint_module():
    root = Path(__file__).resolve().parents[1]
    fake_package = ModuleType("qwen_asr")
    fake_package.__path__ = [str(root / "qwen_asr")]
    fake_startup_logging = ModuleType("qwen_asr.startup_logging")

    class StartupTimer:
        def __init__(self, *_args, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    fake_startup_logging.StartupTimer = StartupTimer
    fake_startup_logging.log_startup = lambda _message: None

    spec = importlib.util.spec_from_file_location(
        "_docker_entrypoint_under_test",
        root / "qwen_asr" / "docker_entrypoint.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load qwen_asr/docker_entrypoint.py")

    module = importlib.util.module_from_spec(spec)
    with patch.dict(
        sys.modules,
        {
            "qwen_asr": fake_package,
            "qwen_asr.startup_logging": fake_startup_logging,
        },
    ):
        spec.loader.exec_module(module)
    return module


ENTRYPOINT = _load_entrypoint_module()


class DockerEntrypointProfileTests(unittest.TestCase):
    def test_balanced_profile_supplies_bounded_piecewise_defaults(self):
        with patch.dict(os.environ, {**BASE_ENV, "QWEN_ASR_PERFORMANCE_PROFILE": "balanced"}, clear=True):
            kwargs = ENTRYPOINT._backend_kwargs()

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
            kwargs = ENTRYPOINT._backend_kwargs()

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
            kwargs = ENTRYPOINT._backend_kwargs()

        self.assertEqual(kwargs["compilation_config"]["cudagraph_capture_sizes"], [1, 2, 4])
        self.assertEqual(kwargs["compilation_config"]["max_cudagraph_capture_size"], 4)

    def test_main_ignores_legacy_api_mode_and_starts_combined_server(self):
        captured = {}
        fake_package = ModuleType("qwen_asr")
        fake_package.__path__ = []
        fake_server = ModuleType("qwen_asr.server")
        fake_server.__path__ = []
        fake_app = ModuleType("qwen_asr.server.app")

        def fake_run_server(**kwargs):
            captured.update(kwargs)

        fake_app.run_server = fake_run_server

        env = {
            "QWEN_ASR_APP": "api",
            "QWEN_ASR_BACKEND": "vllm",
            "QWEN_ASR_MODEL": "Qwen/Qwen3-ASR-0.6B",
            "QWEN_ASR_ALIGNER_MODEL": "Qwen/Qwen3-ForcedAligner-0.6B",
            "QWEN_ASR_ENABLE_ALIGNER": "1",
            "QWEN_ASR_CONCURRENCY": "2",
            "CUDA_VISIBLE_DEVICES": "1",
            "HOST": "127.0.0.1",
            "PORT": "8123",
        }

        with patch.dict(os.environ, env, clear=True), patch.dict(
            sys.modules,
            {
                "qwen_asr": fake_package,
                "qwen_asr.server": fake_server,
                "qwen_asr.server.app": fake_app,
            },
        ):
            self.assertEqual(ENTRYPOINT.main(), 0)

        self.assertEqual(captured["asr_checkpoint"], "Qwen/Qwen3-ASR-0.6B")
        self.assertEqual(captured["aligner_checkpoint"], "Qwen/Qwen3-ForcedAligner-0.6B")
        self.assertEqual(captured["host"], "127.0.0.1")
        self.assertEqual(captured["port"], 8123)
        self.assertEqual(captured["concurrency"], 2)


if __name__ == "__main__":
    unittest.main()
