from __future__ import annotations

import importlib.util
import os
import sys
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import patch


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


class DockerEntrypointTests(unittest.TestCase):
    def test_backend_kwargs_default_to_vllm_gpu_profile(self):
        with patch.dict(os.environ, {}, clear=True):
            kwargs = ENTRYPOINT._backend_kwargs("vllm")

        self.assertEqual(
            kwargs,
            {
                "max_model_len": 2048,
                "gpu_memory_utilization": 0.25,
                "max_inference_batch_size": 2,
                "max_new_tokens": 512,
                "max_num_batched_tokens": 2048,
                "max_num_seqs": 2,
                "dtype": "bfloat16",
                "generation_config": "vllm",
            },
        )

    def test_transformers_kwargs_allow_loader_overrides(self):
        env = {
            "QWEN_ASR_MAX_INFERENCE_BATCH_SIZE": "1",
            "QWEN_ASR_MAX_NEW_TOKENS": "256",
            "QWEN_ASR_TRANSFORMERS_DTYPE": "float16",
            "QWEN_ASR_TRANSFORMERS_DEVICE_MAP": "cuda:1",
            "QWEN_ASR_TORCH_COMPILE": "0",
            "QWEN_ASR_TORCH_COMPILE_BACKEND": "eager",
            "QWEN_ASR_TORCH_COMPILE_MODE": "reduce-overhead",
            "QWEN_ASR_GENERATION_CACHE_IMPLEMENTATION": "static",
            "QWEN_ASR_TORCH_COMPILE_FULLGRAPH": "1",
            "QWEN_ASR_TORCH_COMPILE_DYNAMIC": "1",
            "QWEN_ASR_MODEL_KWARGS": '{"device_map":"cpu","low_cpu_mem_usage":true}',
        }
        with patch.dict(os.environ, env, clear=True):
            kwargs = ENTRYPOINT._backend_kwargs("transformers")

        self.assertEqual(kwargs["max_inference_batch_size"], 1)
        self.assertEqual(kwargs["max_new_tokens"], 256)
        self.assertEqual(kwargs["dtype"], "float16")
        self.assertEqual(kwargs["device_map"], "cpu")
        self.assertTrue(kwargs["low_cpu_mem_usage"])
        self.assertFalse(kwargs["torch_compile"])
        self.assertEqual(kwargs["torch_compile_backend"], "eager")
        self.assertEqual(kwargs["torch_compile_mode"], "reduce-overhead")
        self.assertEqual(kwargs["generation_cache_implementation"], "static")
        self.assertTrue(kwargs["torch_compile_fullgraph"])
        self.assertTrue(kwargs["torch_compile_dynamic"])

    def test_vllm_kwargs_allow_engine_overrides(self):
        env = {
            "QWEN_ASR_GPU_MEMORY_UTILIZATION": "0.4",
            "QWEN_ASR_MAX_NUM_SEQS": "4",
            "QWEN_ASR_ENFORCE_EAGER": "1",
            "QWEN_ASR_BACKEND_KWARGS": '{"enable_prefix_caching":true}',
        }
        with patch.dict(os.environ, env, clear=True):
            kwargs = ENTRYPOINT._backend_kwargs("vllm")

        self.assertEqual(kwargs["gpu_memory_utilization"], 0.4)
        self.assertEqual(kwargs["max_num_seqs"], 4)
        self.assertTrue(kwargs["enforce_eager"])
        self.assertTrue(kwargs["enable_prefix_caching"])

    @staticmethod
    def _fake_server_modules(captured: dict):
        fake_package = ModuleType("qwen_asr")
        fake_package.__path__ = []
        fake_server = ModuleType("qwen_asr.server")
        fake_server.__path__ = []
        fake_app = ModuleType("qwen_asr.server.app")
        fake_app.run_server = lambda **kwargs: captured.update(kwargs)
        return {
            "qwen_asr": fake_package,
            "qwen_asr.server": fake_server,
            "qwen_asr.server.app": fake_app,
        }

    def test_main_uses_vllm_06b_runtime_defaults(self):
        captured = {}
        with patch.dict(os.environ, {}, clear=True), patch.dict(
            sys.modules,
            self._fake_server_modules(captured),
        ):
            self.assertEqual(ENTRYPOINT.main(), 0)

        self.assertEqual(captured["asr_checkpoint"], "Qwen/Qwen3-ASR-0.6B-hf")
        self.assertEqual(captured["backend"], "vllm")
        self.assertEqual(captured["concurrency"], 2)
        self.assertEqual(captured["model_kwargs"]["max_model_len"], 2048)
        self.assertEqual(captured["model_kwargs"]["gpu_memory_utilization"], 0.25)
        self.assertEqual(captured["model_kwargs"]["generation_config"], "vllm")

    def test_main_starts_combined_server_with_native_aligner(self):
        captured = {}
        env = {
            "QWEN_ASR_APP": "api",
            "QWEN_ASR_MODEL": "Qwen/Qwen3-ASR-1.7B-hf",
            "QWEN_ASR_ALIGNER_MODEL": "Qwen/Qwen3-ForcedAligner-0.6B-hf",
            "QWEN_ASR_ENABLE_ALIGNER": "1",
            "QWEN_ASR_CONCURRENCY": "1",
            "CUDA_VISIBLE_DEVICES": "0",
            "HOST": "127.0.0.1",
            "PORT": "8123",
        }
        with patch.dict(os.environ, env, clear=True), patch.dict(
            sys.modules,
            self._fake_server_modules(captured),
        ):
            self.assertEqual(ENTRYPOINT.main(), 0)

        self.assertEqual(captured["asr_checkpoint"], "Qwen/Qwen3-ASR-1.7B-hf")
        self.assertEqual(captured["aligner_checkpoint"], "Qwen/Qwen3-ForcedAligner-0.6B-hf")
        self.assertEqual(captured["aligner_kwargs"]["dtype"], "bfloat16")
        self.assertEqual(captured["aligner_kwargs"]["device_map"], "cuda:0")
        self.assertTrue(captured["aligner_kwargs"]["torch_compile"])
        self.assertEqual(captured["host"], "127.0.0.1")
        self.assertEqual(captured["port"], 8123)

if __name__ == "__main__":
    unittest.main()
