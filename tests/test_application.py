from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

import torch
from fastapi import FastAPI

from qwen_asr.server.application import _coerce_special_types, create_product_app, run_server


class ProductApplicationTests(unittest.TestCase):
    def test_create_product_app_composes_api_before_ui(self) -> None:
        api_app = FastAPI()
        product_app = FastAPI()
        asr = object()
        aligner_runtime = Mock()

        with (
            patch("qwen_asr.server.application.create_openai_app", return_value=api_app) as create_api,
            patch("qwen_asr.server.application.attach_ui", return_value=product_app) as attach_ui,
        ):
            result = create_product_app(
                asr=asr,
                model_name="Qwen/test-model",
                concurrency=2,
                aligner_runtime=aligner_runtime,
                trace_requests=True,
            )

        self.assertIs(result, product_app)
        self.assertIs(create_api.call_args.kwargs["asr"], asr)
        self.assertEqual(create_api.call_args.kwargs["model_name"], "Qwen/test-model")
        self.assertEqual(create_api.call_args.kwargs["concurrency"], 2)
        self.assertTrue(create_api.call_args.kwargs["trace_requests"])
        attach_ui.assert_called_once_with(api_app=api_app)

    def test_model_kwargs_coerce_torch_dtype_without_mutating_input(self) -> None:
        original = {"dtype": "bfloat16", "device_map": "cuda:0"}

        resolved = _coerce_special_types(original)

        self.assertIs(resolved["dtype"], torch.bfloat16)
        self.assertEqual(original["dtype"], "bfloat16")

    def test_server_requires_complete_tls_pair_before_loading_models(self) -> None:
        with self.assertRaisesRegex(ValueError, "Both SSL certfile and SSL keyfile"):
            run_server(
                asr_checkpoint="Qwen/test-model",
                aligner_checkpoint=None,
                load_aligner_at_startup=False,
                settings_path="settings.json",
                backend="vllm",
                model_kwargs={},
                aligner_kwargs={},
                cuda_visible_devices="0",
                host="127.0.0.1",
                port=8000,
                concurrency=1,
                trace_requests=False,
                ssl_certfile="cert.pem",
            )

    def test_persistent_aligner_is_warmed_before_serving(self) -> None:
        asr = Mock()
        aligner_runtime = Mock(load_always=True)
        app = FastAPI()
        with (
            patch("qwen_asr.server.application._load_asr_model", return_value=asr),
            patch("qwen_asr.server.application.RuntimeSettingsStore"),
            patch("qwen_asr.server.application.AlignerRuntime", return_value=aligner_runtime),
            patch("qwen_asr.server.application.create_product_app", return_value=app),
            patch("qwen_asr.server.application.uvicorn.run"),
        ):
            run_server(
                asr_checkpoint="Qwen/test-model",
                aligner_checkpoint="Qwen/test-aligner",
                load_aligner_at_startup=True,
                settings_path="settings.json",
                backend="vllm",
                model_kwargs={},
                aligner_kwargs={},
                cuda_visible_devices="0",
                host="127.0.0.1",
                port=8000,
                concurrency=1,
                trace_requests=False,
            )

        aligner_runtime.load.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
