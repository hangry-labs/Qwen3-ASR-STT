from __future__ import annotations

import unittest
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from qwen_asr.standalone_ui.server import _example_catalog, _read_version_file, create_app


class StandaloneUiTests(unittest.TestCase):
    @staticmethod
    def backend_app() -> FastAPI:
        backend = FastAPI()

        @backend.get("/health")
        async def health() -> dict[str, str]:
            return {"status": "ready"}

        return backend

    def test_example_catalog_exposes_one_existing_file_per_language(self) -> None:
        examples = _example_catalog()
        languages = [example["language"] for example in examples]

        self.assertGreaterEqual(len(examples), 30)
        self.assertEqual(len(languages), len(set(languages)))
        self.assertTrue(all(example["url"].startswith("/example-audio/") for example in examples))

    def test_static_application_and_health_are_available(self) -> None:
        with TestClient(create_app(api_app=self.backend_app())) as client:
            index = client.get("/")
            script = client.get("/static/app.js")
            health = client.get("/health")

        self.assertEqual(index.status_code, 200)
        self.assertIn("Qwen3-ASR-STT", index.text)
        self.assertIn("Qwen3-ASR-STT/tree/main/testbench", index.text)
        self.assertIn('class="qwen-highlight"', index.text)
        self.assertIn('data-text="Qwen3-ASR-STT"', index.text)
        self.assertIn('class="brand-backdrop"', index.text)
        self.assertIn('id="timestamp-support"', index.text)
        self.assertIn('id="model-name"', index.text)
        self.assertIn('id="chunk-size-help" role="tooltip"', index.text)
        self.assertIn('id="unfixed-chunks-help" role="tooltip"', index.text)
        self.assertIn('id="unfixed-tokens-help" role="tooltip"', index.text)
        self.assertNotIn("<span>Temperature</span>", index.text)
        self.assertIn(f"UI v{_read_version_file()}", index.text)
        self.assertNotIn("{{UI_VERSION}}", index.text)
        self.assertEqual(script.status_code, 200)
        self.assertIn("RealtimeRecorder", script.text)
        self.assertEqual(health.json(), {"status": "ready"})

    def test_development_assets_disable_browser_caching(self) -> None:
        with patch.dict("os.environ", {"QWEN_ASR_UI_DEV": "1"}):
            with TestClient(create_app(api_app=self.backend_app())) as client:
                index = client.get("/")
                stylesheet = client.get("/static/styles.css")
                logo = client.get("/brand/logo_small.png")

        self.assertEqual(index.headers["cache-control"], "no-store")
        self.assertEqual(stylesheet.headers["cache-control"], "no-store")
        self.assertEqual(logo.headers["cache-control"], "no-store")

    def test_ui_preserves_in_process_api_routes(self) -> None:
        with TestClient(create_app(api_app=self.backend_app())) as client:
            response = client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ready"})

    def test_unknown_paths_return_not_found(self) -> None:
        with TestClient(create_app(api_app=self.backend_app())) as client:
            response = client.get("/not-allowed")

        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
