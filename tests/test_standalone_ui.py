from __future__ import annotations

import time
import unittest
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from qwen_asr.standalone_ui.server import _example_catalog, _read_version_file, create_app
from qwen_asr.web.gpu import GpuMonitor, read_gpu_stats


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
        gpu_payload = {"gpus": [], "history": {}, "sample_interval_seconds": 1, "idle_timeout_seconds": 60}
        with patch("qwen_asr.standalone_ui.server.GPU_MONITOR.request_snapshot", return_value=gpu_payload):
            with TestClient(create_app(api_app=self.backend_app())) as client:
                index = client.get("/")
                script = client.get("/static/app.js")
                gpu = client.get("/system/gpu")
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
        self.assertIn("gpuWindowMs: 60 * 1000", script.text)
        self.assertIn("GPU_HISTORY_RETENTION_MS = 10 * 60 * 1000", script.text)
        self.assertIn("GPU_POLL_INTERVAL_MS = 1000", script.text)
        self.assertIn("function renderGpuMonitor(", script.text)
        self.assertIn("function addGpuChartGrid(", script.text)
        self.assertIn("function attachGpuChartHover(", script.text)
        self.assertIn("function stopGpuMonitor(", script.text)
        self.assertIn("sessionStorage.setItem(GPU_SESSION_KEY", script.text)
        self.assertNotIn('id="system-refresh"', index.text)
        self.assertEqual(gpu.status_code, 200)
        self.assertIn("gpus", gpu.json())
        self.assertIn("history", gpu.json())
        self.assertIsInstance(gpu.json()["gpus"], list)
        self.assertEqual(gpu.headers["cache-control"], "no-store")
        self.assertEqual(health.json(), {"status": "ready"})

    @patch("qwen_asr.web.gpu.subprocess.run")
    def test_gpu_monitor_parses_nvidia_smi(self, run) -> None:
        run.return_value.returncode = 0
        run.return_value.stdout = (
            "0, NVIDIA RTX Test, 37, 12, 4096, 16384, 52, 30, 61.5, 300, "
            "2400, 3000, 13000, 14000, P2, 5, 16\n"
        )

        stats = read_gpu_stats()

        self.assertEqual(stats[0]["utilization"], 37)
        self.assertEqual(stats[0]["memory_total"], 16384)
        self.assertEqual(stats[0]["name"], "NVIDIA RTX Test")
        self.assertEqual(stats[0]["temperature"], 52)
        self.assertEqual(stats[0]["power"], 61.5)
        self.assertEqual(stats[0]["fan_speed"], 30)
        self.assertEqual(stats[0]["graphics_clock"], 2400)
        self.assertEqual(stats[0]["performance_state"], "P2")

    @patch("qwen_asr.web.gpu.subprocess.run")
    def test_gpu_monitor_keeps_gpu_when_optional_values_are_unavailable(self, run) -> None:
        run.return_value.returncode = 0
        run.return_value.stdout = (
            "0, NVIDIA Compute GPU, 75, N/A, 1024, 8192, 48, [N/A], 125, 250, "
            "1800, N/A, N/A, N/A, P0, 4, 16\n"
        )

        stats = read_gpu_stats()

        self.assertEqual(len(stats), 1)
        self.assertIsNone(stats[0]["fan_speed"])
        self.assertIsNone(stats[0]["memory_utilization"])
        self.assertEqual(stats[0]["power"], 125.0)

    def test_gpu_monitor_samples_until_idle_timeout(self) -> None:
        calls = 0

        def reader():
            nonlocal calls
            calls += 1
            return [{"index": 0, "name": "Test GPU", "utilization": calls}]

        monitor = GpuMonitor(reader, sample_interval=0.01, idle_timeout=0.04, history_seconds=1)
        try:
            first = monitor.request_snapshot()
            self.assertEqual(first["gpus"][0]["utilization"], 1)
            time.sleep(0.03)
            self.assertGreaterEqual(calls, 3)

            time.sleep(0.05)
            stopped_at = calls
            time.sleep(0.03)
            self.assertEqual(calls, stopped_at)
            self.assertGreaterEqual(len(monitor.request_snapshot()["history"]["0"]), 3)
        finally:
            monitor.close()

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
