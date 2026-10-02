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
                audio_editor = client.get("/static/audio-editor.js")
                stylesheet = client.get("/static/styles.css")
                icon_stylesheet = client.get("/static/vendor/lucide/lucide.css")
                icon_font = client.get("/static/vendor/lucide/lucide.woff2")
                product_logo = client.get("/assets/qwen3_asr_logo_horizontal.webp")
                mascot = client.get("/assets/qwen3_asr_favicon.webp")
                labs_logo = client.get("/assets/hangrylabs_logo_horizontal.webp")
                gpu = client.get("/system/gpu")
                health = client.get("/health")

        self.assertEqual(index.status_code, 200)
        self.assertIn("Qwen3-ASR-STT", index.text)
        self.assertIn("Qwen3-ASR-STT/tree/main/testbench", index.text)
        self.assertIn('src="/assets/qwen3_asr_logo_horizontal.webp"', index.text)
        self.assertIn('href="/assets/qwen3_asr_favicon.webp"', index.text)
        self.assertIn('class="collapsed-mascot"', index.text)
        self.assertIn('class="labs-badge"', index.text)
        self.assertIn('src="/assets/hangrylabs_logo_horizontal.webp"', index.text)
        self.assertIn('href="https://github.com/Hangry-Labs/Qwen3-ASR-STT/releases"', index.text)
        self.assertIn('id="hero-toggle"', index.text)
        self.assertIn('id="record-toggle"', index.text)
        self.assertIn('id="realtime-toggle"', index.text)
        self.assertIn('class="icon-button bordered device-refresh"', index.text)
        self.assertNotIn('id="example-load"', index.text)
        self.assertNotIn('id="record-stop"', index.text)
        self.assertNotIn('id="realtime-stop"', index.text)
        self.assertIn("document.documentElement.dataset.headerCollapsed = 'true'", index.text)
        self.assertNotIn('/assets/banner.jpg', index.text)
        self.assertNotIn('class="brand-lockup"', index.text)
        self.assertNotIn("qwen-highlight", index.text)
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
        self.assertIn("function setHeroCollapsed(", script.text)
        self.assertIn("headerCollapsed: state.headerCollapsed", script.text)
        self.assertIn("document.documentElement.dataset.headerCollapsed", script.text)
        self.assertIn("hero.animate(", script.text)
        self.assertIn("function setRecordButton(", script.text)
        self.assertIn("function setRealtimeButton(", script.text)
        self.assertIn("$('#example-select').addEventListener('change'", script.text)
        self.assertIn("recordEditor.isRecording()", script.text)
        self.assertIn("if (realtime.running)", script.text)
        self.assertNotIn('id="system-refresh"', index.text)
        self.assertEqual(audio_editor.status_code, 200)
        self.assertIn("continuousWaveform: true", audio_editor.text)
        self.assertNotIn("continuousWaveformDuration", audio_editor.text)
        self.assertIn("this.wave.setOptions({ minPxPerSec: 0 })", audio_editor.text)
        self.assertEqual(stylesheet.status_code, 200)
        self.assertIn("labs-badge-orbit", stylesheet.text)
        self.assertIn("offset-path", stylesheet.text)
        self.assertIn(".runtime-copy span { display: block", stylesheet.text)
        self.assertIn("@media (min-width: 1101px)", stylesheet.text)
        self.assertIn(".brand-hero { height: 224px; min-height: 224px; }", stylesheet.text)
        self.assertIn('.record-button[data-recording="true"]', stylesheet.text)
        self.assertIn(".device-refresh { flex: 0 0 40px; width: 40px; height: 40px; }", stylesheet.text)
        self.assertEqual(icon_stylesheet.status_code, 200)
        for icon in (
            "activity", "audio-lines", "book-open", "boxes", "check", "chevron-down", "chevron-up",
            "download", "fast-forward", "file-audio", "git-branch", "message-square-text", "mic", "pause",
            "play", "radio", "refresh-cw", "rewind", "rotate-ccw", "scissors", "share-2",
            "sliders-horizontal", "sparkles", "square", "upload", "volume-2", "x",
        ):
            self.assertIn(f".icon-{icon}::before", icon_stylesheet.text)
        self.assertNotIn(".icon-alarm-clock::before", icon_stylesheet.text)
        self.assertEqual(icon_font.status_code, 200)
        self.assertEqual(icon_font.headers["content-type"], "font/woff2")
        self.assertEqual(product_logo.status_code, 200)
        self.assertEqual(product_logo.headers["content-type"], "image/webp")
        self.assertEqual(mascot.status_code, 200)
        self.assertEqual(mascot.headers["content-type"], "image/webp")
        self.assertEqual(labs_logo.status_code, 200)
        self.assertEqual(labs_logo.headers["content-type"], "image/webp")
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
                logo = client.get("/assets/qwen3_asr_favicon.webp")

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
