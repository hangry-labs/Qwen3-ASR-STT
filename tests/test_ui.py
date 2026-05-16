import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from qwen_asr.server.ui import _audio_path_from_gradio, _request_data


class UiCallbackInputTests(unittest.TestCase):
    def test_audio_path_from_gradio_accepts_common_payload_shapes(self):
        self.assertIsNone(_audio_path_from_gradio(None))
        self.assertEqual(_audio_path_from_gradio("sample.wav"), "sample.wav")
        self.assertEqual(_audio_path_from_gradio(Path("sample.wav")), os.fspath(Path("sample.wav")))
        self.assertEqual(_audio_path_from_gradio({"path": "sample.wav"}), "sample.wav")
        self.assertEqual(_audio_path_from_gradio({"name": "sample.wav"}), "sample.wav")
        self.assertEqual(_audio_path_from_gradio(SimpleNamespace(path="sample.wav")), "sample.wav")
        self.assertEqual(_audio_path_from_gradio(SimpleNamespace(name="sample.wav")), "sample.wav")

    def test_request_data_keeps_deterministic_temperature(self):
        data = _request_data(
            model="qwen3-asr",
            language="English",
            response_format="json",
            prompt="",
            timestamp_granularities=[],
        )
        self.assertEqual(data["temperature"], "0")


class UiCallbackSmokeTests(unittest.TestCase):
    def test_transcribe_callback_accepts_filedata_dict(self):
        from qwen_asr.server import ui

        class FakeResponse:
            status_code = 200
            text = '{"text":"ok"}'

        class FakeClient:
            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def post(self, *args, **kwargs):
                return FakeResponse()

        with tempfile.NamedTemporaryFile(suffix=".wav") as audio_file:
            original_client = ui.httpx.Client
            ui.httpx.Client = FakeClient
            try:
                transcript, details, status = ui.transcribe_openai(
                    {"path": audio_file.name},
                    "qwen3-asr",
                    "Auto",
                    "json",
                    "",
                    [],
                    "http://127.0.0.1:8000",
                )
            finally:
                ui.httpx.Client = original_client

        self.assertEqual(transcript, "ok")
        self.assertIn('"text": "ok"', details)
        self.assertIn("HTTP 200", status)


if __name__ == "__main__":
    unittest.main()
