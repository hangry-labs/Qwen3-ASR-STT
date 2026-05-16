import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from qwen_asr.inference.utils import SUPPORTED_LANGUAGES
from qwen_asr.server.app import _audio_path_from_gradio, _load_examples, _request_data, load_example_choice


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

    def test_examples_include_one_fixture_per_supported_language(self):
        examples = _load_examples()
        example_languages = [language for _path, language in examples]
        self.assertEqual(example_languages, SUPPORTED_LANGUAGES)

    def test_realtime_audio_chunk_is_serialized_as_wav(self):
        import numpy as np
        import soundfile as sf

        from qwen_asr.server.app import _wav_bytes_from_audio_chunk

        payload = _wav_bytes_from_audio_chunk((16000, np.zeros((8000,), dtype=np.float32)))
        self.assertIsNotNone(payload)
        audio_bytes, duration = payload
        self.assertAlmostEqual(duration, 0.5, places=2)

        with tempfile.NamedTemporaryFile(suffix=".wav") as audio_file:
            audio_file.write(audio_bytes)
            audio_file.flush()
            audio, sample_rate = sf.read(audio_file.name)
        self.assertEqual(sample_rate, 16000)
        self.assertEqual(audio.shape[0], 8000)

    def test_load_example_choice_returns_audio_and_language(self):
        audio_path, language = load_example_choice("English - 01.mp3", {"English - 01.mp3": ["sample.mp3", "English"]})
        self.assertEqual(audio_path, "sample.mp3")
        self.assertEqual(language, "English")


class UiCallbackSmokeTests(unittest.TestCase):
    def test_transcribe_callback_accepts_filedata_dict(self):
        from qwen_asr.server import app

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
            original_client = app.httpx.Client
            app.httpx.Client = FakeClient
            try:
                transcript, details, status = app.transcribe_openai(
                    {"path": audio_file.name},
                    "qwen3-asr",
                    "Auto",
                    "json",
                    "",
                    [],
                    "http://127.0.0.1:8000",
                )
            finally:
                app.httpx.Client = original_client

        self.assertEqual(transcript, "ok")
        self.assertIn('"text": "ok"', details)
        self.assertIn("HTTP 200", status)


if __name__ == "__main__":
    unittest.main()
