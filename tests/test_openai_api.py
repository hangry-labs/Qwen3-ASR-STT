from __future__ import annotations

import unittest
from dataclasses import dataclass
from typing import Any

from fastapi.testclient import TestClient

from qwen_asr.server.openai_api import create_app


@dataclass
class FakeSpan:
    text: str
    start_time: float
    end_time: float


@dataclass
class FakeAlign:
    items: list[FakeSpan]


@dataclass
class FakeResult:
    text: str
    language: str = "English"
    time_stamps: Any = None


class FakeASR:
    def __init__(self, *, forced_aligner: object | None = None):
        self.forced_aligner = forced_aligner
        self.calls: list[dict[str, Any]] = []

    def transcribe(self, **kwargs):
        self.calls.append(kwargs)
        time_stamps = None
        if kwargs.get("return_time_stamps"):
            time_stamps = FakeAlign(
                [
                    FakeSpan("hello", 0.0, 0.3),
                    FakeSpan("world", 0.31, 0.7),
                ]
            )
        return [FakeResult(text="hello world", language=kwargs.get("language") or "English", time_stamps=time_stamps)]


def _client(asr: FakeASR | None = None) -> TestClient:
    return TestClient(
        create_app(
            asr=asr or FakeASR(),
            model_name="Qwen/Qwen3-ASR-0.6B",
            concurrency=1,
        )
    )


def _files():
    return {"file": ("sample.wav", b"fake audio bytes", "audio/wav")}


class OpenAIApiTests(unittest.TestCase):
    def test_json_transcription_accepts_sdk_style_fields(self):
        asr = FakeASR()
        response = _client(asr).post(
            "/v1/audio/transcriptions",
            files=_files(),
            data={
                "model": "qwen3-asr",
                "language": "en",
                "prompt": "domain vocabulary",
                "temperature": "0",
                "response_format": "json",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"text": "hello world"})
        self.assertEqual(asr.calls[0]["language"], "English")
        self.assertEqual(asr.calls[0]["context"], "domain vocabulary")
        self.assertFalse(asr.calls[0]["return_time_stamps"])

    def test_text_srt_and_vtt_response_formats(self):
        client = _client()

        text = client.post(
            "/v1/audio/transcriptions",
            files=_files(),
            data={"model": "qwen3-asr", "response_format": "text"},
        )
        self.assertEqual(text.status_code, 200)
        self.assertEqual(text.text, "hello world")

        srt = client.post(
            "/v1/audio/transcriptions",
            files=_files(),
            data={"model": "qwen3-asr", "response_format": "srt"},
        )
        self.assertEqual(srt.status_code, 200)
        self.assertIn("00:00:00,000 -->", srt.text)

        vtt = client.post(
            "/v1/audio/transcriptions",
            files=_files(),
            data={"model": "qwen3-asr", "response_format": "vtt"},
        )
        self.assertEqual(vtt.status_code, 200)
        self.assertTrue(vtt.text.startswith("WEBVTT"))

    def test_verbose_json_with_timestamps_requires_aligner_and_returns_words(self):
        no_aligner = _client()
        rejected = no_aligner.post(
            "/v1/audio/transcriptions",
            files=_files(),
            data={
                "model": "qwen3-asr",
                "response_format": "verbose_json",
                "timestamp_granularities[]": "word",
            },
        )
        self.assertEqual(rejected.status_code, 400)
        self.assertIn("error", rejected.json())

        with_aligner = _client(FakeASR(forced_aligner=object()))
        accepted = with_aligner.post(
            "/v1/audio/transcriptions",
            files=_files(),
            data={
                "model": "qwen3-asr",
                "response_format": "verbose_json",
                "timestamp_granularities[]": "word,segment",
            },
        )
        self.assertEqual(accepted.status_code, 200)
        payload = accepted.json()
        self.assertEqual(payload["text"], "hello world")
        self.assertEqual(payload["words"][0]["word"], "hello")
        self.assertEqual(payload["segments"][0]["text"], "hello world")

    def test_stream_true_returns_transcript_events(self):
        with _client().stream(
            "POST",
            "/v1/audio/transcriptions",
            files=_files(),
            data={"model": "qwen3-asr", "stream": "true"},
        ) as response:
            body = response.read().decode("utf-8")

        self.assertEqual(response.status_code, 200)
        self.assertIn("event: transcript.text.delta", body)
        self.assertIn("event: transcript.text.done", body)
        self.assertIn("data: [DONE]", body)

    def test_openai_error_envelope(self):
        response = _client().post(
            "/v1/audio/transcriptions",
            files=_files(),
            data={"model": "wrong-model"},
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"]["type"], "invalid_request_error")

    def test_retrieve_model_alias(self):
        response = _client().get("/v1/models/qwen3-asr")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["id"], "Qwen/Qwen3-ASR-0.6B")

    def test_translation_endpoint_is_explicitly_not_ready(self):
        response = _client().post("/v1/audio/translations", files=_files(), data={"model": "qwen3-asr"})
        self.assertEqual(response.status_code, 501)
        self.assertIn("error", response.json())


if __name__ == "__main__":
    unittest.main()
