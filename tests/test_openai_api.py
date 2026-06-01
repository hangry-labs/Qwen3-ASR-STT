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
        self.streaming_calls: list[Any] = []

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

    def init_streaming_state(self, **kwargs):
        self.calls.append({"streaming_init": kwargs})
        return type("FakeStreamingState", (), {"text": "", "language": "", "chunk_id": 0})()

    def streaming_transcribe(self, pcm16k, state):
        self.streaming_calls.append(pcm16k)
        state.chunk_id += 1
        state.language = "English"
        state.text = f"streamed {state.chunk_id}"
        return state

    def finish_streaming_transcribe(self, state):
        state.chunk_id += 1
        state.language = "English"
        state.text = "streamed final"
        return state


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

    def test_omitted_language_keeps_model_native_auto_detection(self):
        asr = FakeASR()
        response = _client(asr).post(
            "/v1/audio/transcriptions",
            files=_files(),
            data={"model": "qwen3-asr", "response_format": "json"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertIsNone(asr.calls[0]["language"])

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

    def test_realtime_transcription_session_append_and_finish(self):
        asr = FakeASR()
        client = _client(asr)

        created = client.post(
            "/v1/realtime/transcriptions/sessions",
            json={"model": "qwen3-asr", "language": "en", "temperature": 0, "chunk_size_sec": 2.0},
        )
        self.assertEqual(created.status_code, 200)
        session_id = created.json()["id"]
        self.assertTrue(session_id.startswith("rt_"))

        import io

        import numpy as np
        import soundfile as sf

        buffer = io.BytesIO()
        sf.write(buffer, np.zeros((16000,), dtype=np.float32), 16000, format="WAV")

        appended = client.post(
            f"/v1/realtime/transcriptions/sessions/{session_id}/audio",
            files={"file": ("chunk.wav", buffer.getvalue(), "audio/wav")},
        )
        self.assertEqual(appended.status_code, 200)
        self.assertEqual(appended.json()["text"], "streamed 1")
        self.assertEqual(appended.json()["final"], False)

        finished = client.post(f"/v1/realtime/transcriptions/sessions/{session_id}/finish")
        self.assertEqual(finished.status_code, 200)
        self.assertEqual(finished.json()["text"], "streamed final")
        self.assertEqual(finished.json()["final"], True)

        missing = client.post(f"/v1/realtime/transcriptions/sessions/{session_id}/finish")
        self.assertEqual(missing.status_code, 404)


if __name__ == "__main__":
    unittest.main()
