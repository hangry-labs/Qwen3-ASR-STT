from __future__ import annotations

import io
import os
import threading
import time
import unittest
from dataclasses import dataclass
from typing import Any

import numpy as np
import soundfile as sf
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
        self.streaming_inference_calls = 0

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
        chunk_size_samples = round(float(kwargs["chunk_size_sec"]) * 16000)
        return type(
            "FakeStreamingState",
            (),
            {
                "text": "",
                "language": "",
                "chunk_id": 0,
                "buffer": np.zeros((0,), dtype=np.float32),
                "chunk_size_samples": chunk_size_samples,
            },
        )()

    def streaming_transcribe(self, pcm16k, state):
        self.streaming_calls.append(pcm16k)
        state.buffer = np.concatenate([state.buffer, pcm16k])
        while len(state.buffer) >= state.chunk_size_samples:
            state.buffer = state.buffer[state.chunk_size_samples :]
            self.streaming_inference_calls += 1
            state.chunk_id += 1
            state.language = "English"
            state.text = f"streamed {state.chunk_id}"
        return state

    def finish_streaming_transcribe(self, state):
        if len(state.buffer):
            state.buffer = np.zeros((0,), dtype=np.float32)
            self.streaming_inference_calls += 1
            state.chunk_id += 1
            state.language = "English"
            state.text = "streamed final"
        return state


class BlockingASR(FakeASR):
    def __init__(self, delay: float):
        super().__init__()
        self.delay = delay

    def transcribe(self, **kwargs):
        time.sleep(self.delay)
        return super().transcribe(**kwargs)


class TrackingASR(FakeASR):
    def __init__(self):
        super().__init__()
        self._active = 0
        self.max_active = 0
        self._tracking_lock = threading.Lock()

    def _enter(self):
        with self._tracking_lock:
            self._active += 1
            self.max_active = max(self.max_active, self._active)

    def _exit(self):
        with self._tracking_lock:
            self._active -= 1

    def transcribe(self, **kwargs):
        self._enter()
        try:
            time.sleep(0.05)
            return super().transcribe(**kwargs)
        finally:
            self._exit()

    def streaming_transcribe(self, pcm16k, state):
        self._enter()
        try:
            time.sleep(0.05)
            return super().streaming_transcribe(pcm16k, state)
        finally:
            self._exit()


class PausingStreamingASR(FakeASR):
    def __init__(self):
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()

    def streaming_transcribe(self, pcm16k, state):
        self.started.set()
        self.release.wait(timeout=5)
        return super().streaming_transcribe(pcm16k, state)


class PausingFinishASR(FakeASR):
    def __init__(self):
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()
        self.finish_calls = 0

    def finish_streaming_transcribe(self, state):
        self.finish_calls += 1
        self.started.set()
        self.release.wait(timeout=5)
        return super().finish_streaming_transcribe(state)


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


def _wav(seconds: float = 1.0) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, np.zeros((round(16000 * seconds),), dtype=np.float32), 16000, format="WAV")
    return buffer.getvalue()


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

        first = client.post(
            f"/v1/realtime/transcriptions/sessions/{session_id}/audio",
            files={"file": ("chunk.wav", _wav(), "audio/wav")},
        )
        self.assertEqual(first.status_code, 200)
        self.assertEqual(first.json()["text"], "")
        self.assertEqual(first.json()["chunk_id"], 0)
        self.assertEqual(asr.streaming_inference_calls, 0)

        second = client.post(
            f"/v1/realtime/transcriptions/sessions/{session_id}/audio",
            files={"file": ("chunk.wav", _wav(), "audio/wav")},
        )
        self.assertEqual(second.status_code, 200)
        self.assertEqual(second.json()["text"], "streamed 1")
        self.assertEqual(second.json()["chunk_id"], 1)
        self.assertEqual(second.json()["final"], False)
        self.assertEqual(asr.streaming_inference_calls, 1)

        finished = client.post(f"/v1/realtime/transcriptions/sessions/{session_id}/finish")
        self.assertEqual(finished.status_code, 200)
        self.assertEqual(finished.json()["text"], "streamed 1")
        self.assertEqual(finished.json()["final"], True)

        missing = client.post(f"/v1/realtime/transcriptions/sessions/{session_id}/finish")
        self.assertEqual(missing.status_code, 404)

    def test_ordinary_and_realtime_inference_share_one_engine_owner(self):
        from concurrent.futures import ThreadPoolExecutor

        asr = TrackingASR()
        app = create_app(
            asr=asr,
            model_name="Qwen/Qwen3-ASR-0.6B",
            concurrency=2,
            recycle_process=lambda reason: None,
            enable_watchdog=False,
        )
        with TestClient(app) as client:
            created = client.post(
                "/v1/realtime/transcriptions/sessions",
                json={"model": "qwen3-asr", "temperature": 0, "chunk_size_sec": 2.0},
            )
            session_id = created.json()["id"]

            def ordinary():
                return client.post(
                    "/v1/audio/transcriptions",
                    files=_files(),
                    data={"model": "qwen3-asr"},
                )

            def realtime():
                return client.post(
                    f"/v1/realtime/transcriptions/sessions/{session_id}/audio",
                    files={"file": ("chunk.wav", _wav(2.0), "audio/wav")},
                )

            with ThreadPoolExecutor(max_workers=2) as pool:
                responses = [future.result() for future in [pool.submit(ordinary), pool.submit(realtime)]]

        self.assertEqual([response.status_code for response in responses], [200, 200])
        self.assertEqual(asr.max_active, 1)

    def test_inference_timeout_degrades_readiness_and_rejects_later_work(self):
        recycled: list[str] = []
        app = create_app(
            asr=BlockingASR(delay=0.15),
            model_name="Qwen/Qwen3-ASR-0.6B",
            concurrency=2,
            inference_timeout_seconds=0.02,
            recycle_process=recycled.append,
            enable_watchdog=False,
        )
        with TestClient(app) as client:
            session = client.post(
                "/v1/realtime/transcriptions/sessions",
                json={"model": "qwen3-asr", "temperature": 0, "chunk_size_sec": 2.0},
            ).json()
            timed_out = client.post(
                "/v1/audio/transcriptions",
                files=_files(),
                data={"model": "qwen3-asr"},
            )
            ready = client.get("/health/ready")
            live = client.get("/health/live")
            rejected = client.post(
                "/v1/audio/transcriptions",
                files=_files(),
                data={"model": "qwen3-asr"},
            )
            realtime_rejected = client.post(
                f"/v1/realtime/transcriptions/sessions/{session['id']}/audio",
                files={"file": ("empty.wav", b"", "audio/wav")},
            )

            self.assertEqual(timed_out.status_code, 503)
            self.assertEqual(ready.status_code, 503)
            self.assertEqual(ready.json()["reason"], "inference_timeout")
            self.assertEqual(ready.json()["active_inference"], 1)
            self.assertEqual(live.status_code, 200)
            self.assertEqual(rejected.status_code, 503)
            self.assertEqual(realtime_rejected.status_code, 503)
            self.assertEqual(recycled, ["inference_timeout"])

    def test_stale_realtime_session_expires(self):
        app = create_app(
            asr=FakeASR(),
            model_name="Qwen/Qwen3-ASR-0.6B",
            concurrency=1,
            realtime_session_ttl_seconds=0.01,
            recycle_process=lambda reason: None,
            enable_watchdog=False,
        )
        with TestClient(app) as client:
            created = client.post(
                "/v1/realtime/transcriptions/sessions",
                json={"model": "qwen3-asr", "temperature": 0, "chunk_size_sec": 2.0},
            )
            time.sleep(0.03)
            expired = client.post(
                f"/v1/realtime/transcriptions/sessions/{created.json()['id']}/audio",
                files={"file": ("chunk.wav", _wav(), "audio/wav")},
            )

        self.assertEqual(expired.status_code, 404)

    def test_startup_watchdog_probe_sets_real_inference_readiness(self):
        asr = FakeASR()
        app = create_app(
            asr=asr,
            model_name="Qwen/Qwen3-ASR-0.6B",
            concurrency=1,
            recycle_process=lambda reason: None,
            enable_watchdog=True,
        )
        with TestClient(app) as client:
            ready = client.get("/health/ready")

        self.assertEqual(ready.status_code, 200)
        self.assertIsNotNone(ready.json()["last_inference_success_at"])
        self.assertTrue(any(call.get("language") is None for call in asr.calls if "language" in call))

    def test_enabled_watchdog_requires_bundled_fixture(self):
        recycled: list[str] = []
        previous = os.environ.get("QWEN_ASR_WATCHDOG_AUDIO")
        os.environ["QWEN_ASR_WATCHDOG_AUDIO"] = "/missing/watchdog.wav"
        try:
            app = create_app(
                asr=FakeASR(),
                model_name="Qwen/Qwen3-ASR-0.6B",
                concurrency=1,
                recycle_process=recycled.append,
                enable_watchdog=True,
            )
            with self.assertRaisesRegex(RuntimeError, "Startup inference readiness probe failed"):
                with TestClient(app):
                    pass
        finally:
            if previous is None:
                os.environ.pop("QWEN_ASR_WATCHDOG_AUDIO", None)
            else:
                os.environ["QWEN_ASR_WATCHDOG_AUDIO"] = previous

        self.assertEqual(recycled, ["watchdog_failure"])

    def test_delete_rejects_session_with_active_inference(self):
        from concurrent.futures import ThreadPoolExecutor

        asr = PausingStreamingASR()
        app = create_app(
            asr=asr,
            model_name="Qwen/Qwen3-ASR-0.6B",
            concurrency=1,
            recycle_process=lambda reason: None,
            enable_watchdog=False,
        )
        with TestClient(app) as client:
            created = client.post(
                "/v1/realtime/transcriptions/sessions",
                json={"model": "qwen3-asr", "temperature": 0, "chunk_size_sec": 2.0},
            )
            session_id = created.json()["id"]
            with ThreadPoolExecutor(max_workers=1) as pool:
                append = pool.submit(
                    client.post,
                    f"/v1/realtime/transcriptions/sessions/{session_id}/audio",
                    files={"file": ("chunk.wav", _wav(2.0), "audio/wav")},
                )
                try:
                    self.assertTrue(asr.started.wait(timeout=5))
                    deleted = client.delete(f"/v1/realtime/transcriptions/sessions/{session_id}")
                finally:
                    asr.release.set()
                appended = append.result(timeout=5)

        self.assertEqual(deleted.status_code, 409)
        self.assertEqual(appended.status_code, 200)

    def test_concurrent_finish_runs_model_only_once(self):
        from concurrent.futures import ThreadPoolExecutor

        asr = PausingFinishASR()
        app = create_app(
            asr=asr,
            model_name="Qwen/Qwen3-ASR-0.6B",
            concurrency=2,
            recycle_process=lambda reason: None,
            enable_watchdog=False,
        )
        with TestClient(app) as client:
            created = client.post(
                "/v1/realtime/transcriptions/sessions",
                json={"model": "qwen3-asr", "temperature": 0, "chunk_size_sec": 2.0},
            )
            session_id = created.json()["id"]
            buffered = client.post(
                f"/v1/realtime/transcriptions/sessions/{session_id}/audio",
                files={"file": ("chunk.wav", _wav(), "audio/wav")},
            )
            self.assertEqual(buffered.status_code, 200)

            with ThreadPoolExecutor(max_workers=2) as pool:
                first = pool.submit(
                    client.post,
                    f"/v1/realtime/transcriptions/sessions/{session_id}/finish",
                )
                self.assertTrue(asr.started.wait(timeout=5))
                second = pool.submit(
                    client.post,
                    f"/v1/realtime/transcriptions/sessions/{session_id}/finish",
                )
                try:
                    second_result = second.result(timeout=5)
                finally:
                    asr.release.set()
                first_result = first.result(timeout=5)

        self.assertEqual(first_result.status_code, 200)
        self.assertEqual(second_result.status_code, 409)
        self.assertEqual(asr.finish_calls, 1)


if __name__ == "__main__":
    unittest.main()
