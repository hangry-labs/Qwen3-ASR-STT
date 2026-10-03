from __future__ import annotations

import io
import os
import tempfile
import threading
import time
import unittest
from dataclasses import dataclass
from typing import Any
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

from qwen_asr.server.openai_api import _segments_payload, create_app
from qwen_asr.server.aligner_runtime import RuntimeSettingsStore


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


class FakeAlignerRuntime:
    def __init__(self, asr: FakeASR, settings: RuntimeSettingsStore | None = None):
        self.asr = asr
        self.settings = settings
        self.load_always = False
        self.load_calls = 0
        self.unload_calls = 0

    def snapshot(self):
        loaded = self.asr.forced_aligner is not None
        return {
            "configured": True,
            "loaded": loaded,
            "status": "loaded" if loaded else "unloaded",
            "load_always": self.load_always,
            "model": "Qwen/test-aligner",
            "last_error": None,
            "model_supported_languages": [
                "Chinese", "English", "Cantonese", "French", "German", "Italian",
                "Japanese", "Korean", "Portuguese", "Russian", "Spanish",
            ],
            "available_languages": [
                "Chinese", "English", "Cantonese", "French", "German", "Italian",
                "Japanese", "Portuguese", "Russian", "Spanish",
            ],
            "unavailable_languages": {
                "Korean": "Korean timestamps require the optional soynlp tokenizer, which is not bundled in this image."
            },
        }

    def load(self, *, warm_up=True):
        self.load_calls += 1
        self.asr.forced_aligner = object()
        return self.snapshot()

    def unload(self):
        self.unload_calls += 1
        self.asr.forced_aligner = None
        return self.snapshot()

    def set_load_always(self, enabled):
        self.load_always = enabled

class BlockingASR(FakeASR):
    def __init__(self, delay: float):
        super().__init__()
        self.delay = delay

    def transcribe(self, **kwargs):
        time.sleep(self.delay)
        return super().transcribe(**kwargs)


class MissingAlignmentTokenizerASR(FakeASR):
    def transcribe(self, **kwargs):
        raise ImportError("Korean forced alignment requires the soynlp tokenizer")


class UnsupportedAlignmentLanguageASR(FakeASR):
    def transcribe(self, **kwargs):
        raise ValueError("Language 'Turkish' is not supported by the forced aligner")


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
            model_name="Qwen/Qwen3-ASR-0.6B-hf",
            concurrency=1,
        )
    )


def _managed_client(asr: FakeASR, aligner: FakeAlignerRuntime) -> TestClient:
    return TestClient(
        create_app(
            asr=asr,
            model_name="Qwen/Qwen3-ASR-0.6B-hf",
            concurrency=1,
            aligner_runtime=aligner,
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

    def test_timestamp_request_loads_configured_aligner_on_demand(self):
        asr = FakeASR()
        aligner = FakeAlignerRuntime(asr)
        response = _managed_client(asr, aligner).post(
            "/v1/audio/transcriptions",
            files=_files(),
            data={
                "model": "qwen3-asr",
                "response_format": "verbose_json",
                "timestamp_granularities[]": "word",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(aligner.load_calls, 1)
        self.assertTrue(response.json()["words"])

    def test_unsupported_forced_timestamp_language_is_rejected_before_aligner_load(self):
        asr = FakeASR()
        aligner = FakeAlignerRuntime(asr)
        response = _managed_client(asr, aligner).post(
            "/v1/audio/transcriptions",
            files=_files(),
            data={
                "model": "qwen3-asr",
                "language": "Turkish",
                "response_format": "verbose_json",
                "timestamp_granularities[]": "word",
            },
        )

        self.assertEqual(response.status_code, 422)
        self.assertIn("does not support Turkish timestamps", response.json()["error"]["message"])
        self.assertEqual(aligner.load_calls, 0)

    def test_missing_packaged_timestamp_language_dependency_is_rejected_before_aligner_load(self):
        asr = FakeASR()
        aligner = FakeAlignerRuntime(asr)
        response = _managed_client(asr, aligner).post(
            "/v1/audio/transcriptions",
            files=_files(),
            data={
                "model": "qwen3-asr",
                "language": "Korean",
                "response_format": "verbose_json",
                "timestamp_granularities[]": "word",
            },
        )

        self.assertEqual(response.status_code, 422)
        self.assertIn("soynlp tokenizer", response.json()["error"]["message"])
        self.assertEqual(aligner.load_calls, 0)

    def test_missing_language_tokenizer_returns_actionable_error(self):
        asr = MissingAlignmentTokenizerASR(forced_aligner=object())
        response = _client(asr).post(
            "/v1/audio/transcriptions",
            files=_files(),
            data={
                "model": "qwen3-asr",
                "response_format": "verbose_json",
                "timestamp_granularities[]": "word",
            },
        )

        self.assertEqual(response.status_code, 422)
        message = response.json()["error"]["message"]
        self.assertIn("soynlp tokenizer is not bundled", message)
        self.assertIn("without timestamps remains available", message)

    def test_auto_detected_unsupported_timestamp_language_returns_422(self):
        asr = UnsupportedAlignmentLanguageASR(forced_aligner=object())
        response = _client(asr).post(
            "/v1/audio/transcriptions",
            files=_files(),
            data={
                "model": "qwen3-asr",
                "response_format": "verbose_json",
                "timestamp_granularities[]": "word",
            },
        )

        self.assertEqual(response.status_code, 422)
        self.assertIn("not supported by the forced aligner", response.json()["error"]["message"])

    def test_aligner_system_setting_loads_persists_and_unloads(self):
        asr = FakeASR()
        aligner = FakeAlignerRuntime(asr)
        client = _managed_client(asr, aligner)

        enabled = client.put("/system/aligner", json={"load_aligner_always": True})
        self.assertEqual(enabled.status_code, 200)
        self.assertTrue(enabled.json()["loaded"])
        self.assertTrue(enabled.json()["load_always"])

        blocked = client.post("/system/aligner/unload")
        self.assertEqual(blocked.status_code, 409)

        disabled = client.put("/system/aligner", json={"load_aligner_always": False})
        self.assertEqual(disabled.status_code, 200)
        self.assertFalse(disabled.json()["loaded"])
        self.assertFalse(disabled.json()["load_always"])

    def test_realtime_defaults_can_be_persisted(self):
        with tempfile.TemporaryDirectory() as directory:
            asr = FakeASR()
            settings = RuntimeSettingsStore(Path(directory) / "settings.json")
            client = _managed_client(asr, FakeAlignerRuntime(asr, settings))

            saved = client.put(
                "/system/settings/realtime",
                json={"chunk_size_sec": 1.25, "unfixed_chunk_num": 3, "unfixed_token_num": 8},
            )

            self.assertEqual(saved.status_code, 200)
            self.assertEqual(client.get("/system/settings").json()["realtime_defaults"], saved.json()["realtime_defaults"])

    def test_segment_timestamps_split_on_punctuation_and_silence(self):
        result = FakeResult(
            text="Hello world. Another thought after a pause",
            time_stamps=FakeAlign(
                [
                    FakeSpan("Hello", 0.0, 0.3),
                    FakeSpan("world.", 0.31, 0.7),
                    FakeSpan("Another", 0.75, 1.1),
                    FakeSpan("thought", 1.12, 1.5),
                    FakeSpan("after", 2.4, 2.7),
                    FakeSpan("a", 2.72, 2.8),
                    FakeSpan("pause", 2.82, 3.2),
                ]
            ),
        )

        segments = _segments_payload(result)

        self.assertEqual([segment["text"] for segment in segments], ["Hello world.", "Another thought", "after a pause"])
        self.assertEqual([segment["id"] for segment in segments], [0, 1, 2])

    def test_segment_timestamps_project_transcript_punctuation_onto_aligner_words(self):
        result = FakeResult(
            text='I made tea. Then the kettle said "hello!"',
            time_stamps=FakeAlign(
                [
                    FakeSpan("I", 0.0, 0.1),
                    FakeSpan("made", 0.12, 0.3),
                    FakeSpan("tea", 0.31, 0.5),
                    FakeSpan("Then", 0.55, 0.75),
                    FakeSpan("the", 0.76, 0.9),
                    FakeSpan("kettle", 0.91, 1.2),
                    FakeSpan("said", 1.21, 1.4),
                    FakeSpan("hello", 1.41, 1.7),
                ]
            ),
        )

        segments = _segments_payload(result)

        self.assertEqual([segment["text"] for segment in segments], ["I made tea.", "Then the kettle said hello!"])
        self.assertEqual([segment["start"] for segment in segments], [0.0, 0.55])

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
        self.assertEqual(response.json()["id"], "Qwen/Qwen3-ASR-0.6B-hf")

    def test_readiness_reports_timestamp_capability(self):
        without_aligner = _client().get("/health/ready")
        with_aligner = _client(FakeASR(forced_aligner=object())).get("/health/ready")

        self.assertFalse(without_aligner.json()["capabilities"]["timestamps"])
        self.assertEqual(without_aligner.json()["capabilities"]["timestamp_granularities"], [])
        self.assertTrue(with_aligner.json()["capabilities"]["timestamps"])
        self.assertTrue(with_aligner.json()["capabilities"]["timestamps_loaded"])
        self.assertEqual(
            with_aligner.json()["capabilities"]["timestamp_granularities"],
            ["word", "segment"],
        )

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
            model_name="Qwen/Qwen3-ASR-0.6B-hf",
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

    def test_startup_warmup_runs_on_the_engine_owner_thread(self):
        thread_ids: list[int] = []

        class ThreadTrackingASR(FakeASR):
            def transcribe(self, **kwargs):
                thread_ids.append(threading.get_ident())
                return super().transcribe(**kwargs)

        app = create_app(
            asr=ThreadTrackingASR(),
            model_name="Qwen/Qwen3-ASR-0.6B-hf",
            concurrency=1,
            recycle_process=lambda reason: None,
            enable_watchdog=False,
            startup_warmup=lambda: thread_ids.append(threading.get_ident()),
        )
        with TestClient(app) as client:
            response = client.post(
                "/v1/audio/transcriptions",
                files=_files(),
                data={"model": "qwen3-asr"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(thread_ids), 2)
        self.assertEqual(thread_ids[0], thread_ids[1])
        self.assertNotEqual(thread_ids[0], threading.get_ident())

    def test_inference_timeout_degrades_readiness_and_rejects_later_work(self):
        recycled: list[str] = []
        app = create_app(
            asr=BlockingASR(delay=0.15),
            model_name="Qwen/Qwen3-ASR-0.6B-hf",
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
            model_name="Qwen/Qwen3-ASR-0.6B-hf",
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
            model_name="Qwen/Qwen3-ASR-0.6B-hf",
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
                model_name="Qwen/Qwen3-ASR-0.6B-hf",
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
            model_name="Qwen/Qwen3-ASR-0.6B-hf",
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
            model_name="Qwen/Qwen3-ASR-0.6B-hf",
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
