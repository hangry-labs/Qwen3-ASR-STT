from __future__ import annotations

import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import patch

from fastapi.testclient import TestClient
from mcp.server.mcpserver.exceptions import ToolError

from qwen_asr.server.aligner_runtime import RuntimeSettingsStore
from qwen_asr.server.mcp_server import attach_mcp, create_mcp_server
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
    duration: float = 1.25
    time_stamps: Any = None


class FakeASR:
    def __init__(self, *, forced_aligner: object | None = None) -> None:
        self.forced_aligner = forced_aligner
        self.calls: list[dict[str, Any]] = []

    def transcribe(self, **kwargs: Any) -> list[FakeResult]:
        self.calls.append(kwargs)
        time_stamps = None
        if kwargs.get("return_time_stamps"):
            time_stamps = FakeAlign(
                [
                    FakeSpan("hello", 0.0, 0.5),
                    FakeSpan("from", 0.5, 0.8),
                    FakeSpan("mcp", 0.8, 1.25),
                ]
            )
        return [
            FakeResult(
                text="hello from mcp",
                language=kwargs.get("language") or "English",
                time_stamps=time_stamps,
            )
        ]


class FakeAlignerRuntime:
    def __init__(self, asr: FakeASR, settings: RuntimeSettingsStore) -> None:
        self.asr = asr
        self.settings = settings
        self.load_always = settings.load_aligner_always()

    def snapshot(self) -> dict[str, Any]:
        loaded = self.asr.forced_aligner is not None
        return {
            "configured": True,
            "loaded": loaded,
            "status": "loaded" if loaded else "unloaded",
            "load_always": self.load_always,
            "model": "Qwen/test-aligner",
            "last_error": None,
            "model_supported_languages": ["English"],
            "available_languages": ["English"],
            "unavailable_languages": {},
        }

    def load(self, *, warm_up: bool = True) -> dict[str, Any]:
        self.asr.forced_aligner = object()
        return self.snapshot()

    def unload(self) -> dict[str, Any]:
        self.asr.forced_aligner = None
        return self.snapshot()

    def set_load_always(self, enabled: bool) -> None:
        self.settings.set_load_aligner_always(enabled)
        self.load_always = enabled


def _app(
    asr: FakeASR | None = None,
    settings_path: Path | None = None,
):
    resolved_asr = asr or FakeASR()
    settings = RuntimeSettingsStore(settings_path or Path("settings-test.json"))
    return create_app(
        asr=resolved_asr,
        model_name="Qwen/Qwen3-ASR-0.6B-hf",
        concurrency=1,
        enable_watchdog=False,
        aligner_runtime=FakeAlignerRuntime(resolved_asr, settings),
    )


class MCPServerTests(unittest.IsolatedAsyncioTestCase):
    async def test_path_only_tools_expose_health_transcription_and_controls(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            app = _app(settings_path=root / "settings.json")
            server, _ = create_mcp_server(
                api_app=app,
                input_directory=root / "input",
                gpu_snapshot_provider=lambda: {
                    "gpus": [{"index": 0, "name": "Test GPU", "memory_used": 123}],
                    "history": {},
                },
            )

            tools = {tool.name for tool in await server.list_tools()}
            health = await server.call_tool("get_health", {})

        self.assertEqual(
            tools,
            {
                "get_health",
                "transcribe_audio_file",
                "configure_aligner_residency",
                "load_forced_aligner",
                "release_forced_aligner_vram",
                "configure_realtime_defaults",
            },
        )
        self.assertNotIn("transcribe_audio", tools)
        self.assertFalse(health.is_error)
        self.assertEqual(health.structured_content["status"], "ok")
        self.assertEqual(health.structured_content["product"]["version"], "1.1-snapshot")
        self.assertEqual(health.structured_content["gpu"][0]["name"], "Test GPU")
        self.assertIn("inference", health.structured_content)
        self.assertIn("persistent_storage", health.structured_content)

    async def test_mounted_file_tool_uses_shared_transcription_service_and_timestamps(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "sample.wav").write_bytes(b"fake audio bytes")
            asr = FakeASR(forced_aligner=object())
            server, _ = create_mcp_server(
                api_app=_app(asr, root / "settings.json"), input_directory=root
            )

            result = await server.call_tool(
                "transcribe_audio_file",
                {
                    "file_path": "sample.wav",
                    "language": "en",
                    "prompt": "domain vocabulary",
                    "timestamp_granularities": ["word", "segment"],
                },
            )

        self.assertFalse(result.is_error)
        self.assertEqual(result.structured_content["text"], "hello from mcp")
        self.assertEqual(result.structured_content["words"][0]["word"], "hello")
        self.assertEqual(result.structured_content["segments"][0]["text"], "hello from mcp")
        self.assertEqual(asr.calls[0]["context"], "domain vocabulary")
        self.assertEqual(asr.calls[0]["language"], "English")

    async def test_mounted_file_tool_rejects_paths_outside_configured_directory(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "sample.wav").write_bytes(b"fake audio bytes")
            server, _ = create_mcp_server(
                api_app=_app(settings_path=root / "settings.json"),
                input_directory=root,
            )

            accepted = await server.call_tool(
                "transcribe_audio_file", {"file_path": "sample.wav"}
            )
            with self.assertRaisesRegex(ToolError, "must stay inside"):
                await server.call_tool(
                    "transcribe_audio_file", {"file_path": "../outside.wav"}
                )

        self.assertFalse(accepted.is_error)

    async def test_runtime_controls_persist_defaults_and_manage_aligner(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings_path = Path(directory) / "settings.json"
            app = _app(settings_path=settings_path)
            server, _ = create_mcp_server(
                api_app=app,
                input_directory=Path(directory) / "input",
                gpu_snapshot_provider=lambda: {"gpus": [], "history": {}},
            )

            defaults = await server.call_tool(
                "configure_realtime_defaults",
                {
                    "chunk_size_sec": 1.25,
                    "max_window_sec": 20,
                    "unfixed_chunk_num": 3,
                    "unfixed_token_num": 8,
                },
            )
            loaded = await server.call_tool("load_forced_aligner", {})
            persistent = await server.call_tool(
                "configure_aligner_residency", {"load_aligner_always": True}
            )
            with self.assertRaisesRegex(ToolError, "Disable 'Always keep aligner loaded'"):
                await server.call_tool("release_forced_aligner_vram", {})
            released = await server.call_tool(
                "configure_aligner_residency", {"load_aligner_always": False}
            )

            persisted = RuntimeSettingsStore(settings_path)
            persisted_load_always = persisted.load_aligner_always()
            persisted_realtime = persisted.realtime_defaults()

        self.assertEqual(defaults.structured_content["realtime_defaults"]["max_window_sec"], 20.0)
        self.assertTrue(loaded.structured_content["loaded"])
        self.assertTrue(persistent.structured_content["load_always"])
        self.assertFalse(released.structured_content["loaded"])
        self.assertFalse(persisted_load_always)
        self.assertEqual(persisted_realtime["unfixed_token_num"], 8)

    async def test_mounted_file_directory_is_created_for_existing_volume_migration(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            input_directory = Path(directory) / "persistent" / "mcp-input"

            server, configured = create_mcp_server(
                api_app=_app(settings_path=Path(directory) / "settings.json"),
                input_directory=input_directory,
            )

            self.assertEqual(configured, input_directory.resolve())
            self.assertTrue(input_directory.is_dir())
            self.assertIn(
                "transcribe_audio_file",
                {tool.name for tool in await server.list_tools()},
            )


class MCPTransportTests(unittest.TestCase):
    def test_streamable_http_is_disabled_by_default_and_follows_persisted_setting(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            app = _app(settings_path=Path(directory) / "settings.json")
            with patch.dict("os.environ", {"QWEN_ASR_MCP_ALLOWED_HOSTS": "testserver"}):
                attach_mcp(api_app=app, input_directory=Path(directory) / "input")

            request = {
                "headers": {
                    "Accept": "application/json, text/event-stream",
                    "MCP-Protocol-Version": "2025-06-18",
                },
                "json": {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2025-06-18",
                        "capabilities": {},
                        "clientInfo": {"name": "test-client", "version": "1.0"},
                    },
                },
            }

            with TestClient(app) as client:
                disabled = client.post("/mcp", **request)
                enabled_setting = client.put(
                    "/system/settings/mcp", json={"enabled": True}
                )
                enabled = client.post("/mcp", **request)
                client.put("/system/settings/mcp", json={"enabled": False})
                disabled_again = client.post("/mcp", **request)

        self.assertEqual(disabled.status_code, 403)
        self.assertEqual(disabled.json()["error"]["type"], "mcp_access_forbidden")
        self.assertTrue(enabled_setting.json()["mcp"]["enabled"])
        self.assertEqual(enabled.status_code, 200)
        self.assertEqual(
            enabled.json()["result"]["serverInfo"]["name"], "qwen3-asr-stt"
        )
        self.assertEqual(disabled_again.status_code, 403)


if __name__ == "__main__":
    unittest.main()
