from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from qwen_asr.server.aligner_runtime import AlignerRuntime, RuntimeSettingsStore


class _FakeASR:
    forced_aligner = None


class AlignerRuntimeTests(unittest.TestCase):
    def test_setting_is_persisted_and_preserves_other_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.json"
            path.write_text('{"another_setting": 42}\n', encoding="utf-8")
            store = RuntimeSettingsStore(path)

            store.set_load_aligner_always(True)

            self.assertTrue(store.load_aligner_always())
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["another_setting"], 42)

    def test_realtime_defaults_are_validated_and_persisted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.json"
            store = RuntimeSettingsStore(path)

            saved = store.set_realtime_defaults(
                {
                    "chunk_size_sec": 1.25,
                    "max_window_sec": 20,
                    "unfixed_chunk_num": 3,
                    "unfixed_token_num": 8,
                }
            )

            self.assertEqual(saved["chunk_size_sec"], 1.25)
            self.assertEqual(RuntimeSettingsStore(path).realtime_defaults(), saved)
            with self.assertRaises(ValueError):
                store.set_realtime_defaults(
                    {
                        "chunk_size_sec": 0.1,
                        "max_window_sec": 20,
                        "unfixed_chunk_num": 3,
                        "unfixed_token_num": 8,
                    }
                )

    def test_mcp_setting_is_disabled_by_default_and_persisted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.json"
            store = RuntimeSettingsStore(path)

            self.assertFalse(store.mcp_enabled())
            store.set_mcp_enabled(True)

            self.assertTrue(RuntimeSettingsStore(path).mcp_enabled())

    def test_legacy_realtime_defaults_gain_bounded_window_default(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.json"
            path.write_text(
                '{"realtime_defaults":{"chunk_size_sec":1.25,"unfixed_chunk_num":3,"unfixed_token_num":8}}\n',
                encoding="utf-8",
            )

            defaults = RuntimeSettingsStore(path).realtime_defaults()

            self.assertEqual(defaults["chunk_size_sec"], 1.25)
            self.assertEqual(defaults["max_window_sec"], 30.0)

    def test_lazy_load_warmup_and_unload_share_one_model_instance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            calls: list[tuple[str, object]] = []
            aligner = object()
            asr = _FakeASR()

            def loader(checkpoint, **kwargs):
                calls.append((checkpoint, kwargs))
                return aligner

            runtime = AlignerRuntime(
                asr=asr,
                checkpoint="Qwen/test-aligner",
                model_kwargs={"dtype": "bfloat16"},
                settings=RuntimeSettingsStore(Path(directory) / "settings.json"),
                loader=loader,
                warmup=lambda model: calls.append(("warmup", model)),
            )

            self.assertEqual(runtime.snapshot()["status"], "unloaded")
            runtime.load()
            runtime.load()

            self.assertIs(asr.forced_aligner, aligner)
            self.assertEqual(calls[0], ("Qwen/test-aligner", {"dtype": "bfloat16"}))
            self.assertEqual(calls[1], ("warmup", aligner))
            self.assertEqual(len(calls), 2)

            runtime.unload()
            self.assertIsNone(asr.forced_aligner)
            self.assertEqual(runtime.snapshot()["status"], "unloaded")

    def test_persisted_policy_overrides_the_environment_default_on_next_start(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = RuntimeSettingsStore(Path(directory) / "settings.json")
            first = AlignerRuntime(
                asr=_FakeASR(),
                checkpoint="Qwen/test-aligner",
                model_kwargs={},
                settings=store,
                default_load_always=False,
                loader=lambda *_args, **_kwargs: object(),
            )
            first.set_load_always(True)

            restarted = AlignerRuntime(
                asr=_FakeASR(),
                checkpoint="Qwen/test-aligner",
                model_kwargs={},
                settings=RuntimeSettingsStore(store.path),
                default_load_always=False,
                loader=lambda *_args, **_kwargs: object(),
            )

            self.assertTrue(restarted.load_always)


if __name__ == "__main__":
    unittest.main()
