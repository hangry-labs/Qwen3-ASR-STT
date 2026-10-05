# Qwen 3.6 35B-A3B MCP Baseline

Tested on 2026-10-04 with the local `qwen36-35b-hermes:current` Q4_K_M model through its OpenAI-compatible chat-completions endpoint. The target was the live Qwen3-ASR STT MCP Streamable HTTP endpoint backed by the real 0.6B ASR model, optional forced aligner, local GPU telemetry, and persistent runtime settings.

## Result

- 15 cases x 3 repeats = 45 executions
- 45 passed, 0 failed: 100% overall
- Every tool selection and JSON argument was correct for full health/GPU discovery, path-referenced transcription, automatic and forced language, prompt/context, word and segment timestamps, path confinement, missing-file recovery, unsupported languages, saved realtime defaults, aligner residency, and explicit VRAM release
- All executions averaged 3.230 seconds end to end; excluding the first real aligner load and warmup, they averaged 2.232 seconds
- The first word-timestamp request took 47.127 seconds while the aligner loaded and fully warmed; the next two repetitions took 6.202 and 5.824 seconds

The unified URL-reference extension was validated separately on 2026-10-05 against the same Qwen 3.6 35B-A3B model and live 0.6B ASR deployment. All 3/3 executions selected `transcribe_audio_file`, supplied the exact HTTP URL through `file_location`, and returned the correct real transcription. They completed in 2.850, 1.585, and 1.557 seconds end to end.

The final `set_aligner(enabled)` contract was also checked against that model. All 3/3 standalone unload requests selected `enabled=false`, completing in 1.555, 1.207, and 1.161 seconds. A real load-then-unload round trip selected `enabled=true` followed by `enabled=false` and passed in 46.672 seconds, including the first aligner load and warmup. Runtime changes did not alter the user's saved startup preference.

## Contract decision

This recorded baseline predates URL input and exercised shared paths. The current contract keeps the same single `transcribe_audio_file` tool and renames its argument to `file_location`, which accepts either a confined local path or an HTTP(S) URL returned by another tool. There is still no base64 audio tool: only the compact location enters model context.

MCP is disabled by default and enabled through persistent System settings. The benchmark verified the live health, settings, and control tools after opt-in, then restored the canonical realtime defaults and left the aligner unloaded with persistent residency disabled.

The ignored machine-readable report is generated at `testbench/results/local-ai-mcp-latest.json` by the command documented in [`README.md`](README.md).
