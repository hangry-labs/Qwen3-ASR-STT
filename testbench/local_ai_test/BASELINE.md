# Qwen 3.6 35B-A3B MCP Baseline

Tested on 2026-10-04 with the local `qwen36-35b-hermes:current` Q4_K_M model through its OpenAI-compatible chat-completions endpoint. The target was the live Qwen3-ASR STT MCP Streamable HTTP endpoint backed by the real 0.6B ASR model, optional forced aligner, local GPU telemetry, and persistent runtime settings.

## Result

- 15 cases x 3 repeats = 45 executions
- 45 passed, 0 failed: 100% overall
- Every tool selection and JSON argument was correct for full health/GPU discovery, path-only transcription, automatic and forced language, prompt/context, word and segment timestamps, path confinement, missing-file recovery, unsupported languages, saved realtime defaults, aligner residency, and explicit VRAM release
- All executions averaged 3.230 seconds end to end; excluding the first real aligner load and warmup, they averaged 2.232 seconds
- The first word-timestamp request took 47.127 seconds while the aligner loaded and fully warmed; the next two repetitions took 6.202 and 5.824 seconds

## Contract decision

Only path-referenced audio is exposed to MCP. The model receives a small filename or relative path, while the audio remains in the configured shared input directory. There is no base64 audio tool because opaque audio data would become model context, consume large token budgets, and make tool-call generation unreliable.

MCP is disabled by default and enabled through persistent System settings. The benchmark verified the live health, settings, and control tools after opt-in, then restored the canonical realtime defaults and left the aligner unloaded with persistent residency disabled.

The ignored machine-readable report is generated at `testbench/results/local-ai-mcp-latest.json` by the command documented in [`README.md`](README.md).
