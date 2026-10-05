# Local AI MCP Tool-Use Test

This local-only harness measures whether a smaller or quantized chat model can use the real Qwen3-ASR STT MCP tools correctly. It connects to the MCP Streamable HTTP endpoint, converts the discovered tool schemas to OpenAI-compatible function definitions, sends them to a local chat-completions endpoint, executes every returned call through MCP, and feeds real tool results or errors back to the model.

The cases cover full health discovery, shared file locations, language and prompt arguments, word and segment timestamps, path confinement, missing files, unsupported languages, unsupported alignment languages, persistent realtime defaults, current aligner load/unload control, and explicit retry recovery. The discovered `transcribe_audio_file` schema exposes one `file_location` argument for either a shared local path or an HTTP(S) URL; it never asks the model to carry audio bytes or base64.

Start Qwen3-ASR STT, place `english-01.mp3`, `japanese-01.mp3`, and `turkish-01.mp3` in its configured MCP input directory, then run:

```text
python testbench/local_ai_test/run.py --base-url http://127.0.0.1:18080/v1 --mcp-url http://127.0.0.1:8000/mcp --repeats 3
```

Enable MCP in the Qwen3-ASR STT System tab before running the harness. The model is discovered from `/v1/models` unless `--model` is supplied. The latest machine-readable report is written to `testbench/results/local-ai-mcp-latest.json`, which is intentionally ignored by Git because it can contain local endpoint details and model responses.

The URL-reference case is skipped unless `--audio-url` (or `LOCAL_AI_MCP_AUDIO_URL`) supplies a real audio URL reachable from the ASR container. To test a URL returned by a TTS service alongside all standard cases, append `--audio-url http://tts:8000/generated/example.wav`. To run only that contract check, also append `--case url_reference`.

This is an integration benchmark rather than a unit test. A failing case may indicate weak model tool use, unclear MCP descriptions, an unhelpful tool error, or an orchestration compatibility problem. Review each recorded invocation before changing the server contract.
