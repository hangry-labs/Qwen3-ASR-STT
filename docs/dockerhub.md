<p align="center">
  <a href="https://hangry-labs.github.io/Qwen3-ASR-STT/">
    <img src="https://raw.githubusercontent.com/hangry-labs/Qwen3-ASR-STT/main/assets/qwen3_asr_logo_horizontal.webp" alt="Hangry Labs Qwen3-ASR-STT logo" width="900">
  </a>
</p>

<p align="center">
  <a href="https://hangry-labs.github.io/Qwen3-ASR-STT/?lang=en">English</a> ·
  <a href="https://hangry-labs.github.io/Qwen3-ASR-STT/README.nb.html?lang=nb">Norsk bokmål</a> ·
  <a href="https://hangry-labs.github.io/Qwen3-ASR-STT/README.pl.html?lang=pl">Polski</a> ·
  <a href="https://hangry-labs.github.io/Qwen3-ASR-STT/README.ja.html?lang=ja">日本語</a> ·
  <a href="https://hangry-labs.github.io/Qwen3-ASR-STT/README.zh.html?lang=zh">简体中文</a> ·
  <a href="https://hangry-labs.github.io/Qwen3-ASR-STT/README.es.html?lang=es">Español</a>
</p>

<p align="center"><a href="https://hangry-labs.github.io/Qwen3-ASR-STT/examples/?lang=en"><strong>🎧 Listen to transcription examples</strong></a></p>

# Hangry Labs Qwen3-ASR-STT

Local Qwen3-ASR speech to text with a browser Studio, OpenAI-compatible
transcription API, live microphone transcription and optional MCP tools in one
NVIDIA GPU Docker container. Audio stays on infrastructure you control.

## Quick Start

Run the full image:

```bash
docker run --name qwen3-asr-stt --restart unless-stopped -p 8000:8000 --gpus all -e CUDA_VISIBLE_DEVICES=0 -v qwen3_asr_stt_data:/app/persistent hangrylabs/qwen3-asr-stt:latest
```

Open **[http://localhost:8000](http://localhost:8000)**. The browser Studio and
HTTP APIs share the same port; interactive API documentation is at
[http://localhost:8000/docs](http://localhost:8000/docs).

All shell commands are one physical line for Bash, PowerShell and Windows
Command Prompt. Docker creates the named data volume automatically. For access
only from this machine, replace the port mapping with `-p 127.0.0.1:8000:8000`.

The full image bakes both Qwen3-ASR 0.6B and 1.7B checkpoints plus the optional
forced aligner. Default inference uses 0.6B, bf16 weights and the built-in vLLM
backend. After pulling, it can start and transcribe without a model download.
The volume preserves models, compiler caches and settings across replacements;
loaded GPU state is recreated at startup.

### Tiny image

Use the same Studio and APIs with model assets downloaded on first online use:

```bash
docker run --name qwen3-asr-stt --restart unless-stopped -p 8000:8000 --gpus all -e CUDA_VISIBLE_DEVICES=0 -e HF_HUB_OFFLINE=0 -e TRANSFORMERS_OFFLINE=0 -v qwen3_asr_stt_data:/app/persistent hangrylabs/qwen3-asr-stt:latest_tiny
```

Keep the data volume for later runs. The online flags allow missing assets to be
downloaded; after the required assets are cached, offline flags can be restored.
The full and tiny commands are alternative deployments using the same container
name and port.

## What You Get

<p align="center"><img src="https://raw.githubusercontent.com/hangry-labs/Qwen3-ASR-STT/main/assets/ui.webp" alt="Qwen3-ASR-STT Studio with transcription, waveform editing and audio controls"></p>

- **Transcribe:** upload or record audio, edit its waveform, trim, seek, adjust
  playback volume/speed, download audio and inspect transcript responses.
- **Stream:** live microphone transcription over a bounded rolling audio window,
  with configurable update intervals and transcript-stability controls.
- **API:** OpenAI-compatible file transcription, model/language discovery and
  interactive request documentation.
- **System:** inference readiness, model/build identity, GPU telemetry, saved
  realtime defaults and forced-aligner load/release/residency controls.
- **Timestamps and subtitles:** optional word/segment alignment and SRT/VTT
  output for supported alignment languages. The aligner loads on demand.
- **Languages:** transcription across 30 languages with model-native language
  identification. Studio languages are English, Polish, Japanese, Chinese,
  Spanish and German; the public website has six README translations.
- **MCP:** opt-in tools for health, transcription from shared paths or file URLs,
  aligner control and realtime settings, sharing the same loaded ASR model.

The [listening gallery](https://hangry-labs.github.io/Qwen3-ASR-STT/examples/)
pairs 30 existing test recordings with their reference transcripts. It supports
language filtering, playback, seeking, volume and Arabic/Persian right-to-left
content. These are recorded input examples, not live transcription results.

Browser microphone access requires `localhost` or HTTPS. LAN HTTP supports file
uploads and API calls; use a trusted certificate for remote microphone use.

## Runtime Settings

| Setting | Purpose |
| --- | --- |
| Port `8000` | Studio, REST APIs, health and optional MCP in one process. |
| `/app/persistent` | Models, vLLM/TorchInductor compiler caches, saved operator settings and the MCP input directory. |
| `QWEN_ASR_MODEL` | Select the 0.6B default or the baked 1.7B checkpoint. |
| `QWEN_ASR_BACKEND=vllm` | Accelerated production inference; Transformers remains an explicit diagnostic fallback. |
| System → forced aligner | Load, release or save an always-loaded preference for timestamp alignment. |
| `QWEN_ASR_MAX_UPLOAD_MB=100` | Default upload size limit. |
| System → realtime defaults | Save chunk interval, rolling-window duration and transcript-stability settings. |
| System → MCP connectivity | Enable `/mcp` on a trusted deployment; disabled by default. |
| `VLLM_USE_FLASHINFER_SAMPLER=0` | Default native deterministic sampler, avoiding optional sampler JIT startup failures. |

The local API does not authenticate the SDK's placeholder key. Use a trusted
network or an authenticating reverse proxy. MCP's enable switch and
DNS-rebinding guard are deployment safeguards, not user authentication.

## Image Tags

| Tag | Contents |
| --- | --- |
| `latest` | Moving full image with baked ASR and aligner assets. |
| `latest_tiny` | Moving runtime with assets downloaded into the data volume. |
| `v1.0` | Published full release; immutable deployment references are in the release history. |
| `v1.0_tiny` | Matching published tiny release. |

Both image variants are mirrored to
[GHCR](https://github.com/hangry-labs/Qwen3-ASR-STT/pkgs/container/qwen3-asr-stt).
Replace `hangrylabs/qwen3-asr-stt` with `ghcr.io/hangry-labs/qwen3-asr-stt` in any
command to use that registry. Moving tags stay unpinned; see the
[release history](https://hangry-labs.github.io/Qwen3-ASR-STT/#version-history)
for readable release tags pinned to their published top-level image digests.
Snapshot version tags are not published separately.

## API examples and OpenAI-compatible usage

<details markdown="1">

<summary><strong>Show API examples and guidance</strong></summary>

The stable integration endpoint is `POST /v1/audio/transcriptions`, tested with the current OpenAI Python client. Interactive request and response documentation is available at [http://localhost:8000/docs](http://localhost:8000/docs). Uploads are tested with `aac`, `flac`, `mp3`, `mp4`, `mpeg`, `mpga`, `m4a`, `ogg`, `wav`, and `webm`; other formats are passed to the bundled decoder. The configurable default limit is 100 MB. Because the deployment serves one loaded model, `model` may be omitted and that model is selected automatically.

Transcribe an audio file:

```bash
curl -X POST "http://localhost:8000/v1/audio/transcriptions" -F "file=@sample.mp3" -F "model=qwen3-asr" -F "response_format=json"
```

Request exact timestamps:

```bash
curl -X POST "http://localhost:8000/v1/audio/transcriptions" -F "file=@sample.mp3" -F "model=qwen3-asr" -F "response_format=verbose_json" -F "timestamp_granularities[]=word" -F "timestamp_granularities[]=segment"
```

`verbose_json` includes detected language, audio duration, and a coarse whole-audio segment without alignment. Requested word/segment timestamps use the forced aligner. `srt` and `vtt` automatically request accurate segment alignment instead of generating placeholder timings. Timestamp granularities require `response_format=verbose_json`.

Temperature is intentionally fixed at `0`: both backends use deterministic greedy decoding for stable transcription. This is a product choice rather than a claim that the underlying model could never be sampled.

Force a language when needed:

```bash
curl -X POST "http://localhost:8000/v1/audio/transcriptions" -F "file=@sample.mp3" -F "model=qwen3-asr" -F "language=English" -F "response_format=verbose_json"
```

When `language` is omitted, Qwen3-ASR performs model-native language identification. The Whisper component in the image is the Qwen audio feature extractor, not a separate language detector switch.

Use the OpenAI Python client:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="local")

with open("sample.mp3", "rb") as audio:
    result = client.audio.transcriptions.create(
        model="qwen3-asr",
        file=audio,
        response_format="json",
    )

print(result.text)
```

The local server does not authenticate the placeholder `api_key="local"`. Keep it on a trusted local/private network or add an authenticating reverse proxy. For loopback-only Docker access, publish with `-p 127.0.0.1:8000:8000`.

</details>

## MCP integration and endpoint reference

<details markdown="1">

<summary><strong>Show MCP documentation and endpoints</strong></summary>

Both image variants include opt-in stateless MCP Streamable HTTP at `http://localhost:8000/mcp`. Enable **MCP connectivity** in the System tab on a trusted deployment first. It runs inside the same process and shares the loaded model, inference queue, upload limit, GPU monitor, settings, and on-demand aligner with the UI and REST API.

`get_health` reports the deployment, inference metrics, aligner, realtime defaults, current GPU telemetry, storage, and shared directory. `transcribe_audio_file` accepts one `file_location`, which may be a path confined to `/app/persistent/mcp-input` by default or an HTTP(S) file URL. It also accepts optional language, prompt/context, and timestamps. `set_aligner(enabled)` controls only the current in-memory state: `true` loads and warms the aligner, while `false` unloads it and releases its VRAM without changing the user's saved startup preference. Realtime defaults can also be persisted. Make a local file available with:

```bash
docker cp sample.mp3 qwen3-asr-stt:/app/persistent/mcp-input/sample.mp3
```

The tool can use `file_location="sample.mp3"` or pass through a URL returned by another tool, such as `file_location="http://tts:8000/generated/example.wav"`. Only the location enters model context; audio bytes and base64 never do. Set `QWEN_ASR_MCP_INPUT_DIR` to another mounted container directory or to an empty value to disable local paths while retaining URLs. The default directory is created automatically for existing persistent volumes.

URL retrieval uses the same file-size limit, a bounded timeout and redirect count, validates every redirect, and rejects embedded credentials plus link-local, reserved, multicast, and unspecified destinations. It allows private/loopback addresses for local integrations. `QWEN_ASR_MCP_AUDIO_URL_ALLOWED_HOSTS` defaults to `*`; restrict it to exact comma-separated TTS hosts on shared deployments, or set it to an empty value to disable URL input. The URL must be reachable from inside the ASR container.

Localhost is accepted by the MCP endpoint's DNS-rebinding guard by default; add exact comma-separated LAN or reverse-proxy host values with `QWEN_ASR_MCP_ALLOWED_HOSTS`, and origins for clients that send an `Origin` header with `QWEN_ASR_MCP_ALLOWED_ORIGINS`. While disabled, MCP returns `403 Forbidden` because its use has not been authorized by the deployment owner. The enable switch prevents accidental exposure but is not an authentication system.

Useful routes:

- `GET /health` (readiness-compatible alias)
- `GET /health/live`
- `GET /health/ready`
- `GET /metrics/inference`
- `GET /system/gpu`
- `GET /system/aligner`
- `PUT /system/aligner`
- `POST /system/aligner/load`
- `POST /system/aligner/unload`
- `GET /system/settings`
- `PUT /system/settings/realtime`
- `GET /v1/models`
- `GET /v1/models/{model}`
- `POST /v1/audio/transcriptions`
- `GET /v1/audio/supported_languages`
- `POST /v1/realtime/transcriptions/sessions`
- `POST /v1/realtime/transcriptions/sessions/{session_id}/audio`
- `POST /v1/realtime/transcriptions/sessions/{session_id}/finish`
- `DELETE /v1/realtime/transcriptions/sessions/{session_id}`
- `POST /mcp` (MCP Streamable HTTP)

`stream=true` on the file endpoint returns OpenAI-compatible `transcript.text.delta` and `transcript.text.done` SSE events after completed-file inference. The realtime session API used by the Stream tab is different: it accepts audio progressively, repeatedly decodes a configurable recent-audio window, and retains older stable transcript text outside the prompt. This produces live updates while recording and prevents inference latency and context from growing without bound, but its HTTP session protocol is not the OpenAI Realtime WebSocket protocol.

</details>

## Advanced runtime configuration and operations

<details markdown="1">

<summary><strong>Show runtime settings and operational guidance</strong></summary>

Default model:

```text
Qwen/Qwen3-ASR-0.6B-hf
```

The full `latest` image also bakes `Qwen/Qwen3-ASR-1.7B-hf` and `Qwen/Qwen3-ForcedAligner-0.6B-hf`, so every supported model is available after pull without a Hugging Face download.

Run the larger 1.7B model:

```bash
docker run --name qwen3-asr-stt --restart unless-stopped -p 8000:8000 --gpus all -e CUDA_VISIBLE_DEVICES=0 -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e QWEN_ASR_MODEL=Qwen/Qwen3-ASR-1.7B-hf hangrylabs/qwen3-asr-stt:latest
```

Each container starts the browser UI and OpenAI-compatible API together on port 8000.

Selecting timestamps in the UI or requesting `timestamp_granularities` loads the forced aligner on demand. Its first compiled load increased observed VRAM use by approximately 3.3 GiB on an RTX 5070 Ti; releasing it immediately returned about 1.8 GiB, while CUDA/compiler context remained cached until restart. Exact behavior varies by GPU, driver, and runtime settings. The browser forces `verbose_json` while timestamps are selected and releases the aligner after both timestamp options remain unchecked for 60 seconds. System can release it immediately or keep it loaded. To use always-loaded behavior as the initial default when no saved preference exists:

```bash
docker run --name qwen3-asr-stt --restart unless-stopped -p 8000:8000 --gpus all -e CUDA_VISIBLE_DEVICES=0 -e QWEN_ASR_ENABLE_ALIGNER=1 -v qwen3_asr_stt_data:/app/persistent hangrylabs/qwen3-asr-stt:latest
```

The single Qwen aligner supports Chinese, English, Cantonese, French, German, Italian, Japanese, Korean, Portuguese, Russian, and Spanish. Other ASR languages, including Turkish, remain transcribable but cannot produce Qwen forced-alignment timestamps. Japanese alignment includes its required tokenizer. Korean transcription is supported, but Korean timestamp alignment is not packaged to avoid adding its GPLv3-only optional tokenizer to the Apache-2.0 image. Unsupported or unavailable forced-language requests return HTTP 422 before loading the aligner. The System tab also persists the default chunk interval, maximum audio window, and transcript-stability values for new realtime sessions.

Common knobs:

- `QWEN_ASR_MODEL=Qwen/Qwen3-ASR-0.6B-hf`
- `QWEN_ASR_BACKEND=vllm`
- `QWEN_ASR_ENABLE_ALIGNER=0|1`
- `QWEN_ASR_SETTINGS_PATH=/app/persistent/app/settings.json`
- `QWEN_ASR_ALIGNER_LOAD_TIMEOUT_SECONDS=600`
- `QWEN_ASR_MAX_NEW_TOKENS=512`
- `QWEN_ASR_MAX_UPLOAD_MB=100`
- `QWEN_ASR_GPU_MEMORY_UTILIZATION=0.25`
- `QWEN_ASR_MAX_MODEL_LEN=2048`
- `QWEN_ASR_MAX_NUM_BATCHED_TOKENS=2048`
- `QWEN_ASR_MAX_NUM_SEQS=2`
- `QWEN_ASR_VLLM_DTYPE=bfloat16`
- `VLLM_USE_FLASHINFER_SAMPLER=0`
- `QWEN_ASR_TRANSFORMERS_DTYPE=bfloat16`
- `QWEN_ASR_TRANSFORMERS_DEVICE_MAP=cuda:0`
- `QWEN_ASR_TORCH_COMPILE=1`
- `QWEN_ASR_TORCH_COMPILE_BACKEND=inductor`
- `QWEN_ASR_TORCH_COMPILE_MODE=default`
- `QWEN_ASR_TORCH_COMPILE_FULLGRAPH=0`
- `QWEN_ASR_STARTUP_WARMUP=1`
- `QWEN_ASR_STARTUP_WARMUP_TOKENS=512`
- `QWEN_ASR_STARTUP_WARMUP_ITERATIONS=3`
- `QWEN_ASR_INFERENCE_TIMEOUT_SECONDS=120`
- `QWEN_ASR_INFERENCE_QUEUE_TIMEOUT_SECONDS=120`
- `QWEN_ASR_REALTIME_SESSION_TTL_SECONDS=900`
- `QWEN_ASR_ENABLE_MCP=0`
- `QWEN_ASR_MCP_INPUT_DIR=/app/persistent/mcp-input`
- `QWEN_ASR_MCP_AUDIO_URL_ALLOWED_HOSTS=*`
- `QWEN_ASR_MCP_AUDIO_URL_TIMEOUT_SECONDS=60`
- `QWEN_ASR_MCP_AUDIO_URL_MAX_REDIRECTS=3`
- `QWEN_ASR_MCP_DNS_REBINDING_PROTECTION=1`
- `QWEN_ASR_MCP_ALLOWED_HOSTS=127.0.0.1:*,localhost:*,[::1]:*,host.docker.internal:*`
- `QWEN_ASR_MCP_ALLOWED_ORIGINS=http://127.0.0.1:*,http://localhost:*,https://127.0.0.1:*,https://localhost:*`
- `QWEN_ASR_WATCHDOG_ENABLED=1`
- `QWEN_ASR_WATCHDOG_INTERVAL_SECONDS=300`
- `QWEN_ASR_WATCHDOG_TIMEOUT_SECONDS=60`
- `QWEN_ASR_SSL_CERTFILE=/certs/fullchain.pem`
- `QWEN_ASR_SSL_KEYFILE=/certs/privkey.pem`

The System tab persists `mcp_enabled`, `load_aligner_always`, and the realtime `chunk_size_sec`, `max_window_sec`, `unfixed_chunk_num`, and `unfixed_token_num` defaults in `/app/persistent/app/settings.json`. The file can also be edited while the container is stopped. `max_window_sec` accepts 10–60 seconds and defaults to 30. Restart-bound model, backend, GPU, compiler, and health-policy controls remain environment variables.

Startup warmup intentionally makes `/health` wait until vLLM compilation, CUDA graph capture, and three representative decode passes have stabilized the normal generation path. The first API transcription after readiness therefore does not pay lazy initialization cost.

The image uses vLLM's native sampler by default because transcription decoding is deterministic and does not benefit from FlashInfer's optional top-k/top-p sampling JIT. This also avoids startup failures on NAS and alternative NVIDIA container runtimes that provide the versioned CUDA driver library but not its unversioned linker name. Set `VLLM_USE_FLASHINFER_SAMPLER=1` only to opt into that optional JIT path explicitly. For an older image exhibiting `Ninja build failed` followed by `cannot find -lcuda`, add `VLLM_USE_FLASHINFER_SAMPLER=0` to the container environment.

All API, realtime, UI, warmup, and watchdog inference shares one synchronous model owner. If inference exceeds its deadline or generation fails fatally, `/health` and `/health/ready` change to HTTP 503 and the process exits after recording diagnostics. Keep the documented `--restart unless-stopped` policy, or equivalent Kubernetes/systemd supervision, so Docker reloads a clean model automatically. `/health/live` only confirms that the HTTP process is alive.

The single `qwen3_asr_stt_data` volume stores model assets, vLLM and TorchInductor compiler caches, and operator settings under `/app/persistent`. Reuse that name with later image tags to avoid repeated tiny-image downloads and retain settings. Docker initializes a new named volume from the full image's baked contents automatically. Loaded GPU weights and live CUDA graph state are not persisted, so startup profiling and warmup still run.

Existing users can migrate the legacy cache/settings volumes once without downloading models again:

```bash
docker run --rm --entrypoint sh -v qwen3_asr_stt_hf_cache:/legacy/huggingface:ro -v qwen3_asr_stt_torch_compile_cache:/legacy/torchinductor:ro -v qwen3_asr_stt_vllm_cache:/legacy/vllm:ro -v qwen3_asr_stt_settings:/legacy/settings:ro -v qwen3_asr_stt_data:/app/persistent hangrylabs/qwen3-asr-stt:latest_tiny -c "mkdir -p /app/persistent/models/huggingface /app/persistent/cache/torchinductor /app/persistent/cache/vllm /app/persistent/app && cp -an /legacy/huggingface/. /app/persistent/models/huggingface/ && cp -an /legacy/torchinductor/. /app/persistent/cache/torchinductor/ && cp -an /legacy/vllm/. /app/persistent/cache/vllm/ && cp -an /legacy/settings/. /app/persistent/app/"
```

The old volumes remain untouched; delete them only after the new container is healthy.

To use browser microphone recording from another machine, mount a trusted certificate and start the server with HTTPS:

```bash
docker run --name qwen3-asr-stt --restart unless-stopped -p 8000:8000 --gpus all -e CUDA_VISIBLE_DEVICES=0 -e QWEN_ASR_SSL_CERTFILE=/certs/fullchain.pem -e QWEN_ASR_SSL_KEYFILE=/certs/privkey.pem -v /absolute/path/to/certs:/certs:ro hangrylabs/qwen3-asr-stt:latest
```

</details>

## Responsible Use and Privacy

Speech can contain private, identifying, or sensitive information. This image is designed so audio can be processed locally or inside infrastructure you control.

Only transcribe audio you are allowed to process. Do not use this image for covert recording, surveillance, harassment, fraud, or other illegal or unethical activity.

## Benchmarks

The repository publishes [multilingual transcription measurements](https://hangry-labs.github.io/Qwen3-ASR-STT/benchmarks/transcription/BENCHMARKS.html),
[audio robustness results](https://hangry-labs.github.io/Qwen3-ASR-STT/benchmarks/robustness/BENCHMARKS.html)
and [VRAM profiles](https://hangry-labs.github.io/Qwen3-ASR-STT/benchmarks/vram/model_vram.html).
Reports record their model, hardware, corpus and settings; do not compare runs
with different profiles as equivalent measurements.

## Project Links

- [Project website and documentation](https://hangry-labs.github.io/Qwen3-ASR-STT/)
- [Listening gallery](https://hangry-labs.github.io/Qwen3-ASR-STT/examples/)
- [Examples and fixture guide](https://hangry-labs.github.io/Qwen3-ASR-STT/examples/guide/)

- GitHub repository: https://github.com/Hangry-Labs/Qwen3-ASR-STT
- Upstream Qwen3-ASR repository: https://github.com/QwenLM/Qwen3-ASR
- Upstream model collection: https://huggingface.co/collections/Qwen/qwen3-asr
- Hangry Labs: https://hangrylabs.app/

## Attribution

This is an independently maintained Hangry Labs packaging and serving fork of the upstream Qwen3-ASR project.

The upstream Qwen team provides the model architecture, weights, and research. Hangry Labs maintains this Docker-first runtime packaging, local UI/API integration, benchmarking, documentation, and release workflow.

License and attribution are preserved in the repository [LICENSE](https://github.com/hangry-labs/Qwen3-ASR-STT/blob/main/LICENSE) and [third-party notices](https://hangry-labs.github.io/Qwen3-ASR-STT/THIRD_PARTY_NOTICES.html).
