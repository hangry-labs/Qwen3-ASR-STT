<p>
  <img src="https://github.com/Hangry-Labs/Qwen3-ASR-STT/raw/main/logo.jpg" alt="Hangry Labs Qwen3-ASR-STT banner">
</p>

# Hangry Labs Qwen3-ASR-STT

Easy-to-run local speech-to-text Docker images for Qwen3-ASR, with a browser UI and OpenAI-compatible transcription API included.

This Hangry Labs image is built for private local inference. Run the container, open the UI, upload or record audio, or point an OpenAI-compatible client at the local transcription endpoint.

## What You Get

- Browser UI for file upload, recording, realtime microphone transcription, API status, and GPU visibility
- OpenAI-compatible `POST /v1/audio/transcriptions`
- Local realtime transcription session endpoints used by the UI
- Qwen3-ASR 0.6B default model
- Optional Qwen3-ASR 1.7B runtime configuration
- Optional forced aligner support for timestamp output
- Python 3.13 runtime with locked dependencies
- Built-in vLLM 0.26 backend with CUDA graph acceleration
- Transformers 5 diagnostic fallback and native forced aligner
- Offline-friendly full image after pull
- Tiny image variant for persistent model-cache workflows
- Full image includes both `-hf` ASR checkpoints and the forced aligner

## Browser UI

Open the primary browser UI at:

```text
http://localhost:8000
```

<p>
  <img src="https://github.com/Hangry-Labs/Qwen3-ASR-STT/raw/main/docs/ui.jpg" alt="Qwen3-ASR-STT browser UI">
</p>

The responsive UI provides file upload and browser recording, a replaceable waveform editor with playback and trimming controls, bundled multilingual examples, realtime microphone transcription, API status, and GPU monitoring. Model readiness and the UI build version are visible in the header, while timestamp controls report immediately when the optional forced aligner is disabled.

Remote file upload and API calls work over normal LAN HTTP when the port is exposed. Browser microphone recording requires a secure browser origin, so use `localhost` or serve the UI over HTTPS when opening it from another machine.

## Quick Start

Run with NVIDIA GPU support:

```bash
docker volume create qwen3_asr_stt_torch_compile_cache
docker volume create qwen3_asr_stt_vllm_cache
docker run --name qwen3-asr-stt --restart unless-stopped -p 8000:8000 --gpus all \
  -e CUDA_VISIBLE_DEVICES=0 \
  -v qwen3_asr_stt_torch_compile_cache:/app/.cache/torchinductor \
  -v qwen3_asr_stt_vllm_cache:/app/.cache/vllm \
  hangrylabs/qwen3-asr-stt:latest
```

Then open the UI:

```text
http://localhost:8000
```

Health check:

```bash
curl http://localhost:8000/health
```

API docs:

```text
http://localhost:8000/docs
```

## Tiny Image

Use `latest_tiny` when you want runtime dependencies but prefer model assets to live in a persistent cache volume:

```bash
docker volume create qwen3_asr_stt_hf_cache
docker volume create qwen3_asr_stt_torch_compile_cache
docker volume create qwen3_asr_stt_vllm_cache
docker run --name qwen3-asr-stt --restart unless-stopped -p 8000:8000 --gpus all \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e HF_HUB_OFFLINE=0 \
  -e TRANSFORMERS_OFFLINE=0 \
  -v qwen3_asr_stt_hf_cache:/app/.cache/huggingface \
  -v qwen3_asr_stt_torch_compile_cache:/app/.cache/torchinductor \
  -v qwen3_asr_stt_vllm_cache:/app/.cache/vllm \
  hangrylabs/qwen3-asr-stt:latest_tiny
```

The tiny image downloads model assets on first online use, then reuses the mounted cache volumes.

## Image Tags

- `latest` - full baked image
- `latest_tiny` - tiny image with persistent-cache workflow
- `vX.Y.Z` - full release image, for example `v0.1.0`
- `vX.Y.Z_tiny` - tiny release image, for example `v0.1.0_tiny`

Snapshot tags are not published.

## API Example

Transcribe an audio file:

```bash
curl -X POST "http://localhost:8000/v1/audio/transcriptions" \
  -F "file=@sample.mp3" \
  -F "model=qwen3-asr" \
  -F "response_format=json"
```

Force a language when needed:

```bash
curl -X POST "http://localhost:8000/v1/audio/transcriptions" \
  -F "file=@sample.mp3" \
  -F "model=qwen3-asr" \
  -F "language=English" \
  -F "response_format=verbose_json"
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

Useful routes:

- `GET /health` (readiness-compatible alias)
- `GET /health/live`
- `GET /health/ready`
- `GET /metrics/inference`
- `GET /v1/models`
- `GET /v1/models/{model}`
- `POST /v1/audio/transcriptions`
- `GET /v1/audio/supported_languages`
- `POST /v1/realtime/transcriptions/sessions`
- `POST /v1/realtime/transcriptions/sessions/{session_id}/audio`
- `POST /v1/realtime/transcriptions/sessions/{session_id}/finish`
- `DELETE /v1/realtime/transcriptions/sessions/{session_id}`

The realtime session API is local and experimental. It is used by the browser UI, but it is not a full OpenAI Realtime WebSocket implementation.

## Runtime Configuration

Default model:

```text
Qwen/Qwen3-ASR-0.6B-hf
```

The full `latest` image also bakes `Qwen/Qwen3-ASR-1.7B-hf` and `Qwen/Qwen3-ForcedAligner-0.6B-hf`, so every supported model is available after pull without a Hugging Face download.

Run the larger 1.7B model:

```bash
docker run --name qwen3-asr-stt --restart unless-stopped -p 8000:8000 --gpus all \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -e QWEN_ASR_MODEL=Qwen/Qwen3-ASR-1.7B-hf \
  hangrylabs/qwen3-asr-stt:latest
```

Each container starts the browser UI and OpenAI-compatible API together on port 8000.

Enable timestamp output through the forced aligner:

```bash
docker run --name qwen3-asr-stt --restart unless-stopped -p 8000:8000 --gpus all \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e QWEN_ASR_ENABLE_ALIGNER=1 \
  hangrylabs/qwen3-asr-stt:latest
```

Common knobs:

- `QWEN_ASR_MODEL=Qwen/Qwen3-ASR-0.6B-hf`
- `QWEN_ASR_BACKEND=vllm`
- `QWEN_ASR_ENABLE_ALIGNER=0|1`
- `QWEN_ASR_MAX_NEW_TOKENS=512`
- `QWEN_ASR_GPU_MEMORY_UTILIZATION=0.25`
- `QWEN_ASR_MAX_MODEL_LEN=2048`
- `QWEN_ASR_MAX_NUM_BATCHED_TOKENS=2048`
- `QWEN_ASR_MAX_NUM_SEQS=2`
- `QWEN_ASR_VLLM_DTYPE=bfloat16`
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
- `QWEN_ASR_WATCHDOG_ENABLED=1`
- `QWEN_ASR_WATCHDOG_INTERVAL_SECONDS=300`
- `QWEN_ASR_WATCHDOG_TIMEOUT_SECONDS=60`
- `QWEN_ASR_SSL_CERTFILE=/certs/fullchain.pem`
- `QWEN_ASR_SSL_KEYFILE=/certs/privkey.pem`

Startup warmup intentionally makes `/health` wait until vLLM compilation, CUDA graph capture, and three representative decode passes have stabilized the normal generation path. The first API transcription after readiness therefore does not pay lazy initialization cost.

All API, realtime, UI, warmup, and watchdog inference shares one synchronous model owner. If inference exceeds its deadline or generation fails fatally, `/health` and `/health/ready` change to HTTP 503 and the process exits after recording diagnostics. Keep the documented `--restart unless-stopped` policy, or equivalent Kubernetes/systemd supervision, so Docker reloads a clean model automatically. `/health/live` only confirms that the HTTP process is alive.

Persistent Hugging Face cache volumes are useful for the tiny image. Mount `/app/.cache/vllm` to reuse hardware-specific AOT graphs, TorchInductor output, and FlashInfer JIT kernels; `/app/.cache/torchinductor` serves the optional aligner and Transformers fallback. These caches do not preserve loaded GPU weights or live CUDA graph state, and startup profiling/warmup still runs. The full image needs no model cache volume because all supported model assets are baked in.

If you use a persistent Hugging Face cache volume with the full image, seed it from the baked image for offline deployments:

```bash
docker volume create qwen3_asr_stt_hf_cache
docker run --rm \
  --entrypoint sh \
  -v qwen3_asr_stt_hf_cache:/hf-cache \
  hangrylabs/qwen3-asr-stt:latest \
  -c "test -d /app/.cache/huggingface/hub && mkdir -p /hf-cache && cp -an /app/.cache/huggingface/. /hf-cache/"
```

To use browser microphone recording from another machine, mount a trusted certificate and start the server with HTTPS:

```bash
docker run --name qwen3-asr-stt --restart unless-stopped -p 8000:8000 --gpus all \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e QWEN_ASR_SSL_CERTFILE=/certs/fullchain.pem \
  -e QWEN_ASR_SSL_KEYFILE=/certs/privkey.pem \
  -v /path/to/certs:/certs:ro \
  hangrylabs/qwen3-asr-stt:latest
```

## Responsible Use and Privacy

Speech can contain private, identifying, or sensitive information. This image is designed so audio can be processed locally or inside infrastructure you control.

Only transcribe audio you are allowed to process. Do not use this image for covert recording, surveillance, harassment, fraud, or other illegal or unethical activity.

## Project Links

- GitHub repository: https://github.com/Hangry-Labs/Qwen3-ASR-STT
- Upstream Qwen3-ASR repository: https://github.com/QwenLM/Qwen3-ASR
- Upstream model collection: https://huggingface.co/collections/Qwen/qwen3-asr
- Hangry Labs: https://nuggies.website/

## Attribution

This is an independently maintained Hangry Labs packaging and serving fork of the upstream Qwen3-ASR project.

The upstream Qwen team provides the model architecture, weights, and research. Hangry Labs maintains this Docker-first runtime packaging, local UI/API integration, benchmarking, documentation, and release workflow.

License and attribution are preserved in the repository.
