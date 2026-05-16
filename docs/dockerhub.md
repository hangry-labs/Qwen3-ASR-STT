<p>
  <img src="https://github.com/Hangry-Labs/Qwen3-ASR-STT/raw/main/hangrylabs/banner.jpg" alt="Hangry Labs Qwen3-ASR-STT banner">
</p>

# Hangry Labs Qwen3-ASR-STT

Easy-to-run local speech-to-text Docker images for Qwen3-ASR, with a browser UI and OpenAI-compatible transcription API included.

This Hangry Labs image is built for private local inference. Run the container, open the UI, upload or record audio, or point an OpenAI-compatible client at the local transcription endpoint.

## What You Get

- Browser UI for file upload, recording, realtime microphone transcription, API status, and GPU visibility
- OpenAI-compatible `POST /v1/audio/transcriptions`
- Local realtime transcription session endpoints used by the UI
- Qwen3-ASR 1.7B default model
- Optional Qwen3-ASR 0.6B runtime configuration
- Optional forced aligner support for timestamp output
- Python 3.13 runtime with locked dependencies
- vLLM backend by default
- Offline-friendly full image after pull
- Tiny image variant for persistent model-cache workflows

## Browser UI

Open the local UI at:

```text
http://localhost:8000
```

<p>
  <img src="https://github.com/Hangry-Labs/Qwen3-ASR-STT/raw/main/docs/ui.jpg" alt="Qwen3-ASR-STT browser UI">
</p>

The UI supports normal transcription and realtime microphone transcription. It also includes a GPU card so local VRAM and utilization are visible while testing.

## Quick Start

Run with NVIDIA GPU support:

```bash
docker run --rm -p 8000:8000 --gpus all -e CUDA_VISIBLE_DEVICES=0 hangrylabs/qwen3-asr-stt:latest
```

Then open:

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
docker run --rm -p 8000:8000 --gpus all \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e HF_HUB_OFFLINE=0 \
  -e TRANSFORMERS_OFFLINE=0 \
  -v qwen3_asr_stt_hf_cache:/app/.cache/huggingface \
  -v qwen3_asr_stt_vllm_cache:/app/.cache/vllm \
  hangrylabs/qwen3-asr-stt:latest_tiny
```

The tiny image downloads model assets on first online use, then reuses the mounted cache volume.

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

- `GET /health`
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
Qwen/Qwen3-ASR-1.7B
```

Run the smaller 0.6B model:

```bash
docker run --rm -p 8000:8000 --gpus all \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e QWEN_ASR_MODEL=Qwen/Qwen3-ASR-0.6B \
  -e QWEN_ASR_GPU_MEMORY_UTILIZATION=0.38 \
  hangrylabs/qwen3-asr-stt:latest
```

Run API-only mode:

```bash
docker run --rm -p 8000:8000 --gpus all \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e QWEN_ASR_APP=api \
  hangrylabs/qwen3-asr-stt:latest
```

Enable timestamp output through the forced aligner:

```bash
docker run --rm -p 8000:8000 --gpus all \
  -e CUDA_VISIBLE_DEVICES=0 \
  -e QWEN_ASR_ENABLE_ALIGNER=1 \
  hangrylabs/qwen3-asr-stt:latest
```

Common knobs:

- `QWEN_ASR_APP=demo|api`
- `QWEN_ASR_MODEL=Qwen/Qwen3-ASR-1.7B`
- `QWEN_ASR_ENABLE_ALIGNER=0|1`
- `QWEN_ASR_GPU_MEMORY_UTILIZATION=0.53`
- `QWEN_ASR_MAX_MODEL_LEN=4096`
- `QWEN_ASR_MAX_NEW_TOKENS=512`
- `QWEN_ASR_PERFORMANCE_PROFILE=balanced|throughput|custom`

Decoding temperature is fixed at `0` for deterministic transcription.

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
