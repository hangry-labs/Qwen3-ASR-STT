FROM python:3.13-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_ROOT_USER_ACTION=ignore \
    HF_HOME=/app/.cache/huggingface \
    TORCHINDUCTOR_CACHE_DIR=/app/.cache/torchinductor \
    VLLM_CACHE_ROOT=/app/.cache/vllm \
    CUDA_HOME=/usr/local/lib/python3.13/site-packages/nvidia/cu13

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install --extra-index-url https://download.pytorch.org/whl/cu130 -r /app/requirements.txt

RUN ln -sfn lib /usr/local/lib/python3.13/site-packages/nvidia/cu13/lib64 \
    && ln -sfn libcudart.so.13 /usr/local/lib/python3.13/site-packages/nvidia/cu13/lib/libcudart.so

COPY pyproject.toml README.md LICENSE VERSION /app/
COPY qwen_asr /app/qwen_asr
COPY hangrylabs /app/hangrylabs
COPY testbench /app/testbench

RUN python -m pip install -e . --no-deps

FROM base AS baked-builder

ARG QWEN_ASR_PREFETCH_MODELS=Qwen/Qwen3-ASR-0.6B-hf,Qwen/Qwen3-ASR-1.7B-hf
ARG QWEN_ASR_PREFETCH_ALLOW_PATTERNS=
ARG QWEN_ASR_PREFETCH_ALIGNER=1
ARG QWEN_ASR_ALIGNER_MODEL=Qwen/Qwen3-ForcedAligner-0.6B-hf
ARG QWEN_ASR_REQUIRED_MODELS=Qwen/Qwen3-ASR-0.6B-hf,Qwen/Qwen3-ASR-1.7B-hf,Qwen/Qwen3-ForcedAligner-0.6B-hf
ENV QWEN_ASR_PREFETCH_MODELS=${QWEN_ASR_PREFETCH_MODELS} \
    QWEN_ASR_PREFETCH_ALLOW_PATTERNS=${QWEN_ASR_PREFETCH_ALLOW_PATTERNS} \
    QWEN_ASR_PREFETCH_ALIGNER=${QWEN_ASR_PREFETCH_ALIGNER} \
    QWEN_ASR_ALIGNER_MODEL=${QWEN_ASR_ALIGNER_MODEL} \
    QWEN_ASR_REQUIRED_MODELS=${QWEN_ASR_REQUIRED_MODELS}

RUN python -u -m qwen_asr.prefetch_assets \
    && python -c "import os, pathlib, sys; root=pathlib.Path(os.getenv('HF_HOME','/app/.cache/huggingface'))/'hub'; required=[model.strip() for model in os.getenv('QWEN_ASR_REQUIRED_MODELS','').replace(';', ',').split(',') if model.strip()]; missing=[model for model in required if not any((root / ('models--' + model.replace('/', '--')) / 'snapshots').glob('*'))]; print('Validated baked ASR model assets:', ', '.join(required) if required else '(none)'); print('Missing baked ASR model assets:', ', '.join(missing), file=sys.stderr) if missing else None; sys.exit(1 if missing else 0)"

FROM python:3.13-slim AS runtime-base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_ROOT_USER_ACTION=ignore \
    HF_HOME=/app/.cache/huggingface \
    TORCHINDUCTOR_CACHE_DIR=/app/.cache/torchinductor \
    VLLM_CACHE_ROOT=/app/.cache/vllm \
    FLASHINFER_WORKSPACE_BASE=/app/.cache/vllm \
    CUDA_HOME=/usr/local/lib/python3.13/site-packages/nvidia/cu13 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TRANSFORMERS_VERBOSITY=error \
    VLLM_USE_V2_MODEL_RUNNER=0 \
    QWEN_ASR_BACKEND=vllm \
    QWEN_ASR_MODEL=Qwen/Qwen3-ASR-0.6B-hf \
    QWEN_ASR_ALIGNER_MODEL=Qwen/Qwen3-ForcedAligner-0.6B-hf \
    QWEN_ASR_ENABLE_ALIGNER=0 \
    QWEN_ASR_CONCURRENCY=2 \
    QWEN_ASR_GPU_MEMORY_UTILIZATION=0.25 \
    QWEN_ASR_MAX_MODEL_LEN=2048 \
    QWEN_ASR_MAX_NUM_BATCHED_TOKENS=2048 \
    QWEN_ASR_MAX_NUM_SEQS=2 \
    QWEN_ASR_MAX_INFERENCE_BATCH_SIZE=2 \
    QWEN_ASR_MAX_NEW_TOKENS=512 \
    QWEN_ASR_VLLM_DTYPE=bfloat16 \
    QWEN_ASR_GENERATION_CONFIG=vllm \
    QWEN_ASR_TRANSFORMERS_DTYPE=bfloat16 \
    QWEN_ASR_TRANSFORMERS_DEVICE_MAP=cuda:0 \
    QWEN_ASR_TORCH_COMPILE=1 \
    QWEN_ASR_TORCH_COMPILE_BACKEND=inductor \
    QWEN_ASR_TORCH_COMPILE_MODE=default \
    QWEN_ASR_TORCH_COMPILE_FULLGRAPH=0 \
    QWEN_ASR_STARTUP_WARMUP=1 \
    QWEN_ASR_STARTUP_WARMUP_TOKENS=512 \
    QWEN_ASR_STARTUP_WARMUP_ITERATIONS=3 \
    QWEN_ASR_STARTUP_WARMUP_AUDIO=/app/testbench/assets/english/random/01.mp3 \
    QWEN_ASR_INFERENCE_TIMEOUT_SECONDS=120 \
    QWEN_ASR_INFERENCE_QUEUE_TIMEOUT_SECONDS=120 \
    QWEN_ASR_REALTIME_SESSION_TTL_SECONDS=900 \
    QWEN_ASR_RECYCLE_DELAY_SECONDS=2 \
    QWEN_ASR_WATCHDOG_ENABLED=1 \
    QWEN_ASR_WATCHDOG_INTERVAL_SECONDS=300 \
    QWEN_ASR_WATCHDOG_TIMEOUT_SECONDS=60 \
    QWEN_ASR_TRACE_REQUESTS=0 \
    PORT=8000 \
    HOST=0.0.0.0

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=300s --retries=2 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/ready', timeout=4).read()"]

CMD ["python", "-u", "-m", "qwen_asr.docker_entrypoint"]

FROM runtime-base AS tiny

ENV HF_HUB_OFFLINE=0 \
    TRANSFORMERS_OFFLINE=0

COPY --from=base /usr/local /usr/local
COPY --from=base /app /app

FROM runtime-base AS baked

COPY --from=baked-builder /usr/local /usr/local
COPY --from=baked-builder /app /app
