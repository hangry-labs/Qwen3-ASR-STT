FROM python:3.13-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_ROOT_USER_ACTION=ignore \
    HF_HOME=/app/.cache/huggingface \
    VLLM_CACHE_ROOT=/app/.cache/vllm

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE VERSION requirements.txt /app/

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install --extra-index-url https://download.pytorch.org/whl/cu128 -r /app/requirements.txt

COPY qwen_asr /app/qwen_asr

RUN python -m pip install -e . --no-deps

FROM base AS baked-builder

ARG QWEN_ASR_PREFETCH_MODELS=Qwen/Qwen3-ASR-1.7B
ARG QWEN_ASR_PREFETCH_ALLOW_PATTERNS=
ARG QWEN_ASR_PREFETCH_ALIGNER=1
ENV QWEN_ASR_PREFETCH_MODELS=${QWEN_ASR_PREFETCH_MODELS} \
    QWEN_ASR_PREFETCH_ALLOW_PATTERNS=${QWEN_ASR_PREFETCH_ALLOW_PATTERNS} \
    QWEN_ASR_PREFETCH_ALIGNER=${QWEN_ASR_PREFETCH_ALIGNER}

RUN python -u -m qwen_asr.prefetch_assets

FROM python:3.13-slim AS runtime-base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_ROOT_USER_ACTION=ignore \
    HF_HOME=/app/.cache/huggingface \
    VLLM_CACHE_ROOT=/app/.cache/vllm \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TRANSFORMERS_VERBOSITY=error \
    QWEN_ASR_APP=demo \
    QWEN_ASR_BACKEND=vllm \
    QWEN_ASR_MODEL=Qwen/Qwen3-ASR-1.7B \
    QWEN_ASR_ALIGNER_MODEL=Qwen/Qwen3-ForcedAligner-0.6B \
    QWEN_ASR_ENABLE_ALIGNER=0 \
    QWEN_ASR_CONCURRENCY=2 \
    QWEN_ASR_GPU_MEMORY_UTILIZATION=0.53 \
    QWEN_ASR_MAX_MODEL_LEN=4096 \
    QWEN_ASR_MAX_INFERENCE_BATCH_SIZE=2 \
    QWEN_ASR_MAX_NEW_TOKENS=512 \
    QWEN_ASR_GENERATION_CONFIG=vllm \
    QWEN_ASR_LOAD_FORMAT=safetensors \
    QWEN_ASR_KV_CACHE_DTYPE=auto \
    QWEN_ASR_CALCULATE_KV_SCALES=0 \
    QWEN_ASR_ENFORCE_EAGER=0 \
    QWEN_ASR_PERFORMANCE_PROFILE=balanced \
    QWEN_ASR_COMPILATION_MODE= \
    QWEN_ASR_CUDAGRAPH_MODE=PIECEWISE \
    QWEN_ASR_CUDAGRAPH_CAPTURE_SIZES=1,2 \
    QWEN_ASR_MAX_CUDAGRAPH_CAPTURE_SIZE=2 \
    QWEN_ASR_STARTUP_WARMUP=0 \
    QWEN_ASR_STARTUP_WARMUP_TOKENS=1 \
    QWEN_ASR_TRACE_REQUESTS=0 \
    PORT=8000 \
    HOST=0.0.0.0

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

EXPOSE 8000

CMD ["python", "-u", "-m", "qwen_asr.docker_entrypoint"]

FROM runtime-base AS tiny

ENV HF_HUB_OFFLINE=0 \
    TRANSFORMERS_OFFLINE=0

COPY --from=base /usr/local /usr/local
COPY --from=base /app /app

FROM runtime-base AS baked

COPY --from=baked-builder /usr/local /usr/local
COPY --from=baked-builder /app /app
