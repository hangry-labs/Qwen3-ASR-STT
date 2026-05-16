# Qwen3-ASR VRAM Benchmark Log

This document records measured VRAM for Qwen3-ASR Docker inference profiles.
Append one row per successful comparable test so model/runtime choices can be compared directly.

## Test Machine

| Field | Value |
| --- | --- |
| GPU used for tests | NVIDIA GeForce RTX 5060 Ti |
| GPU index | `1` |
| Total VRAM reported by `nvidia-smi` | `16,311 MiB` |
| Host OS context | Windows + Docker Desktop / WSL CUDA runtime |
| Measurement source | `nvidia-smi --query-gpu=index,name,memory.used,memory.total` after server startup |
| Startup success check | `curl.exe -I http://localhost:8000/` or benchmark container equivalent |

## Notes

- The project is inference-only. Training/fine-tuning measurements are out of scope.
- Runtime ASR/translation sampling should remain deterministic: vLLM `SamplingParams(temperature=0.0, max_tokens=...)`.
- All rows in this document must use the same target serving profile unless a new section explicitly says otherwise:
  - `QWEN_ASR_MAX_MODEL_LEN=4096`
  - `QWEN_ASR_CONCURRENCY=2`
  - `QWEN_ASR_MAX_INFERENCE_BATCH_SIZE=2`
  - `QWEN_ASR_MAX_NEW_TOKENS=1024`
- Tune `QWEN_ASR_GPU_MEMORY_UTILIZATION` per model so vLLM reports about `2x` maximum concurrency for 4096-token requests.
- Keep failed, partial, incompatible, and not-yet-measured experiments out of this public comparison table. Record that context in private project notes or issue notes instead.
- KV cache quantization is tracked separately from model weight quantization. vLLM `kv_cache_dtype=auto` means unquantized default cache dtype for the loaded model.
- `QWEN_ASR_ENABLE_ALIGNER=0` means the forced aligner may be baked or cached on disk, but it is not loaded into runtime VRAM.
- GGUF rows should be added only after a GGUF runtime path can run the model as an active service with the same target profile.

## 1.7B Models

| Date | Model | Runtime/backend | Format/weight quantization | KV cache dtype | Calculate KV scales | GPU | VRAM used after startup | Model weight memory | KV cache memory | KV cache tokens | Reported max concurrency | `QWEN_ASR_MAX_MODEL_LEN` | `QWEN_ASR_GPU_MEMORY_UTILIZATION` | `QWEN_ASR_CONCURRENCY` | `QWEN_ASR_MAX_INFERENCE_BATCH_SIZE` | `QWEN_ASR_MAX_NEW_TOKENS` | Aligner loaded | Result / notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-05-15 | `Qwen/Qwen3-ASR-1.7B` | vLLM | safetensors / none | `auto` | `0` | RTX 5060 Ti 16GB | `7,060 MiB` | `3.87 GiB` | `0.94 GiB` | `8,768` | `2.14x` at 4096 tokens/request | `4096` | `0.53` | `2` | `2` | `1024` | no | Adjusted target profile for about two concurrent requests without excess KV cache. Engine init `157.81s`. |

## 0.6B Models

| Date | Model | Runtime/backend | Format/weight quantization | KV cache dtype | Calculate KV scales | GPU | VRAM used after startup | Model weight memory | KV cache memory | KV cache tokens | Reported max concurrency | `QWEN_ASR_MAX_MODEL_LEN` | `QWEN_ASR_GPU_MEMORY_UTILIZATION` | `QWEN_ASR_CONCURRENCY` | `QWEN_ASR_MAX_INFERENCE_BATCH_SIZE` | `QWEN_ASR_MAX_NEW_TOKENS` | Aligner loaded | Result / notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-05-15 | `Qwen/Qwen3-ASR-0.6B` | vLLM | safetensors / none | `auto` | `0` | RTX 5060 Ti 16GB | `4,620 MiB` | `1.53 GiB` | `0.89 GiB` | `8,304` | `2.03x` at 4096 tokens/request | `4096` | `0.38` | `2` | `2` | `1024` | no | Adjusted target profile for about two concurrent requests without excess KV cache. Engine init `156.61s`. |
