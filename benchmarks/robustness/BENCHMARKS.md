# Qwen3-ASR Audio Robustness Benchmarks

Deterministic regression results for silence, noise, echo, and weak speech.

## Baseline decision

The 0.6B baseline confirms the issue specifically when callers force a language: automatic detection rejected every generated non-speech fixture, while forced Chinese or English produced lexical output in 18 of 20 cases. Clean English and Chinese speech remained accurate at 1% source amplitude, so an RMS/volume cutoff would discard valid speech. Severe echo reduced transcription accuracy, while speech buried beneath stronger noise was missed; VAD cannot recover either case.

Keep model-native silence handling unchanged. VAD and energy gating are intentionally not added because silence duration can be meaningful to streaming, forced-alignment, and training-data workflows, while quiet speech demonstrates that amplitude alone is not a safe discriminator. Retain this matrix as an upgrade regression rather than filtering its failures heuristically.

| Version | Model | GPU | Comment | Test time | Passed | Auto non-speech false positives | Forced-language non-speech false positives | Speech passed | Mean speech score | Mean request |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.3.0-snapshot | Qwen/Qwen3-ASR-0.6B-hf | NVIDIA GeForce RTX 5070 Ti (16303 MiB) | Issue #165 model-native baseline | 03.10.2026 22:20:46 | 22/44 | 0/10 | 18/20 | 10/14 | 82.70% | 0.048s |
