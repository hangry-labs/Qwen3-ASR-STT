# Qwen3-ASR Transcription Benchmarks

Append one row per transcription benchmark run.

This benchmark uses the random-only testbench corpus: 30 Qwen3-ASR-supported languages, 10 samples per language, 300 cases total. Each official run starts with 10 mandatory prewarm inferences; prewarm results and time are discarded. Official timing records the measured benchmark run only.

| Version | Model | Test time | Total score % | Bonus % | Total time | Arabic | Cantonese | Chinese | Czech | Danish | Dutch | English | Filipino | Finnish | French | German | Greek | Hindi | Hungarian | Indonesian | Italian | Japanese | Korean | Macedonian | Malay | Persian | Polish | Portuguese | Romanian | Russian | Spanish | Swedish | Thai | Turkish | Vietnamese |
| --- | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.1-snapshot | Qwen/Qwen3-ASR-0.6B | 15.05.2026 19:15:42 | 96.04% | 0.39% | 41.942s | 98.63% | 81.36% | 99.40% | 91.91% | 95.03% | 99.68% | 100.19% | 93.61% | 95.05% | 101.39% | 99.63% | 92.05% | 98.54% | 91.44% | 99.78% | 99.69% | 94.86% | 99.65% | 90.12% | 99.69% | 92.60% | 97.44% | 99.87% | 89.32% | 99.52% | 100.23% | 96.53% | 97.11% | 99.42% | 99.13% |
| 0.1-snapshot | Qwen/Qwen3-ASR-1.7B | 15.05.2026 19:22:15 | 97.64% | 0.40% | 84.751s | 99.85% | 78.47% | 99.20% | 97.75% | 98.97% | 99.88% | 100.19% | 97.66% | 97.62% | 100.93% | 99.87% | 95.47% | 99.81% | 98.76% | 99.86% | 100.20% | 96.97% | 99.38% | 92.68% | 99.46% | 97.68% | 98.25% | 99.73% | 97.13% | 99.99% | 99.83% | 99.28% | 98.30% | 99.55% | 98.57% |
