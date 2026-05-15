# Qwen3-ASR Quality Testbench

This folder contains audio fixtures and expected transcripts for checking Qwen3-ASR inference quality through the OpenAI-compatible transcription API.

## Source

Static assets are copied from:

```text
external fixture corpus assets
```

The generated manifest is:

```text
testbench/manifest.json
```

Generated assets for Qwen-supported languages missing from the static fixture corpus are created through a local fixture generation API.

Each case has:

- `audio`: local fixture path under `testbench/`
- `language`: Qwen3-ASR canonical language name
- `expected_text`: expected transcript from the fixture manifest or the generated fixture text
- `source_file`: original fixture asset path, or `null` for generated fixtures

## Coverage

The current fixture set is random-only. It intentionally excludes intro samples so benchmark scoring is not dominated by foreign product names or project branding.

Coverage is 30 Qwen3-ASR-supported languages with 10 random samples per language, for 300 total cases.

Qwen3-ASR-supported languages generated through the local fixture generation API:

```text
Cantonese, Czech, Danish, Filipino, Finnish, Greek, Hungarian, Macedonian, Malay, Persian, Romanian, Swedish
```

There are currently no Qwen3-ASR-supported languages missing from this testbench.

Fixture-corpus languages intentionally not included because Qwen3-ASR does not list them as supported:

```text
Bengali, Urdu
```

## Run

Start the API server, then run:

```text
python testbench/run_openai_quality.py --base-url http://127.0.0.1:8000 --limit 5
```

Official benchmark runs should use the Taskfile:

```text
task benchmark-transcription-17b
task benchmark-transcription-06b
```

The runner appends benchmark markdown results under:

```text
benchmarks/transcription/
```
