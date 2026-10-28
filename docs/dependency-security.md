# Dependency security review

Reviewed on 2026-10-28 against the repository's 76 open Dependabot alerts.
This records the development source changes, not a claim that existing image
tags have been rebuilt. Published images receive these changes only after a
new build is released.

## Dependency changes

| Dependency | Previous | Selected | Open alerts addressed |
| --- | --- | --- | --- |
| vLLM | 0.26.0 | 0.32.0 | 14 |
| PyTorch | 2.11.0 | 2.13.0 | 2 |
| Pillow | 12.3.0 | 13.0.0 | 41 |
| aiohttp | 3.14.3 | 3.14.5 | 8 |
| certifi | 2026.7.22 | 2026.10.23 | 1 |
| dnspython | 2.8.0 | 2.9.0 | 1 |
| filelock | 4.0.9 | 4.2.4 | 3 |
| OpenTelemetry API | 1.45.0 | 1.45.1 | 1 |
| rubyzip | 2.4.1 | Removed from local website builder | 1 |

The coordinated vLLM upgrade also requires torchvision 0.28.0, Transformers
5.19.0, Hugging Face Hub 1.31.0, and updated kernel dependencies. The CUDA 13.0
compiler pins and disabled optional FlashInfer sampler remain in place.
Security floors in `requirements.in` prevent the lock resolver from selecting
older affected transitive versions.

The local website builder now installs Jekyll 3.10 and only the plugins used by
the site. Removing the `github-pages` umbrella eliminates its unused
`jekyll-remote-theme` and `rubyzip` chain while retaining the GFM parser, local
theme, safe-mode rendering, and language routes. GitHub manages the hosted
Pages build environment independently.

## Remaining upstream constraints

Four alerts remain across two packages. They must not be treated as fixed or
dismissed merely to clear the alert count.

### Accelerate checkpoint index paths

[GHSA-4j2p-28q2-5m79](https://github.com/advisories/GHSA-4j2p-28q2-5m79)
is reported in both `requirements.txt` and `pyproject.toml`. No patched release
is identified by the advisory. Source inspection of Accelerate 1.15.0 still
shows checkpoint `weight_map` filenames joined to the checkpoint directory
without containment validation, so upgrading to it alone is not a verified fix.
The existing 1.14.0 pin is retained.

This product loads operator-selected model repositories at startup or when
loading the aligner; API/MCP clients cannot upload checkpoints or select model
repositories. Full images bake the model assets and run with Hugging Face
offline mode enabled. Continue to use trusted model assets and do not accept
untrusted sharded checkpoint indexes. Tiny images and custom operator-selected
models still depend on the trustworthiness of their model source.

### setuptools compatibility ceiling

vLLM 0.32 requires `setuptools>=78.1.1,<81` on Python 3.13. The fixes for
[GHSA-h35f-9h28-mq5c](https://github.com/advisories/GHSA-h35f-9h28-mq5c)
and [GHSA-xmj3-9gf7-h52w](https://github.com/advisories/GHSA-xmj3-9gf7-h52w)
require 83.0.0 and 84.1.0 respectively. The compatible 80.10.2 pin remains;
overriding vLLM's declared dependency would create an unsupported installation.
The project's isolated build backend separately requires setuptools 84.1.0 or
newer, so source-package builds can use the patched backend without violating
vLLM's runtime requirement.

The reported paths concern source-distribution exclusion on macOS filesystems
and legacy wheel installation on Windows. The supported Docker product builds
and runs on Linux, and transcription requests do not invoke either packaging
operation. Revisit the pin when vLLM supports a patched setuptools release.

## Validation

The candidate Linux/Python 3.13 image was built locally and passed `pip check`.
The Python suite completed 107 tests with two Git-dependent staging checks
skipped in the runtime image; host staging and website language-routing checks
were also exercised. The narrower Ruby builder completed a real safe-mode
Jekyll build.

Live checks used Qwen3-ASR-0.6B on an RTX 5070 Ti with the same cached model
snapshot for both stacks:

| Existing 300-fixture corpus, 30 languages | Previous stack | Updated stack |
| --- | --- | --- |
| Request errors | 0 | 0 |
| Mean normalized-text similarity | 95.99% | 96.20% |
| Fixtures meeting the existing score threshold | 256/300 | 261/300 |
| Mean request time | 125 ms | 116 ms |

260 of 300 transcripts were byte-identical. Individual transcripts can still
change with runtime/kernel updates; these results show no aggregate quality
regression on this corpus, not universal equivalence. Timing is an informal
local comparison with other workloads present, not an isolated performance
benchmark. The 1.7B model was not inference-tested in this review.

Live API checks covered JSON, unaligned verbose JSON, plain text, SSE,
realtime append/finish, SRT/VTT, English/Chinese/Japanese word and segment
alignment, aligner unload, localized UI routes, model discovery, OpenAPI,
and MCP health with both enabled and disabled access.

The full-image asset stage fetched and validated both ASR sizes and the forced
aligner. Its final image export was stopped when the local Docker disk ran out
of unreserved space; only disposable records from that extra build were
removed. A complete new full-image export and offline release qualification
remain to be performed before publishing it.

The authenticated Dependabot alert ranges were compared with the candidate
lock: 72 of the 76 original alerts are addressed, with the four exceptions
described above. A separate `pip-audit` scan reported only Accelerate and
setuptools. That scanner cannot directly identify PyTorch-index `+cu130`
builds on PyPI; their public versions were checked against Dependabot's ranges.
Alerts on GitHub will update after the changes reach the default branch.
