from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import tempfile
import time
import urllib.error
from pathlib import Path
from typing import Any

try:
    from testbench.run_openai_quality import (
        _detect_gpu,
        _gpu_details,
        _gpu_label,
        _md_escape,
        _meaning_text,
        _multipart_request,
        _project_version,
        _score,
    )
except ModuleNotFoundError:  # Direct execution from the testbench directory.
    from run_openai_quality import (  # type: ignore[no-redef]
        _detect_gpu,
        _gpu_details,
        _gpu_label,
        _md_escape,
        _meaning_text,
        _multipart_request,
        _project_version,
        _score,
    )


def _ffmpeg_command(generator: dict[str, Any], sources: dict[str, dict[str, Any]], root: Path, output: Path) -> list[str]:
    kind = generator["kind"]
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]

    if kind == "silence":
        command.extend([
            "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono",
            "-t", str(generator["duration_sec"]),
        ])
    elif kind == "noise":
        source = (
            f"anoisesrc=color={generator['color']}:amplitude={generator['amplitude']}:"
            f"seed={generator['seed']}:r=16000:d={generator['duration_sec']}"
        )
        command.extend(["-f", "lavfi", "-i", source])
        if generator.get("filter"):
            command.extend(["-af", str(generator["filter"])])
    elif kind == "sine":
        source = (
            f"sine=frequency={generator['frequency']}:sample_rate=16000:"
            f"duration={generator['duration_sec']}"
        )
        command.extend(["-f", "lavfi", "-i", source, "-af", f"volume={generator['amplitude']}"])
    elif kind == "speech":
        source = sources[generator["source"]]
        command.extend(["-i", str(root / source["audio"])])
        if generator.get("filter"):
            command.extend(["-af", str(generator["filter"])])
    elif kind == "speech_noise":
        source = sources[generator["source"]]
        noise = (
            f"anoisesrc=color={generator['noise_color']}:amplitude={generator['noise_amplitude']}:"
            f"seed={generator['seed']}:r=16000:d=120"
        )
        mix = (
            f"[0:a]volume={generator['speech_volume']},aresample=16000[s];"
            "[s][1:a]amix=inputs=2:duration=first:normalize=0[out]"
        )
        command.extend([
            "-i", str(root / source["audio"]),
            "-f", "lavfi", "-i", noise,
            "-filter_complex", mix,
            "-map", "[out]",
        ])
    else:
        raise ValueError(f"Unknown fixture generator kind: {kind}")

    command.extend(["-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(output)])
    return command


def _generate_fixture(
    fixture: dict[str, Any],
    sources: dict[str, dict[str, Any]],
    root: Path,
    output_dir: Path,
) -> Path:
    output = output_dir / f"{fixture['id']}.wav"
    command = _ffmpeg_command(fixture["generator"], sources, root, output)
    subprocess.run(command, check=True, capture_output=True, text=True)
    return output


def _expand_cases(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    sources = manifest["sources"]
    for fixture in manifest["fixtures"]:
        generator = fixture["generator"]
        expected = ""
        if fixture["expected"] == "source":
            expected = sources[generator["source"]]["expected_text"]
        for language in fixture["request_languages"]:
            mode = str(language).lower() if language else "auto"
            cases.append({
                "id": f"{fixture['id']}__{mode}",
                "fixture_id": fixture["id"],
                "category": fixture["category"],
                "request_language": language,
                "expected_kind": fixture["expected"],
                "expected_text": expected,
                "minimum_score": float(fixture.get("minimum_score", 100.0)),
            })
    return cases


def _evaluate(case: dict[str, Any], actual: str) -> tuple[bool, float]:
    if case["expected_kind"] == "empty":
        return _meaning_text(actual) == "", 100.0 if _meaning_text(actual) == "" else 0.0
    base_score, _ = _score(case["expected_text"], actual)
    return base_score >= case["minimum_score"], base_score


def _run_case(
    endpoint: str,
    model: str,
    audio_path: Path,
    case: dict[str, Any],
    request_timeout: float,
) -> dict[str, Any]:
    fields = {"model": model, "response_format": "json"}
    if case["request_language"]:
        fields["language"] = case["request_language"]
    result = {
        **case,
        "actual_text": "",
        "detected_language": None,
        "score": 0.0,
        "passed": False,
        "elapsed_sec": None,
        "error": "",
    }
    started = time.perf_counter()
    try:
        response = _multipart_request(endpoint, fields, audio_path, request_timeout)
        actual = str(response.get("text", ""))
        passed, score = _evaluate(case, actual)
        result.update({
            "actual_text": actual,
            "detected_language": response.get("language"),
            "score": round(score, 2),
            "passed": passed,
        })
    except (OSError, urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError) as exc:
        result["error"] = str(exc)
    finally:
        result["elapsed_sec"] = round(time.perf_counter() - started, 3)
    return result


def _metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    non_speech_auto = [
        item for item in results
        if item["expected_kind"] == "empty" and item["request_language"] is None
    ]
    non_speech_forced = [
        item for item in results
        if item["expected_kind"] == "empty" and item["request_language"] is not None
    ]
    speech = [item for item in results if item["expected_kind"] == "source"]
    categories: dict[str, dict[str, int]] = {}
    for item in results:
        category = categories.setdefault(item["category"], {"total": 0, "passed": 0})
        category["total"] += 1
        category["passed"] += int(item["passed"])
    latencies = [float(item["elapsed_sec"]) for item in results if item["elapsed_sec"] is not None]
    return {
        "total": len(results),
        "passed": sum(int(item["passed"]) for item in results),
        "non_speech_auto": {
            "total": len(non_speech_auto),
            "false_positives": sum(not item["passed"] for item in non_speech_auto),
        },
        "non_speech_forced": {
            "total": len(non_speech_forced),
            "false_positives": sum(not item["passed"] for item in non_speech_forced),
        },
        "speech": {
            "total": len(speech),
            "passed": sum(int(item["passed"]) for item in speech),
            "mean_score": round(statistics.fmean(item["score"] for item in speech), 2) if speech else 0.0,
        },
        "categories": categories,
        "mean_request_sec": round(statistics.fmean(latencies), 3) if latencies else None,
    }


def _write_summary(path: Path, output: dict[str, Any], comment: str) -> None:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "# Qwen3-ASR Audio Robustness Benchmarks\n\n"
            "Deterministic regression results for silence, noise, echo, and weak speech.\n\n"
            "| Version | Model | GPU | Comment | Test time | Passed | Auto non-speech false positives | Forced-language non-speech false positives | Speech passed | Mean speech score | Mean request |\n"
            "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |\n",
            encoding="utf-8",
        )
    metrics = output["metrics"]
    row = (
        f"| {_md_escape(output['version'])} | {_md_escape(output['model'])} | {_md_escape(_gpu_label(output['gpu']))} | "
        f"{_md_escape(comment)} | {output['run_time']} | {metrics['passed']}/{metrics['total']} | "
        f"{metrics['non_speech_auto']['false_positives']}/{metrics['non_speech_auto']['total']} | "
        f"{metrics['non_speech_forced']['false_positives']}/{metrics['non_speech_forced']['total']} | "
        f"{metrics['speech']['passed']}/{metrics['speech']['total']} | {metrics['speech']['mean_score']:.2f}% | "
        f"{metrics['mean_request_sec']:.3f}s |\n"
    )
    with path.open("a", encoding="utf-8") as handle:
        handle.write(row)


def _write_details(path: Path, output: dict[str, Any]) -> None:
    metrics = output["metrics"]
    lines = [
        "# Latest Qwen3-ASR Audio Robustness Details",
        "",
        f"- Version: `{output['version']}`",
        f"- Model: `{output['model']}`",
        f"- GPU: `{_gpu_details(output['gpu'])}`",
        f"- Run time: `{output['run_time']}`",
        f"- Passed: `{metrics['passed']}/{metrics['total']}`",
        f"- Auto non-speech false positives: `{metrics['non_speech_auto']['false_positives']}/{metrics['non_speech_auto']['total']}`",
        f"- Forced-language non-speech false positives: `{metrics['non_speech_forced']['false_positives']}/{metrics['non_speech_forced']['total']}`",
        f"- Speech passed: `{metrics['speech']['passed']}/{metrics['speech']['total']}`",
        "",
        "| Fixture | Category | Request language | Result | Score | Expected | Actual |",
        "| --- | --- | --- | --- | ---: | --- | --- |",
    ]
    for item in output["results"]:
        language = item["request_language"] or "Auto"
        expected = "(empty)" if item["expected_kind"] == "empty" else item["expected_text"]
        actual = item["actual_text"] or "(empty)"
        result = "PASS" if item["passed"] else "FAIL"
        lines.append(
            f"| {_md_escape(item['fixture_id'])} | {_md_escape(item['category'])} | {_md_escape(language)} | "
            f"{result} | {item['score']:.2f}% | {_md_escape(expected)} | {_md_escape(actual)} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run deterministic Qwen3-ASR audio robustness fixtures")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--manifest", default="testbench/robustness_manifest.json")
    parser.add_argument("--model", default="qwen3-asr")
    parser.add_argument("--model-label", default="")
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--output", default="testbench/results/robustness-latest.json")
    parser.add_argument("--summary-md", default="benchmarks/robustness/BENCHMARKS.md")
    parser.add_argument("--details-md", default="benchmarks/robustness/DETAILS.md")
    parser.add_argument("--comment", default=os.environ.get("BENCHMARK_COMMENT", ""))
    parser.add_argument("--gpu-index", type=int, default=None)
    parser.add_argument("--no-append", action="store_true")
    parser.add_argument("--strict-exit", action="store_true")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cases = _expand_cases(manifest)
    gpu = _detect_gpu(args.gpu_index)
    if args.gpu_index is not None and gpu is None:
        parser.error(f"GPU index {args.gpu_index} was not found by nvidia-smi")
    print(f"GPU {_gpu_details(gpu)}", flush=True)

    endpoint = args.base_url.rstrip("/") + "/v1/audio/transcriptions"
    results: list[dict[str, Any]] = []
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="qwen-asr-robustness-") as directory:
        generated_dir = Path(directory)
        audio_paths = {
            fixture["id"]: _generate_fixture(fixture, manifest["sources"], root, generated_dir)
            for fixture in manifest["fixtures"]
        }
        for case in cases:
            item = _run_case(endpoint, args.model, audio_paths[case["fixture_id"]], case, args.request_timeout)
            results.append(item)
            status = "PASS" if item["passed"] else "FAIL"
            language = item["request_language"] or "Auto"
            actual = item["actual_text"] or "(empty)"
            print(f"{status} {item['fixture_id']} [{language}] -> {actual!r} ({item['elapsed_sec']:.3f}s)", flush=True)

    output = {
        "base_url": args.base_url,
        "manifest": str(manifest_path),
        "version": _project_version(manifest_path.parent),
        "model": args.model_label or args.model,
        "gpu": gpu,
        "run_time": time.strftime("%d.%m.%Y %H:%M:%S"),
        "elapsed_sec": round(time.perf_counter() - started, 3),
        "metrics": _metrics(results),
        "results": results,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_details(Path(args.details_md), output)
    if not args.no_append:
        _write_summary(Path(args.summary_md), output, args.comment)
    print(f"wrote {output_path}", flush=True)
    passed = output["metrics"]["passed"]
    total = output["metrics"]["total"]
    return 0 if (not args.strict_exit or passed == total) else 1


if __name__ == "__main__":
    raise SystemExit(main())
