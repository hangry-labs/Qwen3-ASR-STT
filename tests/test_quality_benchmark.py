from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import patch

from testbench import run_openai_quality


def test_detect_gpu_returns_requested_device() -> None:
    output = (
        "0, NVIDIA GeForce RTX 5070 Ti, GPU-first, 16303, 610.88\n"
        "1, NVIDIA RTX 6000 Ada Generation, GPU-second, 49140, 610.88\n"
    )

    with patch.object(
        run_openai_quality.subprocess,
        "run",
        return_value=CompletedProcess([], 0, stdout=output, stderr=""),
    ):
        gpu = run_openai_quality._detect_gpu(1)

    assert gpu == {
        "index": 1,
        "name": "NVIDIA RTX 6000 Ada Generation",
        "uuid": "GPU-second",
        "memory_total_mib": 49140,
        "driver_version": "610.88",
    }


def test_detect_gpu_automatically_selects_single_device() -> None:
    output = "0, NVIDIA GeForce RTX 5070 Ti, GPU-only, 16303, 610.88\n"

    with patch.object(
        run_openai_quality.subprocess,
        "run",
        return_value=CompletedProcess([], 0, stdout=output, stderr=""),
    ):
        gpu = run_openai_quality._detect_gpu()

    assert gpu is not None
    assert gpu["name"] == "NVIDIA GeForce RTX 5070 Ti"
    assert run_openai_quality._gpu_label(gpu) == "NVIDIA GeForce RTX 5070 Ti (16303 MiB)"


def test_existing_summary_gains_gpu_column_with_legacy_marker(tmp_path: Path) -> None:
    summary = tmp_path / "BENCHMARKS.md"
    summary.write_text(
        "# Benchmarks\n\n"
        "| Version | Model | Comment | Test time | Total time |\n"
        "| --- | --- | --- | --- | ---: |\n"
        "| 0.2.0 | example/model | baseline | now | 10.000s |\n",
        encoding="utf-8",
    )

    run_openai_quality._ensure_summary_md(summary, [])

    lines = summary.read_text(encoding="utf-8").splitlines()
    assert lines[2] == (
        "| Version | Model | GPU | Comment | Test time | Total time | Audio duration | Mean request | "
        "P95 request | RTF | Realtime speed |"
    )
    assert lines[4] == (
        "| 0.2.0 | example/model | Unknown (legacy run) | baseline | now | 10.000s | Unknown (legacy run) | "
        "Unknown (legacy run) | Unknown (legacy run) | Unknown (legacy run) | Unknown (legacy run) |"
    )


def test_performance_metrics_report_latency_and_realtime_speed() -> None:
    results = [
        {"elapsed_sec": 0.1, "audio_duration_sec": 2.0},
        {"elapsed_sec": 0.2, "audio_duration_sec": 3.0},
        {"elapsed_sec": 0.3, "audio_duration_sec": 5.0},
    ]

    metrics = run_openai_quality._performance_metrics(results, elapsed_sec=1.0)

    assert metrics == {
        "audio_duration_sec": 10.0,
        "audio_duration_cases": 3,
        "mean_request_sec": 0.2,
        "median_request_sec": 0.2,
        "p95_request_sec": 0.29,
        "realtime_factor": 0.1,
        "realtime_speed": 10.0,
    }


def test_performance_metrics_do_not_extrapolate_missing_audio_durations() -> None:
    results = [
        {"elapsed_sec": 0.1, "audio_duration_sec": 2.0},
        {"elapsed_sec": 0.2, "audio_duration_sec": None},
    ]

    metrics = run_openai_quality._performance_metrics(results, elapsed_sec=0.3)

    assert metrics["audio_duration_cases"] == 1
    assert metrics["audio_duration_sec"] is None
    assert metrics["realtime_factor"] is None
    assert metrics["realtime_speed"] is None
