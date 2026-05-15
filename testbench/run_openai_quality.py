from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import time
import tomllib
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from difflib import SequenceMatcher
from typing import Any


def _normalize_text(value: str) -> str:
    value = value.strip()
    value = re.sub(r"\s+", " ", value)
    return value


def _meaning_text(value: str) -> str:
    value = value.casefold()
    value = re.sub(r"\[[^\]]+\]", " ", value)
    value = re.sub(r"[^\w\s]", " ", value, flags=re.UNICODE)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def _special_chars(value: str) -> set[str]:
    return set(re.findall(r"[!?…。،؛؟¡¿'\"“”‘’]", value))


def _score(expected: str, actual: str) -> tuple[float, float]:
    expected_meaning = _meaning_text(expected)
    actual_meaning = _meaning_text(actual)
    base = SequenceMatcher(None, expected_meaning, actual_meaning).ratio() * 100.0
    expected_special = _special_chars(expected)
    actual_special = _special_chars(actual)
    if not actual_special:
        return base, 0.0
    discovered = actual_special if not expected_special else expected_special & actual_special
    bonus = min(10.0, float(len(discovered)) * 2.5)
    return base, bonus


def _append_line(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text("", encoding="utf-8")
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line)


def _language_columns(manifest: dict[str, Any], cases: list[dict[str, Any]]) -> list[str]:
    languages = {
        item["language"]
        for item in manifest.get("supported_languages_with_assets", [])
        if item.get("language")
    }
    languages.update(case["language"] for case in cases)
    return sorted(languages)


def _ensure_summary_md(path: Path, language_columns: list[str]) -> None:
    if path.exists() and path.read_text(encoding="utf-8").strip():
        _ensure_summary_comment_column(path)
        return

    headers = ["Version", "Model", "Comment", "Test time", "Total score %", "Bonus %", "Total time", *language_columns]
    aligns = ["---", "---", "---", "---", "---:", "---:", "---", *(["---:"] * len(language_columns))]
    content = [
        "# Qwen3-ASR Transcription Benchmarks",
        "",
        "Append one row per transcription benchmark run.",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(aligns) + " |",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(content) + "\n", encoding="utf-8")


def _split_md_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _join_md_row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _ensure_summary_comment_column(path: Path) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    header_index = next(
        (index for index, line in enumerate(lines) if line.startswith("|") and "Version" in line and "Model" in line),
        None,
    )
    if header_index is None:
        return
    headers = _split_md_row(lines[header_index])
    if "Comment" in headers:
        return
    try:
        model_index = headers.index("Model")
    except ValueError:
        return
    insert_index = model_index + 1
    for index in range(header_index, len(lines)):
        if not lines[index].startswith("|"):
            continue
        cells = _split_md_row(lines[index])
        if len(cells) <= insert_index:
            continue
        if index == header_index:
            cells.insert(insert_index, "Comment")
        elif index == header_index + 1:
            cells.insert(insert_index, "---")
        else:
            cells.insert(insert_index, "")
        lines[index] = _join_md_row(cells)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _validate_summary_md(path: Path) -> None:
    rows = [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.startswith("|")
    ]
    if not rows:
        return
    expected = len(rows[0].strip().strip("|").split("|"))
    mismatches = [
        (index + 1, len(row.strip().strip("|").split("|")))
        for index, row in enumerate(rows)
        if len(row.strip().strip("|").split("|")) != expected
    ]
    if mismatches:
        detail = ", ".join(f"table row {line}: {count} columns" for line, count in mismatches)
        raise ValueError(f"{path} has inconsistent markdown table columns; expected {expected}: {detail}")


def _md_escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def _project_version(start: Path) -> str:
    for directory in [start, *start.parents]:
        version_path = directory / "VERSION"
        if version_path.exists():
            value = version_path.read_text(encoding="utf-8").strip()
            if value:
                return value
        pyproject_path = directory / "pyproject.toml"
        if pyproject_path.exists():
            data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
            value = data.get("project", {}).get("version", "")
            if value:
                return str(value)
    return "unknown"


def _diff_tokens(value: str) -> list[str]:
    if re.search(r"\s", value):
        return re.findall(r"\s+|\S+", value)
    return list(value)


def _md_diff(expected: str, actual: str) -> tuple[str, str]:
    expected_tokens = _diff_tokens(expected)
    actual_tokens = _diff_tokens(actual)
    matcher = SequenceMatcher(None, expected_tokens, actual_tokens, autojunk=False)
    expected_parts: list[str] = []
    actual_parts: list[str] = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        expected_text = _md_escape("".join(expected_tokens[i1:i2]))
        actual_text = _md_escape("".join(actual_tokens[j1:j2]))
        if tag == "equal":
            expected_parts.append(expected_text)
            actual_parts.append(actual_text)
        else:
            if expected_text:
                expected_parts.append(f"**{expected_text}**")
            if actual_text:
                actual_parts.append(f"**{actual_text}**")

    return "".join(expected_parts), "".join(actual_parts)


def _multipart_request(url: str, fields: dict[str, str], file_path: Path, timeout: float) -> dict[str, Any]:
    boundary = f"----qwen-asr-testbench-{uuid.uuid4().hex}"
    mime = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
    parts: list[bytes] = []

    for name, value in fields.items():
        parts.append(f"--{boundary}\r\n".encode("utf-8"))
        parts.append(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8"))
        parts.append(str(value).encode("utf-8"))
        parts.append(b"\r\n")

    parts.append(f"--{boundary}\r\n".encode("utf-8"))
    parts.append(
        (
            f'Content-Disposition: form-data; name="file"; filename="{file_path.name}"\r\n'
            f"Content-Type: {mime}\r\n\r\n"
        ).encode("utf-8")
    )
    parts.append(file_path.read_bytes())
    parts.append(b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode("utf-8"))

    request = urllib.request.Request(
        url,
        data=b"".join(parts),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = response.read().decode("utf-8")
    return json.loads(payload)


def _run_case(endpoint: str, model: str, root: Path, case: dict[str, Any], request_timeout: float) -> dict[str, Any]:
    audio_path = root / case["audio"]
    expected = case["expected_text"]
    item = {
        "id": case["id"],
        "language": case["language"],
        "audio": case["audio"],
        "expected_text": expected,
        "actual_text": "",
        "exact_match": False,
        "normalized_match": False,
        "error": "",
        "elapsed_sec": None,
    }
    t0 = time.time()
    try:
        response = _multipart_request(
            endpoint,
            {
                "model": model,
                "language": case["language"],
                "response_format": "json",
            },
            audio_path,
            request_timeout,
        )
        actual = str(response.get("text", ""))
        base_score, bonus_score = _score(expected, actual)
        item["actual_text"] = actual
        item["score"] = round(base_score, 2)
        item["bonus"] = round(bonus_score, 2)
        item["total_score"] = round(base_score + bonus_score, 2)
        item["exact_match"] = actual == expected
        item["normalized_match"] = _meaning_text(actual) == _meaning_text(expected)
    except (OSError, urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError) as exc:
        item["error"] = str(exc)
    finally:
        item["elapsed_sec"] = round(time.time() - t0, 3)
    return item


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Qwen3-ASR OpenAI-compatible quality fixtures")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--manifest", default="testbench/manifest.json")
    parser.add_argument("--model", default="qwen3-asr")
    parser.add_argument("--model-label", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--prewarm", type=int, default=0, help="Number of leading cases to run before measured timing")
    parser.add_argument("--request-timeout", type=float, default=120.0, help="Per-request timeout in seconds")
    parser.add_argument("--language", default="", nargs="?", const="")
    parser.add_argument("--output", default="testbench/results/latest.json")
    parser.add_argument("--summary-md", default="benchmarks/transcription/BENCHMARKS.md")
    parser.add_argument("--details-md", default="benchmarks/transcription/DETAILS.md")
    parser.add_argument("--comment", default=os.environ.get("BENCHMARK_COMMENT", ""))
    parser.add_argument("--no-append", action="store_true")
    parser.add_argument("--strict-exit", action="store_true")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    root = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cases = manifest["cases"]
    if args.language:
        cases = [c for c in cases if c["language"].lower() == args.language.lower()]
    if args.limit > 0:
        cases = cases[: args.limit]

    endpoint = args.base_url.rstrip("/") + "/v1/audio/transcriptions"
    prewarm_results = []
    prewarm_started = time.time()
    prewarm_cases = cases[: args.prewarm] if args.prewarm > 0 else []
    for case in prewarm_cases:
        item = _run_case(endpoint, args.model, root, case, args.request_timeout)
        prewarm_results.append(item)
        status = "OK" if not item.get("error") else "ERR"
        print(f"PREWARM {status} {case['id']} {item['elapsed_sec']}s", flush=True)
    prewarm_elapsed = round(time.time() - prewarm_started, 3) if prewarm_cases else 0.0

    results = []
    print(f"MEASURE START cases={len(cases)} prewarm_discarded={len(prewarm_results)}", flush=True)
    started = time.time()
    for case in cases:
        item = _run_case(endpoint, args.model, root, case, args.request_timeout)
        results.append(item)
        status = "PASS" if item.get("total_score", 0.0) >= 90.0 else "FAIL"
        print(f"{status} {case['id']} {item.get('total_score', 0.0)}% {item['elapsed_sec']}s", flush=True)

    passed = sum(1 for item in results if item.get("total_score", 0.0) >= 90.0)
    total_score = sum(item.get("score", 0.0) for item in results) / len(results) if results else 0.0
    total_bonus = sum(item.get("bonus", 0.0) for item in results) / len(results) if results else 0.0
    by_language: dict[str, list[float]] = {}
    for item in results:
        by_language.setdefault(item["language"], []).append(item.get("score", 0.0) + item.get("bonus", 0.0))
    score_per_language = {
        language: round(sum(values) / len(values), 2)
        for language, values in sorted(by_language.items())
    }
    output = {
        "base_url": args.base_url,
        "manifest": str(manifest_path),
        "version": _project_version(manifest_path.parent),
        "model": args.model_label or args.model,
        "total": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "total_score": round(total_score, 2),
        "bonus": round(total_bonus, 2),
        "score_per_language": score_per_language,
        "prewarm": {
            "cases": len(prewarm_results),
            "elapsed_sec": prewarm_elapsed,
            "results": prewarm_results,
        },
        "elapsed_sec": round(time.time() - started, 3),
        "results": results,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not args.no_append:
        run_time = time.strftime("%d.%m.%Y %H:%M:%S")
        language_columns = _language_columns(manifest, cases)
        language_cells = [
            f"{score_per_language[language]:.2f}%"
            if language in score_per_language
            else ""
            for language in language_columns
        ]
        summary_md = Path(args.summary_md)
        _ensure_summary_md(summary_md, language_columns)
        _validate_summary_md(summary_md)
        _append_line(
            summary_md,
            (
                f"| {_md_escape(output['version'])} | {_md_escape(args.model_label or args.model)} | "
                f"{_md_escape(args.comment)} | {run_time} | "
                f"{output['total_score']:.2f}% | {output['bonus']:.2f}% | "
                f"{output['elapsed_sec']:.3f}s | "
                f"{' | '.join(_md_escape(value) for value in language_cells)} |\n"
            ),
        )
        _validate_summary_md(summary_md)

        ranked = sorted(results, key=lambda item: item.get("total_score", 0.0), reverse=True)
        section = [
            "",
            f"## {run_time} - {args.model_label or args.model}",
            "",
            f"- Version: `{output['version']}`",
            f"- Comment: `{args.comment}`" if args.comment else "- Comment: ``",
            f"- Total score: `{output['total_score']:.2f}%`",
            f"- Bonus: `{output['bonus']:.2f}%`",
            f"- Total time: `{output['elapsed_sec']:.3f}s`",
            f"- Cases: `{output['total']}`",
            "",
            "### Best Examples",
            "",
        ]
        for item in ranked[:3]:
            expected_diff, actual_diff = _md_diff(item["expected_text"], item["actual_text"])
            section.extend([
                f"#### {item['id']} - {item.get('total_score', 0.0):.2f}%",
                "",
                f"- Language: `{item['language']}`",
                f"- Expected: {expected_diff}",
                f"- Actual: {actual_diff}",
                "",
            ])
        section.extend(["### Worst Examples", ""])
        for item in sorted(results, key=lambda item: item.get("total_score", 0.0))[:3]:
            expected_diff, actual_diff = _md_diff(item["expected_text"], item["actual_text"])
            section.extend([
                f"#### {item['id']} - {item.get('total_score', 0.0):.2f}%",
                "",
                f"- Language: `{item['language']}`",
                f"- Expected: {expected_diff}",
                f"- Actual: {actual_diff}",
                "",
            ])
        _append_line(Path(args.details_md), "\n".join(section) + "\n")
    print(f"wrote {output_path}")
    return 0 if (not args.strict_exit or passed == len(results)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
