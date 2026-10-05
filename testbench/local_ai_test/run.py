from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CASES = Path(__file__).with_name("cases.json")
DEFAULT_OUTPUT = ROOT / "testbench" / "results" / "local-ai-mcp-latest.json"

SYSTEM_PROMPT = """You are an agent using a local speech-to-text MCP server.
Use the available tools when the request requires deployment health, transcription, or an explicitly requested runtime setting change.
Pass audio through file_location as either a path in the server's shared MCP input directory or an HTTP(S) URL returned by another tool. Never place audio bytes or base64 in tool arguments.
Never invent a transcription or tool result. If a tool returns an error, explain it accurately. Retry only when the user supplied an explicit valid fallback.
Do not change runtime settings unless the user explicitly requests the change. Do not add unsupported arguments."""


def _replace_audio_url(value: Any, audio_url: str) -> Any:
    if isinstance(value, dict):
        return {key: _replace_audio_url(item, audio_url) for key, item in value.items()}
    if isinstance(value, list):
        return [_replace_audio_url(item, audio_url) for item in value]
    if isinstance(value, str):
        return value.replace("{audio_url}", audio_url)
    return value


def _post_json(
    url: str, body: dict[str, Any], timeout: float, api_key: str
) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LLM endpoint returned HTTP {exc.code}: {detail}") from exc


def _get_json(url: str, timeout: float, api_key: str) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {api_key}"},
        method="GET",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _discover_model(base_url: str, timeout: float, api_key: str) -> str:
    payload = _get_json(f"{base_url.rstrip('/')}/models", timeout, api_key)
    models = payload.get("data") or payload.get("models") or []
    for item in models:
        if isinstance(item, dict):
            model = item.get("id") or item.get("name") or item.get("model")
            if model:
                return str(model)
    raise RuntimeError(
        "No model was returned by the local OpenAI-compatible /models endpoint"
    )


def _openai_tools(mcp_tools: list[Any]) -> list[dict[str, Any]]:
    tools = []
    for tool in mcp_tools:
        dumped = tool.model_dump(mode="json", by_alias=True)
        input_schema = (
            dumped.get("inputSchema")
            or dumped.get("input_schema")
            or {"type": "object"}
        )
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": dumped["name"],
                    "description": dumped.get("description")
                    or dumped.get("title")
                    or dumped["name"],
                    "parameters": input_schema,
                },
            }
        )
    return tools


def _content_text(result: Any) -> str:
    structured = getattr(result, "structured_content", None)
    if structured is not None:
        return json.dumps(structured, ensure_ascii=False)
    parts = []
    for item in getattr(result, "content", []) or []:
        text = getattr(item, "text", None)
        if text:
            parts.append(str(text))
        else:
            parts.append(
                json.dumps(
                    item.model_dump(mode="json", by_alias=True), ensure_ascii=False
                )
            )
    return "\n".join(parts) or "{}"


def _safe_arguments(arguments: Any) -> Any:
    return dict(arguments) if isinstance(arguments, dict) else arguments


def _parse_arguments(raw: Any) -> tuple[dict[str, Any] | None, str | None]:
    if isinstance(raw, dict):
        return raw, None
    try:
        parsed = json.loads(str(raw or "{}"))
    except json.JSONDecodeError as exc:
        return None, f"Tool arguments are not valid JSON: {exc.msg}"
    if not isinstance(parsed, dict):
        return None, "Tool arguments must decode to a JSON object"
    return parsed, None


def _tool_error_payload(message: str) -> str:
    return json.dumps(
        {
            "ok": False,
            "error": {
                "code": "invalid_tool_call",
                "message": message,
                "guidance": "Correct the tool name or arguments using the supplied schema. Never invent a tool result.",
            },
        },
        ensure_ascii=False,
    )


def _redact_opaque_text(value: str, limit: int = 2000) -> str:
    def replace(match: re.Match[str]) -> str:
        opaque = match.group(0)
        return f"<redacted opaque value: {len(opaque)} characters>"

    redacted = re.sub(r"[A-Za-z0-9+/]{256,}={0,2}", replace, value)
    if len(redacted) > limit:
        return f"{redacted[:limit]}... <truncated {len(redacted) - limit} characters>"
    return redacted


def _assistant_message(message: dict[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {
        "role": "assistant",
        "content": message.get("content") or "",
    }
    if message.get("tool_calls"):
        cleaned["tool_calls"] = message["tool_calls"]
    return cleaned


async def _run_case(
    *,
    case: dict[str, Any],
    session: ClientSession,
    tool_names: set[str],
    tools: list[dict[str, Any]],
    chat_url: str,
    model: str,
    timeout: float,
    api_key: str,
    max_turns: int,
    max_tokens: int,
) -> dict[str, Any]:
    prompt = str(case["prompt"])
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    invocations: list[dict[str, Any]] = []
    llm_turns: list[dict[str, Any]] = []
    final_text = ""
    stop_reason = "turn_limit"
    started = time.perf_counter()

    for turn in range(max_turns):
        body = {
            "model": model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "temperature": 0,
            "max_tokens": max_tokens,
        }
        try:
            response = await asyncio.to_thread(
                _post_json, chat_url, body, timeout, api_key
            )
        except (OSError, RuntimeError, TimeoutError) as exc:
            llm_error = _redact_opaque_text(str(exc))
            llm_turns.append(
                {
                    "finish_reason": "endpoint_error",
                    "content": "",
                    "tool_call_count": 0,
                    "error": llm_error,
                }
            )
            stop_reason = "llm_endpoint_error"
            break
        choices = response.get("choices") or []
        if not choices:
            raise RuntimeError(f"LLM response contained no choices: {response}")
        choice = choices[0]
        message = choice.get("message") or {}
        llm_turns.append(
            {
                "finish_reason": choice.get("finish_reason"),
                "content": str(message.get("content") or ""),
                "tool_call_count": len(message.get("tool_calls") or []),
            }
        )
        messages.append(_assistant_message(message))
        tool_calls = message.get("tool_calls") or []
        if not tool_calls:
            final_text = str(message.get("content") or "").strip()
            stop_reason = str(choice.get("finish_reason") or "no_tool_call")
            break

        for tool_call in tool_calls:
            call_id = str(tool_call.get("id") or f"call-{turn}-{len(invocations)}")
            function = tool_call.get("function") or {}
            name = str(function.get("name") or "")
            arguments, parse_error = _parse_arguments(function.get("arguments"))
            invocation: dict[str, Any] = {
                "name": name,
                "arguments": _safe_arguments(
                    arguments if arguments is not None else function.get("arguments")
                ),
                "outcome": "error",
                "error": None,
            }

            if parse_error:
                tool_text = _tool_error_payload(parse_error)
                invocation["error"] = parse_error
            elif name not in tool_names:
                error = f"Unknown tool {name!r}. Available tools: {', '.join(sorted(tool_names))}."
                tool_text = _tool_error_payload(error)
                invocation["error"] = error
            else:
                try:
                    result = await session.call_tool(name, arguments=arguments or {})
                    tool_text = _content_text(result)
                    if getattr(result, "is_error", False):
                        invocation["error"] = tool_text
                    else:
                        invocation["outcome"] = "success"
                except Exception as exc:  # noqa: BLE001 - transport failures are benchmark results.
                    tool_text = _tool_error_payload(str(exc))
                    invocation["error"] = str(exc)

            invocation["result"] = tool_text
            invocations.append(invocation)
            messages.append(
                {"role": "tool", "tool_call_id": call_id, "content": tool_text}
            )

    return {
        "id": case["id"],
        "prompt": case.get("stored_prompt") or case["prompt"],
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "invocations": invocations,
        "llm_turns": llm_turns,
        "final_text": final_text,
        "stop_reason": stop_reason,
        "turn_limit_reached": stop_reason == "turn_limit",
    }


def _rule_matches(actual: Any, rule: dict[str, Any]) -> bool:
    if "equals" in rule:
        return actual == rule["equals"]
    if "equals_casefold" in rule:
        return str(actual).casefold() == str(rule["equals_casefold"]).casefold()
    if "one_of" in rule:
        return str(actual).casefold() in {
            str(value).casefold() for value in rule["one_of"]
        }
    if "contains" in rule:
        return str(rule["contains"]).casefold() in str(actual).casefold()
    if "contains_item" in rule:
        return isinstance(actual, list) and rule["contains_item"] in actual
    return False


def _score(case: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    failures: list[str] = []
    invocations = result["invocations"]
    actual_tools = [item["name"] for item in invocations]
    expected_tools = case.get("expected_tools") or []
    if case.get("exact_tool_sequence"):
        if actual_tools != expected_tools:
            failures.append(f"tool sequence {actual_tools!r} != {expected_tools!r}")
    elif actual_tools[: len(expected_tools)] != expected_tools:
        failures.append(
            f"tool prefix {actual_tools!r} does not start with {expected_tools!r}"
        )

    expected_outcomes = case.get("expected_outcomes") or []
    actual_outcomes = [item["outcome"] for item in invocations]
    if actual_outcomes[: len(expected_outcomes)] != expected_outcomes:
        failures.append(
            f"outcomes {actual_outcomes!r} do not start with {expected_outcomes!r}"
        )

    for index, rules in enumerate(case.get("argument_rules") or []):
        if index >= len(invocations):
            failures.append(f"missing invocation {index + 1} for argument checks")
            continue
        arguments = invocations[index].get("arguments")
        if not isinstance(arguments, dict):
            failures.append(f"invocation {index + 1} arguments were not a JSON object")
            continue
        for name, rule in rules.items():
            actual = arguments.get(name)
            matched = _rule_matches(actual, rule)
            if not matched:
                failures.append(
                    f"invocation {index + 1} argument {name!r} failed rule {rule!r}: {actual!r}"
                )

    expected_errors = [
        str(value).casefold() for value in case.get("expected_error_contains") or []
    ]
    if expected_errors:
        observed = "\n".join(
            str(item.get("error") or "") for item in invocations
        ).casefold()
        for expected in expected_errors:
            if expected not in observed:
                failures.append(f"tool error did not contain {expected!r}")

    final_text = str(result.get("final_text") or "")
    if not final_text:
        failures.append("model did not produce a final answer")
    required_final = [
        str(value).casefold() for value in case.get("final_contains_any") or []
    ]
    if required_final and not any(
        value in final_text.casefold() for value in required_final
    ):
        failures.append(f"final answer contained none of {required_final!r}")

    return {"passed": not failures, "failures": failures}


async def _main(args: argparse.Namespace) -> int:
    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    if args.case_ids:
        requested = set(args.case_ids)
        available = {str(case["id"]) for case in cases}
        unknown = sorted(requested - available)
        if unknown:
            raise ValueError(f"Unknown case IDs: {', '.join(unknown)}")
        cases = [case for case in cases if case["id"] in requested]
    url_cases = [case for case in cases if case.get("requires_audio_url")]
    if url_cases and not args.audio_url:
        if args.case_ids:
            raise ValueError("--audio-url is required for the selected URL-reference case")
        skipped = ", ".join(str(case["id"]) for case in url_cases)
        print(f"Skipping URL-reference cases without --audio-url: {skipped}", flush=True)
        cases = [case for case in cases if not case.get("requires_audio_url")]
    elif args.audio_url:
        cases = [_replace_audio_url(case, args.audio_url) for case in cases]
    model = args.model or await asyncio.to_thread(
        _discover_model, args.base_url, args.timeout, args.api_key
    )
    chat_url = f"{args.base_url.rstrip('/')}/chat/completions"

    all_results: list[dict[str, Any]] = []
    async with (
        streamable_http_client(args.mcp_url) as (read_stream, write_stream),
        ClientSession(read_stream, write_stream) as session,
    ):
        initialize = await session.initialize()
        listed = await session.list_tools()
        tool_names = {tool.name for tool in listed.tools}
        tools = _openai_tools(listed.tools)

        print(f"LLM: {model} at {args.base_url}", flush=True)
        print(f"MCP: {initialize.server_info.name} at {args.mcp_url}", flush=True)
        print(f"Tools: {', '.join(sorted(tool_names))}", flush=True)
        for repeat in range(1, args.repeats + 1):
            for case in cases:
                result = await _run_case(
                    case=case,
                    session=session,
                    tool_names=tool_names,
                    tools=tools,
                    chat_url=chat_url,
                    model=model,
                    timeout=args.timeout,
                    api_key=args.api_key,
                    max_turns=args.max_turns,
                    max_tokens=args.max_tokens,
                )
                result["repeat"] = repeat
                result["score"] = _score(case, result)
                all_results.append(result)
                marker = "PASS" if result["score"]["passed"] else "FAIL"
                tools_used = (
                    ", ".join(item["name"] for item in result["invocations"]) or "none"
                )
                print(
                    f"[{marker}] repeat={repeat} case={case['id']} tools={tools_used} time={result['elapsed_seconds']:.3f}s",
                    flush=True,
                )
                for failure in result["score"]["failures"]:
                    print(f"  - {failure}", flush=True)

    passed = sum(1 for result in all_results if result["score"]["passed"])
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "llm_base_url": args.base_url,
        "llm_model": model,
        "mcp_url": args.mcp_url,
        "repeats": args.repeats,
        "cases_per_repeat": len(cases),
        "summary": {
            "passed": passed,
            "failed": len(all_results) - passed,
            "total": len(all_results),
            "pass_rate": round(passed / len(all_results), 4) if all_results else 0.0,
        },
        "results": all_results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["summary"], indent=2), flush=True)
    print(f"Report: {args.output}", flush=True)
    return 0 if passed == len(all_results) else 1


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exercise Qwen3-ASR MCP tool use through a local OpenAI-compatible LLM"
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("LOCAL_AI_BASE_URL", "http://127.0.0.1:18080/v1"),
        help="Local OpenAI-compatible LLM base URL",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("LOCAL_AI_MODEL", ""),
        help="Model name; omitted discovers /models",
    )
    parser.add_argument("--api-key", default=os.getenv("LOCAL_AI_API_KEY", "local"))
    parser.add_argument(
        "--mcp-url", default=os.getenv("LOCAL_MCP_URL", "http://127.0.0.1:8000/mcp")
    )
    parser.add_argument(
        "--audio-url",
        default=os.getenv("LOCAL_AI_MCP_AUDIO_URL", ""),
        help="Reachable HTTP(S) audio URL used by URL-reference cases",
    )
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument(
        "--case",
        action="append",
        dest="case_ids",
        help="Run only this case ID; repeat the option to select multiple cases",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--max-turns", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=float, default=180.0)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main(_arguments())))
