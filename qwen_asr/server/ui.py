# coding=utf-8
from __future__ import annotations

import html
import io
import json
import mimetypes
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable

import gradio as gr
import httpx
import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi.staticfiles import StaticFiles

from qwen_asr.inference.utils import SUPPORTED_LANGUAGES
from qwen_asr.server.openai_api import create_app as create_openai_app
from qwen_asr.startup_logging import StartupTimer, log_startup
from qwen_asr.web.branding import brand_css, brand_header_html
from qwen_asr.web.gpu import gpu_monitor_html

REPO_ROOT = Path(__file__).resolve().parents[2]
ASSET_DIR = REPO_ROOT / "hangrylabs"
BRAND_ASSET_BASE = "/assets/hangrylabs"
TESTBENCH_DIR = REPO_ROOT / "testbench"
MANIFEST_PATH = TESTBENCH_DIR / "manifest.json"

LANGUAGE_CHOICES = ["Auto"] + SUPPORTED_LANGUAGES
RESPONSE_FORMATS = ["json", "text", "verbose_json", "srt", "vtt"]
TIMESTAMP_GRANULARITIES = ["word", "segment"]

GPU_CSS = """
.gpu-monitor {
    margin: 10px 0 4px;
    padding: 12px;
    border: 1px solid rgba(255, 176, 118, 0.22);
    border-radius: 10px;
    background: linear-gradient(135deg, rgba(22, 12, 6, 0.95), rgba(7, 7, 8, 0.96));
    color: #fff3e7;
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.05), 0 12px 24px rgba(0, 0, 0, 0.20);
}

.gpu-monitor-title {
    margin-bottom: 9px;
    font-size: 0.86rem;
    font-weight: 800;
    letter-spacing: 0.02em;
}

.gpu-monitor-muted {
    color: rgba(255, 243, 231, 0.68);
    font-size: 0.86rem;
}

.gpu-card + .gpu-card {
    margin-top: 12px;
    padding-top: 12px;
    border-top: 1px solid rgba(255, 176, 118, 0.15);
}

.gpu-card-head {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: 10px;
    margin-bottom: 8px;
}

.gpu-card-head strong {
    color: #ffb066;
}

.gpu-card-head span,
.gpu-metric-row span,
.gpu-foot span {
    color: rgba(255, 243, 231, 0.68);
    font-size: 0.78rem;
}

.gpu-metric-row,
.gpu-foot {
    display: flex;
    justify-content: space-between;
    gap: 10px;
    font-size: 0.82rem;
}

.gpu-bar {
    overflow: hidden;
    height: 7px;
    margin: 5px 0 8px;
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.09);
}

.gpu-bar span {
    display: block;
    height: 100%;
    border-radius: inherit;
    background: linear-gradient(90deg, #ff6b00, #ffb066);
}

.gpu-vram span {
    background: linear-gradient(90deg, #ffb066, #ffe0bd);
}

.gpu-sparkline {
    display: block;
    width: 100%;
    height: 44px;
    margin-bottom: 8px;
    border-radius: 8px;
    background: linear-gradient(180deg, rgba(255, 255, 255, 0.07), rgba(255, 255, 255, 0.02));
}
"""


def _dtype_from_str(s: str) -> torch.dtype:
    s = (s or "").strip().lower()
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    if s in ("fp16", "float16", "half"):
        return torch.float16
    if s in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {s}. Use bfloat16/float16/float32.")


def _coerce_special_types(d: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in d.items():
        if k == "dtype" and isinstance(v, str):
            out[k] = _dtype_from_str(v)
        else:
            out[k] = v
    return out


def _read_version_file() -> str:
    try:
        return (REPO_ROOT / "VERSION").read_text(encoding="utf-8").strip() or "0.0.0"
    except OSError:
        return "0.0.0"


def _get_cuda_devices() -> list[str]:
    if not torch.cuda.is_available():
        return []
    return [torch.cuda.get_device_name(idx) for idx in range(torch.cuda.device_count())]


def _get_banner_runtime_html(model_name: str, backend: str) -> str:
    version = html.escape(_read_version_file())
    model = html.escape(model_name)
    backend_label = html.escape(backend)
    cuda_devices = _get_cuda_devices()
    if cuda_devices:
        visible = os.getenv("CUDA_VISIBLE_DEVICES", "").strip()
        visible_ids = [part.strip() for part in visible.split(",") if part.strip()] if visible else []
        gpu_lines = []
        for idx, name in enumerate(cuda_devices):
            display_idx = visible_ids[idx] if idx < len(visible_ids) else str(idx)
            gpu_lines.append(f"{html.escape(display_idx)} : {html.escape(name)}")
        hardware = "GPUs :<br>" + "<br>".join(gpu_lines)
    else:
        hardware = "Runtime : CPU"
    return f"v{version} | {backend_label}<br>{model}<br>{hardware}"


def _load_examples() -> list[list[Any]]:
    if not MANIFEST_PATH.exists():
        return []
    try:
        payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []

    by_language: dict[str, str] = {}
    for case in payload.get("cases", []):
        language = str(case.get("language", "") or "")
        audio_rel = str(case.get("audio", "") or "")
        if not language or not audio_rel or language in by_language:
            continue
        audio_path = TESTBENCH_DIR / audio_rel
        if not audio_path.exists():
            continue
        by_language[language] = str(audio_path)
    return [[by_language[language], language] for language in SUPPORTED_LANGUAGES if language in by_language]


def _api_url(base_url: str, path: str) -> str:
    return base_url.rstrip("/") + path


def _audio_path_from_gradio(audio_value: Any) -> str | None:
    if audio_value is None:
        return None
    if isinstance(audio_value, (str, os.PathLike)):
        return os.fspath(audio_value)
    if isinstance(audio_value, dict):
        path = audio_value.get("path") or audio_value.get("name")
        return os.fspath(path) if path else None
    path = getattr(audio_value, "path", None)
    if path:
        return os.fspath(path)
    name = getattr(audio_value, "name", None)
    return os.fspath(name) if name else None


def _format_status(elapsed: float, response_format: str, status_code: int) -> str:
    return f"HTTP {status_code} | {response_format} | {elapsed:.3f}s"


def _extract_text(response_format: str, content: str) -> tuple[str, str]:
    if response_format in {"text", "srt", "vtt"}:
        return content, content
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return content, content
    text = str(payload.get("text", ""))
    return text, json.dumps(payload, ensure_ascii=False, indent=2)


def _request_data(
    *,
    model: str,
    language: str,
    response_format: str,
    prompt: str,
    timestamp_granularities: Iterable[str],
    stream: bool = False,
) -> dict[str, str]:
    data = {
        "model": model,
        "response_format": response_format,
        "temperature": "0",
    }
    if language and language != "Auto":
        data["language"] = language
    if prompt and prompt.strip():
        data["prompt"] = prompt
    if timestamp_granularities:
        data["timestamp_granularities[]"] = ",".join(timestamp_granularities)
    if stream:
        data["stream"] = "true"
    return data


def transcribe_openai(
    audio_path: Any,
    model: str,
    language: str,
    response_format: str,
    prompt: str,
    timestamp_granularities: list[str],
    base_url: str,
) -> tuple[str, str, str]:
    audio_path = _audio_path_from_gradio(audio_path)
    if not audio_path:
        return "", "", "Audio input is required."

    start = time.perf_counter()
    mime = mimetypes.guess_type(audio_path)[0] or "application/octet-stream"
    data = _request_data(
        model=model,
        language=language,
        response_format=response_format,
        prompt=prompt,
        timestamp_granularities=timestamp_granularities,
    )
    with Path(audio_path).open("rb") as audio_file:
        files = {"file": (Path(audio_path).name, audio_file, mime)}
        with httpx.Client(timeout=httpx.Timeout(600.0, connect=10.0)) as client:
            response = client.post(_api_url(base_url, "/v1/audio/transcriptions"), data=data, files=files)
    elapsed = time.perf_counter() - start
    if response.status_code >= 400:
        return "", response.text, _format_status(elapsed, response_format, response.status_code)
    text, details = _extract_text(response_format, response.text)
    return text, details, _format_status(elapsed, response_format, response.status_code)


def stream_openai(
    audio_path: Any,
    model: str,
    language: str,
    prompt: str,
    base_url: str,
):
    audio_path = _audio_path_from_gradio(audio_path)
    if not audio_path:
        yield "", "Audio input is required."
        return

    start = time.perf_counter()
    mime = mimetypes.guess_type(audio_path)[0] or "application/octet-stream"
    data = _request_data(
        model=model,
        language=language,
        response_format="json",
        prompt=prompt,
        timestamp_granularities=[],
        stream=True,
    )
    transcript = ""
    with Path(audio_path).open("rb") as audio_file:
        files = {"file": (Path(audio_path).name, audio_file, mime)}
        with httpx.Client(timeout=httpx.Timeout(None, connect=10.0)) as client:
            with client.stream("POST", _api_url(base_url, "/v1/audio/transcriptions"), data=data, files=files) as response:
                if response.status_code >= 400:
                    yield "", f"HTTP {response.status_code} | {response.read().decode('utf-8', errors='replace')}"
                    return
                for line in response.iter_lines():
                    if not line or not line.startswith("data: "):
                        continue
                    payload = line[6:]
                    if payload == "[DONE]":
                        break
                    try:
                        event = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    if event.get("type") == "transcript.text.delta":
                        transcript += str(event.get("delta", ""))
                        yield transcript, f"Streaming | {time.perf_counter() - start:.3f}s"
                    elif event.get("type") == "transcript.text.done":
                        transcript = str(event.get("text", transcript))
                        yield transcript, f"Done | {time.perf_counter() - start:.3f}s"


def _wav_bytes_from_audio_chunk(audio_chunk: Any) -> tuple[bytes, float] | None:
    if audio_chunk is None:
        return None
    if not isinstance(audio_chunk, tuple) or len(audio_chunk) != 2:
        return None

    sample_rate, samples = audio_chunk
    sample_rate = int(sample_rate)
    audio = np.asarray(samples)
    if audio.size == 0:
        return None
    if audio.ndim > 2:
        audio = audio.reshape(-1)
    if audio.ndim == 2 and audio.shape[0] in {1, 2} and audio.shape[1] > audio.shape[0]:
        audio = audio.T

    duration = float(audio.shape[0]) / float(sample_rate) if sample_rate > 0 else 0.0
    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format="WAV")
    return buffer.getvalue(), duration


def realtime_stream_openai(
    audio_chunk: Any,
    session_id: str | None,
    model: str,
    language: str,
    prompt: str,
    base_url: str,
) -> tuple[str, str, str]:
    wav_payload = _wav_bytes_from_audio_chunk(audio_chunk)
    if wav_payload is None:
        return "", "Listening.", session_id or ""

    audio_bytes, duration = wav_payload
    with httpx.Client(timeout=httpx.Timeout(600.0, connect=10.0)) as client:
        if not session_id:
            data: dict[str, Any] = {
                "model": model,
                "prompt": prompt or "",
                "temperature": 0,
                "chunk_size_sec": 2.0,
                "unfixed_chunk_num": 2,
                "unfixed_token_num": 5,
            }
            if language and language != "Auto":
                data["language"] = language
            response = client.post(_api_url(base_url, "/v1/realtime/transcriptions/sessions"), json=data)
            if response.status_code >= 400:
                return "", response.text, ""
            session_id = str(response.json()["id"])

        files = {"file": ("chunk.wav", audio_bytes, "audio/wav")}
        response = client.post(
            _api_url(base_url, f"/v1/realtime/transcriptions/sessions/{session_id}/audio"),
            files=files,
        )
    if response.status_code >= 400:
        return "", response.text, session_id or ""
    payload = response.json()
    text = str(payload.get("text", ""))
    chunk_id = int(payload.get("chunk_id", 0))
    status = f"Realtime | session {session_id} | chunks {chunk_id} | last audio {duration:.2f}s"
    return text, status, session_id or ""


def finish_realtime_openai(session_id: str | None, base_url: str) -> tuple[str, str, str]:
    if not session_id:
        return "", "No active realtime session.", ""
    with httpx.Client(timeout=httpx.Timeout(600.0, connect=10.0)) as client:
        response = client.post(_api_url(base_url, f"/v1/realtime/transcriptions/sessions/{session_id}/finish"))
    if response.status_code >= 400:
        return "", response.text, session_id
    payload = response.json()
    text = str(payload.get("text", ""))
    return text, f"Finalized | session {session_id} | chunks {payload.get('chunk_id', 0)}", ""


def reset_realtime_openai(session_id: str | None, base_url: str) -> tuple[str, str, str]:
    if session_id:
        with httpx.Client(timeout=10.0) as client:
            client.delete(_api_url(base_url, f"/v1/realtime/transcriptions/sessions/{session_id}"))
    return "", "Realtime session reset.", ""


def refresh_api_status(base_url: str) -> tuple[str, str]:
    with httpx.Client(timeout=10.0) as client:
        health = client.get(_api_url(base_url, "/health"))
        models = client.get(_api_url(base_url, "/v1/models"))
        languages = client.get(_api_url(base_url, "/v1/audio/supported_languages"))
    payload = {
        "health": health.json() if health.headers.get("content-type", "").startswith("application/json") else health.text,
        "models": models.json() if models.headers.get("content-type", "").startswith("application/json") else models.text,
        "languages": languages.json() if languages.headers.get("content-type", "").startswith("application/json") else languages.text,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2), gpu_monitor_html()


def switch_ui_view(view: str) -> tuple[Any, Any, Any, Any, Any]:
    selected = view or "Transcribe"
    return (
        gr.update(visible=selected == "Transcribe"),
        gr.update(visible=selected == "Transcribe"),
        gr.update(visible=selected == "Stream"),
        gr.update(visible=selected == "API"),
        gr.update(visible=selected == "System"),
    )


def build_demo(*, model_name: str, backend: str, openai_base_url: str) -> gr.Blocks:
    app_css = brand_css(BRAND_ASSET_BASE) + GPU_CSS
    header = brand_header_html(
        product_name="Qwen3-ASR-STT",
        description="Offline Qwen3-ASR speech recognition packaged for deterministic local transcription through OpenAI-compatible APIs.",
        links=[
            ("Examples", "https://github.com/Hangry-Labs/Qwen3-ASR-STT/tree/main/testbench"),
            ("GitHub", "https://github.com/Hangry-Labs/Qwen3-ASR-STT"),
            ("API docs", "/docs"),
            ("Models", "/v1/models"),
        ],
        capabilities=["OpenAI-compatible STT", "Offline baked assets", "Streaming responses", "GPU aware"],
        runtime_html=_get_banner_runtime_html(model_name, backend),
        asset_base=BRAND_ASSET_BASE,
    )
    examples = _load_examples()

    with gr.Blocks(title="Qwen3-ASR-STT") as demo:
        gr.HTML(f"<style>{app_css}</style>")
        gr.HTML(header)
        api_base_state = gr.State(openai_base_url)

        with gr.Row():
            with gr.Column(scale=1):
                model = gr.Textbox(value="qwen3-asr", label="Model")
                language = gr.Dropdown(choices=LANGUAGE_CHOICES, value="Auto", label="Language")
                prompt = gr.Textbox(label="Prompt / context", lines=3, placeholder="Optional words, spelling, or context to help recognition.")
                with gr.Group(visible=True) as transcribe_options:
                    response_format = gr.Dropdown(choices=RESPONSE_FORMATS, value="json", label="Response format")
                    timestamp_granularities = gr.CheckboxGroup(
                        choices=TIMESTAMP_GRANULARITIES,
                        value=[],
                        label="Timestamp granularities",
                        info="Requires the forced aligner to be enabled in the container.",
                    )
                temperature = gr.Number(value=0, label="Temperature", interactive=False, info="Deterministic transcription is enforced.")

            with gr.Column(scale=2):
                view_selector = gr.Radio(
                    choices=["Transcribe", "Stream", "API", "System"],
                    value="Transcribe",
                    label="View",
                )
                with gr.Group(visible=True) as transcribe_view:
                    audio_in = gr.Audio(
                        label="Audio input",
                        sources=["upload", "microphone"],
                        type="filepath",
                        format="mp3",
                    )
                    transcribe_btn = gr.Button("Transcribe", variant="primary")
                    transcript = gr.Textbox(label="Transcript", lines=8)
                    details = gr.Textbox(label="Response details", lines=10)
                    status = gr.Textbox(label="Status", lines=1)
                    if examples:
                        gr.Examples(examples=examples, inputs=[audio_in, language], label="Example files")

                with gr.Group(visible=False) as stream_view:
                    stream_audio = gr.Audio(
                        label="Audio input",
                        sources=["upload", "microphone"],
                        type="filepath",
                        format="mp3",
                    )
                    stream_btn = gr.Button("Stream transcription", variant="primary")
                    stream_text = gr.Textbox(label="Streaming transcript", lines=8)
                    stream_status = gr.Textbox(label="Stream status", lines=1)
                    realtime_session = gr.State("")
                    realtime_audio = gr.Audio(
                        label="Realtime microphone",
                        sources=["microphone"],
                        type="numpy",
                        streaming=True,
                    )
                    with gr.Row():
                        realtime_finish = gr.Button("Finalize realtime", variant="secondary")
                        realtime_reset = gr.Button("Reset realtime", variant="secondary")
                    realtime_text = gr.Textbox(label="Realtime transcript", lines=8)
                    realtime_status = gr.Textbox(label="Realtime status", lines=1)

                with gr.Group(visible=False) as api_view:
                    refresh_btn = gr.Button("Refresh API status")
                    api_status = gr.Code(label="OpenAI API status", language="json", value="{}")

                with gr.Group(visible=False) as system_view:
                    gpu_html = gr.HTML(gpu_monitor_html())
                    gpu_refresh = gr.Button("Refresh GPU monitor")

        view_selector.change(
            fn=switch_ui_view,
            inputs=view_selector,
            outputs=[transcribe_options, transcribe_view, stream_view, api_view, system_view],
            queue=False,
        )
        transcribe_btn.click(
            fn=transcribe_openai,
            inputs=[audio_in, model, language, response_format, prompt, timestamp_granularities, api_base_state],
            outputs=[transcript, details, status],
        )
        stream_btn.click(
            fn=stream_openai,
            inputs=[stream_audio, model, language, prompt, api_base_state],
            outputs=[stream_text, stream_status],
        )
        realtime_audio.stream(
            fn=realtime_stream_openai,
            inputs=[realtime_audio, realtime_session, model, language, prompt, api_base_state],
            outputs=[realtime_text, realtime_status, realtime_session],
            stream_every=0.5,
        )
        realtime_audio.stop_recording(
            fn=finish_realtime_openai,
            inputs=[realtime_session, api_base_state],
            outputs=[realtime_text, realtime_status, realtime_session],
        )
        realtime_finish.click(
            fn=finish_realtime_openai,
            inputs=[realtime_session, api_base_state],
            outputs=[realtime_text, realtime_status, realtime_session],
        )
        realtime_reset.click(
            fn=reset_realtime_openai,
            inputs=[realtime_session, api_base_state],
            outputs=[realtime_text, realtime_status, realtime_session],
        )
        refresh_btn.click(fn=refresh_api_status, inputs=api_base_state, outputs=[api_status, gpu_html])
        gpu_refresh.click(fn=gpu_monitor_html, outputs=gpu_html)

    return demo


def run_server(
    *,
    asr_checkpoint: str,
    aligner_checkpoint: str | None,
    backend: str,
    backend_kwargs: Dict[str, Any] | None,
    aligner_kwargs: Dict[str, Any] | None,
    cuda_visible_devices: str,
    host: str,
    port: int,
    concurrency: int,
    share: bool = False,
    ssl_certfile: str | None = None,
    ssl_keyfile: str | None = None,
    ssl_verify: bool = True,
) -> None:
    del share, ssl_certfile, ssl_keyfile, ssl_verify

    log_startup("Gradio OpenAI-backed UI startup entered")
    if cuda_visible_devices.strip():
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices.strip()
        log_startup(f"CUDA_VISIBLE_DEVICES set to {cuda_visible_devices.strip()}")

    resolved_backend_kwargs = _coerce_special_types(backend_kwargs or {})
    resolved_aligner_kwargs = _coerce_special_types(aligner_kwargs or {})

    forced_aligner = aligner_checkpoint if aligner_checkpoint else None

    with StartupTimer("import Qwen3ASRModel"):
        from qwen_asr.inference.qwen3_asr import Qwen3ASRModel

    with StartupTimer(f"load ASR model via {backend} backend"):
        if backend == "transformers":
            asr = Qwen3ASRModel.from_pretrained(
                asr_checkpoint,
                forced_aligner=forced_aligner,
                forced_aligner_kwargs=resolved_aligner_kwargs if forced_aligner else None,
                **resolved_backend_kwargs,
            )
        elif backend == "vllm":
            asr = Qwen3ASRModel.LLM(
                asr_checkpoint,
                forced_aligner=forced_aligner,
                forced_aligner_kwargs=resolved_aligner_kwargs if forced_aligner else None,
                **resolved_backend_kwargs,
            )
        else:
            raise ValueError(f"Unsupported backend: {backend}")

    warmup_enabled = os.getenv("QWEN_ASR_STARTUP_WARMUP", "0").strip().lower() in {"1", "true", "yes", "y"}
    if warmup_enabled:
        warmup_tokens = int(os.getenv("QWEN_ASR_STARTUP_WARMUP_TOKENS", "1"))
        with StartupTimer(f"startup ASR warmup max_new_tokens={warmup_tokens}"):
            asr.warm_up(max_new_tokens=warmup_tokens)

    trace_requests = os.getenv("QWEN_ASR_TRACE_REQUESTS", "0").strip().lower() in {"1", "true", "yes", "y"}
    openai_base_url = os.getenv("QWEN_ASR_UI_OPENAI_BASE_URL", f"http://127.0.0.1:{int(port)}")

    with StartupTimer("create OpenAI API app"):
        app = create_openai_app(
            asr=asr,
            model_name=asr_checkpoint,
            concurrency=concurrency,
            trace_requests=trace_requests,
        )

    if ASSET_DIR.exists():
        app.mount(BRAND_ASSET_BASE, StaticFiles(directory=ASSET_DIR), name="hangrylabs-assets")

    with StartupTimer("build Gradio UI"):
        demo = build_demo(model_name=asr_checkpoint, backend=backend, openai_base_url=openai_base_url)

    with StartupTimer("mount Gradio UI"):
        app = gr.mount_gradio_app(app, demo, path="/")

    log_startup(f"starting UI/API uvicorn on {host}:{port}")
    uvicorn.run(app, host=host, port=port)
