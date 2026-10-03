from __future__ import annotations

import html
import json
import mimetypes
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from qwen_asr.standalone_ui.gpu import GPU_MONITOR


PACKAGE_DIR = Path(__file__).resolve().parent
STATIC_DIR = PACKAGE_DIR / "static"
REPO_ROOT = PACKAGE_DIR.parents[1]
ASSET_DIR = REPO_ROOT / "assets"
TESTBENCH_DIR = REPO_ROOT / "testbench"
MANIFEST_PATH = TESTBENCH_DIR / "manifest.json"

mimetypes.add_type("image/webp", ".webp")
mimetypes.add_type("font/woff2", ".woff2")


def _read_version_file() -> str:
    try:
        return (REPO_ROOT / "VERSION").read_text(encoding="utf-8").strip() or "0.0.0"
    except OSError:
        return "0.0.0"


def _example_catalog() -> list[dict[str, str]]:
    if not MANIFEST_PATH.is_file():
        return []
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    examples: list[dict[str, str]] = []
    seen: set[str] = set()
    for case in manifest.get("cases", []):
        language = str(case.get("language") or "").strip()
        relative_audio = str(case.get("audio") or "").replace("\\", "/").lstrip("/")
        if not language or language in seen or not relative_audio:
            continue
        audio_path = (TESTBENCH_DIR / relative_audio).resolve()
        try:
            audio_path.relative_to(TESTBENCH_DIR.resolve())
        except ValueError:
            continue
        if not audio_path.is_file():
            continue
        seen.add(language)
        examples.append(
            {
                "label": f"{language} - {audio_path.name}",
                "language": language,
                "name": audio_path.name,
                "url": f"/example-audio/{relative_audio}",
            }
        )
    return examples


def attach_ui(*, api_app: FastAPI) -> FastAPI:
    """Add the browser UI to the OpenAI-compatible API application."""
    development_assets = os.getenv("QWEN_ASR_UI_DEV", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    @api_app.middleware("http")
    async def disable_development_asset_cache(request, call_next):
        response = await call_next(request)
        if development_assets and (
            request.url.path == "/"
            or request.url.path.startswith("/static/")
            or request.url.path.startswith("/assets/")
        ):
            response.headers["Cache-Control"] = "no-store"
        return response

    api_app.mount("/static", StaticFiles(directory=STATIC_DIR), name="ui-static")
    if ASSET_DIR.is_dir():
        api_app.mount("/assets", StaticFiles(directory=ASSET_DIR), name="ui-assets")
    if TESTBENCH_DIR.is_dir():
        api_app.mount("/example-audio", StaticFiles(directory=TESTBENCH_DIR), name="ui-examples")

    @api_app.get("/", include_in_schema=False)
    async def index() -> HTMLResponse:
        index_html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
        rendered_html = index_html.replace("{{UI_VERSION}}", html.escape(_read_version_file()))
        return HTMLResponse(rendered_html, headers={"Cache-Control": "no-cache"})

    @api_app.get("/examples", include_in_schema=False)
    async def examples() -> dict[str, list[dict[str, str]]]:
        return {"examples": _example_catalog()}

    @api_app.get("/system/gpu", include_in_schema=False)
    def gpu() -> JSONResponse:
        return JSONResponse(GPU_MONITOR.request_snapshot(), headers={"Cache-Control": "no-store"})

    return api_app
