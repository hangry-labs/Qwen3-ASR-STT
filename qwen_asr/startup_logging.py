from __future__ import annotations

import time
from contextlib import nullcontext
from datetime import datetime
from typing import ContextManager

_PROCESS_START = time.perf_counter()


def log_startup(message: str) -> None:
    elapsed = time.perf_counter() - _PROCESS_START
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[startup +{elapsed:8.3f}s {timestamp}] {message}", flush=True)


class StartupTimer:
    def __init__(self, label: str) -> None:
        self.label = label
        self.started = 0.0

    def __enter__(self) -> "StartupTimer":
        self.started = time.perf_counter()
        log_startup(f"{self.label}: start")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        elapsed = time.perf_counter() - self.started
        status = "failed" if exc_type is not None else "done"
        log_startup(f"{self.label}: {status} in {elapsed:.3f}s")


def optional_timer(label: str, enabled: bool) -> ContextManager[StartupTimer | None]:
    if enabled:
        return StartupTimer(label)
    return nullcontext()
