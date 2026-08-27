from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Callable


DEFAULT_INFERENCE_TIMEOUT_SECONDS = 120.0
DEFAULT_QUEUE_TIMEOUT_SECONDS = 120.0
DEFAULT_RECYCLE_DELAY_SECONDS = 2.0
FATAL_ENGINE_ERROR_MARKERS = (
    "enginecore",
    "engine core",
    "engine process",
    "ipc",
    "worker died",
    "worker failed",
    "cuda error",
    "nccl",
)


class InferenceUnavailableError(RuntimeError):
    pass


class InferenceTimeoutError(InferenceUnavailableError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _log_event(event: str, **fields: Any) -> None:
    payload = {"event": event, "timestamp": _utc_now(), **fields}
    print(f"[inference] {json.dumps(payload, sort_keys=True, default=str)}", flush=True)


def schedule_process_recycle(reason: str, delay_seconds: float = DEFAULT_RECYCLE_DELAY_SECONDS) -> None:
    """Terminate this process after diagnostics so supervision can reload the model."""

    def terminate() -> None:
        delay = max(0.0, float(delay_seconds))
        if delay:
            time.sleep(delay)
        try:
            gpu = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                check=False,
                text=True,
                timeout=5,
            )
            _log_event(
                "process_recycle_gpu_snapshot",
                reason=reason,
                gpu=gpu.stdout.strip(),
                nvidia_smi_error=gpu.stderr.strip(),
            )
        except Exception as exc:
            _log_event("process_recycle_gpu_snapshot_failed", reason=reason, error=repr(exc))

        _log_event("process_recycle", reason=reason, pid=os.getpid())
        os.kill(os.getpid(), signal.SIGTERM)

    threading.Thread(target=terminate, name="qwen-asr-recycler", daemon=True).start()


class InferenceCoordinator:
    """Own all calls that can enter the synchronous offline inference engine."""

    def __init__(
        self,
        *,
        model_name: str,
        capacity: int,
        inference_timeout_seconds: float = DEFAULT_INFERENCE_TIMEOUT_SECONDS,
        queue_timeout_seconds: float = DEFAULT_QUEUE_TIMEOUT_SECONDS,
        recycle_process: Callable[[str], None] | None = None,
        diagnostics_provider: Callable[[], dict[str, Any]] | None = None,
    ) -> None:
        self.model_name = model_name
        self.capacity = max(1, int(capacity))
        self.inference_timeout_seconds = max(0.001, float(inference_timeout_seconds))
        self.queue_timeout_seconds = max(0.001, float(queue_timeout_seconds))
        self._recycle_process = recycle_process or schedule_process_recycle
        self._diagnostics_provider = diagnostics_provider
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="qwen-asr-engine")
        self._background_tasks: set[asyncio.Task[Any]] = set()
        self._engine_lock = asyncio.Lock()
        self._state_lock = asyncio.Lock()
        self._ready = True
        self._reason: str | None = None
        self._active: dict[str, dict[str, Any]] = {}
        self._queued: dict[str, dict[str, Any]] = {}
        self._recycle_scheduled = False
        self._last_success_at: str | None = None
        self._last_error_at: str | None = None
        self._metrics: dict[str, float | int] = {
            "requests_total": 0,
            "success_total": 0,
            "error_total": 0,
            "timeout_total": 0,
            "queue_timeout_total": 0,
            "cancelled_total": 0,
            "rejected_total": 0,
            "queue_seconds_total": 0.0,
            "inference_seconds_total": 0.0,
        }

    async def _reserve(self, request_id: str, entry: dict[str, Any]) -> None:
        async with self._state_lock:
            if not self._ready:
                self._metrics["rejected_total"] += 1
                raise InferenceUnavailableError(self._reason or "inference_unavailable")
            if len(self._active) + len(self._queued) >= self.capacity:
                self._metrics["rejected_total"] += 1
                raise InferenceUnavailableError("inference_capacity_exhausted")
            self._queued[request_id] = entry
            self._metrics["requests_total"] += 1

    async def _mark_degraded(self, reason: str, request_id: str) -> None:
        should_recycle = False
        async with self._state_lock:
            self._ready = False
            self._reason = reason
            self._last_error_at = _utc_now()
            if not self._recycle_scheduled:
                self._recycle_scheduled = True
                should_recycle = True
            snapshot = self._snapshot_unlocked()
            diagnostics = self._diagnostics_unlocked()
        external_diagnostics: dict[str, Any] = {}
        if self._diagnostics_provider is not None:
            try:
                external_diagnostics = self._diagnostics_provider()
            except Exception as exc:
                external_diagnostics = {"diagnostics_error": repr(exc)}
        _log_event(
            "inference_degraded",
            request_id=request_id,
            **snapshot,
            **diagnostics,
            **external_diagnostics,
        )
        if should_recycle:
            self._recycle_process(reason)

    async def mark_degraded(self, reason: str, request_id: str | None = None) -> None:
        await self._mark_degraded(reason, request_id or f"inf_{uuid.uuid4().hex}")

    def _snapshot_unlocked(self) -> dict[str, Any]:
        now = time.monotonic()
        oldest = 0.0
        if self._active:
            oldest = max(now - float(item["started_monotonic"]) for item in self._active.values())
        return {
            "status": "ok" if self._ready else "degraded",
            "model": self.model_name,
            "reason": self._reason,
            "active_inference": len(self._active),
            "queued_inference": len(self._queued),
            "oldest_active_seconds": round(oldest, 3),
            "last_inference_success_at": self._last_success_at,
            "last_inference_error_at": self._last_error_at,
        }

    def _diagnostics_unlocked(self) -> dict[str, Any]:
        now = time.monotonic()
        active = [
            {
                "request_id": request_id,
                "operation": item["operation"],
                "session_id": item["session_id"],
                "active_seconds": round(now - float(item["started_monotonic"]), 3),
                "thread_id": item["thread_id"],
            }
            for request_id, item in self._active.items()
        ]
        queued = [
            {
                "request_id": request_id,
                "operation": item["operation"],
                "session_id": item["session_id"],
                "queued_seconds": round(now - float(item["queued_monotonic"]), 3),
            }
            for request_id, item in self._queued.items()
        ]
        return {"active_requests": active, "queued_requests": queued}

    async def snapshot(self) -> dict[str, Any]:
        async with self._state_lock:
            return self._snapshot_unlocked()

    async def metrics(self) -> dict[str, Any]:
        async with self._state_lock:
            return {**self._metrics, **self._snapshot_unlocked()}

    async def run(
        self,
        operation: str,
        function: Callable[..., Any],
        *args: Any,
        request_id: str | None = None,
        session_id: str | None = None,
        language_mode: str = "auto",
        audio_samples: int | None = None,
        deadline_seconds: float | None = None,
        **kwargs: Any,
    ) -> Any:
        request_id = request_id or f"inf_{uuid.uuid4().hex}"
        queued_at = time.monotonic()
        queued_entry = {
            "request_id": request_id,
            "operation": operation,
            "session_id": session_id,
            "queued_monotonic": queued_at,
        }
        await self._reserve(request_id, queued_entry)
        _log_event(
            "inference_queued",
            request_id=request_id,
            session_id=session_id,
            operation=operation,
            language_mode=language_mode,
            audio_samples=audio_samples,
        )

        try:
            await asyncio.wait_for(self._engine_lock.acquire(), timeout=self.queue_timeout_seconds)
        except TimeoutError as exc:
            async with self._state_lock:
                self._queued.pop(request_id, None)
                self._metrics["queue_timeout_total"] += 1
            await self._mark_degraded("inference_queue_timeout", request_id)
            raise InferenceTimeoutError("inference_queue_timeout") from exc
        except asyncio.CancelledError:
            async with self._state_lock:
                self._queued.pop(request_id, None)
                self._metrics["cancelled_total"] += 1
            raise

        started_at = time.monotonic()
        queue_seconds = started_at - queued_at
        async with self._state_lock:
            self._queued.pop(request_id, None)
            if not self._ready:
                self._engine_lock.release()
                self._metrics["rejected_total"] += 1
                raise InferenceUnavailableError(self._reason or "inference_unavailable")
            self._active[request_id] = {
                **queued_entry,
                "started_monotonic": started_at,
                "thread_id": None,
            }
            self._metrics["queue_seconds_total"] += queue_seconds

        loop = asyncio.get_running_loop()

        def invoke() -> Any:
            thread_id = threading.get_ident()
            loop.call_soon_threadsafe(self._set_thread_id, request_id, thread_id)
            return function(*args, **kwargs)

        work = loop.run_in_executor(self._executor, invoke)
        _log_event(
            "inference_started",
            request_id=request_id,
            session_id=session_id,
            operation=operation,
            language_mode=language_mode,
            audio_samples=audio_samples,
            queue_seconds=round(queue_seconds, 6),
        )

        abandoned = False
        deadline = self.inference_timeout_seconds if deadline_seconds is None else max(0.001, deadline_seconds)
        try:
            try:
                result = await asyncio.wait_for(
                    asyncio.shield(work), timeout=deadline
                )
            except asyncio.CancelledError:
                async with self._state_lock:
                    self._metrics["cancelled_total"] += 1
                _log_event(
                    "inference_client_cancelled",
                    request_id=request_id,
                    session_id=session_id,
                    operation=operation,
                )
                abandoned = True
                monitor = asyncio.create_task(
                    self._monitor_cancelled_work(
                        request_id=request_id,
                        session_id=session_id,
                        operation=operation,
                        started_at=started_at,
                        deadline=deadline,
                        work=work,
                    ),
                    name=f"qwen-asr-cancelled-{request_id}",
                )
                self._background_tasks.add(monitor)
                monitor.add_done_callback(self._background_tasks.discard)
                raise

            inference_seconds = time.monotonic() - started_at
            async with self._state_lock:
                self._metrics["success_total"] += 1
                self._metrics["inference_seconds_total"] += inference_seconds
                self._last_success_at = _utc_now()
            _log_event(
                "inference_finished",
                request_id=request_id,
                session_id=session_id,
                operation=operation,
                inference_seconds=round(inference_seconds, 6),
                cancelled=False,
            )
            return result
        except TimeoutError as exc:
            abandoned = True
            async with self._state_lock:
                self._metrics["timeout_total"] += 1
            await self._mark_degraded("inference_timeout", request_id)
            work.add_done_callback(
                lambda completed: asyncio.create_task(
                    self._release_abandoned(request_id, started_at, completed)
                )
            )
            raise InferenceTimeoutError("inference_timeout") from exc
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            inference_seconds = time.monotonic() - started_at
            async with self._state_lock:
                self._metrics["error_total"] += 1
                self._metrics["inference_seconds_total"] += inference_seconds
                self._last_error_at = _utc_now()
            _log_event(
                "inference_failed",
                request_id=request_id,
                session_id=session_id,
                operation=operation,
                inference_seconds=round(inference_seconds, 6),
                error_type=type(exc).__name__,
            )
            error_text = f"{type(exc).__name__}: {exc}".lower()
            if any(marker in error_text for marker in FATAL_ENGINE_ERROR_MARKERS):
                await self._mark_degraded("inference_engine_error", request_id)
            raise
        finally:
            if not abandoned:
                async with self._state_lock:
                    self._active.pop(request_id, None)
                if self._engine_lock.locked():
                    self._engine_lock.release()

    async def _monitor_cancelled_work(
        self,
        *,
        request_id: str,
        session_id: str | None,
        operation: str,
        started_at: float,
        deadline: float,
        work: asyncio.Future[Any],
    ) -> None:
        timed_out = False
        try:
            elapsed = time.monotonic() - started_at
            remaining = max(0.001, deadline - elapsed)
            await asyncio.wait_for(asyncio.shield(work), timeout=remaining)
            inference_seconds = time.monotonic() - started_at
            async with self._state_lock:
                self._metrics["success_total"] += 1
                self._metrics["inference_seconds_total"] += inference_seconds
                self._last_success_at = _utc_now()
            _log_event(
                "inference_finished",
                request_id=request_id,
                session_id=session_id,
                operation=operation,
                inference_seconds=round(inference_seconds, 6),
                cancelled=True,
            )
        except TimeoutError:
            timed_out = True
            async with self._state_lock:
                self._metrics["timeout_total"] += 1
            await self._mark_degraded("inference_timeout", request_id)
            work.add_done_callback(
                lambda completed: asyncio.create_task(
                    self._release_abandoned(request_id, started_at, completed)
                )
            )
        except Exception as exc:
            inference_seconds = time.monotonic() - started_at
            async with self._state_lock:
                self._metrics["error_total"] += 1
                self._metrics["inference_seconds_total"] += inference_seconds
                self._last_error_at = _utc_now()
            _log_event(
                "inference_failed",
                request_id=request_id,
                session_id=session_id,
                operation=operation,
                inference_seconds=round(inference_seconds, 6),
                error_type=type(exc).__name__,
                cancelled=True,
            )
            error_text = f"{type(exc).__name__}: {exc}".lower()
            if any(marker in error_text for marker in FATAL_ENGINE_ERROR_MARKERS):
                await self._mark_degraded("inference_engine_error", request_id)
        finally:
            if not timed_out:
                async with self._state_lock:
                    self._active.pop(request_id, None)
                if self._engine_lock.locked():
                    self._engine_lock.release()

    async def _release_abandoned(
        self,
        request_id: str,
        started_at: float,
        work: asyncio.Future[Any],
    ) -> None:
        elapsed = time.monotonic() - started_at
        try:
            exception = work.exception()
        except (asyncio.CancelledError, Exception) as exc:
            exception = exc
        async with self._state_lock:
            self._active.pop(request_id, None)
            self._metrics["inference_seconds_total"] += elapsed
        _log_event(
            "timed_out_inference_exited",
            request_id=request_id,
            inference_seconds=round(elapsed, 6),
            error=repr(exception) if exception else None,
        )
        if self._engine_lock.locked():
            self._engine_lock.release()

    def _set_thread_id(self, request_id: str, thread_id: int) -> None:
        active = self._active.get(request_id)
        if active is not None:
            active["thread_id"] = thread_id

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
