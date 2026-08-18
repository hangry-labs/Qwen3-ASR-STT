from __future__ import annotations

import asyncio
import threading
import unittest

from qwen_asr.server.inference_runtime import InferenceCoordinator


class InferenceCoordinatorTests(unittest.IsolatedAsyncioTestCase):
    async def test_cancellation_keeps_engine_owned_until_call_finishes(self):
        started = threading.Event()
        release = threading.Event()
        coordinator = InferenceCoordinator(
            model_name="test-model",
            capacity=2,
            inference_timeout_seconds=1,
            recycle_process=lambda reason: None,
        )

        def blocking_inference():
            started.set()
            release.wait(timeout=1)
            return "done"

        task = asyncio.create_task(coordinator.run("ordinary", blocking_inference))
        await asyncio.to_thread(started.wait, 1)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            _ = await task

        snapshot = await coordinator.snapshot()
        self.assertEqual(snapshot["active_inference"], 1)

        release.set()
        for _ in range(100):
            snapshot = await coordinator.snapshot()
            if snapshot["active_inference"] == 0:
                break
            await asyncio.sleep(0.01)

        snapshot = await coordinator.snapshot()
        metrics = await coordinator.metrics()
        self.assertEqual(snapshot["status"], "ok")
        self.assertEqual(snapshot["active_inference"], 0)
        self.assertEqual(metrics["cancelled_total"], 1)
        self.assertEqual(metrics["success_total"], 1)
        coordinator.shutdown()

    async def test_fatal_engine_error_degrades_and_schedules_recycle(self):
        recycled: list[str] = []
        coordinator = InferenceCoordinator(
            model_name="test-model",
            capacity=1,
            recycle_process=recycled.append,
        )

        def failed_inference():
            raise RuntimeError("EngineCore IPC process is unavailable")

        with self.assertRaises(RuntimeError):
            await coordinator.run("ordinary", failed_inference)

        snapshot = await coordinator.snapshot()
        self.assertEqual(snapshot["status"], "degraded")
        self.assertEqual(snapshot["reason"], "inference_engine_error")
        self.assertEqual(recycled, ["inference_engine_error"])
        coordinator.shutdown()

    async def test_cancelled_call_timeout_degrades_and_schedules_recycle(self):
        started = threading.Event()
        release = threading.Event()
        recycled: list[str] = []
        coordinator = InferenceCoordinator(
            model_name="test-model",
            capacity=1,
            inference_timeout_seconds=0.03,
            recycle_process=recycled.append,
        )

        def blocking_inference():
            started.set()
            release.wait(timeout=1)

        task = asyncio.create_task(coordinator.run("ordinary", blocking_inference))
        await asyncio.to_thread(started.wait, 1)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            _ = await task

        await asyncio.sleep(0.05)
        snapshot = await coordinator.snapshot()
        self.assertEqual(snapshot["status"], "degraded")
        self.assertEqual(snapshot["reason"], "inference_timeout")
        self.assertEqual(snapshot["active_inference"], 1)
        self.assertEqual(recycled, ["inference_timeout"])

        release.set()
        for _ in range(100):
            snapshot = await coordinator.snapshot()
            if snapshot["active_inference"] == 0:
                break
            await asyncio.sleep(0.01)
        self.assertEqual(snapshot["active_inference"], 0)
        coordinator.shutdown()


if __name__ == "__main__":
    unittest.main()
