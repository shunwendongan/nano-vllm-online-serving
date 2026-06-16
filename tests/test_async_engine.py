import asyncio
import json
import os
import tempfile
import unittest

from nanovllm.engine.async_engine import AsyncLLMEngine, RequestFailed, RequestRejected
from nanovllm.sampling_params import SamplingParams


class FakeTokenizer:
    def encode(self, text):
        return text.split()

    def decode(self, token_ids):
        return "".join(f"<{token_id}>" for token_id in token_ids)


class FakeEngine:
    def __init__(self, model, **kwargs):
        self.tokenizer = FakeTokenizer()
        self.requests = {}
        self.exited = False
        self.steps = 0

    def add_request(self, prompt, sampling_params, request_id=None, **kwargs):
        self.requests[request_id] = {
            "tokens": [] if sampling_params.max_tokens == 0 else [101, 102],
            "index": 0,
        }
        return request_id

    def abort_request(self, request_id):
        self.requests.pop(request_id, None)
        return True

    def is_finished(self):
        return not self.requests

    def step(self):
        self.steps += 1
        request_id = next(iter(self.requests))
        request = self.requests[request_id]
        if not request["tokens"]:
            self.requests.pop(request_id)
            return [{
                "request_id": request_id,
                "seq_id": 0,
                "token_id": None,
                "text": "",
                "completion_token_ids": [],
                "finished": True,
                "finish_reason": "cache_warmed",
                "usage": {
                    "prompt_tokens": 3,
                    "input_tokens": 1,
                    "cache_read_input_tokens": 0,
                    "cache_creation_input_tokens": 2,
                },
            }], 3
        token = request["tokens"][request["index"]]
        request["index"] += 1
        finished = request["index"] == len(request["tokens"])
        if finished:
            self.requests.pop(request_id)
        return [{
            "request_id": request_id,
            "seq_id": 0,
            "token_id": token,
            "text": self.tokenizer.decode([token]),
            "completion_token_ids": request["tokens"][:request["index"]],
            "finished": finished,
            "finish_reason": "length" if finished else None,
            "usage": {
                "prompt_tokens": 3,
                "input_tokens": 1,
                "cache_read_input_tokens": 2,
                "cache_creation_input_tokens": 0,
            } if finished else None,
        }], -1

    def metrics(self):
        return {"engine_steps": self.steps}

    def exit(self):
        self.exited = True


class LongOutputEngine(FakeEngine):
    def add_request(self, prompt, sampling_params, request_id=None, **kwargs):
        self.requests[request_id] = {
            "tokens": list(range(100, 100 + sampling_params.max_tokens)),
            "index": 0,
        }
        return request_id


class StuckEngine(FakeEngine):
    def is_finished(self):
        return True

    def step(self):
        raise AssertionError("stuck engine should not step")


class RejectingEngine(FakeEngine):
    def add_request(self, prompt, sampling_params, request_id=None, **kwargs):
        if prompt == "bad":
            raise ValueError("bad request")
        return super().add_request(prompt, sampling_params, request_id=request_id, **kwargs)


class FatalStepEngine(FakeEngine):
    def step(self):
        raise RuntimeError("fatal engine error")


class AsyncEngineTest(unittest.IsolatedAsyncioTestCase):
    async def asyncTearDown(self):
        if hasattr(self, "engine"):
            await self.engine.shutdown()

    async def test_generate_stream_emits_tokens_and_metrics(self):
        self.engine = AsyncLLMEngine("fake", engine_cls=FakeEngine)

        outputs = []
        async for output in self.engine.generate_stream("hello", SamplingParams(max_tokens=2), request_id="r1"):
            outputs.append(output)

        self.assertEqual([output["token_id"] for output in outputs], [101, 102])
        self.assertTrue(outputs[-1]["finished"])
        metrics = self.engine.metrics()
        self.assertEqual(metrics["finished_requests"], 1)
        self.assertEqual(metrics["dequeued_requests"], 1)
        self.assertEqual(metrics["streamed_tokens"], 2)
        self.assertEqual(metrics["decode_tokens"], 2)
        self.assertGreater(metrics["avg_decode_tok_s"], 0)
        self.assertGreater(metrics["recent_decode_tok_s"], 0)
        self.assertGreaterEqual(metrics["avg_queue_wait_s"], 0)
        self.assertGreaterEqual(metrics["max_queue_wait_s"], metrics["avg_queue_wait_s"])
        self.assertGreaterEqual(metrics["recent_queue_wait_p95_s"], 0)
        self.assertEqual(metrics["active_requests"], 0)
        self.assertGreaterEqual(metrics["avg_ttft_s"], 0)
        self.assertGreaterEqual(metrics["recent_ttft_p95_s"], 0)
        self.assertGreaterEqual(metrics["avg_latency_s"], metrics["avg_ttft_s"])
        self.assertGreaterEqual(metrics["max_latency_s"], metrics["avg_latency_s"])
        self.assertGreaterEqual(metrics["recent_latency_p95_s"], metrics["recent_ttft_p95_s"])
        self.assertEqual(metrics["metrics_window_size"], 1024)

    async def test_generate_collects_streamed_tokens(self):
        self.engine = AsyncLLMEngine("fake", engine_cls=FakeEngine)

        output = await self.engine.generate("hello", SamplingParams(max_tokens=2), request_id="r0")

        self.assertEqual(output["trace_id"], "r0")
        self.assertEqual(output["token_ids"], [101, 102])
        self.assertEqual(output["text"], "<101><102>")
        self.assertEqual(output["finish_reason"], "length")
        self.assertEqual(output["usage"]["cache_read_input_tokens"], 2)

    async def test_generate_collects_more_tokens_than_output_queue_capacity(self):
        self.engine = AsyncLLMEngine("fake", engine_cls=LongOutputEngine, output_queue_size=4)

        output = await self.engine.generate("hello", SamplingParams(max_tokens=20), request_id="long")

        self.assertEqual(output["token_ids"], list(range(100, 120)))
        self.assertEqual(output["finish_reason"], "length")
        self.assertEqual(self.engine.metrics()["cancelled_requests"], 0)

    async def test_zero_max_tokens_warms_cache_without_streaming_completion_tokens(self):
        self.engine = AsyncLLMEngine("fake", engine_cls=FakeEngine)

        outputs = []
        async for output in self.engine.generate_stream("hello", SamplingParams(max_tokens=0), request_id="prewarm"):
            outputs.append(output)

        self.assertEqual(len(outputs), 1)
        self.assertIsNone(outputs[0]["token_id"])
        self.assertEqual(outputs[0]["text"], "")
        self.assertEqual(outputs[0]["completion_token_ids"], [])
        self.assertEqual(outputs[0]["finish_reason"], "cache_warmed")
        self.assertEqual(outputs[0]["usage"]["cache_creation_input_tokens"], 2)
        metrics = self.engine.metrics()
        self.assertEqual(metrics["finished_requests"], 1)
        self.assertEqual(metrics["streamed_tokens"], 0)
        self.assertEqual(metrics["prefill_tokens"], 3)
        self.assertEqual(metrics["decode_tokens"], 0)

    async def test_request_log_jsonl_records_lifecycle_without_prompt_text(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "requests.jsonl")
            self.engine = AsyncLLMEngine("fake", engine_cls=FakeEngine, request_log_path=log_path)

            output = await self.engine.generate(
                [1, 2, 3],
                SamplingParams(max_tokens=2),
                request_id="logged",
                trace_id="trace-logged",
                cache_options={
                    "cache_namespace": "tenant-a",
                    "cache_enabled": True,
                    "cacheable_prefix_tokens": 2,
                },
                priority=7,
            )

            with open(log_path, encoding="utf-8") as f:
                records = [json.loads(line) for line in f]

        events = [record["event"] for record in records]
        self.assertEqual(output["token_ids"], [101, 102])
        self.assertEqual(output["trace_id"], "trace-logged")
        self.assertIn("submitted", events)
        self.assertIn("admitted", events)
        self.assertIn("first_token", events)
        self.assertIn("finished", events)
        submitted = next(record for record in records if record["event"] == "submitted")
        admitted = next(record for record in records if record["event"] == "admitted")
        finished = next(record for record in records if record["event"] == "finished")
        self.assertEqual(submitted["request_id"], "logged")
        self.assertEqual(submitted["trace_id"], "trace-logged")
        self.assertEqual(admitted["trace_id"], "trace-logged")
        self.assertEqual(finished["trace_id"], "trace-logged")
        self.assertEqual(submitted["prompt_type"], "token_ids")
        self.assertEqual(submitted["prompt_tokens"], 3)
        self.assertNotIn("prompt", submitted)
        self.assertEqual(submitted["cache_namespace"], "tenant-a")
        self.assertEqual(submitted["priority"], 7)
        self.assertEqual(finished["completion_tokens"], 2)
        self.assertIn("usage", finished)
        self.assertEqual(self.engine.metrics()["request_log_errors"], 0)

    async def test_pending_priority_controls_admission_order(self):
        self.engine = AsyncLLMEngine(
            "fake",
            engine_cls=StuckEngine,
            max_active_requests=0,
        )

        await self.engine.add_request("low", SamplingParams(max_tokens=2), request_id="low", priority=0)
        await self.engine.add_request("high", SamplingParams(max_tokens=2), request_id="high", priority=10)
        await self.engine.add_request("mid", SamplingParams(max_tokens=2), request_id="mid", priority=5)

        self.assertEqual([request.request_id for request in self.engine.pending], ["high", "mid", "low"])

        self.engine.max_active_requests = 1
        for _ in range(20):
            if "high" in self.engine.active:
                break
            await asyncio.sleep(0)

        self.assertIn("high", self.engine.active)
        self.assertEqual([request.request_id for request in self.engine.pending], ["mid", "low"])

    async def test_bad_request_fails_without_stopping_engine_loop(self):
        self.engine = AsyncLLMEngine("fake", engine_cls=RejectingEngine)

        with self.assertRaisesRegex(RequestFailed, "bad request") as raised:
            await self.engine.generate("bad", SamplingParams(max_tokens=2), request_id="bad")

        output = await self.engine.generate("good", SamplingParams(max_tokens=2), request_id="good")
        metrics = self.engine.metrics()

        self.assertEqual(raised.exception.error_type, "request_validation")
        self.assertEqual(output["token_ids"], [101, 102])
        self.assertEqual(metrics["failed_requests"], 1)
        self.assertEqual(metrics["finished_requests"], 1)
        self.assertEqual(metrics["completed_requests"], 2)

    async def test_pending_request_times_out_without_admission(self):
        self.engine = AsyncLLMEngine(
            "fake",
            engine_cls=StuckEngine,
            max_active_requests=0,
            queue_timeout_s=0,
        )

        with self.assertRaisesRegex(RequestFailed, "queue_timeout") as raised:
            await self.engine.generate("hello", SamplingParams(max_tokens=2), request_id="queued")

        metrics = self.engine.metrics()
        self.assertEqual(raised.exception.error_type, "timeout")
        self.assertEqual(metrics["timed_out_requests"], 1)
        self.assertEqual(metrics["completed_requests"], 1)
        self.assertEqual(metrics["dequeued_requests"], 0)

    async def test_active_request_times_out_and_releases_slot(self):
        self.engine = AsyncLLMEngine(
            "fake",
            engine_cls=StuckEngine,
            request_timeout_s=0,
        )

        with self.assertRaisesRegex(RequestFailed, "request_timeout") as raised:
            await self.engine.generate("hello", SamplingParams(max_tokens=2), request_id="active-timeout")

        metrics = self.engine.metrics()
        self.assertEqual(raised.exception.error_type, "timeout")
        self.assertEqual(metrics["timed_out_requests"], 1)
        self.assertEqual(metrics["active_requests"], 0)
        self.assertEqual(metrics["dequeued_requests"], 1)

    async def test_status_reports_ready_and_loop_failure(self):
        self.engine = AsyncLLMEngine("fake", engine_cls=FakeEngine)
        await self.engine.start()

        self.assertTrue(self.engine.status()["ready"])
        await self.engine.shutdown()

        self.engine = AsyncLLMEngine("fake", engine_cls=FatalStepEngine)
        request = await self.engine.add_request("hello", SamplingParams(max_tokens=2), request_id="fatal")
        output = await request.output_queue.get()
        for _ in range(20):
            if self.engine._task is not None and self.engine._task.done():
                break
            await asyncio.sleep(0)

        status = self.engine.status()
        metrics = self.engine.metrics()

        self.assertEqual(output["error_type"], "engine_error")
        self.assertFalse(status["ready"])
        self.assertEqual(metrics["engine_loop_errors"], 1)
        self.assertIn("fatal engine error", metrics["last_engine_error"])

    async def test_fatal_engine_error_fails_active_and_pending_requests(self):
        self.engine = AsyncLLMEngine(
            "fake",
            engine_cls=FatalStepEngine,
            max_active_requests=0,
        )
        active_candidate = await self.engine.add_request(
            "active",
            SamplingParams(max_tokens=2),
            request_id="fatal-active",
        )
        pending = await self.engine.add_request(
            "pending",
            SamplingParams(max_tokens=2),
            request_id="fatal-pending",
        )

        self.engine.max_active_requests = 1
        active_output = await asyncio.wait_for(active_candidate.output_queue.get(), timeout=1)
        pending_output = await asyncio.wait_for(pending.output_queue.get(), timeout=1)
        for _ in range(20):
            if self.engine._task is not None and self.engine._task.done():
                break
            await asyncio.sleep(0)
        metrics = self.engine.metrics()

        self.assertEqual(active_output["error_type"], "engine_error")
        self.assertEqual(pending_output["error_type"], "engine_error")
        self.assertEqual(metrics["engine_loop_errors"], 1)
        self.assertEqual(metrics["failed_requests"], 2)
        self.assertEqual(metrics["completed_requests"], 2)
        self.assertEqual(metrics["active_requests"], 0)
        self.assertEqual(metrics["pending_requests"], 0)
        self.assertFalse(self.engine.status()["ready"])
        self.assertIn("fatal engine error", self.engine.status()["fatal_error"])

    async def test_fatal_engine_error_rejects_new_requests_without_restarting_loop(self):
        self.engine = AsyncLLMEngine("fake", engine_cls=FatalStepEngine)
        request = await self.engine.add_request("hello", SamplingParams(max_tokens=2), request_id="fatal-first")
        await asyncio.wait_for(request.output_queue.get(), timeout=1)
        for _ in range(20):
            if self.engine._task is not None and self.engine._task.done():
                break
            await asyncio.sleep(0)
        failed_task = self.engine._task

        with self.assertRaisesRegex(RequestFailed, "fatal engine error") as raised:
            await self.engine.add_request("after fatal", SamplingParams(max_tokens=2), request_id="fatal-second")
        metrics = self.engine.metrics()

        self.assertEqual(raised.exception.error_type, "engine_error")
        self.assertIs(self.engine._task, failed_task)
        self.assertEqual(metrics["engine_loop_errors"], 1)
        self.assertEqual(metrics["rejected_requests"], 1)
        self.assertFalse(self.engine.status()["ready"])

    async def test_restart_after_fatal_recreates_engine_and_accepts_requests(self):
        self.engine = AsyncLLMEngine("fake", engine_cls=FatalStepEngine)
        request = await self.engine.add_request("hello", SamplingParams(max_tokens=2), request_id="fatal-before-restart")
        await asyncio.wait_for(request.output_queue.get(), timeout=1)
        for _ in range(20):
            if self.engine._task is not None and self.engine._task.done():
                break
            await asyncio.sleep(0)
        failed_engine = self.engine.engine

        self.engine.engine_cls = FakeEngine
        status = await self.engine.restart()
        output = await self.engine.generate(
            "after restart",
            SamplingParams(max_tokens=2),
            request_id="after-restart",
        )

        self.assertTrue(failed_engine.exited)
        self.assertTrue(status["ready"])
        self.assertIsNone(status["fatal_error"])
        self.assertIsNot(self.engine.engine, failed_engine)
        self.assertEqual(output["token_ids"], [101, 102])
        self.assertEqual(self.engine.metrics()["engine_loop_errors"], 1)

    async def test_restart_rejects_when_requests_are_in_flight(self):
        self.engine = AsyncLLMEngine("fake", engine_cls=StuckEngine, max_active_requests=0)
        await self.engine.add_request("queued", SamplingParams(max_tokens=2), request_id="queued")

        with self.assertRaisesRegex(RequestRejected, "in-flight"):
            await self.engine.restart()

        self.assertEqual(self.engine.metrics()["pending_requests"], 1)

    async def test_abort_releases_active_request(self):
        self.engine = AsyncLLMEngine("fake", engine_cls=StuckEngine)
        await self.engine.add_request("hello", SamplingParams(max_tokens=2), request_id="r2")

        for _ in range(10):
            if "r2" in self.engine.active:
                break
            await asyncio.sleep(0)
        self.assertIn("r2", self.engine.active)

        aborted = await self.engine.abort("r2")

        self.assertTrue(aborted)
        metrics = self.engine.metrics()
        self.assertEqual(metrics["cancelled_requests"], 1)
        self.assertEqual(metrics["active_requests"], 0)

    async def test_queue_full_rejects_request(self):
        self.engine = AsyncLLMEngine(
            "fake",
            engine_cls=StuckEngine,
            max_pending_requests=1,
            max_active_requests=0,
        )

        await self.engine.add_request("hello", SamplingParams(max_tokens=2), request_id="r1")

        with self.assertRaises(RequestRejected):
            await self.engine.add_request("hello", SamplingParams(max_tokens=2), request_id="r2")

        self.assertEqual(self.engine.metrics()["rejected_requests"], 1)

    async def test_pending_prompt_token_budget_rejects_excess_queue_load(self):
        self.engine = AsyncLLMEngine(
            "fake",
            engine_cls=StuckEngine,
            max_active_requests=0,
            max_pending_prompt_tokens=3,
        )

        await self.engine.add_request("one two", SamplingParams(max_tokens=2), request_id="r1")

        with self.assertRaisesRegex(RequestRejected, "pending prompt token budget"):
            await self.engine.add_request("three four", SamplingParams(max_tokens=2), request_id="r2")

        metrics = self.engine.metrics()
        self.assertEqual(metrics["pending_prompt_tokens"], 2)
        self.assertEqual(metrics["max_pending_prompt_tokens"], 3)
        self.assertEqual(metrics["rejected_requests"], 1)

    async def test_request_larger_than_active_token_budget_is_rejected(self):
        self.engine = AsyncLLMEngine(
            "fake",
            engine_cls=StuckEngine,
            max_active_tokens=3,
        )

        with self.assertRaisesRegex(RequestRejected, "active token budget"):
            await self.engine.add_request("one two three", SamplingParams(max_tokens=1), request_id="too-large")

        metrics = self.engine.metrics()
        self.assertEqual(metrics["rejected_requests"], 1)
        self.assertEqual(metrics["max_active_tokens"], 3)

    async def test_request_larger_than_namespace_active_token_budget_is_rejected(self):
        self.engine = AsyncLLMEngine(
            "fake",
            engine_cls=StuckEngine,
            max_active_tokens_per_namespace=3,
        )

        with self.assertRaisesRegex(RequestRejected, "namespace active token budget"):
            await self.engine.add_request(
                "one two three",
                SamplingParams(max_tokens=1),
                request_id="too-large-tenant",
                cache_options={"cache_namespace": "tenant-a"},
            )

        metrics = self.engine.metrics()
        self.assertEqual(metrics["rejected_requests"], 1)
        self.assertEqual(metrics["max_active_tokens_per_namespace"], 3)

    async def test_add_request_validates_python_api_inputs(self):
        self.engine = AsyncLLMEngine("fake", engine_cls=StuckEngine)

        invalid_cases = [
            ("", SamplingParams(max_tokens=1), {}, "prompt"),
            ([1, "bad"], SamplingParams(max_tokens=1), {}, "prompt_token_ids"),
            ("hello", SamplingParams(max_tokens=-1), {}, "max_tokens"),
            ("hello", SamplingParams(max_tokens=1, temperature=-1), {}, "temperature"),
            ("hello", SamplingParams(max_tokens=1, ignore_eos="false"), {}, "ignore_eos"),
            ("hello", SamplingParams(max_tokens=1), {"request_timeout_s": -1}, "request_timeout_s"),
            ("hello", SamplingParams(max_tokens=1), {"queue_timeout_s": float("nan")}, "queue_timeout_s"),
            ("hello", SamplingParams(max_tokens=1), {"priority": "high"}, "priority"),
        ]

        for prompt, sampling_params, kwargs, message in invalid_cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(RequestFailed, message) as raised:
                    await self.engine.add_request(prompt, sampling_params, **kwargs)
                self.assertEqual(raised.exception.error_type, "request_validation")

    async def test_default_timeout_configuration_is_validated_on_request(self):
        self.engine = AsyncLLMEngine(
            "fake",
            engine_cls=StuckEngine,
            request_timeout_s=-1,
        )

        with self.assertRaisesRegex(RequestFailed, "request_timeout_s") as raised:
            await self.engine.add_request("hello", SamplingParams(max_tokens=1), request_id="bad-timeout")

        self.assertEqual(raised.exception.error_type, "request_validation")

    async def test_active_token_budget_skips_oversized_pending_request(self):
        self.engine = AsyncLLMEngine(
            "fake",
            engine_cls=StuckEngine,
            max_active_tokens=5,
        )

        await self.engine.add_request("seed", SamplingParams(max_tokens=2), request_id="seed")
        for _ in range(20):
            if "seed" in self.engine.active:
                break
            await asyncio.sleep(0)
        self.assertIn("seed", self.engine.active)

        await self.engine.add_request("big one two", SamplingParams(max_tokens=1), request_id="big", priority=10)
        await self.engine.add_request("small", SamplingParams(max_tokens=1), request_id="small", priority=0)

        for _ in range(20):
            if "small" in self.engine.active:
                break
            await asyncio.sleep(0.002)

        metrics = self.engine.metrics()
        self.assertIn("small", self.engine.active)
        self.assertNotIn("big", self.engine.active)
        self.assertEqual([request.request_id for request in self.engine.pending], ["big"])
        self.assertEqual(metrics["active_estimated_tokens"], 5)
        self.assertEqual(metrics["pending_estimated_tokens"], 4)

    async def test_namespace_active_token_budget_isolates_tenants(self):
        self.engine = AsyncLLMEngine(
            "fake",
            engine_cls=StuckEngine,
            max_active_tokens_per_namespace=5,
        )

        await self.engine.add_request(
            "seed",
            SamplingParams(max_tokens=2),
            request_id="tenant-a-seed",
            cache_options={"cache_namespace": "tenant-a"},
        )
        for _ in range(20):
            if "tenant-a-seed" in self.engine.active:
                break
            await asyncio.sleep(0)
        self.assertIn("tenant-a-seed", self.engine.active)

        await self.engine.add_request(
            "big one two",
            SamplingParams(max_tokens=1),
            request_id="tenant-a-big",
            priority=10,
            cache_options={"cache_namespace": "tenant-a"},
        )
        await self.engine.add_request(
            "other one two",
            SamplingParams(max_tokens=1),
            request_id="tenant-b-fit",
            priority=0,
            cache_options={"cache_namespace": "tenant-b"},
        )

        for _ in range(20):
            if "tenant-b-fit" in self.engine.active:
                break
            await asyncio.sleep(0.002)

        metrics = self.engine.metrics()
        self.assertIn("tenant-b-fit", self.engine.active)
        self.assertNotIn("tenant-a-big", self.engine.active)
        self.assertEqual([request.request_id for request in self.engine.pending], ["tenant-a-big"])
        self.assertEqual(metrics["active_estimated_tokens_by_namespace"], {"tenant-a": 3, "tenant-b": 4})
        self.assertEqual(metrics["pending_estimated_tokens_by_namespace"], {"tenant-a": 4})

    async def test_request_namespace_overrides_cache_namespace_for_resource_budget(self):
        self.engine = AsyncLLMEngine(
            "fake",
            engine_cls=StuckEngine,
            max_active_tokens_per_namespace=5,
        )

        await self.engine.add_request(
            "seed",
            SamplingParams(max_tokens=2),
            request_id="resource-a",
            cache_options={"cache_namespace": "shared-cache"},
            request_namespace="tenant-a",
        )
        await self.engine.add_request(
            "other",
            SamplingParams(max_tokens=2),
            request_id="resource-b",
            cache_options={"cache_namespace": "shared-cache"},
            request_namespace="tenant-b",
        )

        for _ in range(20):
            if {"resource-a", "resource-b"}.issubset(self.engine.active):
                break
            await asyncio.sleep(0.002)

        metrics = self.engine.metrics()
        self.assertIn("resource-a", self.engine.active)
        self.assertIn("resource-b", self.engine.active)
        self.assertEqual(metrics["active_estimated_tokens_by_namespace"], {"tenant-a": 3, "tenant-b": 3})

    async def test_pending_request_namespace_budget_rejects_queue_hog(self):
        self.engine = AsyncLLMEngine(
            "fake",
            engine_cls=StuckEngine,
            max_active_requests=0,
            max_pending_requests_per_namespace=1,
        )

        await self.engine.add_request(
            "one",
            SamplingParams(max_tokens=2),
            request_id="tenant-a-1",
            request_namespace="tenant-a",
        )
        with self.assertRaisesRegex(RequestRejected, "namespace pending request budget"):
            await self.engine.add_request(
                "two",
                SamplingParams(max_tokens=2),
                request_id="tenant-a-2",
                request_namespace="tenant-a",
            )
        accepted = await self.engine.add_request(
            "three",
            SamplingParams(max_tokens=2),
            request_id="tenant-b-1",
            request_namespace="tenant-b",
        )

        metrics = self.engine.metrics()
        self.assertEqual(accepted.request_id, "tenant-b-1")
        self.assertEqual(metrics["pending_requests_by_namespace"], {"tenant-a": 1, "tenant-b": 1})
        self.assertEqual(metrics["max_pending_requests_per_namespace"], 1)
        self.assertEqual(metrics["rejected_requests"], 1)

    async def test_active_request_namespace_budget_skips_busy_tenant(self):
        self.engine = AsyncLLMEngine(
            "fake",
            engine_cls=StuckEngine,
            max_active_requests_per_namespace=1,
        )

        await self.engine.add_request(
            "seed",
            SamplingParams(max_tokens=2),
            request_id="tenant-a-active",
            request_namespace="tenant-a",
        )
        for _ in range(20):
            if "tenant-a-active" in self.engine.active:
                break
            await asyncio.sleep(0)
        self.assertIn("tenant-a-active", self.engine.active)

        await self.engine.add_request(
            "same",
            SamplingParams(max_tokens=2),
            request_id="tenant-a-pending",
            priority=10,
            request_namespace="tenant-a",
        )
        await self.engine.add_request(
            "other",
            SamplingParams(max_tokens=2),
            request_id="tenant-b-active",
            priority=0,
            request_namespace="tenant-b",
        )

        for _ in range(20):
            if "tenant-b-active" in self.engine.active:
                break
            await asyncio.sleep(0.002)

        metrics = self.engine.metrics()
        self.assertIn("tenant-b-active", self.engine.active)
        self.assertNotIn("tenant-a-pending", self.engine.active)
        self.assertEqual([request.request_id for request in self.engine.pending], ["tenant-a-pending"])
        self.assertEqual(metrics["active_requests_by_namespace"], {"tenant-a": 1, "tenant-b": 1})
        self.assertEqual(metrics["pending_requests_by_namespace"], {"tenant-a": 1})
        self.assertEqual(metrics["max_active_requests_per_namespace"], 1)

    async def test_aborted_pending_request_frees_capacity(self):
        self.engine = AsyncLLMEngine(
            "fake",
            engine_cls=StuckEngine,
            max_pending_requests=1,
            max_active_requests=0,
        )

        await self.engine.add_request("hello", SamplingParams(max_tokens=2), request_id="r1")
        aborted = await self.engine.abort("r1")
        second = await self.engine.add_request("hello", SamplingParams(max_tokens=2), request_id="r2")

        self.assertTrue(aborted)
        self.assertEqual(second.request_id, "r2")
        self.assertEqual(self.engine.metrics()["pending_requests"], 1)
        self.assertEqual([request.request_id for request in self.engine.pending], ["r2"])

    async def test_duplicate_request_id_is_rejected(self):
        self.engine = AsyncLLMEngine(
            "fake",
            engine_cls=StuckEngine,
            max_active_requests=0,
        )

        await self.engine.add_request("hello", SamplingParams(max_tokens=2), request_id="dup")

        with self.assertRaises(RequestRejected):
            await self.engine.add_request("hello", SamplingParams(max_tokens=2), request_id="dup")

    async def test_active_limit_keeps_extra_requests_pending(self):
        self.engine = AsyncLLMEngine(
            "fake",
            engine_cls=StuckEngine,
            max_active_requests=1,
        )

        await self.engine.add_request("hello", SamplingParams(max_tokens=2), request_id="a")
        await self.engine.add_request("hello", SamplingParams(max_tokens=2), request_id="b")

        for _ in range(10):
            metrics = self.engine.metrics()
            if metrics["active_requests"] == 1 and metrics["pending_requests"] == 1:
                break
            await asyncio.sleep(0)

        metrics = self.engine.metrics()
        self.assertEqual(metrics["active_requests"], 1)
        self.assertEqual(metrics["pending_requests"], 1)
        self.assertEqual(metrics["dequeued_requests"], 1)

    async def test_drain_rejects_new_requests_until_resume(self):
        self.engine = AsyncLLMEngine(
            "fake",
            engine_cls=StuckEngine,
            max_active_requests=1,
        )

        await self.engine.add_request("hello", SamplingParams(max_tokens=2), request_id="active")
        for _ in range(20):
            if "active" in self.engine.active:
                break
            await asyncio.sleep(0)
        self.assertIn("active", self.engine.active)

        drained_status = self.engine.set_draining(True)

        self.assertTrue(drained_status["draining"])
        self.assertFalse(drained_status["ready"])
        self.assertTrue(self.engine.metrics()["draining"])
        with self.assertRaisesRegex(RequestRejected, "draining"):
            await self.engine.add_request("blocked", SamplingParams(max_tokens=2), request_id="blocked")
        self.assertIn("active", self.engine.active)
        self.assertEqual(self.engine.metrics()["rejected_requests"], 1)

        resumed_status = self.engine.set_draining(False)
        self.assertFalse(resumed_status["draining"])
        self.assertTrue(resumed_status["ready"])
        resumed = await self.engine.add_request("next", SamplingParams(max_tokens=2), request_id="next")

        self.assertEqual(resumed.request_id, "next")
        self.assertEqual(self.engine.metrics()["pending_requests"], 1)

    async def test_slow_consumer_output_queue_cancels_request(self):
        self.engine = AsyncLLMEngine(
            "fake",
            engine_cls=FakeEngine,
            output_queue_size=1,
        )
        request = await self.engine.add_request("hello", SamplingParams(max_tokens=2), request_id="slow")

        for _ in range(20):
            if request.cancelled:
                break
            await asyncio.sleep(0)

        self.assertTrue(request.cancelled)
        self.assertEqual(self.engine.metrics()["cancelled_requests"], 1)


if __name__ == "__main__":
    unittest.main()
