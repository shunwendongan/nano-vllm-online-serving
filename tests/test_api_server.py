import json
import unittest
import warnings
from dataclasses import dataclass
from unittest.mock import patch

try:
    from starlette.exceptions import StarletteDeprecationWarning
    from fastapi.testclient import TestClient
except ModuleNotFoundError:
    StarletteDeprecationWarning = Warning
    TestClient = None

warnings.filterwarnings(
    "ignore",
    category=StarletteDeprecationWarning,
    message="Using `httpx` with `starlette.testclient` is deprecated.*",
)

from nanovllm.entrypoints.openai.api_server import create_app
from nanovllm.engine.async_engine import RequestFailed, RequestRejected


class FakeTokenizer:
    def encode(self, text):
        return str(text).split()

    def decode(self, token_ids):
        return "".join(f"<{token_id}>" for token_id in token_ids)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        parts = [f"{message.get('role', 'user')}: {message.get('content', '')}" for message in messages]
        if add_generation_prompt:
            parts.append("assistant:")
        return "\n".join(parts)


@dataclass
class FakeRequest:
    request_id: str
    trace_id: str
    max_tokens: int = 0


class FakeAsyncEngine:
    instances = []

    def __init__(self, model, **kwargs):
        self.model = model
        self.kwargs = kwargs
        self.tokenizer = FakeTokenizer()
        self.started = False
        self.shutdown_called = False
        self.aborted = []
        self.added_requests = []
        self.draining = False
        self.fatal_error = None
        self.restarted = False
        FakeAsyncEngine.instances.append(self)

    async def start(self):
        self.started = True

    async def shutdown(self):
        self.shutdown_called = True

    def status(self):
        return {
            "started": self.started,
            "ready": self.started and not self.draining and self.fatal_error is None,
            "draining": self.draining,
            "fatal_error": self.fatal_error,
            "engine_loop_running": self.started,
            "engine_loop_errors": 0,
            "last_engine_error": None,
        }

    def metrics(self):
        return {
            "waiting": 1,
            "running": 2,
            "active_requests": 1,
            "pending_requests": 0,
            "num_blocks": 8,
            "free_blocks": 3,
            "cached_blocks": 2,
            "used_blocks": 3,
            "cached_blocks_by_namespace": {"tenant-a": 2},
            "prefix_cache_hits": 4,
            "prefix_cache_misses": 1,
            "prefix_cache_hits_by_namespace": {"tenant-a": 4},
            "prefix_cache_misses_by_namespace": {"tenant-a": 1},
            "cache_read_input_tokens_by_namespace": {"tenant-a": 8},
            "cache_creation_input_tokens_by_namespace": {"tenant-a": 4},
            "prefix_cache_hit_rate": 0.8,
            "evictions": 1,
            "global_quota_evictions": 1,
            "max_cached_blocks": 4,
            "duplicate_cache_blocks_skipped": 1,
            "expired_purges": 0,
            "recent_ttft_p95_s": 0.1,
            "recent_latency_p95_s": 0.2,
            "recent_decode_tok_s": 10.0,
            "scheduler_policy": "alternate",
            "attention_backend": "flash_attn",
            "model_backend": "native",
            "prefix_cache_miss_reasons": {"hash_miss": 1},
        }

    async def cache_inspect(self):
        return {
            "cached_blocks": 2,
            "prefix_cache_hit_rate": 0.8,
            "prefix_cache_miss_reasons": {"hash_miss": 1},
        }

    async def purge_prefix_cache(self, namespace=None, expired_only=False):
        self.purged_namespace = namespace
        self.purged_expired_only = expired_only
        return 2

    def set_draining(self, enabled=True):
        self.draining = bool(enabled)
        return self.status()

    async def restart(self):
        self.fatal_error = None
        self.restarted = True
        self.started = True
        return self.status()

    async def add_request(
        self,
        prompt,
        sampling_params,
        request_id=None,
        cache_options=None,
        trace_id=None,
        request_timeout_s=None,
        queue_timeout_s=None,
        priority=0,
        request_namespace=None,
    ):
        if self.fatal_error is not None:
            raise RequestFailed(self.fatal_error, "engine_error")
        if prompt == "reject":
            raise RequestRejected("request queue is full")
        record = {
            "prompt": prompt,
            "request_id": request_id,
            "trace_id": trace_id,
            "cache_options": cache_options or {},
            "request_timeout_s": request_timeout_s,
            "queue_timeout_s": queue_timeout_s,
            "priority": priority,
            "request_namespace": request_namespace,
            "max_tokens": sampling_params.max_tokens,
            "temperature": sampling_params.temperature,
        }
        self.added_requests.append(record)
        return FakeRequest(request_id, trace_id or request_id, sampling_params.max_tokens)

    async def generate(
        self,
        prompt,
        sampling_params,
        request_id=None,
        cache_options=None,
        **request_options,
    ):
        request = await self.add_request(
            prompt,
            sampling_params,
            request_id=request_id,
            cache_options=cache_options,
            **request_options,
        )
        if sampling_params.max_tokens == 0:
            return {
                "request_id": request.request_id,
                "trace_id": request.trace_id,
                "text": "",
                "token_ids": [],
                "finish_reason": "cache_warmed",
                "usage": {
                    "prompt_tokens": 5,
                    "input_tokens": 1,
                    "cache_read_input_tokens": 0,
                    "cache_creation_input_tokens": 4,
                },
            }
        token_ids = list(range(101, 101 + sampling_params.max_tokens))
        return {
            "request_id": request.request_id,
            "trace_id": request.trace_id,
            "text": self.tokenizer.decode(token_ids),
            "token_ids": token_ids,
            "finish_reason": "length",
            "usage": {
                "prompt_tokens": 5,
                "input_tokens": 1,
                "cache_read_input_tokens": 4,
                "cache_creation_input_tokens": 0,
            },
        }

    async def iter_request(self, request):
        if request.max_tokens == 0:
            yield {
                "request_id": request.request_id,
                "trace_id": request.trace_id,
                "seq_id": 0,
                "token_id": None,
                "text": "",
                "completion_token_ids": [],
                "finished": True,
                "finish_reason": "cache_warmed",
                "usage": {
                    "prompt_tokens": 5,
                    "input_tokens": 1,
                    "cache_read_input_tokens": 0,
                    "cache_creation_input_tokens": 4,
                },
            }
            return
        token_ids = list(range(101, 101 + request.max_tokens))
        for index, token_id in enumerate(token_ids):
            finished = index == len(token_ids) - 1
            yield {
                "request_id": request.request_id,
                "trace_id": request.trace_id,
                "seq_id": 0,
                "token_id": token_id,
                "text": self.tokenizer.decode([token_id]),
                "completion_token_ids": token_ids[: index + 1],
                "finished": finished,
                "finish_reason": "length" if finished else None,
                "usage": {
                    "prompt_tokens": 5,
                    "input_tokens": 1,
                    "cache_read_input_tokens": 4,
                    "cache_creation_input_tokens": 0,
                } if finished else None,
            }

    async def abort(self, request_id):
        self.aborted.append(request_id)
        return True


def sse_payloads(text):
    payloads = []
    for line in text.splitlines():
        if not line.startswith("data:"):
            continue
        data = line[len("data:"):].strip()
        payloads.append(data if data == "[DONE]" else json.loads(data))
    return payloads


@unittest.skipIf(TestClient is None, "fastapi/starlette are required for API server tests")
class ApiServerTest(unittest.TestCase):
    def setUp(self):
        FakeAsyncEngine.instances.clear()
        patcher = patch("nanovllm.entrypoints.openai.api_server.AsyncLLMEngine", FakeAsyncEngine)
        self.addCleanup(patcher.stop)
        patcher.start()
        self.app = create_app("fake-model", max_pending_requests=7, enable_prefix_cache=True)
        self.client_context = TestClient(self.app)
        self.client = self.client_context.__enter__()
        self.addCleanup(self.client_context.__exit__, None, None, None)
        self.engine = FakeAsyncEngine.instances[-1]

    def test_startup_forwards_engine_kwargs_and_health(self):
        self.assertTrue(self.engine.started)
        self.assertEqual(self.engine.kwargs["max_pending_requests"], 7)
        self.assertTrue(self.client.get("/healthz").json()["ok"])
        self.assertTrue(self.client.get("/readyz").json()["ready"])

    def test_generate_returns_full_text_and_forwards_cache_options(self):
        response = self.client.post(
            "/generate",
            json={
                "prompt": "stable prompt",
                "max_tokens": 2,
                "temperature": 0.0,
                "request_id": "req-generate",
                "cache_control": {
                    "type": "ephemeral",
                    "ttl": "5m",
                    "namespace": "tenant-a",
                    "cacheable_prefix_tokens": 2,
                },
                "request_namespace": "tenant-resource",
                "priority": 9,
                "queue_timeout_s": 1,
                "request_timeout_s": 2,
                "trace_id": "trace-generate",
            },
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["request_id"], "req-generate")
        self.assertEqual(body["trace_id"], "trace-generate")
        self.assertEqual(body["token_ids"], [101, 102])
        self.assertEqual(body["usage"]["cache_read_input_tokens"], 4)
        added = self.engine.added_requests[-1]
        self.assertEqual(added["cache_options"]["cache_namespace"], "tenant-a")
        self.assertEqual(added["cache_options"]["cacheable_prefix_tokens"], 2)
        self.assertEqual(added["request_namespace"], "tenant-resource")
        self.assertEqual(added["priority"], 9)
        self.assertEqual(added["queue_timeout_s"], 1)
        self.assertEqual(added["request_timeout_s"], 2)
        self.assertEqual(added["trace_id"], "trace-generate")

    def test_generate_stream_emits_sse_token_chunks(self):
        response = self.client.post(
            "/generate_stream",
            json={"prompt": "hello", "max_tokens": 2, "request_id": "stream-id"},
        )

        self.assertEqual(response.status_code, 200)
        payloads = sse_payloads(response.text)
        self.assertEqual([payload["token_id"] for payload in payloads], [101, 102])
        self.assertEqual([payload["trace_id"] for payload in payloads], ["stream-id", "stream-id"])
        self.assertTrue(payloads[-1]["finished"])
        self.assertIn("stream-id", self.engine.aborted)

    def test_generate_stream_coalesces_after_first_token(self):
        app = create_app("fake-model", stream_interval=2)
        client_context = TestClient(app)
        client = client_context.__enter__()
        self.addCleanup(client_context.__exit__, None, None, None)

        response = client.post(
            "/generate_stream",
            json={"prompt": "hello", "max_tokens": 4, "request_id": "coalesce-id"},
        )

        self.assertEqual(response.status_code, 200)
        payloads = sse_payloads(response.text)
        self.assertEqual([payload["text"] for payload in payloads], ["<101>", "<102><103>", "<104>"])
        self.assertEqual([payload["token_ids"] for payload in payloads], [[101], [102, 103], [104]])
        self.assertTrue(payloads[-1]["finished"])

    def test_openai_completion_non_stream_and_stream_shapes(self):
        response = self.client.post(
            "/v1/completions",
            json={"prompt": "hello", "max_tokens": 2, "request_id": "cmpl-id"},
        )
        body = response.json()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(body["object"], "text_completion")
        self.assertEqual(body["trace_id"], "cmpl-id")
        self.assertEqual(body["choices"][0]["text"], "<101><102>")
        self.assertEqual(body["usage"]["completion_tokens"], 2)
        self.assertEqual(body["usage"]["cache_read_input_tokens"], 4)

        stream = self.client.post(
            "/v1/completions",
            json={"prompt": "hello", "stream": True, "max_tokens": 2, "request_id": "cmpl-stream"},
        )
        payloads = sse_payloads(stream.text)
        self.assertEqual(payloads[-1], "[DONE]")
        self.assertEqual(payloads[0]["object"], "text_completion")
        self.assertEqual(payloads[0]["trace_id"], "cmpl-stream")
        self.assertEqual(payloads[0]["choices"][0]["text"], "<101>")

    def test_openai_chat_completion_collects_message_cache_breakpoints(self):
        response = self.client.post(
            "/v1/chat/completions",
            json={
                "messages": [
                    {"role": "system", "content": "stable", "cache_control": {"type": "ephemeral"}},
                    {"role": "user", "content": "question"},
                ],
                "max_tokens": 2,
                "request_id": "chat-id",
            },
        )
        body = response.json()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(body["object"], "chat.completion")
        self.assertEqual(body["choices"][0]["message"]["content"], "<101><102>")
        self.assertTrue(self.engine.added_requests[-1]["cache_options"]["cache_breakpoint_tokens"])

    def test_metrics_cache_admin_and_rejection_paths(self):
        metrics = self.client.get("/metrics").json()
        self.assertEqual(metrics["prefix_cache_hit_rate"], 0.8)

        prometheus = self.client.get("/metrics/prometheus")
        self.assertEqual(prometheus.status_code, 200)
        self.assertIn('nanovllm_cached_blocks_by_namespace{namespace="tenant-a"} 2', prometheus.text)

        cache_stats = self.client.get("/cache/stats").json()
        self.assertEqual(cache_stats["cached_blocks_by_namespace"], {"tenant-a": 2})

        cache_inspect = self.client.get("/cache/inspect").json()
        self.assertEqual(cache_inspect["prefix_cache_miss_reasons"], {"hash_miss": 1})

        purge = self.client.post("/cache/purge", json={"namespace": "tenant-a", "expired_only": True}).json()
        self.assertEqual(purge["purged_blocks"], 2)
        self.assertEqual(self.engine.purged_namespace, "tenant-a")
        self.assertTrue(self.engine.purged_expired_only)

        drain = self.client.post("/admin/drain").json()
        self.assertFalse(drain["ready"])
        self.assertEqual(self.client.get("/readyz").status_code, 503)
        resume = self.client.post("/admin/resume").json()
        self.assertTrue(resume["ready"])

        rejected = self.client.post("/generate", json={"prompt": "reject", "max_tokens": 2})
        self.assertEqual(rejected.status_code, 429)

    def test_cache_prewarm_for_prompt_and_chat_payloads(self):
        prompt_response = self.client.post(
            "/cache/prewarm",
            json={
                "prompt": "stable prompt",
                "request_id": "prewarm-prompt",
                "cache_control": {
                    "type": "ephemeral",
                    "ttl": "1h",
                    "namespace": "tenant-a",
                },
            },
        )
        prompt_body = prompt_response.json()

        self.assertEqual(prompt_response.status_code, 200)
        self.assertEqual(prompt_body["request_id"], "prewarm-prompt")
        self.assertEqual(prompt_body["finish_reason"], "cache_warmed")
        self.assertEqual(prompt_body["usage"]["cache_creation_input_tokens"], 4)
        self.assertEqual(prompt_body["cache"]["namespace"], "tenant-a")
        self.assertEqual(prompt_body["cache"]["ttl_seconds"], 3600)
        self.assertEqual(self.engine.added_requests[-1]["max_tokens"], 0)

        chat_response = self.client.post(
            "/cache/prewarm",
            json={
                "messages": [
                    {"role": "system", "content": "stable", "cache_control": {"type": "ephemeral"}},
                    {"role": "user", "content": "question"},
                ],
                "request_id": "prewarm-chat",
            },
        )
        chat_body = chat_response.json()

        self.assertEqual(chat_response.status_code, 200)
        self.assertEqual(chat_body["finish_reason"], "cache_warmed")
        self.assertTrue(chat_body["cache"]["breakpoints"])

    def test_engine_fatal_error_maps_to_500_for_streaming_admission(self):
        self.engine.fatal_error = "fatal engine error"

        self.assertEqual(self.client.get("/readyz").status_code, 503)
        simple_stream = self.client.post(
            "/generate_stream",
            json={"prompt": "hello", "max_tokens": 2, "request_id": "fatal-simple-stream"},
        )
        simple_generate_stream_flag = self.client.post(
            "/generate",
            json={"prompt": "hello", "stream": True, "max_tokens": 2, "request_id": "fatal-generate-stream"},
        )
        completion_stream = self.client.post(
            "/v1/completions",
            json={"prompt": "hello", "stream": True, "max_tokens": 2, "request_id": "fatal-cmpl-stream"},
        )
        chat_stream = self.client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
                "max_tokens": 2,
                "request_id": "fatal-chat-stream",
            },
        )

        self.assertEqual(simple_stream.status_code, 500)
        self.assertEqual(simple_generate_stream_flag.status_code, 500)
        self.assertEqual(completion_stream.status_code, 500)
        self.assertEqual(chat_stream.status_code, 500)

    def test_bad_request_payloads_map_to_http_400(self):
        bad_generate = self.client.post("/generate", json={"prompt": "hello", "max_tokens": -1})
        bad_stream = self.client.post(
            "/generate_stream",
            json={"prompt": "hello", "max_tokens": 2, "queue_timeout_s": -1},
        )
        bad_completion = self.client.post(
            "/v1/completions",
            json={"prompt": "hello", "max_tokens": 2, "temperature": -1},
        )
        bad_chat = self.client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 2,
                "ignore_eos": "maybe",
            },
        )

        self.assertEqual(bad_generate.status_code, 400)
        self.assertEqual(bad_stream.status_code, 400)
        self.assertEqual(bad_completion.status_code, 400)
        self.assertEqual(bad_chat.status_code, 400)

    def test_admin_restart_clears_fatal_state(self):
        self.engine.fatal_error = "fatal engine error"
        self.assertEqual(self.client.get("/readyz").status_code, 503)

        restarted = self.client.post("/admin/restart").json()

        self.assertTrue(restarted["ok"])
        self.assertTrue(restarted["ready"])
        self.assertIsNone(restarted["fatal_error"])
        self.assertTrue(self.engine.restarted)
        self.assertEqual(self.client.get("/readyz").status_code, 200)


if __name__ == "__main__":
    unittest.main()
