from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from nanovllm.engine.async_engine import RequestFailed, RequestRejected
from nanovllm.models.gpt_oss_compat import inspect_gpt_oss_config
from nanovllm.sampling_params import SamplingParams


_STOP = object()


@dataclass
class HFAutoRequest:
    request_id: str
    trace_id: str
    prompt: str | list[int]
    sampling_params: SamplingParams
    output_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    created_at: float = field(default_factory=time.time)
    first_token_at: float | None = None
    finished: bool = False
    cancelled: bool = False


class HFAutoAsyncEngine:
    """Minimal Transformers-backed serving path for gpt-oss smoke validation.

    This path intentionally does not claim nano-vLLM continuous batching,
    paged-KV scheduling, or custom-kernel performance. It is a compatibility
    bridge for server-side gpt-oss validation while the native path remains
    fail-fast.
    """

    def __init__(self, model: str, output_queue_size: int = 16, **kwargs):
        self.model = model
        self.output_queue_size = output_queue_size
        self.kwargs = kwargs
        self.tokenizer = None
        self.model_obj = None
        self.started = False
        self.draining = False
        self.active: dict[str, HFAutoRequest] = {}
        self.tasks: dict[str, asyncio.Task] = {}
        self.submitted_requests = 0
        self.finished_requests = 0
        self.cancelled_requests = 0
        self.failed_requests = 0
        self.streamed_tokens = 0
        self.total_latency_s = 0.0
        self.total_ttft_s = 0.0
        self.ttft_count = 0
        self.compatibility_report: dict[str, Any] | None = None

    async def start(self):
        if self.started:
            return
        try:
            import torch
            from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        except Exception as exc:
            raise RequestFailed(
                "model_backend='hf_auto' requires torch and transformers on the server",
                "engine_error",
            ) from exc
        config = AutoConfig.from_pretrained(self.model, trust_remote_code=True)
        self.compatibility_report = inspect_gpt_oss_config(config).to_dict()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model, trust_remote_code=True)
        self.model_obj = AutoModelForCausalLM.from_pretrained(
            self.model,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True,
        )
        self.model_obj.eval()
        self.started = True

    async def shutdown(self):
        for request_id in list(self.active):
            await self.abort(request_id)
        self.started = False
        self.model_obj = None
        self.tokenizer = None

    async def restart(self):
        if self.active:
            raise RequestRejected("engine has in-flight requests; drain or cancel before restart")
        await self.shutdown()
        await self.start()
        return self.status()

    def status(self):
        return {
            "started": self.started,
            "ready": self.started and not self.draining,
            "draining": self.draining,
            "fatal_error": None,
            "engine_loop_running": self.started,
            "engine_loop_errors": 0,
            "last_engine_error": None,
            "model_backend": "hf_auto",
        }

    def set_draining(self, enabled: bool = True):
        self.draining = bool(enabled)
        return self.status()

    def metrics(self):
        completed = self.finished_requests + self.failed_requests
        avg_latency = self.total_latency_s / completed if completed else 0.0
        avg_ttft = self.total_ttft_s / self.ttft_count if self.ttft_count else 0.0
        return {
            "model_backend": "hf_auto",
            "attention_backend": "transformers",
            "scheduler_policy": "hf_auto",
            "waiting": 0,
            "running": len(self.active),
            "active_requests": len(self.active),
            "active_requests_by_namespace": {},
            "active_estimated_tokens": 0,
            "active_estimated_tokens_by_namespace": {},
            "pending_requests": 0,
            "pending_requests_by_namespace": {},
            "pending_prompt_tokens": 0,
            "pending_estimated_tokens": 0,
            "pending_estimated_tokens_by_namespace": {},
            "submitted_requests": self.submitted_requests,
            "finished_requests": self.finished_requests,
            "cancelled_requests": self.cancelled_requests,
            "failed_requests": self.failed_requests,
            "streamed_tokens": self.streamed_tokens,
            "prefill_tokens": 0,
            "decode_tokens": self.streamed_tokens,
            "avg_prefill_tok_s": 0.0,
            "avg_decode_tok_s": 0.0,
            "recent_prefill_tok_s": 0.0,
            "recent_decode_tok_s": 0.0,
            "avg_latency_s": avg_latency,
            "recent_latency_p50_s": avg_latency,
            "recent_latency_p95_s": avg_latency,
            "recent_latency_p99_s": avg_latency,
            "avg_ttft_s": avg_ttft,
            "recent_ttft_p50_s": avg_ttft,
            "recent_ttft_p95_s": avg_ttft,
            "recent_ttft_p99_s": avg_ttft,
            "avg_queue_wait_s": 0.0,
            "recent_queue_wait_p50_s": 0.0,
            "recent_queue_wait_p95_s": 0.0,
            "num_blocks": 0,
            "free_blocks": 0,
            "used_blocks": 0,
            "cached_blocks": 0,
            "cached_blocks_by_namespace": {},
            "prefix_cache_hit_rate": 0.0,
            "prefix_cache_hits": 0,
            "prefix_cache_misses": 0,
            "prefix_cache_hits_by_namespace": {},
            "prefix_cache_misses_by_namespace": {},
            "prefix_cache_miss_reasons": {},
            "cache_read_input_tokens_by_namespace": {},
            "cache_creation_input_tokens_by_namespace": {},
            "preemptions": 0,
            "evictions": 0,
            "global_quota_evictions": 0,
            "namespace_quota_evictions": 0,
            "duplicate_cache_blocks_skipped": 0,
            "expired_purges": 0,
            "gpt_oss_compatibility": self.compatibility_report or {},
        }

    async def purge_prefix_cache(self, namespace: str | None = None, expired_only: bool = False):
        return 0

    async def cache_inspect(self):
        return {
            "model_backend": "hf_auto",
            "prefix_cache_supported": False,
            "prefix_cache_miss_reasons": {},
            "cached_blocks": 0,
            "prefix_cache_hit_rate": 0.0,
            "gpt_oss_compatibility": self.compatibility_report or {},
        }

    async def add_request(
        self,
        prompt: str | list[int],
        sampling_params: SamplingParams,
        request_id: str | None = None,
        trace_id: str | None = None,
        **kwargs,
    ):
        await self.start()
        if self.draining:
            raise RequestRejected("engine is draining")
        request_id = request_id or uuid.uuid4().hex
        if request_id in self.active:
            raise RequestRejected(f"duplicate request_id: {request_id}")
        request = HFAutoRequest(
            request_id=request_id,
            trace_id=trace_id or request_id,
            prompt=prompt,
            sampling_params=sampling_params,
            output_queue=asyncio.Queue(maxsize=self.output_queue_size),
        )
        self.submitted_requests += 1
        self.active[request_id] = request
        self.tasks[request_id] = asyncio.create_task(self._run_request(request))
        return request

    async def abort(self, request_id: str):
        request = self.active.pop(request_id, None)
        task = self.tasks.pop(request_id, None)
        if request is None:
            return False
        request.cancelled = True
        request.finished = True
        if task is not None:
            task.cancel()
        self.cancelled_requests += 1
        await self._put_or_drop(request, _STOP)
        return True

    async def iter_request(self, request: HFAutoRequest) -> AsyncIterator[dict[str, Any]]:
        try:
            while True:
                item = await request.output_queue.get()
                if item is _STOP:
                    break
                yield item
                if item.get("finished"):
                    break
        finally:
            if not request.finished:
                await self.abort(request.request_id)

    async def generate(
        self,
        prompt: str | list[int],
        sampling_params: SamplingParams,
        request_id: str | None = None,
        trace_id: str | None = None,
        **kwargs,
    ):
        token_ids = []
        final = None
        async for output in self.generate_stream(
            prompt,
            sampling_params,
            request_id=request_id,
            trace_id=trace_id,
            **kwargs,
        ):
            if output.get("token_id") is not None:
                token_ids.append(output["token_id"])
            final = output
        return {
            "request_id": final["request_id"] if final else request_id,
            "trace_id": final.get("trace_id") if final else trace_id,
            "text": self.tokenizer.decode(token_ids, skip_special_tokens=True) if token_ids else "",
            "token_ids": token_ids,
            "finish_reason": final.get("finish_reason") if final else None,
            "usage": final.get("usage") if final else None,
        }

    async def generate_stream(self, prompt: str | list[int], sampling_params: SamplingParams, **kwargs):
        request = await self.add_request(prompt, sampling_params, **kwargs)
        async for output in self.iter_request(request):
            yield output

    async def _put_or_drop(self, request: HFAutoRequest, item: Any):
        try:
            request.output_queue.put_nowait(item)
        except asyncio.QueueFull:
            await self.abort(request.request_id)

    async def _run_request(self, request: HFAutoRequest):
        assert self.tokenizer is not None and self.model_obj is not None
        started = time.time()
        try:
            import torch

            device = next(self.model_obj.parameters()).device
            if isinstance(request.prompt, list):
                input_ids = torch.tensor([request.prompt], device=device)
                prompt_len = len(request.prompt)
            else:
                encoded = self.tokenizer(request.prompt, return_tensors="pt")
                input_ids = encoded["input_ids"].to(device)
                prompt_len = int(input_ids.shape[-1])
            is_cache_prewarm = request.sampling_params.max_tokens == 0
            if is_cache_prewarm:
                completion_ids = []
            else:
                with torch.inference_mode():
                    output_ids = self.model_obj.generate(
                        input_ids=input_ids,
                        max_new_tokens=request.sampling_params.max_tokens,
                        do_sample=request.sampling_params.temperature > 0,
                        temperature=max(request.sampling_params.temperature, 1e-5),
                        pad_token_id=getattr(self.tokenizer, "eos_token_id", None),
                    )
                completion_ids = output_ids[0, prompt_len:].detach().cpu().tolist()
            for index, token_id in enumerate(completion_ids):
                if request.cancelled:
                    break
                if request.first_token_at is None:
                    request.first_token_at = time.time()
                    self.total_ttft_s += request.first_token_at - request.created_at
                    self.ttft_count += 1
                self.streamed_tokens += 1
                await self._put_or_drop(request, {
                    "request_id": request.request_id,
                    "trace_id": request.trace_id,
                    "token_id": token_id,
                    "text": self.tokenizer.decode([token_id], skip_special_tokens=True),
                    "completion_token_ids": completion_ids[: index + 1],
                    "finished": False,
                    "finish_reason": None,
                })
            request.finished = True
            latency = time.time() - started
            self.total_latency_s += latency
            self.finished_requests += 1
            await self._put_or_drop(request, {
                "request_id": request.request_id,
                "trace_id": request.trace_id,
                "token_id": None,
                "text": "",
                "completion_token_ids": completion_ids,
                "finished": True,
                "finish_reason": "cache_warmed" if is_cache_prewarm else ("length" if completion_ids else "stop"),
                "usage": {
                    "prompt_tokens": prompt_len,
                    "input_tokens": prompt_len,
                    "cache_read_input_tokens": 0,
                    "cache_creation_input_tokens": 0,
                },
            })
        except asyncio.CancelledError:
            return
        except Exception as exc:
            request.finished = True
            self.failed_requests += 1
            await self._put_or_drop(request, {
                "request_id": request.request_id,
                "trace_id": request.trace_id,
                "error": str(exc),
                "error_type": "engine_error",
                "finished": True,
                "finish_reason": "error",
            })
        finally:
            self.active.pop(request.request_id, None)
            self.tasks.pop(request.request_id, None)
            await self._put_or_drop(request, _STOP)
