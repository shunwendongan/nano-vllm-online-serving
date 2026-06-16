import asyncio
import json
import math
import os
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, AsyncIterator

from nanovllm.sampling_params import SamplingParams

if TYPE_CHECKING:
    from nanovllm.engine.llm_engine import LLMEngine


_STOP = object()


class RequestRejected(RuntimeError):
    pass


class RequestFailed(RuntimeError):
    def __init__(self, message: str, error_type: str = "engine_error"):
        super().__init__(message)
        self.error_type = error_type


@dataclass
class AsyncRequest:
    request_id: str
    trace_id: str
    prompt: str | list[int]
    sampling_params: SamplingParams
    cache_options: dict[str, Any] = field(default_factory=dict)
    output_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    created_at: float = field(default_factory=time.time)
    admitted_at: float | None = None
    first_token_at: float | None = None
    finished_at: float | None = None
    request_timeout_s: float | None = None
    queue_timeout_s: float | None = None
    priority: int = 0
    namespace: str = ""
    prompt_token_count: int = 0
    estimated_token_count: int = 0
    finished: bool = False
    cancelled: bool = False


class AsyncLLMEngine:
    def __init__(
        self,
        model: str,
        engine_cls=None,
        max_pending_requests: int = 1024,
        max_active_requests: int | None = None,
        output_queue_size: int = 1024,
        request_timeout_s: float | None = None,
        queue_timeout_s: float | None = None,
        request_log_path: str | None = None,
        max_pending_prompt_tokens: int | None = None,
        max_active_tokens: int | None = None,
        max_active_tokens_per_namespace: int | None = None,
        max_pending_requests_per_namespace: int | None = None,
        max_active_requests_per_namespace: int | None = None,
        metrics_window_size: int = 1024,
        **engine_kwargs,
    ):
        self.model = model
        self.engine_kwargs = engine_kwargs
        self.engine_cls = engine_cls
        self.max_pending_requests = max_pending_requests
        self.max_active_requests = max_active_requests
        self.output_queue_size = output_queue_size
        self.default_request_timeout_s = request_timeout_s
        self.default_queue_timeout_s = queue_timeout_s
        self.request_log_path = request_log_path
        self.max_pending_prompt_tokens = max_pending_prompt_tokens
        self.max_active_tokens = max_active_tokens
        self.max_active_tokens_per_namespace = (
            max_active_tokens_per_namespace
            if max_active_tokens_per_namespace and max_active_tokens_per_namespace > 0
            else None
        )
        self.max_pending_requests_per_namespace = (
            max_pending_requests_per_namespace
            if max_pending_requests_per_namespace and max_pending_requests_per_namespace > 0
            else None
        )
        self.max_active_requests_per_namespace = (
            max_active_requests_per_namespace
            if max_active_requests_per_namespace and max_active_requests_per_namespace > 0
            else None
        )
        self.metrics_window_size = max(1, int(metrics_window_size))
        self.engine: "LLMEngine | None" = None
        self.pending: deque[AsyncRequest] = deque()
        self.pending_by_id: dict[str, AsyncRequest] = {}
        self.active: dict[str, AsyncRequest] = {}
        self.draining = False
        self._task: asyncio.Task | None = None
        self._shutdown = asyncio.Event()
        self.submitted_requests = 0
        self.dequeued_requests = 0
        self.finished_requests = 0
        self.cancelled_requests = 0
        self.failed_requests = 0
        self.timed_out_requests = 0
        self.rejected_requests = 0
        self.streamed_tokens = 0
        self.prefill_tokens = 0
        self.decode_tokens = 0
        self.prefill_elapsed_s = 0.0
        self.decode_elapsed_s = 0.0
        self.total_ttft_s = 0.0
        self.max_ttft_s = 0.0
        self.ttft_count = 0
        self.total_latency_s = 0.0
        self.max_latency_s = 0.0
        self.total_queue_wait_s = 0.0
        self.max_queue_wait_s = 0.0
        self.loop_errors = 0
        self.last_engine_error: str | None = None
        self.fatal_error: str | None = None
        self.request_log_errors = 0
        self.last_request_log_error: str | None = None
        self.recent_ttft_s = deque(maxlen=self.metrics_window_size)
        self.recent_latency_s = deque(maxlen=self.metrics_window_size)
        self.recent_queue_wait_s = deque(maxlen=self.metrics_window_size)
        self.recent_prefill_steps = deque(maxlen=self.metrics_window_size)
        self.recent_decode_steps = deque(maxlen=self.metrics_window_size)

    async def start(self):
        if self.fatal_error is not None:
            return
        if self.engine is None:
            if self.engine_cls is None:
                from nanovllm.engine.llm_engine import LLMEngine
                self.engine_cls = LLMEngine
            self.engine = self.engine_cls(self.model, **self.engine_kwargs)
        if self._task is None or self._task.done():
            self._shutdown.clear()
            self._task = asyncio.create_task(self._run_loop())

    async def shutdown(self):
        await self._stop_loop()
        if self.engine is not None:
            self.engine.exit()
            self.engine = None

    async def _stop_loop(self):
        self._shutdown.set()
        if self._task is not None:
            try:
                await self._task
            except Exception:
                pass
            self._task = None

    async def restart(self):
        if self.active or self.pending_by_id:
            raise RequestRejected("engine has in-flight requests; drain or cancel before restart")
        await self._stop_loop()
        if self.engine is not None:
            self.engine.exit()
            self.engine = None
        self.fatal_error = None
        self._shutdown.clear()
        try:
            await self.start()
        except Exception as exc:
            self.loop_errors += 1
            self.last_engine_error = str(exc)
            self.fatal_error = str(exc)
            self._log_event("engine_restart_failed", error=str(exc), error_type="engine_error")
            raise RequestFailed(str(exc), "engine_error") from exc
        self._log_event("engine_restarted")
        return self.status()

    @property
    def tokenizer(self):
        assert self.engine is not None
        return self.engine.tokenizer

    def metrics(self):
        if self.engine is None:
            return {"draining": self.draining}
        completed_requests = self.finished_requests + self.failed_requests + self.timed_out_requests
        return {
            **self.engine.metrics(),
            "draining": self.draining,
            "active_requests": len(self.active),
            "active_requests_by_namespace": self._active_requests_by_namespace(),
            "active_estimated_tokens": self._active_estimated_tokens(),
            "active_estimated_tokens_by_namespace": self._active_estimated_tokens_by_namespace(),
            "pending_requests": len(self.pending_by_id),
            "pending_requests_by_namespace": self._pending_requests_by_namespace(),
            "pending_prompt_tokens": self._pending_prompt_tokens(),
            "pending_estimated_tokens": self._pending_estimated_tokens(),
            "pending_estimated_tokens_by_namespace": self._pending_estimated_tokens_by_namespace(),
            "submitted_requests": self.submitted_requests,
            "dequeued_requests": self.dequeued_requests,
            "completed_requests": completed_requests,
            "finished_requests": self.finished_requests,
            "cancelled_requests": self.cancelled_requests,
            "failed_requests": self.failed_requests,
            "timed_out_requests": self.timed_out_requests,
            "rejected_requests": self.rejected_requests,
            "streamed_tokens": self.streamed_tokens,
            "prefill_tokens": self.prefill_tokens,
            "decode_tokens": self.decode_tokens,
            "avg_prefill_tok_s": self.prefill_tokens / self.prefill_elapsed_s if self.prefill_elapsed_s else 0.0,
            "avg_decode_tok_s": self.decode_tokens / self.decode_elapsed_s if self.decode_elapsed_s else 0.0,
            "recent_prefill_tok_s": self._recent_tok_s(self.recent_prefill_steps),
            "recent_decode_tok_s": self._recent_tok_s(self.recent_decode_steps),
            "avg_ttft_s": self.total_ttft_s / self.ttft_count if self.ttft_count else 0.0,
            "max_ttft_s": self.max_ttft_s,
            "recent_ttft_p50_s": self._percentile(self.recent_ttft_s, 0.50),
            "recent_ttft_p95_s": self._percentile(self.recent_ttft_s, 0.95),
            "recent_ttft_p99_s": self._percentile(self.recent_ttft_s, 0.99),
            "avg_queue_wait_s": self.total_queue_wait_s / self.dequeued_requests if self.dequeued_requests else 0.0,
            "max_queue_wait_s": self.max_queue_wait_s,
            "recent_queue_wait_p50_s": self._percentile(self.recent_queue_wait_s, 0.50),
            "recent_queue_wait_p95_s": self._percentile(self.recent_queue_wait_s, 0.95),
            "avg_latency_s": self.total_latency_s / completed_requests if completed_requests else 0.0,
            "max_latency_s": self.max_latency_s,
            "recent_latency_p50_s": self._percentile(self.recent_latency_s, 0.50),
            "recent_latency_p95_s": self._percentile(self.recent_latency_s, 0.95),
            "recent_latency_p99_s": self._percentile(self.recent_latency_s, 0.99),
            "engine_loop_running": self._task is not None and not self._task.done(),
            "engine_loop_errors": self.loop_errors,
            "last_engine_error": self.last_engine_error,
            "fatal_error": self.fatal_error,
            "request_log_errors": self.request_log_errors,
            "last_request_log_error": self.last_request_log_error,
            "max_pending_requests": self.max_pending_requests,
            "max_active_requests": self.max_active_requests,
            "default_request_timeout_s": self.default_request_timeout_s,
            "default_queue_timeout_s": self.default_queue_timeout_s,
            "max_pending_prompt_tokens": self.max_pending_prompt_tokens,
            "max_active_tokens": self.max_active_tokens,
            "max_active_tokens_per_namespace": self.max_active_tokens_per_namespace,
            "max_pending_requests_per_namespace": self.max_pending_requests_per_namespace,
            "max_active_requests_per_namespace": self.max_active_requests_per_namespace,
            "metrics_window_size": self.metrics_window_size,
        }

    def status(self):
        loop_running = self._task is not None and not self._task.done()
        ready = self.engine is not None and loop_running and not self.draining and self.fatal_error is None
        return {
            "started": self.engine is not None,
            "ready": ready,
            "draining": self.draining,
            "fatal_error": self.fatal_error,
            "engine_loop_running": loop_running,
            "engine_loop_errors": self.loop_errors,
            "last_engine_error": self.last_engine_error,
        }

    def set_draining(self, enabled: bool = True):
        self.draining = bool(enabled)
        self._log_event("drain_started" if self.draining else "drain_stopped")
        return self.status()

    def _prompt_stats(self, prompt: str | list[int]):
        if isinstance(prompt, str):
            return {"prompt_type": "text", "prompt_chars": len(prompt)}
        return {"prompt_type": "token_ids", "prompt_tokens": len(prompt)}

    def _request_log_fields(self, request: AsyncRequest | None):
        if request is None:
            return {}
        cache_options = request.cache_options or {}
        return {
            "request_id": request.request_id,
            "trace_id": request.trace_id,
            "age_s": max(time.time() - request.created_at, 0.0),
            "request_timeout_s": request.request_timeout_s,
            "queue_timeout_s": request.queue_timeout_s,
            "cache_namespace": cache_options.get("cache_namespace"),
            "cache_enabled": cache_options.get("cache_enabled", True),
            "cacheable_prefix_tokens": cache_options.get("cacheable_prefix_tokens"),
            "cache_breakpoint_tokens": cache_options.get("cache_breakpoint_tokens"),
            "sampling_max_tokens": getattr(request.sampling_params, "max_tokens", None),
            "priority": request.priority,
            "namespace": request.namespace,
            "prompt_token_count": request.prompt_token_count,
            "estimated_token_count": request.estimated_token_count,
        }

    def _pending_prompt_tokens(self):
        return sum(request.prompt_token_count for request in self.pending_by_id.values())

    def _pending_estimated_tokens(self):
        return sum(request.estimated_token_count for request in self.pending_by_id.values())

    def _active_estimated_tokens(self):
        return sum(request.estimated_token_count for request in self.active.values())

    def _estimated_tokens_by_namespace(self, requests):
        result: dict[str, int] = {}
        for request in requests:
            result[request.namespace] = result.get(request.namespace, 0) + request.estimated_token_count
        return result

    def _request_count_by_namespace(self, requests):
        result: dict[str, int] = {}
        for request in requests:
            result[request.namespace] = result.get(request.namespace, 0) + 1
        return result

    def _pending_estimated_tokens_by_namespace(self):
        return self._estimated_tokens_by_namespace(self.pending_by_id.values())

    def _active_estimated_tokens_by_namespace(self):
        return self._estimated_tokens_by_namespace(self.active.values())

    def _pending_requests_by_namespace(self):
        return self._request_count_by_namespace(self.pending_by_id.values())

    def _active_requests_by_namespace(self):
        return self._request_count_by_namespace(self.active.values())

    def _active_estimated_tokens_for_namespace(self, namespace: str):
        return sum(
            request.estimated_token_count
            for request in self.active.values()
            if request.namespace == namespace
        )

    def _pending_requests_for_namespace(self, namespace: str):
        return sum(1 for request in self.pending_by_id.values() if request.namespace == namespace)

    def _active_requests_for_namespace(self, namespace: str):
        return sum(1 for request in self.active.values() if request.namespace == namespace)

    def _can_admit_active(self, request: AsyncRequest):
        if request.cancelled:
            return True
        if self.max_active_requests is not None and len(self.active) >= self.max_active_requests:
            return False
        if (
            self.max_active_requests_per_namespace is not None
            and self._active_requests_for_namespace(request.namespace) >= self.max_active_requests_per_namespace
        ):
            return False
        if (
            self.max_active_tokens is not None
            and self._active_estimated_tokens() + request.estimated_token_count > self.max_active_tokens
        ):
            return False
        if (
            self.max_active_tokens_per_namespace is not None
            and self._active_estimated_tokens_for_namespace(request.namespace) + request.estimated_token_count
            > self.max_active_tokens_per_namespace
        ):
            return False
        return True

    @staticmethod
    def _percentile(values, p: float):
        if not values:
            return 0.0
        sorted_values = sorted(values)
        index = min(len(sorted_values) - 1, int(round((len(sorted_values) - 1) * p)))
        return sorted_values[index]

    @staticmethod
    def _recent_tok_s(steps):
        tokens = sum(item[0] for item in steps)
        elapsed = sum(item[1] for item in steps)
        return tokens / elapsed if elapsed else 0.0

    @staticmethod
    def _request_validation_error(message: str):
        raise RequestFailed(message, "request_validation")

    def _validate_timeout(self, value: float | None, name: str):
        if value is None:
            return None
        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise RequestFailed(f"{name} must be a number", "request_validation") from exc
        if not math.isfinite(value):
            self._request_validation_error(f"{name} must be finite")
        if value < 0:
            self._request_validation_error(f"{name} must be non-negative")
        return value

    def _validate_sampling_params(self, sampling_params: SamplingParams):
        try:
            sampling_params.max_tokens = int(sampling_params.max_tokens)
        except (TypeError, ValueError) as exc:
            raise RequestFailed("max_tokens must be an integer", "request_validation") from exc
        if sampling_params.max_tokens < 0:
            self._request_validation_error("max_tokens must be non-negative")
        try:
            sampling_params.temperature = float(sampling_params.temperature)
        except (TypeError, ValueError) as exc:
            raise RequestFailed("temperature must be a number", "request_validation") from exc
        if not math.isfinite(sampling_params.temperature):
            self._request_validation_error("temperature must be finite")
        if sampling_params.temperature < 0:
            self._request_validation_error("temperature must be non-negative")
        if not isinstance(sampling_params.ignore_eos, bool):
            self._request_validation_error("ignore_eos must be a boolean")

    def _prompt_token_count(self, prompt: str | list[int]):
        if isinstance(prompt, str):
            assert self.engine is not None
            return len(self.engine.tokenizer.encode(prompt))
        if isinstance(prompt, list):
            if not all(isinstance(token_id, int) for token_id in prompt):
                self._request_validation_error("prompt_token_ids must be integers")
            return len(prompt)
        self._request_validation_error("prompt must be a string or token id list")

    def _log_event(self, event: str, request: AsyncRequest | None = None, **fields):
        if not self.request_log_path:
            return
        record = {
            "ts": time.time(),
            "event": event,
            "model": self.model,
            **self._request_log_fields(request),
            **fields,
        }
        try:
            directory = os.path.dirname(os.path.abspath(self.request_log_path))
            if directory:
                os.makedirs(directory, exist_ok=True)
            with open(self.request_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        except Exception as exc:
            self.request_log_errors += 1
            self.last_request_log_error = str(exc)

    async def purge_prefix_cache(self, namespace: str | None = None, expired_only: bool = False):
        await self.start()
        assert self.engine is not None
        return self.engine.purge_prefix_cache(namespace=namespace, expired_only=expired_only)

    async def cache_inspect(self):
        await self.start()
        assert self.engine is not None
        return self.engine.cache_inspect()

    async def add_request(
        self,
        prompt: str | list[int],
        sampling_params: SamplingParams,
        request_id: str | None = None,
        trace_id: str | None = None,
        cache_options: dict[str, Any] | None = None,
        request_timeout_s: float | None = None,
        queue_timeout_s: float | None = None,
        priority: int = 0,
        request_namespace: str | None = None,
    ) -> AsyncRequest:
        await self.start()
        await self._expire_pending()
        resolved_request_id = request_id or uuid.uuid4().hex
        resolved_trace_id = str(trace_id or resolved_request_id)
        if self.fatal_error is not None:
            self.rejected_requests += 1
            self._log_event(
                "rejected",
                request_id=resolved_request_id,
                trace_id=resolved_trace_id,
                reason="engine_fatal_error",
            )
            raise RequestFailed(self.fatal_error, "engine_error")
        if self.draining:
            self.rejected_requests += 1
            self._log_event(
                "rejected",
                request_id=resolved_request_id,
                trace_id=resolved_trace_id,
                reason="engine_draining",
            )
            raise RequestRejected("engine is draining")
        self._validate_sampling_params(sampling_params)
        request_timeout_s = self._validate_timeout(
            self.default_request_timeout_s if request_timeout_s is None else request_timeout_s,
            "request_timeout_s",
        )
        queue_timeout_s = self._validate_timeout(
            self.default_queue_timeout_s if queue_timeout_s is None else queue_timeout_s,
            "queue_timeout_s",
        )
        try:
            priority = int(priority)
        except (TypeError, ValueError) as exc:
            raise RequestFailed("priority must be an integer", "request_validation") from exc
        if len(self.pending_by_id) >= self.max_pending_requests:
            self.rejected_requests += 1
            self._log_event(
                "rejected",
                request_id=resolved_request_id,
                trace_id=resolved_trace_id,
                reason="queue_full",
            )
            raise RequestRejected("request queue is full")
        if resolved_request_id in self.pending_by_id or resolved_request_id in self.active:
            self.rejected_requests += 1
            self._log_event(
                "rejected",
                request_id=resolved_request_id,
                trace_id=resolved_trace_id,
                reason="duplicate_request_id",
            )
            raise RequestRejected(f"duplicate request_id: {resolved_request_id}")
        namespace = request_namespace or (cache_options or {}).get("cache_namespace") or ""
        if (
            self.max_pending_requests_per_namespace is not None
            and self._pending_requests_for_namespace(namespace) >= self.max_pending_requests_per_namespace
        ):
            self.rejected_requests += 1
            self._log_event(
                "rejected",
                request_id=resolved_request_id,
                trace_id=resolved_trace_id,
                reason="pending_request_namespace_budget",
                namespace=namespace,
                pending_requests_for_namespace=self._pending_requests_for_namespace(namespace),
                max_pending_requests_per_namespace=self.max_pending_requests_per_namespace,
            )
            raise RequestRejected("namespace pending request budget exceeded")
        prompt_token_count = self._prompt_token_count(prompt)
        if prompt_token_count <= 0:
            self._request_validation_error("prompt must contain at least one token")
        estimated_token_count = prompt_token_count + max(0, getattr(sampling_params, "max_tokens", 0))
        if (
            self.max_pending_prompt_tokens is not None
            and self._pending_prompt_tokens() + prompt_token_count > self.max_pending_prompt_tokens
        ):
            self.rejected_requests += 1
            self._log_event(
                "rejected",
                request_id=resolved_request_id,
                trace_id=resolved_trace_id,
                reason="pending_prompt_token_budget",
                prompt_token_count=prompt_token_count,
                pending_prompt_tokens=self._pending_prompt_tokens(),
                max_pending_prompt_tokens=self.max_pending_prompt_tokens,
            )
            raise RequestRejected("pending prompt token budget exceeded")
        if self.max_active_tokens is not None and estimated_token_count > self.max_active_tokens:
            self.rejected_requests += 1
            self._log_event(
                "rejected",
                request_id=resolved_request_id,
                trace_id=resolved_trace_id,
                reason="single_request_active_token_budget",
                estimated_token_count=estimated_token_count,
                max_active_tokens=self.max_active_tokens,
            )
            raise RequestRejected("request exceeds active token budget")
        if (
            self.max_active_tokens_per_namespace is not None
            and estimated_token_count > self.max_active_tokens_per_namespace
        ):
            self.rejected_requests += 1
            self._log_event(
                "rejected",
                request_id=resolved_request_id,
                trace_id=resolved_trace_id,
                reason="single_request_namespace_active_token_budget",
                namespace=namespace,
                estimated_token_count=estimated_token_count,
                max_active_tokens_per_namespace=self.max_active_tokens_per_namespace,
            )
            raise RequestRejected("request exceeds namespace active token budget")
        request = AsyncRequest(
            request_id=resolved_request_id,
            trace_id=resolved_trace_id,
            prompt=prompt,
            sampling_params=sampling_params,
            cache_options=cache_options or {},
            output_queue=asyncio.Queue(maxsize=self.output_queue_size),
            request_timeout_s=request_timeout_s,
            queue_timeout_s=queue_timeout_s,
            priority=priority,
            namespace=namespace,
            prompt_token_count=prompt_token_count,
            estimated_token_count=estimated_token_count,
        )
        self.submitted_requests += 1
        self.pending_by_id[request.request_id] = request
        self._enqueue_pending(request)
        self._log_event("submitted", request, **self._prompt_stats(prompt))
        return request

    def _enqueue_pending(self, request: AsyncRequest):
        if not self.pending or request.priority <= self.pending[-1].priority:
            self.pending.append(request)
            return
        for index, pending_request in enumerate(self.pending):
            if request.priority > pending_request.priority:
                self.pending.insert(index, request)
                return
        self.pending.append(request)

    async def abort(self, request_id: str):
        pending_request = self.pending_by_id.get(request_id)
        if pending_request is not None:
            pending_request.cancelled = True
            pending_request.finished = True
            self.pending_by_id.pop(request_id, None)
            try:
                self.pending.remove(pending_request)
            except ValueError:
                pass
            self.cancelled_requests += 1
            self._log_event("cancelled", pending_request, reason="pending_abort")
            await self._put_or_drop(pending_request, _STOP)
            return True
        request = self.active.get(request_id)
        if request is not None:
            request.cancelled = True
            request.finished = True
            if self.engine is not None:
                self.engine.abort_request(request_id)
            await self._put_or_drop(request, _STOP)
            self.active.pop(request_id, None)
            self.cancelled_requests += 1
            self._log_event("cancelled", request, reason="active_abort")
            return True
        return False

    async def generate(
        self,
        prompt: str | list[int],
        sampling_params: SamplingParams,
        request_id: str | None = None,
        trace_id: str | None = None,
        cache_options: dict[str, Any] | None = None,
        request_timeout_s: float | None = None,
        queue_timeout_s: float | None = None,
        priority: int = 0,
        request_namespace: str | None = None,
    ):
        token_ids = []
        final: dict[str, Any] | None = None
        async for output in self.generate_stream(
            prompt,
            sampling_params,
            request_id=request_id,
            trace_id=trace_id,
            cache_options=cache_options,
            request_timeout_s=request_timeout_s,
            queue_timeout_s=queue_timeout_s,
            priority=priority,
            request_namespace=request_namespace,
        ):
            if output.get("error"):
                raise RequestFailed(output["error"], output.get("error_type", "engine_error"))
            if output.get("token_id") is not None:
                token_ids.append(output["token_id"])
            final = output
        assert self.engine is not None
        return {
            "request_id": final["request_id"] if final else request_id,
            "trace_id": final.get("trace_id") if final else trace_id,
            "text": self.engine.tokenizer.decode(token_ids),
            "token_ids": token_ids,
            "finish_reason": final["finish_reason"] if final else None,
            "usage": final.get("usage") if final else None,
        }

    async def generate_stream(
        self,
        prompt: str | list[int],
        sampling_params: SamplingParams,
        request_id: str | None = None,
        trace_id: str | None = None,
        cache_options: dict[str, Any] | None = None,
        request_timeout_s: float | None = None,
        queue_timeout_s: float | None = None,
        priority: int = 0,
        request_namespace: str | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        request = await self.add_request(
            prompt,
            sampling_params,
            request_id=request_id,
            trace_id=trace_id,
            cache_options=cache_options,
            request_timeout_s=request_timeout_s,
            queue_timeout_s=queue_timeout_s,
            priority=priority,
            request_namespace=request_namespace,
        )
        async for output in self.iter_request(request):
            yield output

    async def iter_request(self, request: AsyncRequest) -> AsyncIterator[dict[str, Any]]:
        try:
            while True:
                item = await request.output_queue.get()
                if item is _STOP:
                    break
                yield item
                if item.get("finished"):
                    request.finished = True
                    break
        finally:
            if not request.finished:
                await self.abort(request.request_id)

    async def _drain_pending(self):
        assert self.engine is not None
        while True:
            await self._expire_pending()
            if not self.pending:
                break
            if self.max_active_requests is not None and len(self.active) >= self.max_active_requests:
                break
            request = self._pop_next_admissible_pending()
            if request is None:
                break
            self.pending_by_id.pop(request.request_id, None)
            if request.cancelled:
                await self._put_or_drop(request, _STOP)
                continue
            self._record_queue_wait(request)
            try:
                self.engine.add_request(
                    request.prompt,
                    request.sampling_params,
                    request_id=request.request_id,
                    **request.cache_options,
                )
            except Exception as exc:
                await self._fail_request(request, exc)
                continue
            self.active[request.request_id] = request
            self._log_event(
                "admitted",
                request,
                queue_wait_s=(request.admitted_at - request.created_at) if request.admitted_at else None,
            )

    def _pop_next_admissible_pending(self):
        for request in list(self.pending):
            if self._can_admit_active(request):
                self.pending.remove(request)
                return request
        return None

    async def _expire_pending(self):
        now = time.time()
        for request in list(self.pending):
            if request.cancelled or request.queue_timeout_s is None:
                continue
            if now - request.created_at >= request.queue_timeout_s:
                self.pending.remove(request)
                self.pending_by_id.pop(request.request_id, None)
                await self._timeout_request(request, "queue_timeout")

    async def _expire_active(self):
        now = time.time()
        for request in list(self.active.values()):
            if request.cancelled or request.request_timeout_s is None:
                continue
            if now - request.created_at >= request.request_timeout_s:
                if self.engine is not None:
                    self.engine.abort_request(request.request_id)
                self.active.pop(request.request_id, None)
                await self._timeout_request(request, "request_timeout")

    def _record_queue_wait(self, request: AsyncRequest):
        request.admitted_at = time.time()
        queue_wait = request.admitted_at - request.created_at
        self.dequeued_requests += 1
        self.total_queue_wait_s += queue_wait
        self.max_queue_wait_s = max(self.max_queue_wait_s, queue_wait)
        self.recent_queue_wait_s.append(queue_wait)

    async def _fail_request(
        self,
        request: AsyncRequest,
        exc: Exception,
        error_type: str = "request_validation",
    ):
        request.finished = True
        request.finished_at = time.time()
        self.failed_requests += 1
        latency = request.finished_at - request.created_at
        self.total_latency_s += latency
        self.max_latency_s = max(self.max_latency_s, latency)
        self.recent_latency_s.append(latency)
        self._log_event("failed", request, error=str(exc), error_type=error_type)
        await self._put_or_drop(request, {
            "request_id": request.request_id,
            "trace_id": request.trace_id,
            "error": str(exc),
            "error_type": error_type,
            "finished": True,
            "finish_reason": "error",
        })
        await self._put_or_drop(request, _STOP)

    async def _timeout_request(self, request: AsyncRequest, reason: str):
        request.finished = True
        request.finished_at = time.time()
        self.timed_out_requests += 1
        latency = request.finished_at - request.created_at
        self.total_latency_s += latency
        self.max_latency_s = max(self.max_latency_s, latency)
        self.recent_latency_s.append(latency)
        self._log_event("timeout", request, reason=reason, latency_s=latency)
        await self._put_or_drop(request, {
            "request_id": request.request_id,
            "trace_id": request.trace_id,
            "error": reason,
            "error_type": "timeout",
            "finished": True,
            "finish_reason": "timeout",
        })
        await self._put_or_drop(request, _STOP)

    async def _put_or_drop(self, request: AsyncRequest, item: Any):
        try:
            request.output_queue.put_nowait(item)
            if item is not _STOP and not (isinstance(item, dict) and item.get("finished")):
                await asyncio.sleep(0)
            return True
        except asyncio.QueueFull:
            if item is _STOP:
                while True:
                    try:
                        request.output_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                request.output_queue.put_nowait(_STOP)
                return True
            request.cancelled = True
            request.finished = True
            if self.engine is not None:
                self.engine.abort_request(request.request_id)
            self.active.pop(request.request_id, None)
            self.pending_by_id.pop(request.request_id, None)
            self.cancelled_requests += 1
            self._log_event("cancelled", request, reason="slow_consumer")
            await self._put_or_drop(request, _STOP)
            return False

    async def _run_loop(self):
        assert self.engine is not None
        while not self._shutdown.is_set():
            try:
                await self._expire_pending()
                await self._expire_active()
                await self._drain_pending()
                await self._expire_active()
                if self.engine.is_finished():
                    await asyncio.sleep(0.001)
                    continue
                step_started_at = time.perf_counter()
                outputs, num_tokens = self.engine.step()
                step_elapsed = max(time.perf_counter() - step_started_at, 1e-9)
                if num_tokens > 0:
                    self.prefill_tokens += num_tokens
                    self.prefill_elapsed_s += step_elapsed
                    self.recent_prefill_steps.append((num_tokens, step_elapsed))
                elif num_tokens < 0:
                    self.decode_tokens += -num_tokens
                    self.decode_elapsed_s += step_elapsed
                    self.recent_decode_steps.append((-num_tokens, step_elapsed))
                for output in outputs:
                    request = self.active.get(output["request_id"])
                    if request is None:
                        continue
                    output["trace_id"] = request.trace_id
                    if output.get("token_id") is not None:
                        if request.first_token_at is None:
                            request.first_token_at = time.time()
                            ttft = request.first_token_at - request.created_at
                            self.total_ttft_s += ttft
                            self.max_ttft_s = max(self.max_ttft_s, ttft)
                            self.ttft_count += 1
                            self.recent_ttft_s.append(ttft)
                            self._log_event("first_token", request, ttft_s=ttft)
                        self.streamed_tokens += 1
                    delivered = await self._put_or_drop(request, output)
                    if not delivered:
                        continue
                    if output["finished"]:
                        request.finished = True
                        request.finished_at = time.time()
                        self.finished_requests += 1
                        latency = request.finished_at - request.created_at
                        self.total_latency_s += latency
                        self.max_latency_s = max(self.max_latency_s, latency)
                        self.recent_latency_s.append(latency)
                        self._log_event(
                            "finished",
                            request,
                            finish_reason=output.get("finish_reason"),
                            completion_tokens=len(output.get("completion_token_ids", [])),
                            usage=output.get("usage"),
                            latency_s=latency,
                        )
                        await self._put_or_drop(request, _STOP)
                        self.active.pop(request.request_id, None)
                await self._expire_active()
            except Exception as exc:
                self.loop_errors += 1
                self.last_engine_error = str(exc)
                self.fatal_error = str(exc)
                self._log_event("engine_loop_failed", error=str(exc), error_type="engine_error")
                for request in list(self.active.values()):
                    await self._fail_request(request, exc, error_type="engine_error")
                self.active.clear()
                for request in list(self.pending):
                    await self._fail_request(request, exc, error_type="engine_error")
                self.pending.clear()
                self.pending_by_id.clear()
                raise
