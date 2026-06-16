import json
import math
import os
import re
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator

from nanovllm.engine.async_engine import AsyncLLMEngine, RequestFailed, RequestRejected
from nanovllm.entrypoints.protocol import (
    cache_options_from_payload,
    chat_prompt_and_cache_options,
    prompt_from_payload,
    request_options_from_payload,
    sampling_params_from_payload,
)


def _json_sse(payload: dict):
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _normalize_stream_interval(value):
    try:
        interval = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"stream_interval must be an integer, got {value!r}") from exc
    if interval < 1:
        raise ValueError("stream_interval must be >= 1")
    return interval


def _merge_stream_outputs(outputs: list[dict]):
    if len(outputs) == 1:
        output = dict(outputs[0])
        if output.get("token_id") is not None:
            output.setdefault("token_ids", [output["token_id"]])
        return output

    merged = dict(outputs[-1])
    token_ids = [output["token_id"] for output in outputs if output.get("token_id") is not None]
    merged["text"] = "".join(output.get("text", "") for output in outputs)
    merged["token_ids"] = token_ids
    if token_ids:
        merged["token_id"] = token_ids[-1]
    return merged


async def _iter_stream_outputs(engine, engine_request, stream_interval: int):
    buffer: list[dict] = []
    buffered_tokens = 0
    emitted_first_token = False

    async for output in engine.iter_request(engine_request):
        token_id = output.get("token_id")
        if token_id is None:
            if buffer:
                yield _merge_stream_outputs(buffer)
                buffer = []
                buffered_tokens = 0
            yield output
            continue

        buffer.append(output)
        buffered_tokens += 1
        should_flush = output.get("finished") or output.get("error")
        if not emitted_first_token:
            emitted_first_token = True
            should_flush = True
        elif buffered_tokens >= stream_interval:
            should_flush = True

        if should_flush:
            yield _merge_stream_outputs(buffer)
            buffer = []
            buffered_tokens = 0

    if buffer:
        yield _merge_stream_outputs(buffer)


def _model_name(model: str):
    return os.path.basename(os.path.normpath(model)) or model


def _openai_usage(prompt_usage: dict | None, completion_tokens: int):
    prompt_usage = prompt_usage or {}
    prompt_tokens = prompt_usage.get("prompt_tokens", 0)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "input_tokens": prompt_usage.get("input_tokens", prompt_tokens),
        "cache_read_input_tokens": prompt_usage.get("cache_read_input_tokens", 0),
        "cache_creation_input_tokens": prompt_usage.get("cache_creation_input_tokens", 0),
    }


def _openai_model_list(model_name: str, created: int | None = None):
    return {
        "object": "list",
        "data": [{
            "id": model_name,
            "object": "model",
            "created": created or 0,
            "owned_by": "nano-vllm",
        }],
    }


def _sanitize_metric_name(name: str):
    sanitized = re.sub(r"[^a-zA-Z0-9_:]", "_", name)
    if not sanitized or not re.match(r"[a-zA-Z_:]", sanitized[0]):
        sanitized = "_" + sanitized
    return sanitized


def _escape_label_value(value: str):
    return value.replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


def _metric_value(value):
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        value = float(value)
        if math.isfinite(value):
            return value
    return None


def _prometheus_metrics(metrics: dict, prefix: str = "nanovllm"):
    lines = []
    emitted_types: set[str] = set()
    for key in sorted(metrics):
        value = metrics[key]
        metric_name = _sanitize_metric_name(f"{prefix}_{key}")
        if isinstance(value, dict):
            for label_value, item in sorted(value.items(), key=lambda pair: str(pair[0])):
                numeric = _metric_value(item)
                if numeric is None:
                    continue
                if metric_name not in emitted_types:
                    lines.append(f"# TYPE {metric_name} gauge")
                    emitted_types.add(metric_name)
                namespace = _escape_label_value(str(label_value))
                lines.append(f'{metric_name}{{namespace="{namespace}"}} {numeric:g}')
            continue
        numeric = _metric_value(value)
        if numeric is None:
            continue
        if metric_name not in emitted_types:
            lines.append(f"# TYPE {metric_name} gauge")
            emitted_types.add(metric_name)
        lines.append(f"{metric_name} {numeric:g}")
    return "\n".join(lines) + ("\n" if lines else "")


def create_app(model: str, **engine_kwargs):
    try:
        from fastapi import FastAPI, HTTPException, Request
        from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
    except ImportError as exc:
        raise RuntimeError("fastapi and uvicorn are required for nanovllm.serve") from exc

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await app.state.engine.start()
        try:
            yield
        finally:
            await app.state.engine.shutdown()

    stream_interval = _normalize_stream_interval(engine_kwargs.pop("stream_interval", 1))
    model_backend = engine_kwargs.pop("model_backend", "native")
    app = FastAPI(title="nano-vLLM", lifespan=lifespan)
    if model_backend == "hf_auto":
        from nanovllm.engine.hf_auto_engine import HFAutoAsyncEngine

        app.state.engine = HFAutoAsyncEngine(model, **engine_kwargs)
    elif model_backend == "native":
        app.state.engine = AsyncLLMEngine(model, model_backend=model_backend, **engine_kwargs)
    else:
        raise ValueError(f"unsupported model_backend: {model_backend}")
    app.state.model_backend = model_backend
    app.state.model_name = _model_name(model)
    app.state.created = int(time.time())
    app.state.stream_interval = stream_interval

    def rejected(exc: RequestRejected):
        return HTTPException(status_code=429, detail=str(exc))

    def failed(exc: RequestFailed):
        if exc.error_type == "request_validation":
            status_code = 400
        elif exc.error_type == "timeout":
            status_code = 408
        else:
            status_code = 500
        return HTTPException(status_code=status_code, detail=str(exc))

    def bad_request(exc: ValueError):
        return HTTPException(status_code=400, detail=str(exc))

    async def simple_stream(request: Request, engine_request) -> AsyncIterator[str]:
        request_id = engine_request.request_id
        try:
            async for output in _iter_stream_outputs(
                app.state.engine,
                engine_request,
                app.state.stream_interval,
            ):
                if await request.is_disconnected():
                    await app.state.engine.abort(request_id)
                    break
                yield _json_sse({
                    "request_id": output["request_id"],
                    "trace_id": output.get("trace_id"),
                    "token_id": output.get("token_id"),
                    "token_ids": output.get("token_ids"),
                    "text": output.get("text", ""),
                    "finished": output.get("finished", False),
                    "finish_reason": output.get("finish_reason"),
                    "usage": output.get("usage"),
                    "error": output.get("error"),
                    "error_type": output.get("error_type"),
                })
        finally:
            await app.state.engine.abort(request_id)

    async def add_simple_request(payload: dict):
        prompt = prompt_from_payload(payload)
        sampling_params = sampling_params_from_payload(payload)
        cache_options = cache_options_from_payload(payload, tokenizer=app.state.engine.tokenizer, prompt=prompt)
        request_id = payload.get("request_id") or uuid.uuid4().hex
        return await app.state.engine.add_request(
            prompt,
            sampling_params,
            request_id=request_id,
            cache_options=cache_options,
            **request_options_from_payload(payload),
        )

    @app.post("/generate")
    async def generate(request: Request):
        payload = await request.json()
        if payload.get("stream", False):
            try:
                engine_request = await add_simple_request(payload)
            except RequestRejected as exc:
                raise rejected(exc) from exc
            except RequestFailed as exc:
                raise failed(exc) from exc
            except ValueError as exc:
                raise bad_request(exc) from exc
            return StreamingResponse(simple_stream(request, engine_request), media_type="text/event-stream")
        try:
            output = await app.state.engine.generate(
                prompt_from_payload(payload),
                sampling_params_from_payload(payload),
                request_id=payload.get("request_id"),
                cache_options=cache_options_from_payload(
                    payload,
                    tokenizer=app.state.engine.tokenizer,
                    prompt=prompt_from_payload(payload),
                ),
                **request_options_from_payload(payload),
            )
        except RequestRejected as exc:
            raise rejected(exc) from exc
        except RequestFailed as exc:
            raise failed(exc) from exc
        except ValueError as exc:
            raise bad_request(exc) from exc
        return JSONResponse(output)

    @app.post("/generate_stream")
    async def generate_stream(request: Request):
        payload = await request.json()
        try:
            engine_request = await add_simple_request(payload)
        except RequestRejected as exc:
            raise rejected(exc) from exc
        except RequestFailed as exc:
            raise failed(exc) from exc
        except ValueError as exc:
            raise bad_request(exc) from exc
        return StreamingResponse(simple_stream(request, engine_request), media_type="text/event-stream")

    @app.post("/cancel/{request_id}")
    async def cancel(request_id: str):
        cancelled = await app.state.engine.abort(request_id)
        return JSONResponse({"request_id": request_id, "cancelled": cancelled})

    @app.get("/cache/stats")
    async def cache_stats():
        metrics = app.state.engine.metrics()
        return JSONResponse({
            key: metrics.get(key)
            for key in (
                "num_blocks",
                "free_blocks",
                "cached_blocks",
                "used_blocks",
                "cached_blocks_by_namespace",
                "prefix_cache_hits",
                "prefix_cache_misses",
                "prefix_cache_hits_by_namespace",
                "prefix_cache_misses_by_namespace",
                "cache_read_input_tokens_by_namespace",
                "cache_creation_input_tokens_by_namespace",
                "prefix_cache_hit_rate",
                "evictions",
                "global_quota_evictions",
                "max_cached_blocks",
                "duplicate_cache_blocks_skipped",
                "expired_purges",
            )
        })

    @app.get("/cache/inspect")
    async def cache_inspect():
        return JSONResponse(await app.state.engine.cache_inspect())

    @app.post("/cache/purge")
    async def cache_purge(request: Request):
        try:
            payload = await request.json()
        except Exception:
            payload = {}
        namespace = payload.get("namespace")
        expired_only = payload.get("expired_only", False)
        purged_blocks = await app.state.engine.purge_prefix_cache(
            namespace=namespace,
            expired_only=expired_only,
        )
        return JSONResponse({
            "namespace": namespace,
            "expired_only": expired_only,
            "purged_blocks": purged_blocks,
        })

    @app.post("/cache/prewarm")
    async def cache_prewarm(request: Request):
        payload = await request.json()
        payload = {**payload, "max_tokens": 0}
        try:
            if "messages" in payload and "prompt" not in payload and "prompt_token_ids" not in payload:
                prompt, cache_options = chat_prompt_and_cache_options(
                    app.state.engine.tokenizer,
                    payload.get("messages", []),
                    payload,
                )
            else:
                prompt = prompt_from_payload(payload)
                cache_options = cache_options_from_payload(
                    payload,
                    tokenizer=app.state.engine.tokenizer,
                    prompt=prompt,
                )
            output = await app.state.engine.generate(
                prompt,
                sampling_params_from_payload(payload),
                request_id=payload.get("request_id"),
                cache_options=cache_options,
                **request_options_from_payload(payload),
            )
        except RequestRejected as exc:
            raise rejected(exc) from exc
        except RequestFailed as exc:
            raise failed(exc) from exc
        except ValueError as exc:
            raise bad_request(exc) from exc
        return JSONResponse({
            "request_id": output["request_id"],
            "trace_id": output.get("trace_id"),
            "finish_reason": output["finish_reason"],
            "usage": output.get("usage"),
            "cache": {
                "namespace": cache_options.get("cache_namespace"),
                "breakpoints": cache_options.get("cache_breakpoint_tokens") or [],
                "ttl_seconds": cache_options.get("cache_ttl_seconds"),
                "enabled": cache_options.get("cache_enabled", True),
            },
        })

    @app.post("/admin/drain")
    async def admin_drain():
        return JSONResponse({"ok": True, **app.state.engine.set_draining(True)})

    @app.post("/admin/resume")
    async def admin_resume():
        return JSONResponse({"ok": True, **app.state.engine.set_draining(False)})

    @app.post("/admin/restart")
    async def admin_restart():
        try:
            status = await app.state.engine.restart()
        except RequestRejected as exc:
            raise rejected(exc) from exc
        except RequestFailed as exc:
            raise failed(exc) from exc
        return JSONResponse({"ok": True, **status})

    @app.get("/admin/state")
    async def admin_state():
        return JSONResponse({"ok": True, **app.state.engine.status()})

    @app.get("/healthz")
    async def healthz():
        return JSONResponse({"ok": True, **app.state.engine.status()})

    @app.get("/readyz")
    async def readyz():
        status = app.state.engine.status()
        if not status["ready"]:
            raise HTTPException(status_code=503, detail=status)
        return JSONResponse({"ok": True, **status})

    @app.get("/v1/models")
    async def models():
        return JSONResponse(_openai_model_list(app.state.model_name, app.state.created))

    @app.post("/v1/completions")
    async def completions(request: Request):
        payload = await request.json()
        try:
            prompt = prompt_from_payload(payload)
            if isinstance(prompt, list) and prompt and isinstance(prompt[0], str):
                prompt = prompt[0]
            sampling_params = sampling_params_from_payload(payload)
            cache_options = cache_options_from_payload(payload, tokenizer=app.state.engine.tokenizer, prompt=prompt)
        except ValueError as exc:
            raise bad_request(exc) from exc
        response_id = f"cmpl-{uuid.uuid4().hex}"
        engine_request_id = payload.get("request_id") or uuid.uuid4().hex
        created = int(time.time())
        if payload.get("stream", False):
            try:
                engine_request = await app.state.engine.add_request(
                    prompt,
                    sampling_params,
                    request_id=engine_request_id,
                    cache_options=cache_options,
                    **request_options_from_payload(payload),
                )
            except RequestRejected as exc:
                raise rejected(exc) from exc
            except RequestFailed as exc:
                raise failed(exc) from exc
            except ValueError as exc:
                raise bad_request(exc) from exc

            async def stream():
                try:
                    async for output in _iter_stream_outputs(
                        app.state.engine,
                        engine_request,
                        app.state.stream_interval,
                    ):
                        if await request.is_disconnected():
                            await app.state.engine.abort(engine_request_id)
                            break
                        yield _json_sse({
                            "id": response_id,
                            "request_id": engine_request_id,
                            "trace_id": output.get("trace_id"),
                            "object": "text_completion",
                            "created": created,
                            "model": app.state.model_name,
                            "choices": [{
                                "index": 0,
                                "text": output.get("text", ""),
                                "finish_reason": output.get("finish_reason") if output.get("finished") else None,
                            }],
                            "usage": _openai_usage(output.get("usage"), len(output.get("completion_token_ids", [])))
                            if output.get("finished") else None,
                        })
                    yield "data: [DONE]\n\n"
                finally:
                    await app.state.engine.abort(engine_request_id)
            return StreamingResponse(stream(), media_type="text/event-stream")
        try:
            output = await app.state.engine.generate(
                prompt,
                sampling_params,
                request_id=engine_request_id,
                cache_options=cache_options,
                **request_options_from_payload(payload),
            )
        except RequestRejected as exc:
            raise rejected(exc) from exc
        except RequestFailed as exc:
            raise failed(exc) from exc
        return JSONResponse({
            "id": response_id,
            "request_id": engine_request_id,
            "trace_id": output.get("trace_id"),
            "object": "text_completion",
            "created": created,
            "model": app.state.model_name,
            "choices": [{
                "index": 0,
                "text": output["text"],
                "finish_reason": output["finish_reason"],
            }],
            "usage": {
                **_openai_usage(output.get("usage"), len(output["token_ids"])),
            },
        })

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        payload = await request.json()
        try:
            messages = payload.get("messages", [])
            prompt, cache_options = chat_prompt_and_cache_options(app.state.engine.tokenizer, messages, payload)
            sampling_params = sampling_params_from_payload(payload)
        except ValueError as exc:
            raise bad_request(exc) from exc
        response_id = f"chatcmpl-{uuid.uuid4().hex}"
        engine_request_id = payload.get("request_id") or uuid.uuid4().hex
        created = int(time.time())
        if payload.get("stream", False):
            try:
                engine_request = await app.state.engine.add_request(
                    prompt,
                    sampling_params,
                    request_id=engine_request_id,
                    cache_options=cache_options,
                    **request_options_from_payload(payload),
                )
            except RequestRejected as exc:
                raise rejected(exc) from exc
            except RequestFailed as exc:
                raise failed(exc) from exc
            except ValueError as exc:
                raise bad_request(exc) from exc

            async def stream():
                try:
                    async for output in _iter_stream_outputs(
                        app.state.engine,
                        engine_request,
                        app.state.stream_interval,
                    ):
                        if await request.is_disconnected():
                            await app.state.engine.abort(engine_request_id)
                            break
                        yield _json_sse({
                            "id": response_id,
                            "request_id": engine_request_id,
                            "trace_id": output.get("trace_id"),
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": app.state.model_name,
                            "choices": [{
                                "index": 0,
                                "delta": {"content": output.get("text", "")},
                                "finish_reason": output.get("finish_reason") if output.get("finished") else None,
                            }],
                            "usage": _openai_usage(output.get("usage"), len(output.get("completion_token_ids", [])))
                            if output.get("finished") else None,
                        })
                    yield "data: [DONE]\n\n"
                finally:
                    await app.state.engine.abort(engine_request_id)
            return StreamingResponse(stream(), media_type="text/event-stream")
        try:
            output = await app.state.engine.generate(
                prompt,
                sampling_params,
                request_id=engine_request_id,
                cache_options=cache_options,
                **request_options_from_payload(payload),
            )
        except RequestRejected as exc:
            raise rejected(exc) from exc
        except RequestFailed as exc:
            raise failed(exc) from exc
        return JSONResponse({
            "id": response_id,
            "request_id": engine_request_id,
            "trace_id": output.get("trace_id"),
            "object": "chat.completion",
            "created": created,
            "model": app.state.model_name,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": output["text"]},
                "finish_reason": output["finish_reason"],
            }],
            "usage": {
                **_openai_usage(output.get("usage"), len(output["token_ids"])),
            },
        })

    @app.get("/metrics")
    async def metrics():
        return app.state.engine.metrics()

    @app.get("/metrics/prometheus")
    async def metrics_prometheus():
        return PlainTextResponse(
            _prometheus_metrics(app.state.engine.metrics()),
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )

    return app
