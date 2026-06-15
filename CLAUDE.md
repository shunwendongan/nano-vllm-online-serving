# CLAUDE.md

## Project Memory

This repo is a nano-vLLM modification project. The priority is enterprise-style online inference, with current-stage optional CUDA attention and gpt-oss compatibility work behind explicit switches.

Current stable direction:

- FastAPI serving.
- Async request queue.
- Continuous batching.
- Paged KV block manager.
- Exact prefix cache with namespace, TTL, quota, and usage metrics.
- Prefix-cache miss diagnostics through `/cache/inspect`.
- Optional `attention_backend="cuda_ext"` decode attention backend.
- Optional `model_backend="hf_auto"` Transformers bridge for gpt-oss smoke tests.
- OpenAI-compatible completions and chat completions.
- Colab-based CUDA validation.

## Do Not Assume Local CUDA

The local Windows machine is for non-GPU development and regression tests only.

Use this local command:

```powershell
.\scripts\run_local_tests.ps1
```

Use this CUDA command on Colab or a GPU server:

```bash
python scripts/validate_online_gpu.py --model <model_dir>
```

If CUDA, torch, flash-attn, triton, or transformers are missing locally, that is expected for this machine.

## Serving Requirements

Preserve these API contracts:

- `POST /generate`
- `POST /generate_stream`
- `POST /v1/completions`
- `POST /v1/chat/completions`
- `GET /metrics`
- `GET /metrics/prometheus`
- `GET /cache/stats`
- `GET /cache/inspect`
- `POST /cache/prewarm`
- `POST /cache/purge`

Streaming endpoints use SSE. OpenAI-compatible streaming ends with `[DONE]`.

## Cache Semantics

Prompt caching is inspired by Anthropic-style controls:

- `cache_control`
- `cacheable_prefix_tokens`
- `cache_breakpoints`
- `cache_ttl_seconds`
- `cache_namespace`
- `disable_cache`
- `cache_enabled`

`max_tokens=0` is a cache prewarm path, not an invalid request.

`request_namespace` is for admission and tenant resource budgets. `cache_namespace` is for prefix-cache sharing. Keep them separate.

## Implementation Priorities

1. Make serving behavior correct and observable.
2. Improve scheduler policy and cache diagnostics.
3. Validate on GPU through Colab.
4. Keep CUDA attention and gpt-oss work explicit, measurable, and separate from the stable path.

Do not replace flash-attn with custom attention by default. Use custom kernels as explicit benchmark or experimental backends.
Do not describe `hf_auto` gpt-oss results as native nano-vLLM continuous-batching performance.

## Failure And Logging Rules

- 400: validation error.
- 408: timeout.
- 429: queue, tenant, or admission budget rejection.
- 500: engine or runtime failure.
- Prompt text should not be written to request lifecycle logs.
- Cancellation and disconnects must release request resources.
- Drain should reject new requests but not corrupt active work.

## Documentation To Keep Current

- `ONLINE_SERVING.md`: operational serving docs.
- `docs/REQUIREMENTS.md`: product and enterprise requirements.
- `docs/TECHNICAL_ROADMAP.md`: staged implementation roadmap.
- `AGENTS.md`: Codex-oriented repository rules.
- `CLAUDE.md`: Claude Code-oriented repository memory.
