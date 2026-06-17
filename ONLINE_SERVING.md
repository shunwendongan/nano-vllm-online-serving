# nano-vLLM Online Serving Notes

This branch turns the original offline batch runner into a minimal online serving stack.

## Serving

Start the HTTP server:

```bash
python -m nanovllm.serve --model ~/huggingface/Qwen3-0.6B --host 127.0.0.1 --port 8000
```

Current-stage experimental switches are explicit and off the stable path by default:

```bash
python -m nanovllm.serve --model ~/huggingface/Qwen3-0.6B --attention-backend cuda_ext
python -m nanovllm.serve --model ~/models/gpt-oss-20b --model-backend hf_auto --host 0.0.0.0 --port 8000
```

`attention_backend="flash_attn"` remains the default reference path. `model_backend="native"` remains the nano-vLLM scheduler/model path; `hf_auto` is a Transformers bridge for gpt-oss smoke validation and must not be reported as nano-vLLM continuous-batching performance.

Before starting a real GPU server, check the runtime:

```bash
python -m nanovllm.check_runtime --model ~/huggingface/Qwen3-0.6B --tensor-parallel-size 1
```

Supported endpoints:

- `POST /generate`
- `POST /generate_stream`
- `POST /cancel/{request_id}`
- `GET /cache/stats`
- `GET /cache/inspect`
- `POST /cache/purge`
- `POST /cache/prewarm`
- `POST /admin/drain`
- `POST /admin/resume`
- `POST /admin/restart`
- `GET /admin/state`
- `POST /v1/completions`
- `POST /v1/chat/completions`
- `GET /v1/models`
- `GET /healthz`
- `GET /readyz`
- `GET /metrics`
- `GET /metrics/prometheus`

Streaming endpoints use server-sent events. OpenAI-compatible streams end with `data: [DONE]`.
All request styles accept an optional `request_id`; `/cancel/{request_id}` aborts a pending or active request and releases scheduler/KV resources.
`/healthz` reports process-level liveness, while `/readyz` returns HTTP 503 if the background engine loop has failed or the instance is in drain mode.
`/admin/drain` rejects new requests and flips readiness to 503 while allowing already pending or active requests to finish. `/admin/resume` accepts new traffic again. This is intended for rolling deploys and safe instance removal behind a load balancer.
If the background engine loop enters a fatal state, readiness stays at 503 and new requests fail fast instead of silently restarting the broken loop. `/admin/restart` closes the current engine, recreates it, and clears the fatal state only when there are no active or pending requests. Use it after operator inspection or automated supervisor policy decides the process can try an in-place recovery; otherwise restart the process.
`/cache/stats` reports prefix-cache block state and cache usage by namespace, and also purges expired idle blocks before returning. `/cache/inspect` returns aggregate cache diagnostics such as miss reasons and block counts without prompt text or token content. `/cache/purge` accepts `{"namespace": "tenant-a"}` to remove idle cached blocks for one namespace, `{}` to purge all idle cached blocks, or `{"expired_only": true}` to remove only expired idle blocks. Live in-flight KV blocks are not interrupted.
Requests can override server defaults with `request_timeout_s` and `queue_timeout_s`. Timeout responses use `finish_reason="timeout"` and non-streaming HTTP requests map them to HTTP 408.
Requests can also set integer `priority`; higher priority requests are admitted from the async HTTP queue before lower priority requests, while already-running decode work remains scheduler-controlled.
Requests can set `trace_id`; when omitted, it defaults to `request_id`. The value is returned by `/generate`, SSE chunks, OpenAI-compatible responses, and JSONL lifecycle logs.
For multi-tenant deployments, `request_namespace` controls request admission budgets and resource metrics. It is intentionally separate from `cache_namespace`, which controls prefix-cache sharing. If `request_namespace` is omitted, the engine falls back to `cache_namespace`.

Production-facing queue controls:

- `--max-pending-requests`: bounds the HTTP-to-engine waiting queue and returns HTTP 429 when full.
- `--max-pending-requests-per-namespace`: bounds queued requests per `request_namespace`, preventing one tenant from occupying the async admission queue with many short requests.
- `--max-pending-prompt-tokens`: bounds total prompt tokens held in the async pending queue before requests are admitted to the engine.
- `--max-active-requests`: bounds requests admitted from HTTP queue into the engine scheduler.
- `--max-active-requests-per-namespace`: bounds active engine requests per `request_namespace`, so a busy tenant cannot consume every decode slot even when each request is small.
- `--max-active-tokens`: bounds estimated active tokens already admitted to the engine. Each request is estimated as prompt tokens plus `max_tokens`. If a large pending request does not currently fit, smaller later requests can still be admitted to avoid head-of-line blocking.
- `--max-active-tokens-per-namespace`: applies the same estimated-token admission budget per `request_namespace`, so one tenant cannot consume all active KV capacity. This is an admission gate and does not preempt already-running requests.
- `--output-queue-size`: bounds each request stream queue; slow consumers are cancelled instead of blocking the engine loop.
- `--queue-timeout-s`: bounds how long a request can wait in the async pending queue.
- `--request-timeout-s`: bounds total request lifetime from submission to completion.
- `--request-log-path`: writes JSONL request lifecycle events for observability. The log records request ids, state transitions, latency, token counts, and cache settings, but not prompt text.
- `--metrics-window-size`: controls the rolling window used for recent latency and throughput metrics.
- `--stream-interval`: coalesces streaming token chunks after the first token. The default `1` preserves per-token SSE behavior; values such as `4` or `8` reduce HTTP/SSE write overhead for high-concurrency throughput probes while keeping first-token flush immediate.

Deployment controls:

- `--scheduler-fairness`: selects `alternate`, `fcfs`, `decode_first`, `prefill_first`, or `cache_aware_lpm`.
- `--attention-backend`: selects `flash_attn` or the optional `cuda_ext` decode attention backend. Prefill still uses flash-attn in the current implementation.
- `--model-backend`: selects `native` or `hf_auto`. Use `hf_auto` for current-stage gpt-oss server smoke tests.
- `--distributed-backend`: torch distributed backend, default `nccl`.
- `--distributed-init-method`: process-group init address, default `tcp://127.0.0.1:2333`; change this when running multiple nano-vLLM servers on one host.
- `--cuda-device-offset`: maps rank 0 to `cuda:<offset>`, useful for multi-instance serving.
- `--ipc-shm-name`: shared memory name used by tensor-parallel workers; change this for multiple servers.
- `--prefix-cache-min-tokens`: skip prefix-cache writes for short prompts so cache capacity is kept for long reusable contexts.
- `--max-cached-blocks`: caps total idle prefix-cache blocks across all namespaces. `0` means unlimited. This keeps free KV pages available for live prefill/decode even when many tenants share the service.
- `--max-cached-blocks-per-namespace`: caps idle prefix-cache blocks per namespace. `0` means unlimited. Live in-flight KV blocks are not evicted by this quota.
- `--min-prefill-chunk-tokens`: lower bound for dynamic prefill chunking under decode pressure.
- `--kvcache-watermark-blocks`: reserve KV blocks for decode when prefill and decode compete, reducing preemption and latency spikes.
- `--enforce-eager`: recommended for first validation of online serving and chunked prefill; enable CUDA graphs only after the dynamic block-table path is verified on the target GPU.

## Continuous Batching

`AsyncLLMEngine` owns a single synchronous `LLMEngine` and feeds it from an async request queue. The scheduler runs at iteration granularity:

- new requests can enter between decode iterations;
- long prompts are split by `max_prefill_chunk_tokens`;
- when active decode requests exist, prefill chunks shrink toward `min_prefill_chunk_tokens` to reduce latency spikes;
- prefill admission respects `kvcache_watermark_blocks` while decode can consume the reserve;
- decode batches keep generating one token per active sequence;
- preemption releases live blocks and lets cached full blocks remain reusable.

## Paged KV Cache

`BlockManager` manages the KV cache as fixed-size physical blocks. Each `Sequence` keeps a logical block table that is passed to flash-attn as `block_table`, so the default attention path still uses the proven paged KV interface instead of a new custom attention kernel.

Block states:

- `free`: available for allocation;
- `used`: referenced by a running sequence;
- `cached`: no live reference, but hash-addressable as a reusable prompt prefix.

The cache supports ref counts, LRU eviction, global cache quota, namespace isolation, and TTL expiry.
Prefix-cache miss reasons are counted for `cache_disabled`, `namespace_mismatch`, `ttl_expired`, `prefix_shorter_than_min`, `no_full_block_at_breakpoint`, `hash_miss`, and `token_guard_mismatch`. Use `/cache/inspect` when debugging why a repeated prompt did not reuse cached KV blocks.
Idle cached blocks can also be purged explicitly through `/cache/purge`, which is useful for tenant offboarding, prompt-policy changes, TTL cleanup, or emergency memory recovery.
For multi-tenant serving, `max_cached_blocks` caps the total idle prefix-cache footprint, while `max_cached_blocks_per_namespace` prevents one namespace from occupying the full shared prefix-cache pool. Quota eviction removes least-recently-accessed idle cached blocks only; running sequences keep their live block references.
When concurrent requests compute the same cacheable prefix, only one canonical physical block is retained in prefix cache; duplicate block writes are skipped and reported as `duplicate_cache_blocks_skipped`.

## Prompt Cache Controls

Requests can provide Anthropic-style prompt cache controls:

```json
{
  "prompt": "stable context ... user question",
  "cache_control": {
    "type": "ephemeral",
    "ttl": "5m",
    "namespace": "tenant-a",
    "cacheable_prefix_tokens": 2048
  }
}
```

Equivalent top-level fields are also accepted:

- `cacheable_prefix_tokens`
- `cache_breakpoints`
- `cache_ttl_seconds`
- `cache_namespace`
- `request_namespace`
- `disable_cache`
- `cache_enabled`

Multiple cache breakpoints are supported for long prompts with several reusable regions:

```json
{
  "prompt": "tools ... system ... examples ... user question",
  "cache_breakpoints": [2048, 4096, 8192],
  "cache_control": {"type": "ephemeral", "ttl": "5m", "namespace": "tenant-a"}
}
```

The API also accepts `cache_control.cache_breakpoints`, `cache_control.breakpoints`, and `cacheable_prefix_tokens` as a list. The engine keeps at most the four largest positive breakpoints, matching the practical Anthropic-style limit while keeping the paged KV allocator simple. Internally, the largest breakpoint is the maximum cacheable KV boundary; smaller breakpoints are retained in request metadata and logs so clients can express stable prefix boundaries without changing the default flash-attn paged KV path.

For sensitive or non-reusable prompts, disable shared prefix-cache reads and writes:

```json
{
  "prompt": "private context ...",
  "cache_control": {"type": "no-store"}
}
```

`{"disable_cache": true}` is equivalent. The request still uses normal live KV cache for generation, but no idle prefix-cache block is read or retained after the request.

For chat requests, `cache_control` on a message or content block marks a cache breakpoint. Multiple message/content markers are collected, tokenized, capped to four breakpoints, and passed into the engine as `cache_breakpoint_tokens`.

To prewarm a reusable prompt cache without generating completion tokens, send `max_tokens: 0` to `/generate`, `/v1/completions`, or `/v1/chat/completions`. The service runs the prefill path, writes eligible full KV blocks into the paged prefix cache, returns `finish_reason: "cache_warmed"`, and reports zero completion tokens. A clearer operational endpoint is also available:

```json
POST /cache/prewarm
{
  "prompt": "stable system prompt ... few-shot examples ...",
  "cache_control": {
    "type": "ephemeral",
    "ttl": "1h",
    "namespace": "tenant-a"
  }
}
```

`/cache/prewarm` also accepts chat `messages`; message-level `cache_control` markers are converted to the same cache breakpoints as normal chat completions. This is useful in Colab or service startup scripts: first prewarm long stable contexts, then run online traffic and inspect `cache_read_input_tokens`, `cache_creation_input_tokens`, and `prefix_cache_hit_rate`.

Responses include prompt-cache usage fields inspired by Anthropic prompt caching:

- `prompt_tokens`: logical prompt length.
- `input_tokens`: prompt tokens that were not read from cache or written as a reusable cache block.
- `cache_read_input_tokens`: prompt tokens served from cached KV blocks.
- `cache_creation_input_tokens`: prompt tokens newly written into reusable cached KV blocks.

In OpenAI-compatible responses these fields are added under `usage` together with `completion_tokens` and `total_tokens`. In `/generate` and streaming chunks, the same usage object is emitted on the final output event.

## Local vs Colab Validation

This project has two validation layers:

- Local Windows validation uses the existing Python 3.12 interpreter at `C:\Users\Administrator\AppData\Local\Programs\Python\Python312\python.exe`.
- Local validation does not require CUDA, `torch`, `flash-attn`, `triton`, or `transformers`; it covers API mocks, scheduler behavior, block-manager logic, async queue behavior, metrics/export helpers, scripts, and CLI wiring.
- CUDA/model validation is expected to run on a GPU host such as Google Colab, where `nvidia-smi`, `torch`, `flash-attn`, `triton`, and `transformers` are available.

Run the local non-GPU regression suite from the repo root:

```powershell
.\scripts\run_local_tests.ps1
```

The script checks Python `>=3.10,<3.13`, verifies local test packages (`pytest`, `fastapi`, `starlette`, `uvicorn`, `httpx`), reports missing optional GPU/model packages without failing, then runs `unittest`, `pytest`, `compileall`, `git diff --check`, and CLI help checks.

If `nvidia-smi` is not available on the machine, do not treat local results as proof of real model throughput, flash-attn paged KV behavior, or CUDA graph safety. Use Colab or another CUDA host for the full online serving validation:

```bash
python scripts/validate_online_gpu.py --model <model_dir>
```

On Colab, install the project and GPU dependencies first, then run `validate_online_gpu.py`. A passing CUDA run must start the server, reach `/readyz`, validate `/generate`, `/generate_stream`, `/cache/prewarm`, `/metrics`, `/metrics/prometheus`, `/cache/stats`, observe a prefix-cache hit through either `cache_read_input_tokens > 0` or `prefix_cache_hits` growth, and run `bench_online.py --stream --fetch-metrics`.

For repeatable Colab experiments, use the checked-in config files:

```bash
bash scripts/setup_colab_gpu.sh configs/colab/qwen3_native_flash_attn_baseline.env
python scripts/run_colab_config.py --config configs/colab/qwen3_native_flash_attn_baseline.env
```

Each run writes timestamped artifacts under `reports/colab/<experiment>/<run_id>/`. See `docs/COLAB_BENCHMARKS.md` for the config matrix, gpt-oss `hf_auto` smoke flow, native Qwen baseline, scheduler comparison, and CUDA extension experiment.

`scripts/run_colab_config.py` applies shell environment overrides for known runtime
keys and records them in `resolved_config.json`. For example,
`BENCHMARK_CONCURRENCY=64 python scripts/run_colab_config.py --config ...` now
changes the effective benchmark command instead of silently using the `.env`
value.

## Benchmarking

After starting the server, run a simple concurrent benchmark:

```bash
python bench_online.py --url http://127.0.0.1:8000 --requests 128 --concurrency 16 --max-tokens 128
```

For first-token latency, use SSE streaming mode:

```bash
python bench_online.py --url http://127.0.0.1:8000 --stream --requests 128 --concurrency 16 --max-tokens 128 --fetch-metrics
```

The benchmark reports request throughput, completion token throughput, HTTP/error status counts, prompt-cache read/create token totals, average/p50/p95/max latency, and streaming TTFT. It reads `usage.completion_tokens` for OpenAI-compatible endpoints and falls back to `/generate` `token_ids`. Add `--cache-namespace tenant-a` to repeatedly hit the same prompt cache namespace and inspect `/metrics` for prefix cache hit rate.
For long-context cache benchmarks, pass an explicit boundary with `--cacheable-prefix-tokens 2048` or multiple boundaries with `--cache-breakpoints 2048,4096,8192`.

`GET /metrics` includes backend name, scheduler policy, queue depth, pending prompt-token pressure, active estimated-token pressure, per-namespace request pressure, per-namespace estimated-token pressure, average/max async queue wait, timeout counters, KV block state, prompt-cache hit/miss counters, prefix-cache miss reasons, per-namespace prompt-cache read/create token counters, namespace quota evictions, scheduler `preemptions`, `prefill_watermark_delays`, policy decision counters, request counters, engine-loop health/error state, TTFT/latency summaries, and aggregate `avg_prefill_tok_s` / `avg_decode_tok_s` measured inside the online engine loop. It also exposes rolling-window metrics such as `recent_ttft_p95_s`, `recent_latency_p95_s`, `recent_queue_wait_p95_s`, `recent_prefill_tok_s`, and `recent_decode_tok_s`, which are better suited for live load tests than lifetime averages. `GET /metrics/prometheus` exports the numeric subset in Prometheus text format, including namespace dictionaries as `namespace="..."` labeled time series.

Benchmark runs can enforce basic SLO gates:

```bash
python bench_online.py --url http://127.0.0.1:8000 --stream --requests 128 --concurrency 16 --fetch-metrics --fail-on-errors --slo-ttft-p95-s 1.0 --slo-latency-p95-s 8.0
```

`bench_online.py` reports backend, scheduler policy, model name, p50/p95/p99 latency, p50/p95/p99 TTFT for streaming, error rate, completion token throughput, prompt-cache usage, optional server metrics, and `slo_pass` / `slo_failures`. Use `--report-json-path` and `--report-markdown-path` to save benchmark artifacts for resume-safe reporting.
The saved reports include an automatic `bottleneck_analysis` section with findings and optimization suggestions. See `docs/PERFORMANCE_ANALYSIS.md` for the Colab gpt-oss `hf_auto` smoke test flow and native Qwen follow-up matrix.

For a full CUDA-host validation run, use:

```bash
python scripts/validate_online_gpu.py --model ~/huggingface/Qwen3-0.6B --attention-backend cuda_ext
```

The validation script runs `nanovllm.check_runtime`, starts `python -m nanovllm.serve`, checks `/readyz` and `/v1/models`, sends one non-streaming request, sends one SSE streaming request, verifies required `/metrics`, `/metrics/prometheus`, `/cache/stats`, and cache diagnostics fields, runs a long-prompt prefix-cache probe that requires the warm request to report `cache_read_input_tokens` or a positive `prefix_cache_hits` delta, then runs `bench_online.py --stream --fetch-metrics`. It forwards deployment parameters such as tensor parallel size, CUDA device offset, distributed backend/init method, GPU memory utilization, scheduler fairness, attention backend, model backend, prefix-cache controls, KV experimental switches, op backend, queue limits, token budgets, and SLO gates. It defaults to `--enforce-eager` for the first pass; use `--no-enforce-eager` only after the eager serving path is correct on the target GPU. Use `--skip-cache-probe` only when intentionally validating a no-prefix-cache configuration.
When `--model-backend hf_auto` is used, native prefix-cache prewarm/probe validation is skipped and benchmark artifacts default to `reports/gpt_oss_hf_auto_bench.json` and `reports/gpt_oss_hf_auto_bench.md` for gpt-oss model ids.

## Experimental Hooks

The default path keeps correctness first:

- `kv_cache_dtype="auto"` keeps the model dtype.
- `kv_cache_dtype="fp8_e4m3"` and `kv_cache_dtype="fp8_e5m2"` are reserved but intentionally fail fast until compatible scales/dequant support is implemented.
- `kv_compression` reserves switches for KIVI, SnapKV, H2O, StreamingLLM, TurboQuant, and paged eviction experiments; non-`none` values fail fast instead of silently becoming no-ops.
- `op_backend="torch|triton|cuda_ext"` selects low-level KV-store/RMS/SwiGLU helper routes where implemented.
- `attention_backend="cuda_ext"` is a current-stage optional decode attention backend. It includes dense MHA, GQA/MQA, streaming softmax, and paged decode CUDA extension entrypoints under `nanovllm/kernels/cuda_ext/`. On non-CUDA hosts it fails fast; on servers it must be compared against flash-attn before any performance claim.
- `model_backend="hf_auto"` is a current-stage Transformers backend for gpt-oss smoke tests. It is not the native nano-vLLM scheduler path and should be reported separately.

GPU validation still needs to be run on a CUDA host:

1. `python -m nanovllm.check_runtime --model <model> --tensor-parallel-size <tp>`
2. `python -m nanovllm.serve --model <model> --enforce-eager --attention-backend flash_attn`
3. `python bench_online.py --stream --requests 32 --concurrency 4 --fetch-metrics --report-json-path bench.json --report-markdown-path bench.md`
4. `python scripts/validate_online_gpu.py --model <model> --attention-backend cuda_ext`
5. `python -m nanovllm.serve --model <gpt_oss_20b_dir> --model-backend hf_auto --host 0.0.0.0 --port 8000`
6. Repeat without `--enforce-eager` only after outputs and streaming behavior match.
