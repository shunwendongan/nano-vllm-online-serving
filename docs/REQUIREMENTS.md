# nano-vLLM Enterprise Serving Requirements

## Summary

This project is no longer only a minimal nano-vLLM study repo. The next phase should make it a small but coherent online inference system that is easy to inspect, test locally, and validate later on a CUDA host.

The product priority is:

1. Serving correctness and operability.
2. Continuous batching and paged KV behavior.
3. Prefix cache and request governance for multi-tenant use.
4. CUDA attention and model-backend experiments behind explicit switches.

The default path must stay exact and stable. Approximate KV compression and custom kernels are opt-in experiments.

## Users And Use Cases

- Local developer on Windows:
  - Runs non-GPU regression tests.
  - Reviews scheduler, protocol, cache, and API behavior without CUDA.
  - Prepares Colab validation runs.
- GPU experimenter on Colab or a lab server:
  - Starts the HTTP server with a small model.
  - Validates flash-attn paged KV behavior, prefix cache hits, streaming, and throughput.
  - Benchmarks kernel backends.
- Enterprise serving engineer:
  - Needs predictable request admission, cancellation, drain, restart, metrics, and logs.
  - Needs tenant isolation between request budgets and cache sharing.
  - Needs failure modes to be explicit rather than silent.
- Agent-assisted developer using Codex or Claude Code:
  - Needs project rules that prevent local CUDA assumptions.
  - Needs consistent commands for local and GPU validation.
  - Needs clear boundaries between serving changes and kernel experiments.

## Current State

Already implemented:

- Async HTTP serving with `/generate`, `/generate_stream`, `/v1/completions`, `/v1/chat/completions`.
- SSE streaming and request cancellation on disconnect.
- Async engine with pending and active queues.
- Iteration-level scheduler with prefill/decode separation and chunked prefill.
- Paged KV block table path using flash-attn block tables.
- Prefix cache with namespace, TTL, quota, LRU eviction, and usage accounting.
- Cache prewarm through `max_tokens=0` and `/cache/prewarm`.
- Metrics, Prometheus export, admin drain/resume/restart, and JSONL lifecycle logging.
- Local non-GPU tests and Colab GPU validation script.

Known constraints:

- This Windows machine has no CUDA and should not be treated as proof of real model serving.
- `torch`, `flash-attn`, `triton`, and `transformers` are required only for real GPU validation.
- FP8 KV cache and approximate KV compression switches are reserved and must fail fast until implemented.
- `attention_backend="cuda_ext"` is available as a current-stage CUDA server experiment, but local Windows can only test import/config/fail-fast behavior.
- `model_backend="hf_auto"` is available for gpt-oss smoke tests and is not equivalent to the native nano-vLLM scheduler path.

## Functional Requirements

### Serving API

- Keep `/generate` and `/generate_stream` as the simple API.
- Keep `/v1/completions` and `/v1/chat/completions` as the OpenAI-compatible API.
- Streaming must use SSE and finish with `[DONE]` for OpenAI-compatible endpoints.
- Add or document `trace_id` as a request correlation field. If omitted, reuse `request_id`.
- Preserve `request_namespace` for admission budgets and `cache_namespace` for prefix-cache sharing.
- Keep prompt text out of request lifecycle logs.

### Request Lifecycle

- Requests move through submitted, admitted, first_token, finished, failed, timeout, cancelled.
- Queue timeout maps to HTTP 408.
- Admission rejection maps to HTTP 429.
- Validation errors map to HTTP 400.
- Engine/runtime failures map to HTTP 500.
- Slow consumers are cancelled to avoid unbounded output queues.
- Drain rejects new traffic but lets existing active work finish.
- Restart is allowed only when no requests are in flight.

### Continuous Batching

- New requests can enter while decode requests are running.
- Long prompts use chunked prefill to avoid blocking decode.
- Decode requests should normally produce one token per engine iteration.
- Scheduler policy must remain configurable.
- First production default remains fairness-oriented, not maximum benchmark throughput.

Planned policy names:

- `fcfs`: simple baseline.
- `decode_first`: lower TTFT and decode jitter under load.
- `prefill_first`: useful for cache prewarm or prompt-heavy traffic.
- `cache_aware_lpm`: longest-prefix-match admission strategy with starvation guard.

### Paged KV And Prefix Cache

- Continue using fixed-size physical KV blocks and per-sequence logical block tables.
- Keep exact KV reuse as the default.
- Prefix cache must respect namespace, TTL, no-store, and quota controls.
- Cached blocks must never be overwritten while still referenced.
- Live request KV blocks must not be evicted by idle cache quota cleanup.
- Add diagnostics that explain prefix-cache misses, especially namespace mismatch, TTL expiry, no-store, short prefix, and partial block boundary.
- `/cache/inspect` must expose aggregate cache diagnostics without prompt or token content.

### Kernel And Backend Experiments

- Keep flash-attn as the default attention backend.
- Keep PyTorch/Triton/CUDA extension backend selection behind explicit config.
- Provide `attention_backend="cuda_ext"` for server-side decode attention experiments.
- Keep prefill on the flash-attn reference path in the first CUDA extension integration.
- Require fail-fast behavior when CUDA, compiled extension, dtype, or shape support is missing.
- Prioritize low-risk kernels first:
  - KV cache store.
  - RMSNorm.
  - SiluAndMul.
- Add FlashInfer attention only as an optional backend after the CUDA extension and benchmark harness are stable.

### gpt-oss Compatibility

- Add `model_backend="native|hf_auto"`.
- Detect gpt-oss configs and model names.
- For native execution, fail fast with a clear message until MoE, MXFP4, harmony formatting, GQA/MQA layout, and weight mapping are implemented.
- For current-stage server smoke tests, allow `hf_auto` to load through Transformers on a CUDA host.
- Keep `hf_auto` metrics and reports separate from native nano-vLLM scheduler performance.

### Enterprise Observability

Expose or preserve metrics for:

- waiting/running/pending/active request counts.
- active and pending token pressure.
- TTFT, latency, queue wait, prefill tok/s, decode tok/s.
- timeout, cancellation, rejection, slow consumer, and engine failure counters.
- KV free/used/cached blocks.
- prefix cache hits, misses, hit rate, read/create tokens, eviction counts.
- per-namespace request and cache pressure.
- scheduler policy, policy decision counters, backend name, and prefix-cache miss reasons.

## Non-Goals

- Do not turn the repo into a full vLLM replacement.
- Do not add TensorRT-LLM as a hard dependency.
- Do not default-enable approximate KV compression.
- Do not replace flash-attn by default; custom CUDA attention must remain opt-in and benchmarked against the reference path.
- Do not treat local Python-only tests as GPU serving proof.
- Do not add auth, billing, or production gateway features unless requested later.

## Success Criteria

Local Windows success:

- `scripts\run_local_tests.ps1` passes.
- `pytest` and `unittest` pass.
- `compileall`, CLI help, and `git diff --check` pass.
- No CUDA package is required for local non-GPU tests.

Colab/GPU success:

- `python scripts/validate_online_gpu.py --model <model_dir>` passes.
- `/readyz`, `/generate`, `/generate_stream`, `/cache/prewarm`, `/cache/stats`, `/metrics`, and `/metrics/prometheus` work.
- Prefix cache probe shows `cache_read_input_tokens > 0` or increasing `prefix_cache_hits`.
- `bench_online.py --stream --fetch-metrics` reports TTFT, latency, throughput, and cache usage.

Engineering success:

- New features are covered by focused tests.
- Experimental switches either work with tests or fail fast.
- Documentation says which host can validate which behavior.
- Agent tools can follow the repository rules without re-learning the same constraints.

## Reference Landscape

- vLLM: PagedAttention, continuous batching, OpenAI-compatible serving.
- SGLang: RadixAttention, cache-aware serving, high-throughput scheduler design.
- TensorRT-LLM: in-flight batching, KV cache stats, production runtime ideas.
- FlashInfer: serving-focused LLM kernels.
- FlashAttention: default exact attention foundation.
- Triton, CUTLASS, ThunderKittens, xFormers, Liger Kernel: kernel implementation and benchmarking references.
- OpenAI Codex: AGENTS.md, skills, MCP, hooks, subagents, open-source workflow ideas.
- Anthropic Claude Code: CLAUDE.md-style repo memory, hooks, MCP, subagents, prompt caching semantics.
- Papers and projects to track: PagedAttention, KIVI, SnapKV, H2O, StreamingLLM, LMCache, PagedEviction.
