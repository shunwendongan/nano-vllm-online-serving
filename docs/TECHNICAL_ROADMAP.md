# nano-vLLM Technical Roadmap

## Summary

The roadmap is serving-first and kernel-gradual. The stable path remains:

`FastAPI -> AsyncLLMEngine -> LLMEngine -> Scheduler -> BlockManager -> ModelRunner -> flash-attn`

The next engineering work should improve this path before replacing kernels.

## Phase 1: Documentation And Agent Guardrails

Deliverables:

- `docs/REQUIREMENTS.md`
- `docs/TECHNICAL_ROADMAP.md`
- `AGENTS.md`
- `CLAUDE.md`

Acceptance:

- Local regression script passes.
- Docs clearly distinguish local non-GPU validation from Colab/GPU validation.
- Agent guidance prevents accidental CUDA claims on this machine.

## Phase 2: Serving Hardening

Implementation focus:

- Add explicit `trace_id` support at protocol and async-engine levels. Default to `request_id` when absent.
- Ensure lifecycle logs include request id, trace id, namespace, queue wait, TTFT, latency, finish reason, cache controls, and token counts.
- Keep prompt text out of logs.
- Document and test status mapping:
  - 400 for request validation.
  - 408 for timeout.
  - 429 for queue or tenant budget rejection.
  - 500 for engine failures.
- Add tests for drain, resume, restart, active abort, disconnect cancellation, slow consumer cancellation, and duplicate id rejection.

Default behavior:

- `request_namespace` controls resource budgets.
- `cache_namespace` controls prefix-cache sharing.
- If `request_namespace` is omitted, fall back to `cache_namespace`.
- `trace_id` does not affect scheduling or cache identity.

## Phase 3: Scheduler Strategy Layer

Keep the current scheduler as the baseline and make policy selection explicit.

Planned policies:

- `fcfs`: deterministic baseline for debugging.
- `decode_first`: prioritize decode tokens when latency matters.
- `prefill_first`: useful for cache prewarm or prompt-heavy batch testing.
- `alternate`: current fairness-oriented behavior.
- `cache_aware_lpm`: admits requests with longer reusable prefix first while enforcing starvation limits.

Implementation rules:

- Do not rewrite `Scheduler` all at once.
- Extract policy choice behind small helper methods.
- Preserve existing tests before adding cache-aware behavior.
- Add starvation tests for cache-aware scheduling.
- Track per-step decision counters in metrics.

Acceptance:

- New requests can enter while decode runs.
- Long prompt chunked prefill does not block existing decode indefinitely.
- KV pressure triggers preemption or delay according to policy.
- Cache-aware policy improves prefix-cache reuse in synthetic tests without starving misses.

## Phase 4: KV Cache Diagnostics And Exact Prefix Reuse

The first production-grade cache target is better exact reuse, not approximate compression.

Implementation focus:

- Add prefix miss reason counters:
  - namespace mismatch.
  - no-store or cache disabled.
  - TTL expired.
  - prefix shorter than minimum.
  - no full block at breakpoint.
  - hash miss.
  - token collision guard mismatch.
- Add `/cache/inspect` for aggregate diagnostic state only, without prompt text.
- Plan radix/prefix-tree metadata for faster longest-prefix-match discovery.
- Keep block hash and token guard checks to prevent wrong reuse.

Experimental KV features:

- `kivi_exp`
- `snapkv_exp`
- `h2o_exp`
- `streamingllm_exp`
- `paged_eviction_exp`

Rules:

- Experiments must be off by default.
- Experiments must declare accuracy risk.
- Experiments must have a fallback to exact KV.
- Local tests cover config and fail-fast behavior; GPU tests cover numeric or generation drift.

## Phase 5: CUDA Attention Backend Integration

Default:

- `attention_backend="flash_attn"`
- `op_backend="torch"`
- `kv_cache_dtype="auto"`

Current-stage additions:

- Add `attention_backend="cuda_ext"` as an explicit server-side experiment.
- Keep `flash_attn` as the reference backend.
- Keep prefill on `flash_attn_varlen_func` in the first integration.
- Route decode through the optional CUDA extension only when `attention_backend="cuda_ext"`.
- Implement extension entrypoints under `nanovllm/kernels/cuda_ext/`:
  - dense MHA correctness kernel.
  - GQA/MQA correctness kernel.
  - streaming-softmax attention kernel.
  - paged decode attention over `block_tables`, `context_lens`, `k_cache`, and `v_cache`.
- Unsupported dtype, layout, shape, or missing CUDA extension must fail fast instead of silently claiming acceleration.
- Implement backend tests in three layers:
  - import/config tests on local Windows.
  - shape/stride/fallback tests without CUDA where possible.
  - numeric and throughput tests on CUDA.

Kernel priority:

1. KV store: low risk, directly connected to paged cache.
2. RMSNorm and fused add-RMSNorm: simple numerical contract.
3. SiluAndMul: simple MLP activation fusion.
4. CUDA extension paged decode attention: compare with flash-attn decode on GPU.
5. FlashInfer paged or ragged attention: optional future backend after the custom extension path is measurable.

Acceptance:

- Backends are selected through config, not imports with side effects.
- Missing optional packages produce clear errors.
- Reference backend and experimental backend are numerically compared on GPU.
- Benchmark output includes backend name, model, scheduler policy, prompt length, concurrency, TTFT, decode throughput, and cache stats.

## Phase 6: gpt-oss Current-Stage Compatibility

Supported first-class targets:

- Qwen3 small models.
- MiniCPM/CPM path already present in this repo.

Current-stage gpt-oss target:

- Add `model_backend="native|hf_auto"`.
- Keep `native` as the nano-vLLM model/scheduler path.
- Add `hf_auto` as a Transformers bridge for gpt-oss-20b smoke tests on a CUDA server.
- Detect gpt-oss from `model_type`, `architectures`, or model path hints.
- Fail fast in the native path when gpt-oss is detected, because MoE and MXFP4 are not implemented natively.
- Provide skeleton files for future native support without pretending it is complete.

Expected gaps for gpt-oss-style models:

- MoE routing and expert parallel strategy.
- GQA/MQA layout compatibility.
- Longer context and cache pressure.
- MXFP4 or other low-bit weight loading and dequantization.
- Harmony/chat format and tokenizer compatibility.
- Weight-name mapping for router, experts, attention, and norms.

Claude models:

- Claude weights are not open-source targets for this repo.
- Borrow public prompt caching and agent engineering semantics only.

## Validation Matrix

Local Windows:

```powershell
.\scripts\run_local_tests.ps1
```

Expected:

- Python `>=3.10,<3.13`.
- Required local packages import.
- Optional GPU packages may be missing.
- `unittest`, `pytest`, `compileall`, `git diff --check`, and CLI help checks pass.

Colab or CUDA host:

```bash
python scripts/validate_online_gpu.py --model <model_dir>
```

Expected:

- Runtime check sees CUDA, torch, flash-attn, triton, transformers.
- Server reaches `/readyz`.
- Simple and OpenAI-compatible generation work.
- SSE streaming emits incremental chunks.
- Cache prewarm and prefix cache probe succeed.
- Benchmark reports TTFT, latency, throughput, and cache usage.
- CUDA extension runs numeric correctness and throughput comparisons before any performance claim.
- gpt-oss-20b `hf_auto` smoke test is reported separately from native nano-vLLM scheduler benchmarks.

## Implementation Discipline

- Keep changes scoped to one subsystem at a time.
- Add tests before or with behavior changes.
- Preserve the stable flash-attn path when adding new backends.
- Prefer fail-fast errors over silent no-ops for reserved features.
- Do not claim GPU correctness from local non-GPU tests.
- Do not describe `hf_auto` gpt-oss results as native continuous batching.
- Keep docs updated when CLI flags, endpoints, cache semantics, or validation commands change.
