# CloudStudio baseline and bottleneck analysis - 2026-06-16

## Scope

- Repository: `shunwendongan/nano-vllm-online-serving`
- CloudStudio workspace: `https://cloudstudio.net/a/36187923176894464/edit`
- Final code commit used on CloudStudio: `f733698`
- Model: `Qwen/Qwen3-0.6B`
- Local validation: `scripts/run_local_tests.ps1` passed with `142 passed, 1 skipped`; `unittest discover` passed with `143 tests OK (skipped=1)`.
- GPU validation: A10 and A100 CloudStudio runs under `reports/cloudstudio`.

Local Windows validation only covers control-plane behavior, API/config compatibility, and test logic. Throughput, TTFT, CUDA extension behavior, FlashAttention, and GPU memory conclusions are based on CloudStudio GPU runs.

## Code fixes made before successful baseline

1. CUDA extension builds now select a supported host compiler:
   - Prefer `NANOVLLM_CUDA_HOST_COMPILER`, then `g++-12`, then `g++-11`.
   - Fall back to `-allow-unsupported-compiler` only when no supported compiler is found.
   - CloudStudio setup installs `g++-12` for CUDA extension runs.

2. CUDA extension attention now uses a valid KV-cache dtype contract:
   - `kv_cache_dtype=float32` is supported.
   - `attention_backend=cuda_ext` with `auto` KV dtype resolves to `float32`.
   - The wrapper casts query tensors to `float32`, returns the original output dtype, and rejects non-`float32` KV cache instead of silently copying the whole cache.

3. Prefix cache is disabled for the current CUDA extension experiment:
   - The existing `cuda_ext` path is decode-focused and float32-KV based.
   - Prefix-cache prefill reuse still depends on the FlashAttention low-precision path, so incompatible configs now fail fast.

4. Benchmark summarization no longer reports intentionally disabled prefix cache as a missing-cache failure.

5. The A100 long-context probe was bounded to the model length:
   - `CACHE_PROBE_REPETITIONS=4096`, because `8192` repetitions produced `24589` Qwen tokens and exceeded `max_model_len=16384`.

## Successful runs

| Experiment | GPU | Backend | Policy | Requests | Concurrency | Req/s | Completion tok/s | TTFT p95 | Latency p95 | Prefix hit | Preemptions | Evictions |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| qwen3_native_flash_attn_baseline | A10 | flash_attn | alternate | 32 | 4 | 1.3792 | 88.2688 | 0.2534s | 3.0309s | 0.8000 | 0 | 0 |
| qwen3_native_prefill_first | A10 | flash_attn | prefill_first | 32 | 4 | 1.4411 | 92.2311 | 0.1831s | 2.8877s | 0.8000 | 0 | 0 |
| qwen3_native_decode_first | A10 | flash_attn | decode_first | 32 | 4 | 0.7335 | 46.9410 | 2.8046s | 5.5280s | 0.8000 | 0 | 0 |
| qwen3_native_cache_aware_lpm | A10 | flash_attn | cache_aware_lpm | 32 | 4 | 1.3524 | 86.5560 | 0.2644s | 3.0716s | 0.8000 | 0 | 0 |
| qwen3_native_cuda_ext_decode | A10 | cuda_ext | alternate | 32 | 4 | 1.3815 | 88.4175 | 0.1894s | 3.0224s | 0.0000 | 0 | 0 |
| qwen3_native_a100_high_concurrency | A100 | flash_attn | decode_first | 512 | 64 | 6.4498 | 825.5748 | 4.4083s | 10.0676s | 0.8000 | 0 | 0 |
| qwen3_native_a100_long_context | A100 | flash_attn | cache_aware_lpm | 128 | 16 | 3.1180 | 399.1034 | 0.2433s | 5.2566s | 0.9796 | 0 | 0 |

## Main findings

1. The baseline now runs end to end.
   - All seven benchmark JSON reports completed with zero request errors.
   - The final CloudStudio summary reports `Report count: 7`.

2. On A10, `prefill_first` is the best small-workload policy among the measured runs.
   - Best latency: `2.8877s` p95 latency and `0.1831s` p95 TTFT.
   - `decode_first` is poor for this small workload: throughput drops to `46.9410 tok/s`, TTFT p95 increases to `2.8046s`, and latency p95 increases to `5.5280s`.

3. The current `cuda_ext` path is not a production win yet.
   - It matches baseline-level throughput on this small A10 run, but only after using float32 KV cache and disabling prefix cache.
   - Its prefix-cache hit rate is intentionally `0.0`.
   - It should be treated as a smoke-test/development backend until it supports low-precision paged KV and prefix-cache-compatible prefill/decode behavior.

4. A100 high-concurrency throughput scales, but latency and TTFT become the main service bottleneck.
   - At concurrency 64 and 512 requests, throughput reaches `825.5748 completion tok/s`.
   - p95 latency is `10.0676s`; p95 TTFT is `4.4083s`.
   - Queueing failures, preemption, eviction, and engine loop errors were not observed.
   - This points to decode iteration scheduling, Python/async orchestration, HTTP/SSE emission, or per-token synchronization overhead rather than KV capacity exhaustion.

5. Prefix cache works well on the A100 long-context probe once the prompt length is valid.
   - Prefix hit rate: `0.9796`.
   - Cache creation input tokens: `12288`.
   - Cache read input tokens: `24576`.
   - p95 TTFT stays low at `0.2433s`, showing the long-prefix reuse path is effective.

6. KV pressure was not reached in these successful runs.
   - A10 FlashAttention runs had `661` KV blocks with no preemptions or evictions.
   - A100 FlashAttention runs had `1227` KV blocks with no preemptions or evictions.
   - Larger models, higher active-token budgets, higher concurrency, or longer generated outputs are needed to find the true KV-cache pressure knee.

## Optimization priorities

1. Scheduler defaults:
   - Use `prefill_first` or `alternate` for normal small/medium concurrency workloads.
   - Avoid `decode_first` as a default unless a concurrency sweep proves it helps a decode-heavy workload.

2. Decode path:
   - Reduce per-token Python and async overhead in the engine loop.
   - Add or expand CUDA graph capture for stable decode shapes.
   - Batch HTTP/SSE flushes where product latency requirements allow.
   - Add a whole-program timeline with Nsight Systems before kernel-level tuning.

3. Prefix cache:
   - Keep explicit cache namespaces and cache breakpoints in benchmark and production clients.
   - Keep long reusable prefixes block-aligned when possible.
   - Tune `PREFIX_CACHE_MIN_TOKENS` with real traffic, not only synthetic prompts.

4. KV/admission control:
   - Sweep `MAX_ACTIVE_TOKENS`, `MAX_NUM_BATCHED_TOKENS`, `MAX_NUM_SEQS`, and `KVCACHE_WATERMARK_BLOCKS`.
   - Add an admission controller that protects p95 latency instead of only maximizing throughput.
   - Run larger models or higher active-token workloads to force preemption/eviction and identify the real memory knee.

5. CUDA extension backend:
   - Implement bf16/fp16 paged decode instead of float32 KV.
   - Remove the query cast and float32 KV memory expansion.
   - Integrate a prefix-cache-compatible path or keep fail-fast guards.
   - Compare against FlashAttention only after feature parity.

## Reproducibility notes

- Summary command used on CloudStudio:
  `python scripts/summarize_benchmarks.py --root reports/cloudstudio`
- CloudStudio report paths:
  - `reports/cloudstudio/qwen3_native_flash_attn_baseline/20260616-045931`
  - `reports/cloudstudio/qwen3_native_prefill_first/20260616-050125`
  - `reports/cloudstudio/qwen3_native_decode_first/20260616-050018`
  - `reports/cloudstudio/qwen3_native_cache_aware_lpm/20260616-050210`
  - `reports/cloudstudio/qwen3_native_cuda_ext_decode/20260616-053728`
  - `reports/cloudstudio/qwen3_native_a100_high_concurrency/20260616-055512`
  - `reports/cloudstudio/qwen3_native_a100_long_context/20260616-055937`

After the final extraction, the A100 CloudStudio app was stopped to avoid continuing compute-hour consumption.
