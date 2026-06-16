# CloudStudio Benchmark Runbook

Use this runbook for the CloudStudio GPU workspace. The configs under
`configs/cloudstudio/` keep model files and reports inside the repository
workspace instead of using Colab-only `/content` paths.

## Setup

From the repository root:

```bash
nvidia-smi
bash scripts/setup_colab_gpu.sh configs/cloudstudio/qwen3_native_flash_attn_baseline.env
```

The setup script checks Python `>=3.10,<3.13`, installs project/runtime
dependencies, downloads `Qwen/Qwen3-0.6B` into `.cache/models/Qwen3-0.6B`, and
fails fast if CUDA, torch, flash-attn, triton, transformers, or the model path
are not usable.

## A10 Baseline

```bash
python scripts/run_colab_config.py \
  --config configs/cloudstudio/qwen3_native_flash_attn_baseline.env
```

If this fails, inspect:

```bash
cat reports/cloudstudio/qwen3_native_flash_attn_baseline/*/validation_output.txt
```

`validate_online_gpu.py` now fails quickly when the server exits before
`/readyz`, and the JSON output includes `server_log_tail` so startup failures
are diagnosable.

## One-Command Matrix

For A10:

```bash
bash scripts/run_cloudstudio_matrix.sh a10
```

After switching CloudStudio to A100:

```bash
bash scripts/run_cloudstudio_matrix.sh a100
```

Set `SKIP_SETUP=1` to reuse an already prepared environment:

```bash
SKIP_SETUP=1 bash scripts/run_cloudstudio_matrix.sh a10
```

The matrix runner writes an aggregate report after success or failure:

```text
reports/cloudstudio/summary.json
reports/cloudstudio/summary.md
```

## A10 Optimized Probe

The A10 bottleneck report showed low queue wait, no KV preemptions, and low GPU
SM/memory utilization. Use the checked-in optimized probe to exercise larger
decode batches with `prefill_first`, CUDA graph decode, and reduced SSE flush
frequency:

```bash
python scripts/run_colab_config.py \
  --config configs/cloudstudio/qwen3_native_a10_prefill_first_c64_r128.env
```

For short sweeps, override only the probe dimensions from the shell. Overrides
are recorded in `resolved_config.json`:

```bash
BENCHMARK_CONCURRENCY=96 BENCHMARK_REQUESTS=192 \
python scripts/run_colab_config.py \
  --config configs/cloudstudio/qwen3_native_a10_prefill_first_c64_r128.env
```

## Policy Sweep

Run after the baseline passes:

```bash
bash scripts/run_colab_sweep.sh \
  configs/cloudstudio/qwen3_native_flash_attn_baseline.env \
  configs/cloudstudio/qwen3_native_decode_first.env \
  configs/cloudstudio/qwen3_native_prefill_first.env \
  configs/cloudstudio/qwen3_native_cache_aware_lpm.env
```

Compare `*_bench.json` and `*_bench.md` across runs for:

- `ttft_p95_s`
- `latency_p95_s`
- `completion_tok_per_s`
- `server_prefix_cache_hit_rate`
- `server_preemptions`
- `server_evictions`
- `bottleneck_analysis`

You can regenerate the aggregate summary at any time:

```bash
python scripts/summarize_benchmarks.py --root reports/cloudstudio
```

## A100 Stress Runs

After switching CloudStudio to A100:

```bash
python scripts/run_colab_config.py \
  --config configs/cloudstudio/qwen3_native_a100_high_concurrency.env

python scripts/run_colab_config.py \
  --config configs/cloudstudio/qwen3_native_a100_long_context.env
```

Use high-concurrency results to find throughput and queueing limits. Use
long-context results to inspect prefill cost, prefix-cache reuse, KV pressure,
preemptions, and evictions.

## CUDA Extension

Run only after the flash-attn baseline is correct on the same GPU:

```bash
python scripts/run_colab_config.py \
  --config configs/cloudstudio/qwen3_native_cuda_ext_decode.env
```

Treat `cuda_ext` as experimental until output correctness and benchmark
stability match the flash-attn baseline.
