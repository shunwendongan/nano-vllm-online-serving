# CloudStudio GPU Benchmark Configs

These configs are the CloudStudio/A10 equivalents of `configs/colab/*.env`.
They avoid Colab-only `/content` paths and keep downloaded models plus reports
inside the repository workspace.

Recommended order:

1. `qwen3_native_flash_attn_baseline.env`
2. `qwen3_native_decode_first.env`
3. `qwen3_native_prefill_first.env`
4. `qwen3_native_cache_aware_lpm.env`
5. `qwen3_native_cuda_ext_decode.env`
6. `qwen3_native_a10_prefill_first_c64_r128.env` for the A10 optimized
   high-concurrency probe.
7. `qwen3_native_a100_high_concurrency.env` after switching CloudStudio to A100.
8. `qwen3_native_a100_long_context.env` after switching CloudStudio to A100.

Run from the repository root:

```bash
bash scripts/setup_colab_gpu.sh configs/cloudstudio/qwen3_native_flash_attn_baseline.env
python scripts/run_colab_config.py --config configs/cloudstudio/qwen3_native_flash_attn_baseline.env
```

After the baseline passes:

```bash
bash scripts/run_colab_sweep.sh \
  configs/cloudstudio/qwen3_native_flash_attn_baseline.env \
  configs/cloudstudio/qwen3_native_decode_first.env \
  configs/cloudstudio/qwen3_native_prefill_first.env \
  configs/cloudstudio/qwen3_native_cache_aware_lpm.env
```

Each run writes artifacts under `reports/cloudstudio/<experiment>/<run_id>/`.

For the A10 optimized probe:

```bash
python scripts/run_colab_config.py --config configs/cloudstudio/qwen3_native_a10_prefill_first_c64_r128.env
```

`scripts/run_colab_config.py` also records shell environment overrides in
`resolved_config.json`, so short probes can safely adjust a checked-in config:

```bash
BENCHMARK_CONCURRENCY=96 BENCHMARK_REQUESTS=192 \
python scripts/run_colab_config.py --config configs/cloudstudio/qwen3_native_a10_prefill_first_c64_r128.env
```

For A100 stress tests:

```bash
python scripts/run_colab_config.py --config configs/cloudstudio/qwen3_native_a100_high_concurrency.env
python scripts/run_colab_config.py --config configs/cloudstudio/qwen3_native_a100_long_context.env
```
