# Colab Benchmark Configs

These `.env` files drive `scripts/run_colab_config.py`, which converts stable
experiment settings into `scripts/validate_online_gpu.py` commands and writes
timestamped artifacts under `reports/colab/<experiment>/<run_id>/`.

Recommended order:

1. `gpt_oss_hf_auto_smoke.env`: compatibility smoke test for gpt-oss through
   Transformers. Do not use its numbers as native nano-vLLM scheduler results.
2. `qwen3_native_flash_attn_baseline.env`: native nano-vLLM baseline with
   flash-attn, paged KV, prefix cache, and continuous batching.
3. `qwen3_native_decode_first.env` and `qwen3_native_prefill_first.env`:
   scheduler policy comparisons against the baseline.
4. `qwen3_native_cache_aware_lpm.env`: scheduler policy comparison against the
   flash-attn baseline.
5. `qwen3_native_cuda_ext_decode.env`: optional CUDA decode attention backend
   experiment after the baseline works.

Run one config on Colab:

```bash
bash scripts/setup_colab_gpu.sh configs/colab/qwen3_native_flash_attn_baseline.env
python scripts/run_colab_config.py --config configs/colab/qwen3_native_flash_attn_baseline.env
```

Inspect the generated run directory for:

- `resolved_config.json`
- `command.txt`
- `validation_output.txt`
- benchmark JSON and Markdown reports from `bench_online.py`
- `online_requests.jsonl`
