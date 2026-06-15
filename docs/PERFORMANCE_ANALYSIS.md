# nano-vLLM Performance Analysis Playbook

## Summary

Local Windows validation proves non-GPU logic only. This machine does not have
`nvidia-smi`, CUDA, torch, flash-attn, triton, or transformers for real model
throughput. Treat performance numbers as valid only when collected on Colab or a
CUDA server.

The first GPU smoke target is:

- Colab A100 or L4.
- HuggingFace online model id: `openai/gpt-oss-20b`.
- `model_backend="hf_auto"`.
- Streaming TTFT and throughput through `bench_online.py`.

`hf_auto` is a Transformers compatibility path. Do not describe its benchmark as
native nano-vLLM continuous batching, paged KV, prefix cache, or CUDA extension
performance.

## Colab gpt-oss hf_auto Smoke

Run from the repository root after installing runtime dependencies:

```bash
nvidia-smi
python -m nanovllm.check_runtime \
  --model openai/gpt-oss-20b \
  --model-backend hf_auto
```

Start serving:

```bash
python -m nanovllm.serve \
  --model openai/gpt-oss-20b \
  --model-backend hf_auto \
  --host 0.0.0.0 \
  --port 8000
```

Run the benchmark:

```bash
python bench_online.py \
  --url http://127.0.0.1:8000 \
  --stream \
  --requests 32 \
  --concurrency 4 \
  --max-tokens 64 \
  --fetch-metrics \
  --model-name gpt-oss-20b \
  --backend hf_auto \
  --scheduler-policy hf_auto \
  --report-json-path reports/gpt_oss_hf_auto_bench.json \
  --report-markdown-path reports/gpt_oss_hf_auto_bench.md
```

The same report paths are used automatically by:

```bash
python scripts/validate_online_gpu.py \
  --model openai/gpt-oss-20b \
  --model-backend hf_auto \
  --benchmark-requests 32 \
  --benchmark-concurrency 4 \
  --max-tokens 64
```

The config-driven equivalent is:

```bash
bash scripts/setup_colab_gpu.sh configs/colab/gpt_oss_hf_auto_smoke.env
python scripts/run_colab_config.py --config configs/colab/gpt_oss_hf_auto_smoke.env
```

Native Qwen benchmark configs live under `configs/colab/`; the detailed runbook
is `docs/COLAB_BENCHMARKS.md`.

For `hf_auto`, native prefix-cache prewarm/probe checks are skipped because this
path does not use nano-vLLM's paged prefix cache. `/cache/inspect` still returns
aggregate compatibility fields for API shape validation.

## Bottleneck Rules

`bench_online.py` writes `bottleneck_analysis` into JSON and a matching
`Bottleneck Analysis` section into Markdown.

- High errors or timeouts: inspect GPU memory, server logs, model download/auth,
  HTTP timeout, and Colab runtime stability before interpreting throughput.
- High TTFT share: check warmup, prompt length, prefill cost, model placement,
  and CPU/disk offload.
- Latency much higher than TTFT: check decode throughput, `max_tokens`,
  concurrency, and backend overhead.
- Zero prefix-cache hit rate under `hf_auto`: expected and not a native cache
  failure.
- Preemptions or evictions under native backend: indicates KV pressure; lower
  concurrency or tokens, tune scheduler policy, and inspect `/cache/inspect`.

## Native Follow-Up Matrix

Run these separately after the gpt-oss smoke:

- Qwen3 native + flash-attn baseline.
- Qwen3 native with prefix-cache cold vs warm prompts.
- Qwen3 native policy sweep: `alternate`, `decode_first`, `prefill_first`,
  `cache_aware_lpm`.
- Qwen3 native `flash_attn` vs `cuda_ext` decode after CUDA extension numeric
  correctness is proven.

Only native Qwen results should be used for resume claims about continuous
batching, paged KV cache, prefix cache, scheduler policy, or custom CUDA
attention speedups.
