# Colab Benchmark Runbook

Use this runbook to collect GPU-side data for the modified nano-vLLM serving
stack. Local Windows tests only validate non-GPU logic; TTFT, throughput,
flash-attn, CUDA graph, and CUDA extension behavior must be measured on Colab or
another CUDA host.

## Setup

Use a GPU runtime first. A CPU runtime is not enough.

```bash
nvidia-smi
python --version
```

Install dependencies and, for native Qwen configs, download the model:

```bash
bash scripts/setup_colab_gpu.sh configs/colab/qwen3_native_flash_attn_baseline.env
```

For gpt-oss smoke validation:

```bash
bash scripts/setup_colab_gpu.sh configs/colab/gpt_oss_hf_auto_smoke.env
```

If the repository is private, clone it with a Colab secret such as
`GITHUB_TOKEN`. Do not print the token in the notebook.

## Dry Run

Before spending GPU time, verify the resolved command:

```bash
python scripts/run_colab_config.py \
  --config configs/colab/qwen3_native_flash_attn_baseline.env \
  --dry-run
```

The dry run writes `resolved_config.json` and `command.txt` but does not start
the server.

## Single Experiments

Native flash-attn baseline:

```bash
python scripts/run_colab_config.py \
  --config configs/colab/qwen3_native_flash_attn_baseline.env
```

Cache-aware scheduler variant:

```bash
python scripts/run_colab_config.py \
  --config configs/colab/qwen3_native_cache_aware_lpm.env
```

Decode-first and prefill-first scheduler variants:

```bash
python scripts/run_colab_config.py \
  --config configs/colab/qwen3_native_decode_first.env

python scripts/run_colab_config.py \
  --config configs/colab/qwen3_native_prefill_first.env
```

CUDA decode attention experiment:

```bash
python scripts/run_colab_config.py \
  --config configs/colab/qwen3_native_cuda_ext_decode.env
```

gpt-oss Transformers compatibility smoke:

```bash
python scripts/run_colab_config.py \
  --config configs/colab/gpt_oss_hf_auto_smoke.env
```

## Sweep

After the baseline passes, run a small bounded sweep:

```bash
bash scripts/run_colab_sweep.sh \
  configs/colab/qwen3_native_flash_attn_baseline.env \
  configs/colab/qwen3_native_decode_first.env \
  configs/colab/qwen3_native_prefill_first.env \
  configs/colab/qwen3_native_cache_aware_lpm.env
```

Run the CUDA extension config separately after numerical correctness and import
fail-fast behavior are understood on the target server.

## Artifacts

Each run writes a timestamped directory:

```text
reports/colab/<experiment>/<YYYYmmdd-HHMMSS>/
```

Expected files:

- `resolved_config.json`: config, command list, and shell command.
- `command.txt`: exact command to reproduce the run.
- `validation_output.txt`: full output from `validate_online_gpu.py`.
- benchmark JSON: metrics and automatic bottleneck analysis.
- benchmark Markdown: report suitable for copying into notes.
- `online_requests.jsonl`: request lifecycle log without prompt text.

Use benchmark JSON/Markdown for resume numbers only after confirming the run used
`model_backend=native` for scheduler, paged KV, prefix cache, and CUDA backend
claims. `model_backend=hf_auto` is a Transformers compatibility path.

## Interpreting Results

Look at these fields first:

- `ttft_p50_s`, `ttft_p95_s`, `ttft_p99_s`
- `latency_p50_s`, `latency_p95_s`, `latency_p99_s`
- `completion_tok_per_s`
- `server_prefix_cache_hit_rate`
- `cache_read_input_tokens`, `cache_creation_input_tokens`
- `server_preemptions`, `server_evictions`
- `bottleneck_analysis`

High TTFT usually points to prefill, warmup, prompt length, model placement, or
CPU/offload overhead. Latency much larger than TTFT usually points to decode
throughput, concurrency, `max_tokens`, or backend overhead. Prefix hit rate of
zero under `hf_auto` is expected and is not a native cache failure.
