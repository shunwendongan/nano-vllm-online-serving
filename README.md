# nano-vLLM Online Serving Lab

基于 nano-vLLM 改造的轻量级 LLM Serving 实验项目，重点覆盖在线推理、continuous batching、Paged KV cache、prefix cache diagnostics、OpenAI-style API、Colab GPU 验证和 benchmark 报告闭环。

这个仓库适合作为学习和实习项目展示：代码结构保留 nano-vLLM 的教学可读性，同时补上企业在线推理系统里常见的请求生命周期、调度、缓存、指标和实验配置。

## 共享与协作

本项目由 GitHub 账号 `shunwendongan` 维护，并共享给 Codex coding agent 参与代码分析、性能瓶颈定位、实验配置整理、优化实现和 PR 辅助交付。仓库访问权限仍以 GitHub 实际 collaborator/settings 为准。

## 主要能力

- 在线推理服务：FastAPI HTTP API、SSE streaming、OpenAI-compatible completions/chat。
- 异步请求生命周期：`request_id`、`trace_id`、pending/active queue、cancel、timeout、drain/restart。
- Continuous batching：iteration-level scheduling，新请求可在 decode 过程中进入系统。
- Chunked prefill：长 prompt 分块 prefill，降低对 decode 的阻塞。
- Paged KV cache：基于 block table 管理 KV cache，支持 live/cached/free block 状态。
- Prefix cache：exact prefix cache、TTL、namespace、quota、LRU eviction、miss reason diagnostics。
- Metrics：TTFT、latency、queue wait、prefill/decode tok/s、cache hit rate、preemption、eviction。
- Benchmark：`bench_online.py` 输出 JSON 和 Markdown 报告，并附带瓶颈分析建议。
- Colab 配置：用 `configs/colab/*.env` 固化 GPU 验证参数，方便复现实验。
- 实验 backend：`attention_backend=flash_attn|cuda_ext`，`model_backend=native|hf_auto`。

## 当前边界

本地 Windows 环境用于非 GPU 回归测试，不证明真实模型吞吐。真实性能数据需要在 Google Colab 或 CUDA 服务器上运行。

- `model_backend=native` 才代表 nano-vLLM 自研 scheduler、Paged KV cache、prefix cache 路径。
- `model_backend=hf_auto` 是 Transformers 兼容路径，用于 gpt-oss smoke test，不等同于 native continuous batching 性能。
- `attention_backend=cuda_ext` 是显式实验路径，默认稳定路径仍是 `flash_attn`。
- FP8/KIVI/SnapKV/H2O/StreamingLLM 等 KV 压缩入口保留为实验开关，默认不启用。

## 项目结构

```text
nanovllm/
  serve.py                         # HTTP server CLI
  check_runtime.py                 # CUDA/依赖/模型路径检查
  engine/
    async_engine.py                # async queue, lifecycle, metrics
    llm_engine.py                  # native engine step loop
    scheduler.py                   # continuous batching scheduler
    block_manager.py               # paged KV and prefix cache manager
    model_runner.py                # model execution, CUDA graph, KV tensors
    hf_auto_engine.py              # Transformers compatibility backend
  entrypoints/openai/api_server.py # /generate, SSE, OpenAI-style API
  layers/
    attention.py                   # flash-attn / cuda_ext dispatch
    cuda_attention.py              # optional CUDA extension wrapper
  kernels/cuda_ext/                # CUDA attention extension skeleton
  models/                          # Qwen/CPM/gpt-oss compatibility layer

configs/colab/                     # GPU experiment configs
scripts/
  run_local_tests.ps1              # local non-GPU regression
  setup_colab_gpu.sh               # Colab dependency/model setup
  run_colab_config.py              # config-driven GPU validation
  validate_online_gpu.py           # server + API + benchmark validation
bench_online.py                    # online benchmark client
docs/                              # requirements, roadmap, runbooks
tests/                             # non-GPU unit/API/mock tests
```

## 本地非 GPU 测试

本机只跑 Python 逻辑、API mock、scheduler、block manager、async engine、脚本和 CLI wiring。

```powershell
.\scripts\run_local_tests.ps1
```

也可以手动安装本地测试依赖：

```powershell
python -m pip install -r requirements-local.txt
python -m pytest tests -q
```

本地测试不会安装 `torch`、`flash-attn`、`triton` 或 CUDA 相关依赖。

## Colab GPU 快速开始

在 Colab 中先切换到 GPU runtime，然后从仓库根目录运行：

```bash
nvidia-smi
bash scripts/setup_colab_gpu.sh configs/colab/qwen3_native_flash_attn_baseline.env
python scripts/run_colab_config.py --config configs/colab/qwen3_native_flash_attn_baseline.env
```

每次运行会生成：

```text
reports/colab/<experiment>/<run_id>/
  resolved_config.json
  command.txt
  validation_output.txt
  *_bench.json
  *_bench.md
  online_requests.jsonl
```

`reports/` 默认不会提交到 Git。

## 推荐实验顺序

先跑 native baseline：

```bash
python scripts/run_colab_config.py \
  --config configs/colab/qwen3_native_flash_attn_baseline.env
```

再跑 scheduler policy sweep：

```bash
bash scripts/run_colab_sweep.sh \
  configs/colab/qwen3_native_flash_attn_baseline.env \
  configs/colab/qwen3_native_decode_first.env \
  configs/colab/qwen3_native_prefill_first.env \
  configs/colab/qwen3_native_cache_aware_lpm.env
```

gpt-oss smoke test 单独记录：

```bash
bash scripts/setup_colab_gpu.sh configs/colab/gpt_oss_hf_auto_smoke.env
python scripts/run_colab_config.py --config configs/colab/gpt_oss_hf_auto_smoke.env
```

CUDA extension decode attention 实验在 baseline 跑通后再运行：

```bash
python scripts/run_colab_config.py \
  --config configs/colab/qwen3_native_cuda_ext_decode.env
```

详细说明见 [docs/COLAB_BENCHMARKS.md](docs/COLAB_BENCHMARKS.md) 和 [docs/PERFORMANCE_ANALYSIS.md](docs/PERFORMANCE_ANALYSIS.md)。

## CloudStudio GPU 快速开始

CloudStudio/A10 工作区优先使用仓库内路径，避免 Colab 的 `/content` 目录假设：

```bash
nvidia-smi
bash scripts/setup_colab_gpu.sh configs/cloudstudio/qwen3_native_flash_attn_baseline.env
python scripts/run_colab_config.py --config configs/cloudstudio/qwen3_native_flash_attn_baseline.env
```

一键运行 A10 baseline + scheduler sweep：

```bash
bash scripts/run_cloudstudio_matrix.sh a10
```

切换到 A100 后可运行：

```bash
bash scripts/run_cloudstudio_matrix.sh a100
```

详细说明见 [docs/CLOUDSTUDIO_BENCHMARKS.md](docs/CLOUDSTUDIO_BENCHMARKS.md)。

## 启动服务

Native nano-vLLM serving：

```bash
python -m nanovllm.serve \
  --model /content/models/Qwen3-0.6B \
  --host 0.0.0.0 \
  --port 8000 \
  --model-backend native \
  --attention-backend flash_attn
```

gpt-oss Transformers compatibility serving：

```bash
python -m nanovllm.serve \
  --model openai/gpt-oss-20b \
  --model-backend hf_auto \
  --host 0.0.0.0 \
  --port 8000
```

## API 示例

Simple generation：

```bash
curl -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Explain paged KV cache in one sentence.","max_tokens":64}'
```

Streaming：

```bash
curl -N -X POST http://127.0.0.1:8000/generate_stream \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Explain continuous batching.","max_tokens":64}'
```

OpenAI-style chat：

```bash
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nano-vllm",
    "messages": [{"role": "user", "content": "Explain prefix cache."}],
    "max_tokens": 64
  }'
```

Metrics：

```bash
curl http://127.0.0.1:8000/metrics
curl http://127.0.0.1:8000/cache/inspect
curl http://127.0.0.1:8000/metrics/prometheus
```

## Benchmark

```bash
python bench_online.py \
  --url http://127.0.0.1:8000 \
  --stream \
  --requests 32 \
  --concurrency 4 \
  --max-tokens 64 \
  --fetch-metrics \
  --report-json-path reports/manual_bench.json \
  --report-markdown-path reports/manual_bench.md
```

报告会包含：

- TTFT p50/p95/p99
- latency p50/p95/p99
- completion token throughput
- error rate
- prefix cache hit rate
- cache read/create tokens
- preemption/eviction counters
- automatic bottleneck analysis

## 文档

- [ONLINE_SERVING.md](ONLINE_SERVING.md): serving API、queue controls、cache controls、benchmark 说明。
- [docs/COLAB_BENCHMARKS.md](docs/COLAB_BENCHMARKS.md): Colab 实验运行手册。
- [docs/PERFORMANCE_ANALYSIS.md](docs/PERFORMANCE_ANALYSIS.md): 性能指标解释和瓶颈判断规则。
- [docs/REQUIREMENTS.md](docs/REQUIREMENTS.md): 产品需求和企业痛点。
- [docs/TECHNICAL_ROADMAP.md](docs/TECHNICAL_ROADMAP.md): 技术路线和后续改造方向。
- [AGENTS.md](AGENTS.md), [CLAUDE.md](CLAUDE.md): agent 协作和测试边界说明。

## 简历表述建议

可以真实描述：

- 实现轻量 LLM online serving stack，支持 HTTP/SSE/OpenAI-style API。
- 扩展 continuous batching scheduler，支持 chunked prefill 和多种调度策略。
- 实现 Paged KV block manager 和 exact prefix cache，支持 TTL、namespace、quota、LRU eviction 和 miss reason diagnostics。
- 构建 TTFT/latency/throughput/cache metrics 和 Colab benchmark 报告闭环。
- 增加 gpt-oss `hf_auto` smoke path，并明确 native MoE/MXFP4 适配边界。

不要在没有 Colab/服务器实测前声称具体 GPU 提速百分比。

## License

MIT. See [LICENSE](LICENSE).
