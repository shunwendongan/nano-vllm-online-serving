# AGENTS.md

## Repository Role

This repository is a modified nano-vLLM project focused on online inference, continuous batching, paged KV cache, prefix caching, an optional CUDA attention backend, and current-stage gpt-oss compatibility.

The active package is the top-level `nanovllm/` directory. Do not treat old nested copies or experiment folders as the main implementation.

## Local Environment

- Current local development host is Windows.
- This machine has no CUDA runtime for real model serving.
- Local tests are Python-only or mocked serving tests.
- Do not install GPU dependencies locally unless the user explicitly asks.
- Real torch, flash-attn, triton, CUDA graph, and throughput validation belongs on Colab or another CUDA host.

## Required Local Validation

From the repository root, use:

```powershell
.\scripts\run_local_tests.ps1
```

This is the default pre-handoff check after documentation, protocol, scheduler, block manager, async engine, API, script, or test changes.

The script is expected to:

- use Python 3.12 on this machine.
- check local packages such as pytest, fastapi, starlette, uvicorn, and httpx.
- report missing GPU packages without failing local validation.
- run unittest, pytest, compileall, git diff whitespace checks, and CLI help checks.

## CUDA Validation

Use a CUDA host for:

```bash
python scripts/validate_online_gpu.py --model <model_dir>
```

Do not state that GPU serving is validated unless this script or an equivalent CUDA run passed.

## Engineering Rules

- Serving correctness comes before defaulting to custom kernels.
- Keep flash-attn as the stable reference attention path.
- `attention_backend="cuda_ext"` is opt-in and must be validated on CUDA before any speedup claim.
- `model_backend="hf_auto"` is a gpt-oss smoke-test bridge, not native nano-vLLM continuous batching.
- Keep approximate KV compression off by default.
- Reserved switches must fail fast instead of becoming silent no-ops.
- Preserve request cancellation, drain, restart, timeout, and queue-budget behavior.
- Keep prompt text out of request lifecycle logs.
- Keep `request_namespace` for resource budgets separate from `cache_namespace` for prefix-cache sharing.
- Add or update tests for scheduler, KV cache, API protocol, metrics, and scripts when changing behavior.

## Documentation Rules

- Update `ONLINE_SERVING.md` when endpoints, CLI flags, metrics, or cache semantics change.
- Update `docs/REQUIREMENTS.md` when product scope or enterprise requirements change.
- Update `docs/TECHNICAL_ROADMAP.md` when implementation sequencing changes.
- Keep local-vs-Colab validation boundaries explicit.

## Kernel Work Rules

- Add kernel backends behind explicit config.
- Always compare experimental kernels against the stable reference backend.
- For local Windows, test import/config/error paths only.
- For CUDA hosts, test numeric equivalence and benchmark output.
- For this stage, CUDA attention extension work may exist under `nanovllm/kernels/cuda_ext/`, but it must stay optional and benchmarked against flash-attn.

## Git Hygiene

- The worktree may contain unrelated user changes.
- Do not revert changes that were not made for the current task.
- Avoid broad refactors unless required by the requested change.
- Prefer small, reviewable changes with tests.
