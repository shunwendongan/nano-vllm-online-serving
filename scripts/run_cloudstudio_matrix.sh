#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-a10}"
SKIP_SETUP="${SKIP_SETUP:-0}"

summarize_reports() {
  python scripts/summarize_benchmarks.py --root reports/cloudstudio || true
}

trap 'status=$?; summarize_reports; exit ${status}' EXIT

case "${MODE}" in
  a10|A10)
    RUN_A100=0
    ;;
  a100|A100)
    RUN_A100=1
    ;;
  *)
    echo "Usage: bash scripts/run_cloudstudio_matrix.sh [a10|a100]" >&2
    exit 2
    ;;
esac

BASELINE="configs/cloudstudio/qwen3_native_flash_attn_baseline.env"

if [[ "${SKIP_SETUP}" != "1" && "${SKIP_SETUP}" != "true" ]]; then
  echo "==> CloudStudio setup"
  bash scripts/setup_colab_gpu.sh "${BASELINE}"
fi

echo "==> CloudStudio A10/native baseline and scheduler sweep"
bash scripts/run_colab_sweep.sh \
  "${BASELINE}" \
  configs/cloudstudio/qwen3_native_decode_first.env \
  configs/cloudstudio/qwen3_native_prefill_first.env \
  configs/cloudstudio/qwen3_native_cache_aware_lpm.env

echo "==> Optional CUDA extension decode experiment"
python scripts/run_colab_config.py --config configs/cloudstudio/qwen3_native_cuda_ext_decode.env

if [[ "${RUN_A100}" != "1" ]]; then
  echo "==> A10 optimized high-concurrency probe"
  python scripts/run_colab_config.py --config configs/cloudstudio/qwen3_native_a10_prefill_first_c64_r128.env
fi

if [[ "${RUN_A100}" == "1" ]]; then
  echo "==> A100 high-concurrency and long-context stress runs"
  python scripts/run_colab_config.py --config configs/cloudstudio/qwen3_native_a100_high_concurrency.env
  python scripts/run_colab_config.py --config configs/cloudstudio/qwen3_native_a100_long_context.env
fi

echo "CloudStudio matrix finished. Reports are under reports/cloudstudio/."
