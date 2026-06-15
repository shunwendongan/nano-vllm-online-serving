#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-}"
if [[ -n "${CONFIG_PATH}" ]]; then
  if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "Config not found: ${CONFIG_PATH}" >&2
    exit 1
  fi
  set -a
  # shellcheck disable=SC1090
  source "${CONFIG_PATH}"
  set +a
fi

echo "Python: $(python --version)"
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi was not found. Switch the Colab runtime to GPU before running model validation." >&2
  exit 1
fi
nvidia-smi

python -m pip install -U pip setuptools wheel packaging ninja
python -m pip install -e . --no-deps

if ! python - <<'PY'
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("torch") else 1)
PY
then
  if [[ "${INSTALL_TORCH:-auto}" == "0" || "${INSTALL_TORCH:-auto}" == "false" ]]; then
    echo "torch is missing and INSTALL_TORCH is disabled." >&2
    exit 1
  fi
  python -m pip install torch --index-url https://download.pytorch.org/whl/cu121
fi

python -m pip install -U \
  "transformers>=4.51.0" \
  "accelerate>=0.30.0" \
  "triton>=3.0.0" \
  "xxhash" \
  "fastapi>=0.110.0" \
  "uvicorn>=0.30.0" \
  "httpx" \
  "pytest" \
  "sentencepiece" \
  "protobuf" \
  "huggingface_hub[cli]"

if [[ -n "${PIP_EXTRA_PACKAGES:-}" ]]; then
  python -m pip install ${PIP_EXTRA_PACKAGES}
fi

INSTALL_FLASH="${INSTALL_FLASH_ATTN:-auto}"
if [[ "${INSTALL_FLASH}" == "auto" ]]; then
  if [[ "${MODEL_BACKEND:-native}" == "hf_auto" ]]; then
    INSTALL_FLASH="0"
  else
    INSTALL_FLASH="1"
  fi
fi

if [[ "${INSTALL_FLASH}" == "1" || "${INSTALL_FLASH}" == "true" ]]; then
  export MAX_JOBS="${MAX_JOBS:-2}"
  python -m pip install flash-attn --no-build-isolation
fi

if [[ -n "${HF_MODEL_ID:-}" && -n "${HF_LOCAL_DIR:-}" ]]; then
  mkdir -p "${HF_LOCAL_DIR}"
  huggingface-cli download "${HF_MODEL_ID}" --local-dir "${HF_LOCAL_DIR}"
fi

python -m nanovllm.check_runtime \
  --model "${MODEL:-${HF_LOCAL_DIR:-}}" \
  --model-backend "${MODEL_BACKEND:-native}" \
  --attention-backend "${ATTENTION_BACKEND:-flash_attn}" \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE:-1}" \
  --cuda-device-offset "${CUDA_DEVICE_OFFSET:-0}" \
  --distributed-backend "${DISTRIBUTED_BACKEND:-nccl}" || true

echo "Colab setup finished."
