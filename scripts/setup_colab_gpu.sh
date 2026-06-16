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

PYTHON_BIN="${SETUP_PYTHON:-${PYTHON:-python}}"
PYTHON_VERSION="$("${PYTHON_BIN}" --version)"
echo "Python: ${PYTHON_VERSION}"
"${PYTHON_BIN}" - <<'PY'
import sys

if not ((3, 10) <= sys.version_info[:2] < (3, 13)):
    raise SystemExit(
        f"Python {sys.version.split()[0]} is unsupported; use Python >=3.10,<3.13 "
        "for torch/flash-attn/nano-vLLM GPU validation."
    )
PY

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi was not found. Switch the Colab runtime to GPU before running model validation." >&2
  exit 1
fi
nvidia-smi

install_cuda_host_compiler() {
  if command -v g++-12 >/dev/null 2>&1 || command -v g++-11 >/dev/null 2>&1; then
    return 0
  fi
  if command -v apt-get >/dev/null 2>&1 && [[ "$(id -u)" == "0" ]]; then
    echo "Installing g++-12 for CUDA extension builds."
    apt-get update
    apt-get install -y g++-12
  fi
}

if [[ "${ATTENTION_BACKEND:-flash_attn}" == "cuda_ext" || "${OP_BACKEND:-torch}" == "cuda_ext" ]]; then
  install_cuda_host_compiler
fi

"${PYTHON_BIN}" -m pip install -U pip setuptools wheel packaging ninja
"${PYTHON_BIN}" -m pip install -e . --no-deps

if ! "${PYTHON_BIN}" - <<'PY'
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("torch") else 1)
PY
then
  if [[ "${INSTALL_TORCH:-auto}" == "0" || "${INSTALL_TORCH:-auto}" == "false" ]]; then
    echo "torch is missing and INSTALL_TORCH is disabled." >&2
    exit 1
  fi
  "${PYTHON_BIN}" -m pip install torch --index-url "${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
fi

"${PYTHON_BIN}" -m pip install -U \
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
  "${PYTHON_BIN}" -m pip install ${PIP_EXTRA_PACKAGES}
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
  if ! "${PYTHON_BIN}" -m pip install "${FLASH_ATTN_PACKAGE:-flash-attn}" --no-build-isolation; then
    echo "flash-attn installation failed. Check CUDA toolkit, torch CUDA version, Python version, and MAX_JOBS." >&2
    exit 1
  fi
fi

download_hf_model() {
  local model_id="$1"
  local local_dir="$2"
  if command -v hf >/dev/null 2>&1; then
    if hf download "${model_id}" --local-dir "${local_dir}"; then
      return 0
    fi
    echo "hf download failed; retrying with huggingface_hub.snapshot_download." >&2
  fi
  "${PYTHON_BIN}" - "${model_id}" "${local_dir}" <<'PY'
import sys
from huggingface_hub import snapshot_download

snapshot_download(repo_id=sys.argv[1], local_dir=sys.argv[2])
PY
}

if [[ -n "${HF_MODEL_ID:-}" ]]; then
  if [[ -z "${HF_LOCAL_DIR:-}" ]]; then
    HF_SAFE_NAME="${HF_MODEL_ID//\//__}"
    HF_LOCAL_DIR="${MODEL_CACHE_DIR:-${PWD}/.cache/hf_models}/${HF_SAFE_NAME}"
    export HF_LOCAL_DIR
    if [[ -z "${MODEL:-}" && "${MODEL_BACKEND:-native}" == "native" ]]; then
      MODEL="${HF_LOCAL_DIR}"
      export MODEL
    fi
  fi
  mkdir -p "${HF_LOCAL_DIR}"
  if [[ "${SKIP_MODEL_DOWNLOAD:-0}" == "1" || "${SKIP_MODEL_DOWNLOAD:-0}" == "true" ]]; then
    echo "Skipping HuggingFace model download because SKIP_MODEL_DOWNLOAD=${SKIP_MODEL_DOWNLOAD}."
  else
    download_hf_model "${HF_MODEL_ID}" "${HF_LOCAL_DIR}"
  fi
fi

if ! "${PYTHON_BIN}" -m nanovllm.check_runtime \
  --model "${MODEL:-${HF_LOCAL_DIR:-}}" \
  --model-backend "${MODEL_BACKEND:-native}" \
  --attention-backend "${ATTENTION_BACKEND:-flash_attn}" \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE:-1}" \
  --cuda-device-offset "${CUDA_DEVICE_OFFSET:-0}" \
  --distributed-backend "${DISTRIBUTED_BACKEND:-nccl}"; then
  if [[ "${ALLOW_RUNTIME_CHECK_FAILURE:-0}" == "1" || "${ALLOW_RUNTIME_CHECK_FAILURE:-0}" == "true" ]]; then
    echo "Runtime check failed but ALLOW_RUNTIME_CHECK_FAILURE=${ALLOW_RUNTIME_CHECK_FAILURE}; continuing." >&2
  else
    echo "Runtime check failed. Fix the environment before running validate_online_gpu.py." >&2
    exit 1
  fi
fi

echo "Colab setup finished."
