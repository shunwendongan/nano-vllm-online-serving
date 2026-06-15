#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 1 ]]; then
  echo "Usage: bash scripts/run_colab_sweep.sh configs/colab/*.env" >&2
  exit 2
fi

for config in "$@"; do
  echo ""
  echo "==> Running ${config}"
  python scripts/run_colab_config.py --config "${config}"
done
