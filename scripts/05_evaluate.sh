#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

require_command conda
require_command timeout
require_command nvidia-smi

if ! nvidia-smi -L >/dev/null 2>&1; then
    die "没有检测到 GPU"
fi

if [[ "$#" -eq 0 ]]; then
    cat <<'USAGE'
用法：
  bash scripts/05_evaluate.sh \
    --config configs/b0_no_retrieval.yaml \
    --input /path/to/test.parquet \
    --limit 1
USAGE
    exit 2
fi

export PYTHONPATH="${REPO_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1

EVAL_TIMEOUT_SECONDS="${EVAL_TIMEOUT_SECONDS:-1800}"

timeout \
  --signal=TERM \
  --kill-after=10s \
  "${EVAL_TIMEOUT_SECONDS}s" \
  conda run --no-capture-output -n searchr1 \
  python -m searchrl_eval.evaluate "$@"
