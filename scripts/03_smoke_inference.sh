#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

require_command conda
require_command nvidia-smi
require_command timeout

QUESTION="${QUESTION:-${1:-Who wrote the novel Pride and Prejudice?}}"
EXPECTED_ANSWER="${EXPECTED_ANSWER:-Jane Austen}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
SEED="${SEED:-42}"
SMOKE_TIMEOUT_SECONDS="${SMOKE_TIMEOUT_SECONDS:-900}"

if [[ ! "${MAX_NEW_TOKENS}" =~ ^[1-9][0-9]*$ ]]; then
    die "MAX_NEW_TOKENS 必须是正整数"
fi

if [[ ! "${SEED}" =~ ^[0-9]+$ ]]; then
    die "SEED 必须是非负整数"
fi

if [[ ! "${SMOKE_TIMEOUT_SECONDS}" =~ ^[1-9][0-9]*$ ]]; then
    die "SMOKE_TIMEOUT_SECONDS 必须是正整数"
fi

if [[ ! -s "${MODEL_PATH}/config.json" ]]; then
    die "模型目录无效：${MODEL_PATH}"
fi

if ! gpu_info="$(nvidia-smi -L 2>&1)" || [[ -z "${gpu_info}" ]]; then
    die "没有检测到 GPU。请使用 A800 有卡模式开机。"
fi

if ! conda env list | awk '{print $1}' | grep -Fxq searchr1; then
    die "Conda 环境 searchr1 不存在，请先执行 scripts/01_create_envs.sh --yes"
fi

if ! conda run -n searchr1 python -c \
    'import torch, transformers; assert torch.cuda.is_available()' \
    >/dev/null 2>&1; then
    die "searchr1 环境不完整，或 PyTorch 无法使用 CUDA"
fi

RUN_ID="b0_smoke_$(date +%Y%m%d_%H%M%S)"
OUTPUT_FILE="${OUTPUT_DIR}/${RUN_ID}.json"
LOG_FILE="${LOG_DIR}/${RUN_ID}.log"

export RUN_ID
export OUTPUT_FILE
export QUESTION
export EXPECTED_ANSWER
export MAX_NEW_TOKENS
export SEED

# 强制离线，防止推理时意外访问 Hugging Face。
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1

exec > >(tee -a "${LOG_FILE}") 2>&1

printf '===== B0 minimal inference =====\n'
printf 'Run ID: %s\n' "${RUN_ID}"
printf 'GPU: %s\n' "${gpu_info}"
printf 'Model: %s\n' "${MODEL_PATH}"
printf 'Question: %s\n' "${QUESTION}"
printf 'Expected answer: %s\n' "${EXPECTED_ANSWER}"
printf 'Timeout: %s seconds\n' "${SMOKE_TIMEOUT_SECONDS}"
printf 'Output: %s\n' "${OUTPUT_FILE}"
printf 'Log: %s\n\n' "${LOG_FILE}"

set +e
timeout \
    --signal=TERM \
    --kill-after=10s \
    "${SMOKE_TIMEOUT_SECONDS}s" \
    conda run --no-capture-output -n searchr1 \
    python "${REPO_DIR}/src/searchrl_eval/smoke_inference.py"

exit_code=$?
set -e

if (( exit_code == 124 || exit_code == 137 )); then
    printf '[ERROR] 推理超过 %s 秒，已按硬超时停止。\n' \
        "${SMOKE_TIMEOUT_SECONDS}" >&2
    exit "${exit_code}"
fi

if (( exit_code != 0 )); then
    printf '[ERROR] 最小推理失败，退出码：%s\n' "${exit_code}" >&2
    printf '[ERROR] 查看日志：%s\n' "${LOG_FILE}" >&2
    exit "${exit_code}"
fi

if [[ ! -s "${OUTPUT_FILE}" ]]; then
    die "推理命令成功退出，但没有生成结果文件：${OUTPUT_FILE}"
fi

printf '\n===== Saved result =====\n'
cat "${OUTPUT_FILE}"
printf '\n\n[PASS] B0 最小推理完成。\n'
