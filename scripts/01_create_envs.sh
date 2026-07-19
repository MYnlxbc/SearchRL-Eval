#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

CONFIRMED=0
REPAIR=0

for arg in "$@"; do
    case "${arg}" in
        --yes)
            CONFIRMED=1
            ;;
        --repair)
            REPAIR=1
            ;;
        *)
            die "未知参数：${arg}"
            ;;
    esac
done

if (( CONFIRMED == 0 )); then
    cat <<'USAGE'
本脚本会创建两个 Conda 环境并下载、编译依赖。

首次安装：
  bash scripts/01_create_envs.sh --yes

修复或更新已有环境：
  bash scripts/01_create_envs.sh --yes --repair

脚本不会自动重试失败的安装。
USAGE
    exit 2
fi

require_command conda
bash "${SCRIPT_DIR}/00_check_assets.sh"

env_exists() {
    conda env list | awk '{print $1}' | grep -Fxq "$1"
}

prepare_conda_env() {
    local name="$1"
    local spec="$2"

    if env_exists "${name}"; then
        if (( REPAIR == 1 )); then
            log "更新已有环境：${name}"
            conda env update \
                --name "${name}" \
                --file "${spec}" \
                --prune
        else
            log "环境已存在，跳过创建：${name}"
        fi
    else
        log "创建环境：${name}"
        conda env create --file "${spec}"
    fi
}

prepare_conda_env \
    searchr1 \
    "${REPO_DIR}/environment/searchr1.yml"

prepare_conda_env \
    retriever \
    "${REPO_DIR}/environment/retriever.yml"

if ! conda run -n searchr1 python -c \
    'import torch, vllm, verl' >/dev/null 2>&1; then

    log "安装 Search-R1 运行依赖"
    conda run -n searchr1 \
        python -m pip install --upgrade pip

    conda run -n searchr1 \
        python -m pip install "numpy<2"

    conda run -n searchr1 \
        python -m pip install \
        torch==2.4.0 \
        --index-url https://download.pytorch.org/whl/cu121

    conda run -n searchr1 \
        python -m pip install vllm==0.6.3

    conda run -n searchr1 \
        python -m pip install -e "${SEARCHR1_CODE_DIR}"

    env MAX_JOBS="${MAX_JOBS:-1}" \
        conda run -n searchr1 \
        python -m pip install \
        flash-attn \
        --no-build-isolation

    conda run -n searchr1 \
        python -m pip install wandb pyyaml requests
else
    log "Search-R1 核心依赖已存在，跳过安装"
fi

log "验证 Search-R1 环境"
conda run -n searchr1 python - <<'PY'
import torch
import vllm
import verl

print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("vLLM:", vllm.__version__)
print("verl:", verl.__file__)
PY

log "验证 Retriever 环境"
conda run -n retriever python - <<'PY'
import torch
import faiss
import datasets
import transformers
import fastapi
import uvicorn

print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("faiss:", getattr(faiss, "__version__", "unknown"))
print("datasets:", datasets.__version__)
print("transformers:", transformers.__version__)
PY

log "两个环境均验证完成"
