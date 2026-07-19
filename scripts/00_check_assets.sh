#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

failures=0

pass() {
    printf '[PASS] %s\n' "$*"
}

fail() {
    printf '[FAIL] %s\n' "$*" >&2
    failures=$((failures + 1))
}

check_dir() {
    local path="$1"
    local label="$2"

    if [[ -d "${path}" ]]; then
        pass "${label}: ${path}"
    else
        fail "${label}不存在: ${path}"
    fi
}

check_file() {
    local path="$1"
    local label="$2"

    if [[ -s "${path}" ]]; then
        pass "${label}: ${path} ($(du -h "${path}" | awk '{print $1}'))"
    else
        fail "${label}不存在或为空: ${path}"
    fi
}

check_min_size() {
    local path="$1"
    local minimum="$2"
    local label="$3"

    if [[ ! -f "${path}" ]]; then
        fail "${label}不存在: ${path}"
        return
    fi

    local actual
    actual="$(stat -c '%s' "${path}")"

    if (( actual >= minimum )); then
        pass "${label}大小正常: ${actual} bytes"
    else
        fail "${label}可能不完整: ${actual} bytes，小于 ${minimum}"
    fi
}

printf '===== SearchRL-Eval asset check =====\n'

check_dir "${SEARCHR1_CODE_DIR}" "Search-R1 源码"
check_file "${SEARCHR1_CODE_DIR}/pyproject.toml" "Search-R1 pyproject"
check_file \
    "${SEARCHR1_CODE_DIR}/search_r1/search/retrieval_server.py" \
    "检索服务入口"

check_dir "${MODEL_PATH}" "Qwen 模型目录"
check_file "${MODEL_PATH}/config.json" "Qwen config.json"

check_dir "${RETRIEVER_MODEL_PATH}" "E5 模型目录"
check_file "${RETRIEVER_MODEL_PATH}/config.json" "E5 config.json"

check_dir "${QA_DATA_DIR}" "NQ/HotpotQA 数据目录"

if compgen -G "${QA_DATA_DIR}/*.parquet" >/dev/null; then
    pass "发现 QA parquet 文件"
else
    fail "未发现 QA parquet 文件: ${QA_DATA_DIR}/*.parquet"
fi

# 官方完整 E5 Flat 索引约 64.56 GB。
check_min_size "${INDEX_PATH}" 64000000000 "Wiki-18 E5 Flat 索引"

# 当前解压语料约 14.39 GB。
check_min_size "${CORPUS_PATH}" 14000000000 "Wiki-18 JSONL 语料"

if command -v conda >/dev/null 2>&1; then
    pass "Conda: $(conda --version)"
else
    fail "未找到 Conda"
fi

if command -v nvidia-smi >/dev/null 2>&1; then
    if gpu_info="$(nvidia-smi -L 2>&1)" && [[ -n "${gpu_info}" ]]; then
        pass "GPU 可见"
        printf '%s\n' "${gpu_info}"
    else
        fail "nvidia-smi 存在，但没有检测到 GPU"
    fi
else
    fail "未找到 nvidia-smi"
fi

printf '\n===== Resource summary =====\n'
free -h || true
df -h "${SEARCHRL_ROOT}" || true

printf '\n===== Runtime policy =====\n'
printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES}"
printf 'FAISS_GPU=%s\n' "${FAISS_GPU}"
printf 'Retriever URL=http://%s:%s/retrieve\n' \
    "${RETRIEVER_HOST}" "${RETRIEVER_PORT}"

if (( failures > 0 )); then
    printf '\n[RESULT] 检查失败，共 %d 项。\n' "${failures}" >&2
    exit 1
fi

printf '\n[RESULT] 所有必要资产检查通过。\n'
