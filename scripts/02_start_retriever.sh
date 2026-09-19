#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

require_command conda
require_command curl
require_command setsid
require_command nvidia-smi

# Some hosted environments export OMP_NUM_THREADS=0.  OpenMP treats that as
# invalid and silently falls back to a single thread, which needlessly slows
# CPU FAISS Flat search.  Keep the value configurable while guaranteeing a
# valid positive default.
FAISS_CPU_THREADS="${FAISS_CPU_THREADS:-12}"

if ! [[ "${OMP_NUM_THREADS:-}" =~ ^[1-9][0-9]*$ ]]; then
    export OMP_NUM_THREADS="${FAISS_CPU_THREADS}"
fi

export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${OMP_NUM_THREADS}}"

PID_FILE="${LOG_DIR}/retriever.pid"
LOG_POINTER="${LOG_DIR}/retriever.current_log"
BASE_URL="${RETRIEVER_URL%/retrieve}"
STARTUP_TIMEOUT_SECONDS="${RETRIEVER_STARTUP_TIMEOUT_SECONDS:-3600}"

if [[ "${RETRIEVER_PORT}" != "8000" ]]; then
    die "官方 retrieval_server.py 固定使用8000端口"
fi

if [[ "${FAISS_GPU}" != "0" ]]; then
    die "单卡实验必须使用 FAISS_GPU=0，避免60 GiB索引占满显存"
fi

if [[ ! -s "${INDEX_PATH}" ]]; then
    die "索引不存在：${INDEX_PATH}"
fi

if [[ ! -s "${CORPUS_PATH}" ]]; then
    die "语料不存在：${CORPUS_PATH}"
fi

if [[ ! -s "${RETRIEVER_MODEL_PATH}/config.json" ]]; then
    die "E5模型目录无效：${RETRIEVER_MODEL_PATH}"
fi

if [[ ! -s \
    "${REPO_DIR}/src/searchrl_eval/lazy_retrieval_server.py" ]]; then
    die "检索服务入口不存在"
fi

if [[ -f "${PID_FILE}" ]]; then
    old_pid="$(cat "${PID_FILE}")"

    if kill -0 "${old_pid}" 2>/dev/null; then
        log "检索服务似乎已运行，PID=${old_pid}"

        if curl -fsS "${BASE_URL}/openapi.json" >/dev/null 2>&1; then
            log "检索服务健康检查通过"
            exit 0
        fi

        die "PID仍存在但服务未就绪，请先检查日志"
    fi

    rm -f "${PID_FILE}"
fi

if curl -fsS "${BASE_URL}/openapi.json" >/dev/null 2>&1; then
    die "端口8000已经存在其他服务"
fi

log "验证 retriever 环境"

conda run --no-capture-output -n retriever python - <<'PY'
import torch
import faiss
import datasets
import transformers

print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))
print("faiss:", getattr(faiss, "__version__", "unknown"))
print("datasets:", datasets.__version__)
print("transformers:", transformers.__version__)

assert torch.cuda.is_available()
PY

printf '\n===== Resource status before startup =====\n'
free -h
df -h "${SEARCHRL_ROOT}"
nvidia-smi \
    --query-gpu=name,memory.total,memory.used \
    --format=csv,noheader

export CUDA_VISIBLE_DEVICES
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

# Arrow缓存必须放数据盘，避免占满系统盘。
export HF_HOME="${SEARCHRL_ROOT}/cache/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"

mkdir -p "${HF_DATASETS_CACHE}"

RUN_TAG="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/retriever_${RUN_TAG}.log"

command=(
    conda run
    --no-capture-output
    -n retriever
    python
    "${REPO_DIR}/src/searchrl_eval/lazy_retrieval_server.py"
    --index_path "${INDEX_PATH}"
    --corpus_path "${CORPUS_PATH}"
    --topk "${RETRIEVER_TOPK}"
    --retriever_name e5
    --retriever_model "${RETRIEVER_MODEL_PATH}"
)

log "启动检索服务"
log "FAISS索引：CPU"
log "E5编码器：GPU 0"
log "日志：${LOG_FILE}"

nohup setsid "${command[@]}" \
    >"${LOG_FILE}" 2>&1 </dev/null &

retriever_pid=$!

printf '%s\n' "${retriever_pid}" >"${PID_FILE}"
printf '%s\n' "${LOG_FILE}" >"${LOG_POINTER}"

log "PID=${retriever_pid}"
log "等待服务就绪，最长 ${STARTUP_TIMEOUT_SECONDS} 秒"

elapsed=0

while (( elapsed < STARTUP_TIMEOUT_SECONDS )); do
    if ! kill -0 "${retriever_pid}" 2>/dev/null; then
        printf '[ERROR] 检索服务进程提前结束\n' >&2
        tail -100 "${LOG_FILE}" >&2
        exit 1
    fi

    if curl -fsS "${BASE_URL}/openapi.json" >/dev/null 2>&1; then
        printf '\n[PASS] 检索服务已就绪\n'
        printf 'PID=%s\n' "${retriever_pid}"
        printf 'URL=%s\n' "${RETRIEVER_URL}"
        printf 'LOG=%s\n' "${LOG_FILE}"
        exit 0
    fi

    if (( elapsed % 60 == 0 )); then
        printf '\n[INFO] 已等待 %d 秒\n' "${elapsed}"
        tail -5 "${LOG_FILE}" || true
        df -h "${SEARCHRL_ROOT}" | tail -1
    fi

    sleep 10
    elapsed=$((elapsed + 10))
done

warn "等待超时，但检索进程仍保留运行"
warn "请继续查看日志：tail -f ${LOG_FILE}"
exit 2
