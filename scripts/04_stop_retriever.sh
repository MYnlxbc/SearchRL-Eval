#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

PID_FILE="${LOG_DIR}/retriever.pid"

if [[ ! -f "${PID_FILE}" ]]; then
    log "没有检索服务 PID 文件，无需停止"
    exit 0
fi

pid="$(cat "${PID_FILE}")"

if ! kill -0 "${pid}" 2>/dev/null; then
    warn "PID=${pid} 已不存在，清理陈旧 PID 文件"
    rm -f "${PID_FILE}"
    exit 0
fi

printf '即将停止检索服务：\n'
ps -p "${pid}" -o pid,ppid,etime,stat,cmd

pgid="$(ps -o pgid= -p "${pid}" | tr -d ' ')"

if [[ -z "${pgid}" ]]; then
    die "无法获取进程组 ID"
fi

kill -TERM -- "-${pgid}"

for _ in $(seq 1 30); do
    if ! kill -0 "${pid}" 2>/dev/null; then
        rm -f "${PID_FILE}"
        printf '[PASS] 检索服务已停止\n'
        exit 0
    fi

    sleep 1
done

warn "发送 SIGTERM 后30秒进程仍存在"
warn "未自动使用 SIGKILL，请人工检查：ps -ef --forest"
exit 2
