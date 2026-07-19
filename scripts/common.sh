#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PATHS_FILE="${PATHS_FILE:-${REPO_DIR}/configs/paths.env}"

log() {
    printf '[INFO] %s\n' "$*"
}

warn() {
    printf '[WARN] %s\n' "$*" >&2
}

die() {
    printf '[ERROR] %s\n' "$*" >&2
    exit 1
}

require_command() {
    command -v "$1" >/dev/null 2>&1 ||
        die "找不到命令：$1"
}

if [[ ! -f "${PATHS_FILE}" ]]; then
    die "缺少路径配置：${PATHS_FILE}。请复制 configs/paths.env.example。"
fi

# shellcheck disable=SC1090
source "${PATHS_FILE}"

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"
