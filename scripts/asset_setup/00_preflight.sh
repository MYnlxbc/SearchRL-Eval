#!/usr/bin/env bash
set -euo pipefail

resolve_root() {
  if [[ -n "${SEARCHRL_ROOT:-}" ]]; then
    printf '%s\n' "$SEARCHRL_ROOT"
  elif [[ -d /root/autodl-tmp && -w /root/autodl-tmp ]]; then
    printf '%s\n' /root/autodl-tmp/searchrl_eval
  elif [[ -d /data && -w /data ]]; then
    printf '%s\n' /data/searchrl_eval
  else
    printf '%s\n' "$HOME/searchrl_eval"
  fi
}

ROOT="$(resolve_root)"
mkdir -p "$ROOT/logs" "$ROOT/manifests"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$ROOT/logs/preflight_${STAMP}.log"

exec > >(tee "$LOG") 2>&1

echo "[INFO] timestamp=$(date --iso-8601=seconds)"
echo "[INFO] hostname=$(hostname)"
echo "[INFO] root=$ROOT"
echo "[INFO] user=$(id -un)"
echo "[INFO] cpu_cores=$(nproc)"
echo "[INFO] kernel=$(uname -srmo)"

echo "[CHECK] GPU"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
else
  echo "[FAIL] nvidia-smi not found"
fi

echo "[CHECK] Memory"
free -h || true

echo "[CHECK] Disk"
df -h "$ROOT"

FREE_KB="$(df -Pk "$ROOT" | awk 'NR==2 {print $4}')"
FREE_GB="$((FREE_KB / 1024 / 1024))"
MEM_KB="$(awk '/MemTotal/ {print $2}' /proc/meminfo)"
MEM_GB="$((MEM_KB / 1024 / 1024))"

echo "[INFO] free_disk_gb=$FREE_GB"
echo "[INFO] total_memory_gb=$MEM_GB"

if (( FREE_GB < 40 )); then
  echo "[FAIL] Less than 40GB free. Do not start downloads."
  exit 2
elif (( FREE_GB < 180 )); then
  echo "[WARN] Core assets only. Do not run 02_download_wiki18.sh."
elif (( FREE_GB < 250 )); then
  echo "[WARN] Full Wiki-18 is allowed but disk margin is limited. Monitor df -h."
else
  echo "[PASS] Disk capacity is suitable for full download."
fi

if (( MEM_GB < 96 )); then
  echo "[WARN] RAM below 96GB. Full CPU Flat FAISS may be unsuitable."
else
  echo "[PASS] RAM meets the preferred CPU-retrieval threshold."
fi

echo "[CHECK] Tools"
for cmd in python3 git curl gzip sha256sum; do
  if command -v "$cmd" >/dev/null 2>&1; then
    echo "[PASS] $cmd=$(command -v "$cmd")"
  else
    echo "[MISSING] $cmd"
  fi
done

echo "[DONE] preflight_log=$LOG"

