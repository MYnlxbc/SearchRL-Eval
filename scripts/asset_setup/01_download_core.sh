#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${SEARCHRL_ROOT:-}" ]]; then
  ROOT="$SEARCHRL_ROOT"
elif [[ -d /root/autodl-tmp && -w /root/autodl-tmp ]]; then
  ROOT=/root/autodl-tmp/searchrl_eval
elif [[ -d /data && -w /data ]]; then
  ROOT=/data/searchrl_eval
else
  ROOT="$HOME/searchrl_eval"
fi

mkdir -p "$ROOT/logs" "$ROOT/cache/huggingface"
FREE_KB="$(df -Pk "$ROOT" | awk 'NR==2 {print $4}')"
FREE_GB="$((FREE_KB / 1024 / 1024))"
if (( FREE_GB < 40 )); then
  echo "[FAIL] Need at least 40GB free for core assets; found ${FREE_GB}GB."
  exit 2
fi

VENV="$ROOT/.download-venv"
if [[ ! -x "$VENV/bin/python" ]]; then
  python3 -m venv "$VENV"
fi
"$VENV/bin/python" -m pip install --upgrade pip
"$VENV/bin/python" -m pip install --upgrade huggingface_hub

export HF_HOME="$ROOT/cache/huggingface"
export HF_HUB_DISABLE_TELEMETRY=1

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$ROOT/logs/download_core_${STAMP}.log"

"$VENV/bin/python" "$SCRIPT_DIR/download_hf_assets.py" \
  --root "$ROOT" \
  --group core \
  --workers "${HF_DOWNLOAD_WORKERS:-4}" 2>&1 | tee "$LOG"

echo "[DONE] core_log=$LOG"

