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

VENV="$ROOT/.download-venv"
if [[ ! -x "$VENV/bin/python" ]]; then
  echo "[FAIL] Download environment missing. Run 01_download_core.sh first."
  exit 2
fi

"$VENV/bin/python" "$SCRIPT_DIR/verify_assets.py" --root "$ROOT"

