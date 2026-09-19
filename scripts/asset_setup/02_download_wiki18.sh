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

mkdir -p "$ROOT/logs" "$ROOT/retrieval"
FREE_KB="$(df -Pk "$ROOT" | awk 'NR==2 {print $4}')"
FREE_GB="$((FREE_KB / 1024 / 1024))"
if (( FREE_GB < 180 )); then
  echo "[FAIL] Need at least 180GB free before Wiki-18 download; found ${FREE_GB}GB."
  exit 2
fi

VENV="$ROOT/.download-venv"
if [[ ! -x "$VENV/bin/python" ]]; then
  echo "[FAIL] Download environment missing. Run 01_download_core.sh first."
  exit 2
fi

export HF_HOME="$ROOT/cache/huggingface"
export HF_HUB_DISABLE_TELEMETRY=1
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$ROOT/logs/download_wiki18_${STAMP}.log"

exec > >(tee "$LOG") 2>&1

"$VENV/bin/python" "$SCRIPT_DIR/download_hf_assets.py" \
  --root "$ROOT" \
  --group wiki \
  --workers "${HF_DOWNLOAD_WORKERS:-4}"

PART_DIR="$ROOT/retrieval/wiki-18-e5-index"
CORPUS_GZ="$ROOT/retrieval/wiki-18-corpus/wiki-18.jsonl.gz"
INDEX_OUT="$ROOT/retrieval/e5_Flat.index"
CORPUS_OUT="$ROOT/retrieval/wiki-18.jsonl"

for file in "$PART_DIR/part_aa" "$PART_DIR/part_ab" "$CORPUS_GZ"; do
  if [[ ! -s "$file" ]]; then
    echo "[FAIL] Missing or empty: $file"
    exit 3
  fi
done

if [[ ! -s "$INDEX_OUT" ]]; then
  echo "[INFO] Combining index parts. This may take time and extra disk space."
  cat "$PART_DIR/part_aa" "$PART_DIR/part_ab" > "${INDEX_OUT}.tmp"
  mv "${INDEX_OUT}.tmp" "$INDEX_OUT"
else
  echo "[SKIP] Existing index: $INDEX_OUT"
fi

if [[ ! -s "$CORPUS_OUT" ]]; then
  echo "[INFO] Decompressing corpus."
  gzip -dc "$CORPUS_GZ" > "${CORPUS_OUT}.tmp"
  mv "${CORPUS_OUT}.tmp" "$CORPUS_OUT"
else
  echo "[SKIP] Existing corpus: $CORPUS_OUT"
fi

ls -lh "$INDEX_OUT" "$CORPUS_OUT" "$CORPUS_GZ"
df -h "$ROOT"
echo "[DONE] wiki_log=$LOG"

