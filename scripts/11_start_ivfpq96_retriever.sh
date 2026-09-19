#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export INDEX_PATH="${INDEX_PATH:-${ROOT}/../retrieval/e5_IVFPQ96.index}"

if [[ ! -f "${INDEX_PATH}" ]]; then
  echo "Compressed index is missing: ${INDEX_PATH}" >&2
  exit 1
fi

echo "Starting real retriever with compressed index: ${INDEX_PATH}"
exec bash "${ROOT}/scripts/02_start_retriever.sh"
