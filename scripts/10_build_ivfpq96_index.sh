#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_INDEX="${SOURCE_INDEX:-${ROOT}/../retrieval/e5_Flat.index}"
OUTPUT_INDEX="${OUTPUT_INDEX:-${ROOT}/../retrieval/e5_IVFPQ96.index}"

echo "Source: ${SOURCE_INDEX}"
echo "Output: ${OUTPUT_INDEX}"
echo "This is CPU-only and may take hours. Do not run the full retriever or GRPO concurrently."

exec conda run --no-capture-output -n retriever \
  python "${ROOT}/tools/build_ivfpq_index.py" \
  --source "${SOURCE_INDEX}" \
  --output "${OUTPUT_INDEX}" \
  --nlist 4096 \
  --m 96 \
  --nbits 8 \
  --nprobe 32 \
  --train-size 200000 \
  --add-block-size 50000 \
  --threads 12 \
  --verify-queries 100 \
  --verify-topk 10
