#!/usr/bin/env bash
set -Eeuo pipefail

# Record device-wide GPU memory while a training PID is alive.  Device-wide
# usage deliberately includes the retriever, vLLM and all Ray workers.

if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "Usage: $0 TRAIN_PID OUTPUT_LOG [INTERVAL_SECONDS]" >&2
    exit 2
fi

train_pid="$1"
output_log="$2"
interval_seconds="${3:-5}"

if ! [[ "$train_pid" =~ ^[1-9][0-9]*$ ]]; then
    echo "TRAIN_PID must be a positive integer" >&2
    exit 2
fi

if ! [[ "$interval_seconds" =~ ^[1-9][0-9]*$ ]]; then
    echo "INTERVAL_SECONDS must be a positive integer" >&2
    exit 2
fi

mkdir -p "$(dirname "$output_log")"
peak_mib=0
peak_timestamp=""

while kill -0 "$train_pid" 2>/dev/null; do
    sample="$(nvidia-smi --query-gpu=timestamp,memory.used,memory.total --format=csv,noheader,nounits)"
    timestamp="$(awk -F ', ' '{print $1}' <<<"$sample")"
    used_mib="$(awk -F ', ' '{print $2}' <<<"$sample")"
    total_mib="$(awk -F ', ' '{print $3}' <<<"$sample")"

    if (( used_mib > peak_mib )); then
        peak_mib="$used_mib"
        peak_timestamp="$timestamp"
    fi

    printf '%s used_mib=%s total_mib=%s peak_mib=%s peak_timestamp=%s\n' \
        "$timestamp" "$used_mib" "$total_mib" "$peak_mib" "$peak_timestamp" \
        >>"$output_log"
    sleep "$interval_seconds"
done

printf 'FINAL peak_mib=%s peak_gib=%.2f peak_timestamp=%s monitored_pid=%s\n' \
    "$peak_mib" "$(awk -v mib="$peak_mib" 'BEGIN { print mib / 1024 }')" \
    "$peak_timestamp" "$train_pid" >>"$output_log"
