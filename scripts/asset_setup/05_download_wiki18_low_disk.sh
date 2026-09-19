#!/usr/bin/env bash
set -Eeuo pipefail

export LC_ALL=C
umask 022

die() {
  echo "[FAIL] $*" >&2
  exit 1
}

file_size() {
  stat -c '%s' -- "$1"
}

require_command() {
  command -v "$1" >/dev/null 2>&1 ||
    die "Missing command: $1"
}

verify_sha256() {
  local expected="$1"
  local file="$2"
  local actual

  echo "[INFO] Verifying SHA-256: $file"
  actual="$(sha256sum -- "$file")"
  actual="${actual%% *}"

  [[ "$actual" == "$expected" ]] ||
    die "SHA-256 mismatch: $file expected=$expected actual=$actual"

  echo "[OK] SHA-256 verified: $file"
}

verify_prefix_sha256() {
  local expected="$1"
  local file="$2"
  local bytes="$3"
  local actual

  echo "[INFO] Verifying first $bytes bytes: $file"
  actual="$(head -c "$bytes" -- "$file" | sha256sum)"
  actual="${actual%% *}"

  [[ "$actual" == "$expected" ]] ||
    die "Prefix SHA-256 mismatch: $file expected=$expected actual=$actual"

  echo "[OK] Prefix SHA-256 verified: $file"
}

download_file() {
  local url="$1"
  local destination="$2"
  local expected_size="$3"
  local actual_size=0
  local resume_args=()

  mkdir -p "$(dirname "$destination")"

  if [[ -L "$destination" ]]; then
    die "Refusing to write through symbolic link: $destination"
  fi

  if [[ -f "$destination" ]]; then
    actual_size="$(file_size "$destination")"

    if (( actual_size == expected_size )); then
      echo "[SKIP] Download already complete: $destination"
      return 0
    fi

    if (( actual_size > expected_size )); then
      die "Existing file is larger than expected: $destination size=$actual_size expected=$expected_size"
    fi

    echo "[INFO] Resuming download: $destination"
    echo "[INFO] Existing bytes: $actual_size / $expected_size"
    resume_args=(--continue-at -)
  else
    echo "[INFO] Starting download: $destination"
  fi

  curl \
    --fail \
    --location \
    --retry 20 \
    --retry-delay 5 \
    --connect-timeout 30 \
    --speed-limit 1024 \
    --speed-time 120 \
    "${resume_args[@]}" \
    --output "$destination" \
    "$url"

  actual_size="$(file_size "$destination")"

  [[ "$actual_size" -eq "$expected_size" ]] ||
    die "Downloaded size mismatch: $destination size=$actual_size expected=$expected_size"

  echo "[OK] Download complete: $destination"
}

append_index_files() {
  local part_a="$1"
  local part_b="$2"
  local output="$3"
  local part_a_original_size="$4"
  local part_b_size="$5"
  local expected_total_size="$6"

  local current_a_size
  local current_b_size
  local already_appended
  local final_size

  [[ -f "$part_a" && ! -L "$part_a" ]] ||
    die "Missing or invalid part_aa: $part_a"

  [[ -f "$part_b" && ! -L "$part_b" ]] ||
    die "Missing or invalid part_ab: $part_b"

  current_a_size="$(file_size "$part_a")"
  current_b_size="$(file_size "$part_b")"

  [[ "$current_b_size" -eq "$part_b_size" ]] ||
    die "Incorrect part_ab size: $current_b_size expected=$part_b_size"

  if (( current_a_size < part_a_original_size )); then
    die "part_aa is incomplete: size=$current_a_size expected-at-least=$part_a_original_size"
  fi

  if (( current_a_size > expected_total_size )); then
    die "part_aa is larger than final index: size=$current_a_size expected=$expected_total_size"
  fi

  already_appended=$((current_a_size - part_a_original_size))

  if (( already_appended > part_b_size )); then
    die "Invalid append state: appended=$already_appended part_ab_size=$part_b_size"
  fi

  # 如果上一次合并中断，检查已经追加的内容是否与 part_ab 一致。
  if (( already_appended > 0 )); then
    echo "[INFO] Checking $already_appended previously appended bytes."

    if ! cmp -n "$already_appended" \
      <(tail -c "+$((part_a_original_size + 1))" -- "$part_a") \
      "$part_b"; then
      die "Previously appended data does not match part_ab"
    fi

    echo "[OK] Existing appended bytes are valid."
  fi

  if (( already_appended < part_b_size )); then
    echo "[INFO] Appending part_ab from byte offset $already_appended"
    tail -c "+$((already_appended + 1))" -- "$part_b" >> "$part_a"
  else
    echo "[SKIP] part_ab was already completely appended."
  fi

  final_size="$(file_size "$part_a")"

  [[ "$final_size" -eq "$expected_total_size" ]] ||
    die "Combined index size mismatch: size=$final_size expected=$expected_total_size"

  mv -- "$part_a" "$output"
  rm -f -- "$part_b"

  echo "[OK] Index created: $output"
}

decompress_corpus() {
  local compressed="$1"
  local output="$2"
  local expected_size="$3"
  local temporary="${output}.tmp"
  local actual_size

  [[ -f "$compressed" && ! -L "$compressed" ]] ||
    die "Missing or invalid corpus archive: $compressed"

  echo "[INFO] Testing gzip archive."
  gzip -t -- "$compressed"

  # gzip 数据无法从任意位置安全续解压，因此失败后重新解压。
  rm -f -- "$temporary"

  echo "[INFO] Decompressing corpus."
  gzip -dc -- "$compressed" > "$temporary"

  actual_size="$(file_size "$temporary")"

  [[ "$actual_size" -eq "$expected_size" ]] ||
    die "Corpus size mismatch: size=$actual_size expected=$expected_size"

  mv -- "$temporary" "$output"
  rm -f -- "$compressed"

  echo "[OK] Corpus created: $output"
}

main() {
  local root
  local endpoint
  local stamp
  local log
  local free_kb
  local free_bytes
  local managed_bytes=0
  local budget_bytes
  local budget_gib
  local min_budget_gib
  local size
  local file
  local index_done=0
  local corpus_done=0
  local current_a_size=0

  if [[ -n "${SEARCHRL_ROOT:-}" ]]; then
    root="$SEARCHRL_ROOT"
  elif [[ -d /root/autodl-tmp && -w /root/autodl-tmp ]]; then
    root=/root/autodl-tmp/searchrl_eval
  elif [[ -d /data && -w /data ]]; then
    root=/data/searchrl_eval
  else
    root="$HOME/searchrl_eval"
  fi

  mkdir -p "$root/logs" "$root/retrieval"

  stamp="$(date +%Y%m%d_%H%M%S)"
  log="$root/logs/download_wiki18_low_disk_${stamp}.log"

  exec > >(tee -a "$log") 2>&1
  trap 'rc=$?; echo "[FAIL] Script stopped at line $LINENO with exit code $rc"; exit "$rc"' ERR

  for file in curl sha256sum head tail cmp gzip stat df awk tee; do
    require_command "$file"
  done

  endpoint="${HF_ENDPOINT:-https://huggingface.co}"
  endpoint="${endpoint%/}"

  readonly PART_A_SIZE=42949672960
  readonly PART_B_SIZE=21609402413
  readonly INDEX_SIZE=64559075373
  readonly CORPUS_GZ_SIZE=5123307260
  readonly CORPUS_SIZE=14393579520

  readonly PART_A_SHA="a8a6a246951da4bbc8771a223283ef61963882a32864d9044ec00abb90fc3023"
  readonly PART_B_SHA="b6d9bc943626fe7cb44de4c849e9379e7f272ab216c0552acbcf2390cc033c11"
  readonly CORPUS_GZ_SHA="7abd929223399cd63c52b499f289bf4f9039be1e9f8c43e1cb3938305b2317db"

  readonly PART_DIR="$root/retrieval/wiki-18-e5-index"
  readonly PART_A="$PART_DIR/part_aa"
  readonly PART_B="$PART_DIR/part_ab"

  readonly CORPUS_DIR="$root/retrieval/wiki-18-corpus"
  readonly CORPUS_GZ="$CORPUS_DIR/wiki-18.jsonl.gz"

  readonly INDEX_OUT="$root/retrieval/e5_Flat.index"
  readonly CORPUS_OUT="$root/retrieval/wiki-18.jsonl"
  readonly CORPUS_TMP="${CORPUS_OUT}.tmp"

  readonly PART_A_URL="${endpoint}/datasets/PeterJinGo/wiki-18-e5-index/resolve/main/part_aa?download=true"
  readonly PART_B_URL="${endpoint}/datasets/PeterJinGo/wiki-18-e5-index/resolve/main/part_ab?download=true"
  readonly CORPUS_URL="${endpoint}/datasets/PeterJinGo/wiki-18-corpus/resolve/main/wiki-18.jsonl.gz?download=true"

  echo "[INFO] root=$root"
  echo "[INFO] endpoint=$endpoint"
  echo "[INFO] log=$log"

  # 检查已经完成的最终文件。
  if [[ -e "$INDEX_OUT" || -L "$INDEX_OUT" ]]; then
    [[ -f "$INDEX_OUT" && ! -L "$INDEX_OUT" ]] ||
      die "Invalid index output: $INDEX_OUT"

    size="$(file_size "$INDEX_OUT")"
    [[ "$size" -eq "$INDEX_SIZE" ]] ||
      die "Existing index has incorrect size: $size expected=$INDEX_SIZE"

    index_done=1
    rm -f -- "$PART_A" "$PART_B"
    echo "[SKIP] Existing valid index: $INDEX_OUT"
  fi

  if [[ -e "$CORPUS_OUT" || -L "$CORPUS_OUT" ]]; then
    [[ -f "$CORPUS_OUT" && ! -L "$CORPUS_OUT" ]] ||
      die "Invalid corpus output: $CORPUS_OUT"

    size="$(file_size "$CORPUS_OUT")"
    [[ "$size" -eq "$CORPUS_SIZE" ]] ||
      die "Existing corpus has incorrect size: $size expected=$CORPUS_SIZE"

    corpus_done=1
    rm -f -- "$CORPUS_GZ" "$CORPUS_TMP"
    echo "[SKIP] Existing valid corpus: $CORPUS_OUT"
  fi

  if (( index_done == 1 && corpus_done == 1 )); then
    ls -lh "$INDEX_OUT" "$CORPUS_OUT"
    echo "[DONE] Wiki-18 files already complete."
    exit 0
  fi

  # 可用空间加上本流程已有文件，代表本流程可管理的总空间。
  free_kb="$(df -Pk "$root" | awk 'NR==2 {print $4}')"
  free_bytes=$((free_kb * 1024))

  for file in \
    "$PART_A" \
    "$PART_B" \
    "$CORPUS_GZ" \
    "$INDEX_OUT" \
    "$CORPUS_OUT" \
    "$CORPUS_TMP"; do
    if [[ -f "$file" ]]; then
      size="$(file_size "$file")"
      managed_bytes=$((managed_bytes + size))
    fi
  done

  budget_bytes=$((free_bytes + managed_bytes))
  budget_gib=$((budget_bytes / 1024 / 1024 / 1024))
  min_budget_gib="${WIKI_MIN_BUDGET_GIB:-86}"

  echo "[INFO] Immediately free: $((free_bytes / 1024 / 1024 / 1024)) GiB"
  echo "[INFO] Existing managed Wiki files: $((managed_bytes / 1024 / 1024 / 1024)) GiB"
  echo "[INFO] Effective Wiki storage budget: ${budget_gib} GiB"

  if (( budget_gib < min_budget_gib )); then
    die "Need at least ${min_budget_gib} GiB effective budget; found ${budget_gib} GiB"
  fi

  # 阶段一：索引。语料尚未下载，降低合并时的峰值空间。
  if (( index_done == 0 )); then
    mkdir -p "$PART_DIR"

    if [[ -f "$PART_A" ]]; then
      current_a_size="$(file_size "$PART_A")"
    else
      current_a_size=0
    fi

    if (( current_a_size <= PART_A_SIZE )); then
      download_file "$PART_A_URL" "$PART_A" "$PART_A_SIZE"
    elif (( current_a_size <= INDEX_SIZE )); then
      echo "[INFO] Found partially combined part_aa: size=$current_a_size"
    else
      die "part_aa is larger than final index: $current_a_size"
    fi

    download_file "$PART_B_URL" "$PART_B" "$PART_B_SIZE"

    # part_aa 可能已经包含部分 part_ab，因此只校验原始前缀。
    verify_prefix_sha256 "$PART_A_SHA" "$PART_A" "$PART_A_SIZE"
    verify_sha256 "$PART_B_SHA" "$PART_B"

    append_index_files \
      "$PART_A" \
      "$PART_B" \
      "$INDEX_OUT" \
      "$PART_A_SIZE" \
      "$PART_B_SIZE" \
      "$INDEX_SIZE"

    index_done=1
  fi

  # 阶段二：语料。此时索引分片已经清理。
  if (( corpus_done == 0 )); then
    mkdir -p "$CORPUS_DIR"

    download_file "$CORPUS_URL" "$CORPUS_GZ" "$CORPUS_GZ_SIZE"
    verify_sha256 "$CORPUS_GZ_SHA" "$CORPUS_GZ"

    decompress_corpus "$CORPUS_GZ" "$CORPUS_OUT" "$CORPUS_SIZE"
    corpus_done=1
  fi

  echo "[INFO] Final files:"
  ls -lh "$INDEX_OUT" "$CORPUS_OUT"

  echo "[INFO] Remaining disk space:"
  df -h "$root"

  echo "[DONE] wiki_log=$log"
}

main "$@"