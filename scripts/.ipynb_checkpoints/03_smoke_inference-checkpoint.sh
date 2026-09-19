cd /root/autodl-tmp/searchrl_eval/SearchRL-Eval

cat > src/searchrl_eval/smoke_inference.py <<'PY'
"""B0: local Qwen2.5-3B-Instruct minimal inference."""

from __future__ import annotations

import json
import os
import re
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_answer(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff\s]", "", text)
    return re.sub(r"\s+", " ", text)


def main() -> None:
    model_path = Path(os.environ["MODEL_PATH"])
    output_file = Path(os.environ["OUTPUT_FILE"])
    question = os.environ["QUESTION"]
    expected_answer = os.environ.get("EXPECTED_ANSWER", "")
    run_id = os.environ["RUN_ID"]
    seed = int(os.environ.get("SEED", "42"))
    max_new_tokens = int(os.environ.get("MAX_NEW_TOKENS", "64"))

    output_file.parent.mkdir(parents=True, exist_ok=True)

    result = {
        "run_id": run_id,
        "experiment": "B0_no_retrieval_smoke",
        "status": "running",
        "started_at_utc": utc_now(),
        "model_path": str(model_path),
        "question": question,
        "expected_answer": expected_answer,
        "seed": seed,
        "generation_config": {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "num_beams": 1,
        },
    }

    try:
        if not model_path.is_dir():
            raise FileNotFoundError(f"Model directory not found: {model_path}")

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; start the A800 instance first.")

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        gpu_name = torch.cuda.get_device_name(0)
        gpu_total_gib = (
            torch.cuda.get_device_properties(0).total_memory / 1024**3
        )

        print(f"[INFO] GPU: {gpu_name}")
        print(f"[INFO] Model: {model_path}")
        print(f"[INFO] Question: {question}")
        print("[INFO] Loading tokenizer...")

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True,
            use_fast=True,
        )

        print("[INFO] Loading model in bfloat16...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        )
        model.to("cuda")
        model.eval()

        messages = [
            {
                "role": "system",
                "content": (
                    "Answer the question accurately and concisely. "
                    "Return only the answer."
                ),
            },
            {"role": "user", "content": question},
        ]

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {key: value.to("cuda") for key, value in inputs.items()}

        input_token_count = int(inputs["input_ids"].shape[-1])

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        print("[INFO] Starting deterministic generation...")
        started = time.perf_counter()

        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        torch.cuda.synchronize()
        latency_seconds = time.perf_counter() - started

        generated_ids = output_ids[0, input_token_count:]
        prediction = tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        ).strip()

        generated_token_count = int(generated_ids.shape[-1])
        peak_gpu_memory_gib = (
            torch.cuda.max_memory_allocated() / 1024**3
        )

        normalized_prediction = normalize_answer(prediction)
        normalized_expected = normalize_answer(expected_answer)

        result.update(
            {
                "status": "success",
                "prediction": prediction,
                "input_token_count": input_token_count,
                "generated_token_count": generated_token_count,
                "latency_seconds": round(latency_seconds, 4),
                "tokens_per_second": round(
                    generated_token_count / max(latency_seconds, 1e-9),
                    4,
                ),
                "peak_gpu_memory_gib": round(peak_gpu_memory_gib, 4),
                "gpu_name": gpu_name,
                "gpu_total_memory_gib": round(gpu_total_gib, 2),
                "torch_version": torch.__version__,
                "transformers_version": transformers.__version__,
                "smoke_pass": bool(prediction),
                "expected_answer_mentioned": (
                    bool(normalized_expected)
                    and normalized_expected in normalized_prediction
                ),
            }
        )

        print(f"[RESULT] Prediction: {prediction}")
        print(f"[RESULT] Latency: {latency_seconds:.4f} seconds")
        print(f"[RESULT] Peak GPU memory: {peak_gpu_memory_gib:.4f} GiB")

        if not prediction:
            raise RuntimeError("Model returned an empty prediction.")

    except Exception as exc:
        result.update(
            {
                "status": "failed",
                "smoke_pass": False,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        print(f"[ERROR] {type(exc).__name__}: {exc}")
        raise

    finally:
        result["finished_at_utc"] = utc_now()
        output_file.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"[INFO] Result saved to: {output_file}")


if __name__ == "__main__":
    main()
PY

cat > scripts/03_smoke_inference.sh <<'SH'
#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

require_command conda
require_command nvidia-smi
require_command timeout

QUESTION="${QUESTION:-${1:-Who wrote the novel Pride and Prejudice?}}"
EXPECTED_ANSWER="${EXPECTED_ANSWER:-Jane Austen}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
SEED="${SEED:-42}"
SMOKE_TIMEOUT_SECONDS="${SMOKE_TIMEOUT_SECONDS:-900}"

if [[ ! "${MAX_NEW_TOKENS}" =~ ^[1-9][0-9]*$ ]]; then
    die "MAX_NEW_TOKENS 必须是正整数"
fi

if [[ ! "${SEED}" =~ ^[0-9]+$ ]]; then
    die "SEED 必须是非负整数"
fi

if [[ ! "${SMOKE_TIMEOUT_SECONDS}" =~ ^[1-9][0-9]*$ ]]; then
    die "SMOKE_TIMEOUT_SECONDS 必须是正整数"
fi

if [[ ! -s "${MODEL_PATH}/config.json" ]]; then
    die "模型目录无效：${MODEL_PATH}"
fi

if ! gpu_info="$(nvidia-smi -L 2>&1)" || [[ -z "${gpu_info}" ]]; then
    die "没有检测到 GPU。请使用 A800 有卡模式开机。"
fi

if ! conda env list | awk '{print $1}' | grep -Fxq searchr1; then
    die "Conda 环境 searchr1 不存在，请先执行 scripts/01_create_envs.sh --yes"
fi

if ! conda run -n searchr1 python -c \
    'import torch, transformers; assert torch.cuda.is_available()' \
    >/dev/null 2>&1; then
    die "searchr1 环境不完整，或 PyTorch 无法使用 CUDA"
fi

RUN_ID="b0_smoke_$(date +%Y%m%d_%H%M%S)"
OUTPUT_FILE="${OUTPUT_DIR}/${RUN_ID}.json"
LOG_FILE="${LOG_DIR}/${RUN_ID}.log"

export RUN_ID
export OUTPUT_FILE
export QUESTION
export EXPECTED_ANSWER
export MAX_NEW_TOKENS
export SEED

# 强制离线，防止推理时意外访问 Hugging Face。
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
export PYTHONUNBUFFERED=1

exec > >(tee -a "${LOG_FILE}") 2>&1

printf '===== B0 minimal inference =====\n'
printf 'Run ID: %s\n' "${RUN_ID}"
printf 'GPU: %s\n' "${gpu_info}"
printf 'Model: %s\n' "${MODEL_PATH}"
printf 'Question: %s\n' "${QUESTION}"
printf 'Expected answer: %s\n' "${EXPECTED_ANSWER}"
printf 'Timeout: %s seconds\n' "${SMOKE_TIMEOUT_SECONDS}"
printf 'Output: %s\n' "${OUTPUT_FILE}"
printf 'Log: %s\n\n' "${LOG_FILE}"

set +e
timeout \
    --signal=TERM \
    --kill-after=10s \
    "${SMOKE_TIMEOUT_SECONDS}s" \
    conda run --no-capture-output -n searchr1 \
    python "${REPO_DIR}/src/searchrl_eval/smoke_inference.py"

exit_code=$?
set -e

if (( exit_code == 124 || exit_code == 137 )); then
    printf '[ERROR] 推理超过 %s 秒，已按硬超时停止。\n' \
        "${SMOKE_TIMEOUT_SECONDS}" >&2
    exit "${exit_code}"
fi

if (( exit_code != 0 )); then
    printf '[ERROR] 最小推理失败，退出码：%s\n' "${exit_code}" >&2
    printf '[ERROR] 查看日志：%s\n' "${LOG_FILE}" >&2
    exit "${exit_code}"
fi

if [[ ! -s "${OUTPUT_FILE}" ]]; then
    die "推理命令成功退出，但没有生成结果文件：${OUTPUT_FILE}"
fi

printf '\n===== Saved result =====\n'
cat "${OUTPUT_FILE}"
printf '\n\n[PASS] B0 最小推理完成。\n'
SH

chmod +x scripts/03_smoke_inference.sh

echo "===== Syntax check ====="
bash -n scripts/03_smoke_inference.sh

conda run -n searchr1 python -m py_compile \
    src/searchrl_eval/smoke_inference.py 2>/dev/null \
    || python -m py_compile src/searchrl_eval/smoke_inference.py

echo "03_smoke_inference.sh 已补全"