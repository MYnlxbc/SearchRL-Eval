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
