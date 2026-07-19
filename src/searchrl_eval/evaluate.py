"""Unified B0/B1/B2 evaluation entry."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests
import torch
import yaml
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from searchrl_eval.schema import EvalExample, EvalRecord, append_jsonl


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def to_plain(value: Any) -> Any:
    """Convert Arrow/Numpy containers to regular Python objects."""

    if hasattr(value, "tolist"):
        value = value.tolist()

    if isinstance(value, dict):
        return {str(k): to_plain(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [to_plain(v) for v in value]

    return value


def to_string_list(value: Any) -> List[str]:
    value = to_plain(value)

    if value is None:
        return []

    if isinstance(value, str):
        return [value]

    if isinstance(value, dict):
        if "target" in value:
            return to_string_list(value["target"])
        return [json.dumps(value, ensure_ascii=False)]

    if isinstance(value, list):
        answers = []
        for item in value:
            answers.extend(to_string_list(item))
        return [answer for answer in answers if answer]

    return [str(value)]


def extract_prompt_question(row: Dict[str, Any]) -> str:
    if row.get("question"):
        return str(row["question"]).strip()

    prompt = to_plain(row.get("prompt"))

    if isinstance(prompt, list):
        contents = []

        for message in prompt:
            if isinstance(message, dict) and message.get("content"):
                contents.append(str(message["content"]))

        if contents:
            content = contents[-1]

            if "Question:" in content:
                return content.rsplit("Question:", 1)[-1].strip()

            return content.strip()

    if isinstance(prompt, str):
        if "Question:" in prompt:
            return prompt.rsplit("Question:", 1)[-1].strip()
        return prompt.strip()

    raise ValueError("Cannot extract question from row")


def normalize_example(row: Dict[str, Any], row_index: int) -> EvalExample:
    row = to_plain(row)

    dataset = str(row.get("data_source") or row.get("dataset") or "unknown")

    extra_info = row.get("extra_info") or {}
    if not isinstance(extra_info, dict):
        extra_info = {}

    split = str(extra_info.get("split") or row.get("split") or "unknown")
    source_index = extra_info.get("index", row_index)

    explicit_id = (
        row.get("question_id")
        or row.get("_id")
        or row.get("id")
    )

    question_id = str(
        explicit_id
        if explicit_id is not None
        else f"{dataset}:{split}:{source_index}"
    )

    reward_model = row.get("reward_model") or {}
    ground_truth: Any = None

    if isinstance(reward_model, dict):
        ground_truth = reward_model.get("ground_truth")

    gold_answers = to_string_list(
        row.get("golden_answers")
        or row.get("answers")
        or ground_truth
    )

    try:
        normalized_index: Optional[int] = int(source_index)
    except (TypeError, ValueError):
        normalized_index = None

    return EvalExample(
        question_id=question_id,
        dataset=dataset,
        question=extract_prompt_question(row),
        gold_answers=gold_answers,
        source_split=split,
        source_index=normalized_index,
    )


def load_examples(
    input_path: Path,
    dataset_filter: Optional[str],
    start_index: int,
    limit: Optional[int],
) -> Iterable[EvalExample]:
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    suffix = input_path.suffix.lower()

    if suffix == ".parquet":
        dataset = load_dataset(
            "parquet",
            data_files=str(input_path),
            split="train",
        )
    elif suffix in {".json", ".jsonl"}:
        dataset = load_dataset(
            "json",
            data_files=str(input_path),
            split="train",
        )
    else:
        raise ValueError(f"Unsupported input type: {suffix}")

    yielded = 0

    for row_index in range(start_index, len(dataset)):
        example = normalize_example(dataset[row_index], row_index)

        if dataset_filter and example.dataset.lower() != dataset_filter.lower():
            continue

        yield example
        yielded += 1

        if limit is not None and yielded >= limit:
            break


class LocalGenerator:
    def __init__(self, model_path: Path, config: Dict[str, Any]) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")

        if not model_path.is_dir():
            raise FileNotFoundError(f"Model directory not found: {model_path}")

        self.model_path = model_path
        self.generation_config = config["generation"]

        attention = config["model"].get("attention", "sdpa")

        print(f"[INFO] Loading tokenizer: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True,
            use_fast=True,
        )

        print(f"[INFO] Loading model with attention={attention}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            attn_implementation=attention,
        )
        self.model.to("cuda")
        self.model.eval()

    def generate(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: Optional[int] = None,
    ) -> Tuple[str, int, int, float]:
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {key: value.to("cuda") for key, value in inputs.items()}

        prompt_tokens = int(inputs["input_ids"].shape[-1])
        token_limit = int(
            max_new_tokens
            or self.generation_config.get("max_new_tokens", 64)
        )

        torch.cuda.synchronize()
        started = time.perf_counter()

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=token_limit,
                do_sample=False,
                num_beams=1,
                use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        torch.cuda.synchronize()
        latency = time.perf_counter() - started

        generated_ids = output_ids[0, prompt_tokens:]
        prediction = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        ).strip()

        return (
            prediction,
            prompt_tokens,
            int(generated_ids.shape[-1]),
            latency,
        )


class RetrieverClient:
    def __init__(self, url: str, topk: int) -> None:
        self.url = url
        self.topk = topk

    def retrieve(
        self,
        query: str,
        round_index: int,
    ) -> Tuple[List[Dict[str, Any]], float]:
        started = time.perf_counter()

        response = requests.post(
            self.url,
            json={
                "queries": [query],
                "topk": self.topk,
                "return_scores": True,
            },
            timeout=120,
        )
        response.raise_for_status()

        latency = time.perf_counter() - started
        payload = response.json()
        result = payload.get("result")

        if not isinstance(result, list) or not result:
            raise ValueError("Retriever returned an invalid result")

        documents = []

        for rank, item in enumerate(result[0], start=1):
            if isinstance(item, dict) and "document" in item:
                document = item.get("document") or {}
                score = item.get("score")
            else:
                document = item if isinstance(item, dict) else {}
                score = None

            contents = str(document.get("contents") or "")
            title = str(document.get("title") or "")
            text = str(document.get("text") or "")

            if not text and contents:
                parts = contents.split("\n", 1)
                if not title:
                    title = parts[0].strip('"')
                text = parts[1] if len(parts) > 1 else contents

            documents.append(
                {
                    "round_index": round_index,
                    "rank": rank,
                    "query": query,
                    "document_id": (
                        str(document["id"])
                        if document.get("id") is not None
                        else None
                    ),
                    "title": title,
                    "text": text,
                    "score": (
                        float(score)
                        if score is not None
                        else None
                    ),
                }
            )

        return documents, latency


def build_context(documents: List[Dict[str, Any]]) -> str:
    sections = []

    for document in documents:
        text = document.get("text", "")[:1500]
        sections.append(
            f"[Document {document['round_index']}.{document['rank']}]\n"
            f"Title: {document.get('title', '')}\n"
            f"{text}"
        )

    return "\n\n".join(sections)


def answer_messages(
    question: str,
    documents: List[Dict[str, Any]],
) -> List[Dict[str, str]]:
    if not documents:
        user_content = question
    else:
        user_content = (
            f"Question: {question}\n\n"
            f"Retrieved evidence:\n{build_context(documents)}"
        )

    return [
        {
            "role": "system",
            "content": (
                "Answer the question accurately and concisely. "
                "Return only the final answer."
            ),
        },
        {"role": "user", "content": user_content},
    ]


def followup_messages(
    question: str,
    documents: List[Dict[str, Any]],
) -> List[Dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Generate one short search query that would retrieve missing "
                "information needed to answer the question. "
                "If no more information is needed, output NONE. "
                "Return only the query or NONE."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question: {question}\n\n"
                f"Current evidence:\n{build_context(documents)}"
            ),
        },
    ]


def clean_followup_query(text: str) -> str:
    text = text.replace("<search>", "").replace("</search>", "").strip()
    return text.splitlines()[0][:300] if text else ""


def run_example(
    example: EvalExample,
    baseline: str,
    config: Dict[str, Any],
    generator: LocalGenerator,
    retriever: Optional[RetrieverClient],
) -> Dict[str, Any]:
    documents: List[Dict[str, Any]] = []
    retrieval_rounds = 0
    retrieval_latency = 0.0
    generation_latency = 0.0
    prompt_tokens = 0
    generated_tokens = 0

    if baseline in {"B1", "B2"}:
        if retriever is None:
            raise RuntimeError("Retriever is required for B1/B2")

        first_documents, first_latency = retriever.retrieve(
            example.question,
            round_index=1,
        )
        documents.extend(first_documents)
        retrieval_latency += first_latency
        retrieval_rounds = 1

    if baseline == "B2":
        query_text, query_prompt_tokens, query_generated_tokens, query_latency = (
            generator.generate(
                followup_messages(example.question, documents),
                max_new_tokens=int(
                    config["generation"].get(
                        "query_max_new_tokens",
                        32,
                    )
                ),
            )
        )

        prompt_tokens += query_prompt_tokens
        generated_tokens += query_generated_tokens
        generation_latency += query_latency

        followup_query = clean_followup_query(query_text)

        if followup_query and followup_query.upper() != "NONE":
            second_documents, second_latency = retriever.retrieve(
                followup_query,
                round_index=2,
            )
            documents.extend(second_documents)
            retrieval_latency += second_latency
            retrieval_rounds = 2

    prediction, answer_prompt_tokens, answer_generated_tokens, answer_latency = (
        generator.generate(
            answer_messages(example.question, documents)
        )
    )

    prompt_tokens += answer_prompt_tokens
    generated_tokens += answer_generated_tokens
    generation_latency += answer_latency

    return {
        "prediction": prediction,
        "retrieved_documents": documents,
        "retrieval_rounds": retrieval_rounds,
        "retrieval_latency_seconds": retrieval_latency,
        "generation_latency_seconds": generation_latency,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified SearchRL-Eval runner"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dataset")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = yaml.safe_load(
        args.config.read_text(encoding="utf-8")
    )
    baseline = str(config["baseline"]).upper()

    if baseline not in {"B0", "B1", "B2"}:
        raise ValueError(f"Unsupported baseline: {baseline}")

    model_env = config["model"].get("path_env", "MODEL_PATH")
    model_path = Path(os.environ[model_env])

    retriever_url = os.environ.get("RETRIEVER_URL")
    retrieval_config = config["retrieval"]

    run_id = f"{baseline.lower()}_{utc_timestamp()}"
    output_path = args.output or Path("outputs") / f"{run_id}.jsonl"
    summary_path = output_path.with_suffix(".summary.json")

    seed = int(config["generation"].get("seed", 42))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    generator = LocalGenerator(model_path, config)

    retriever = None
    if retrieval_config.get("enabled"):
        if not retriever_url:
            raise RuntimeError("RETRIEVER_URL is not configured")

        retriever = RetrieverClient(
            retriever_url,
            int(retrieval_config.get("topk", 3)),
        )

    print(f"[INFO] Run ID: {run_id}")
    print(f"[INFO] Baseline: {baseline}")
    print(f"[INFO] Input: {args.input}")
    print(f"[INFO] Output: {output_path}")

    success_count = 0
    failure_count = 0
    total_started = time.perf_counter()

    for example in load_examples(
        args.input,
        args.dataset,
        args.start_index,
        args.limit,
    ):
        record = EvalRecord(
            run_id=run_id,
            baseline=baseline,
            dataset=example.dataset,
            question_id=example.question_id,
            question=example.question,
            gold_answers=example.gold_answers,
            model_path=str(model_path),
            retriever_url=(
                retriever_url
                if retrieval_config.get("enabled")
                else None
            ),
        )

        example_started = time.perf_counter()

        try:
            torch.cuda.reset_peak_memory_stats()

            result = run_example(
                example,
                baseline,
                config,
                generator,
                retriever,
            )

            for key, value in result.items():
                setattr(record, key, value)

            record.latency_seconds = (
                time.perf_counter() - example_started
            )
            record.peak_gpu_memory_gib = round(
                torch.cuda.max_memory_allocated() / 1024**3,
                4,
            )

            if not record.prediction:
                raise RuntimeError("Empty prediction")

            record.mark_success()
            success_count += 1
            print(
                f"[PASS] {record.question_id}: "
                f"{record.prediction}"
            )

        except Exception as exc:
            record.latency_seconds = (
                time.perf_counter() - example_started
            )
            record.mark_error(exc)
            failure_count += 1
            print(
                f"[FAIL] {record.question_id}: "
                f"{record.error_type}: {record.error_message}"
            )

        finally:
            append_jsonl(output_path, record)

    summary = {
        "run_id": run_id,
        "baseline": baseline,
        "input": str(args.input),
        "output": str(output_path),
        "success_count": success_count,
        "failure_count": failure_count,
        "elapsed_seconds": round(
            time.perf_counter() - total_started,
            4,
        ),
    }

    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if success_count == 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
