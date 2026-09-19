#!/usr/bin/env python3
"""Evaluate one GRPO actor with the same tag-driven adaptive RAG protocol."""

from __future__ import annotations

import argparse
import ast
import json
import re
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from searchrl_eval.eval_utils import normalize_answer_for_metrics
from searchrl_eval.schema import EvalRecord, append_jsonl


SYSTEM_PROMPT = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
INVALID_OBSERVATION = (
    "\nMy previous action is invalid. If I want to search, I should put the query "
    "between <search> and </search>. If I want to give the final answer, I should "
    "put the answer between <answer> and </answer>. Let me try again.\n"
)
ACTION_PATTERN = re.compile(r"<(search|answer)>(.*?)</\1>", re.DOTALL)


def now_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def parse_list(value: Any) -> list[str]:
    """Decode manifest gold fields without treating serialized aliases as one answer."""
    if isinstance(value, str):
        stripped = value.strip()
        quoted_pattern = r"'(?:\\.|[^'\\])*'|\"(?:\\.|[^\"\\])*\""
        quoted_items = re.findall(quoted_pattern, stripped, flags=re.DOTALL)
        remainder = re.sub(quoted_pattern, "", stripped, flags=re.DOTALL)
        if (
            len(quoted_items) > 1
            and stripped.startswith("[")
            and stripped.endswith("]")
            and not remainder.strip("[] \t\r\n,")
        ):
            return [ast.literal_eval(item) for item in quoted_items]
        try:
            return parse_list(ast.literal_eval(value))
        except (ValueError, SyntaxError):
            return [value]
    if isinstance(value, (list, tuple)):
        answers: list[str] = []
        for item in value:
            answers.extend(parse_list(item))
        return [answer for answer in answers if answer]
    return [str(value)] if value is not None else []


def prompt_from_row(row: dict[str, Any]) -> str:
    raw = row.get("prompt")
    if isinstance(raw, str):
        try:
            parsed = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            parsed = None
        if isinstance(parsed, list):
            for message in reversed(parsed):
                if isinstance(message, dict) and message.get("content"):
                    return str(message["content"])
    question = str(row.get("question") or "").strip()
    if not question:
        raise ValueError("Manifest row has no prompt/question")
    return (
        "Answer the given question. You must conduct reasoning inside <think> and </think> "
        "first every time you get new information. After reasoning, if you find you lack some "
        "knowledge, you can call a search engine by <search> query </search> and it will return "
        "the top searched results between <information> and </information>. You can search as "
        "many times as your want. If you find no further external knowledge needed, you can directly "
        "provide the answer inside <answer> and </answer>, without detailed illustrations. "
        f"For example, <answer> Beijing </answer>. Question: {question}"
    )


def action_from_text(text: str) -> tuple[str | None, str, str]:
    """Return (action, content, response retained by the training environment)."""
    match = ACTION_PATTERN.search(text)
    if not match:
        return None, "", text
    action, content = match.group(1), match.group(2).strip()
    end = match.end()
    return action, content, text[:end]


def token_f1(prediction: str, gold: str) -> float:
    prediction_tokens = normalize_answer_for_metrics(prediction).split()
    gold_tokens = normalize_answer_for_metrics(gold).split()
    if not prediction_tokens or not gold_tokens:
        return float(prediction_tokens == gold_tokens)
    common = Counter(prediction_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    if not overlap:
        return 0.0
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


class AdaptiveEvaluator:
    def __init__(self, model_path: Path, retriever_url: str, topk: int) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, local_files_only=True, trust_remote_code=True, use_fast=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        ).to("cuda").eval()
        self.retriever_url = retriever_url
        self.topk = topk

    def generate(self, input_ids: torch.Tensor) -> tuple[str, torch.Tensor, float]:
        input_ids = input_ids[:, -1024:].to("cuda")
        attention_mask = torch.ones_like(input_ids)
        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            output = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=128,
                do_sample=False,
                num_beams=1,
                use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        torch.cuda.synchronize()
        generated = output[:, input_ids.shape[1] :]
        return self.tokenizer.decode(generated[0], skip_special_tokens=True), generated.cpu(), time.perf_counter() - started

    def search(self, query: str, round_index: int) -> tuple[str, list[dict[str, Any]], float]:
        started = time.perf_counter()
        response = requests.post(
            self.retriever_url,
            json={"queries": [query], "topk": self.topk, "return_scores": True},
            timeout=120,
        )
        response.raise_for_status()
        result = response.json()["result"][0]
        docs: list[dict[str, Any]] = []
        text_parts: list[str] = []
        for rank, item in enumerate(result, start=1):
            document = item.get("document", {})
            contents = str(document.get("contents") or "")
            title, _, text = contents.partition("\n")
            docs.append(
                {
                    "round_index": round_index,
                    "rank": rank,
                    "query": query,
                    "document_id": str(document.get("id")) if document.get("id") is not None else None,
                    "title": title.strip('"'),
                    "text": text or contents,
                    "score": float(item["score"]) if item.get("score") is not None else None,
                }
            )
            text_parts.append(f"Doc {rank}(Title: {title}) {text}")
        return "\n".join(text_parts), docs, time.perf_counter() - started

    def evaluate(self, row: dict[str, Any]) -> dict[str, Any]:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt_from_row(row)}]
        initial = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        state = self.tokenizer(initial, add_special_tokens=False, return_tensors="pt")["input_ids"]
        docs: list[dict[str, Any]] = []
        trajectory: list[str] = []
        retrieval_latency = generation_latency = 0.0
        prompt_tokens = generated_tokens = 0
        valid_actions = valid_searches = 0
        searches = 0
        answer = ""
        stop_reason = "max_turns_no_answer"

        # Two environment actions plus the manager's one final answer attempt.
        for action_index in range(3):
            prompt_tokens += int(state.shape[1])
            raw, generated, latency = self.generate(state)
            generation_latency += latency
            generated_tokens += int(generated.shape[1])
            action, content, retained = action_from_text(raw)
            trajectory.append(retained)
            retained_ids = self.tokenizer(retained, add_special_tokens=False, return_tensors="pt")["input_ids"]
            state = torch.cat([state, retained_ids], dim=1)[:, -1024:]

            if action == "answer":
                valid_actions += 1
                answer = content
                stop_reason = "model_answer"
                break
            if action == "search" and searches < 2 and action_index < 2:
                valid_actions += 1
                valid_searches += 1
                searches += 1
                info, found_docs, latency = self.search(content, searches)
                retrieval_latency += latency
                docs.extend(found_docs)
                observation = f"\n\n<information>{info.strip()}</information>\n\n"
                obs_ids = self.tokenizer(observation, add_special_tokens=False, return_tensors="pt")["input_ids"]
                state = torch.cat([state, obs_ids], dim=1)[:, -1024:]
                continue
            if action == "search":
                stop_reason = "max_searches_no_answer"
                break
            if action_index < 2:
                obs_ids = self.tokenizer(INVALID_OBSERVATION, add_special_tokens=False, return_tensors="pt")["input_ids"]
                state = torch.cat([state, obs_ids], dim=1)[:, -1024:]
                continue
            stop_reason = "invalid_final_action"

        return {
            "prediction": answer,
            "retrieved_documents": docs,
            "retrieval_rounds": searches,
            "retrieval_stop_reason": stop_reason,
            "retrieval_latency_seconds": retrieval_latency,
            "generation_latency_seconds": generation_latency,
            "prompt_tokens": prompt_tokens,
            "generated_tokens": generated_tokens,
            "valid_actions": valid_actions,
            "valid_searches": valid_searches,
            "raw_trajectory": trajectory,
        }


def summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["dataset"]].append(row)
    report: dict[str, Any] = {}
    for dataset, values in grouped.items():
        success = [row for row in values if row["status"] == "success"]
        report[dataset] = {
            "n": len(values),
            "success": len(success),
            "strict_em_percent": round(100 * sum(row.get("strict_em", row.get("extracted_em", False)) for row in success) / max(len(success), 1), 2),
            "token_f1_percent": round(100 * sum(row.get("token_f1", 0.0) for row in success) / max(len(success), 1), 2),
            "avg_retrieval_rounds": round(sum(row.get("retrieval_rounds", 0) for row in success) / max(len(success), 1), 3),
            "avg_latency_seconds": round(sum(row.get("latency_seconds", 0.0) for row in success) / max(len(success), 1), 3),
            "invalid_or_no_answer": sum(row.get("retrieval_stop_reason") in {"invalid_final_action", "max_turns_no_answer"} for row in success),
        }
    all_success = [row for row in rows if row["status"] == "success"]
    report["overall"] = {
        "n": len(rows),
        "success": len(all_success),
        "strict_em_percent": round(100 * sum(row.get("strict_em", row.get("extracted_em", False)) for row in all_success) / max(len(all_success), 1), 2),
        "token_f1_percent": round(100 * sum(row.get("token_f1", 0.0) for row in all_success) / max(len(all_success), 1), 2),
        "avg_retrieval_rounds": round(sum(row.get("retrieval_rounds", 0) for row in all_success) / max(len(all_success), 1), 3),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--retriever-url", default="http://127.0.0.1:8000/retrieve")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--datasets",
        help="Comma-separated data_source values to evaluate (for example: nq,hotpotqa).",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    existing_ids: set[str] = set()
    if args.resume and args.output.exists():
        for line in args.output.read_text(encoding="utf-8").splitlines():
            if line.strip():
                existing_ids.add(json.loads(line)["question_id"])

    rows = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.datasets:
        requested = {name.strip() for name in args.datasets.split(",") if name.strip()}
        available = {str(row.get("data_source") or "unknown") for row in rows}
        unknown = requested - available
        if unknown:
            raise ValueError(f"Requested datasets are absent from manifest: {sorted(unknown)}")
        rows = [row for row in rows if str(row.get("data_source") or "unknown") in requested]
    if args.limit is not None:
        rows = rows[: args.limit]
    evaluator = AdaptiveEvaluator(args.checkpoint, args.retriever_url, args.topk)
    run_id = f"grpo500_adaptive_{now_id()}"
    for index, row in enumerate(rows, start=1):
        question_id = str(row["id"])
        if question_id in existing_ids:
            continue
        golds = parse_list(row.get("golden_answers") or row.get("reward_model", {}).get("ground_truth", {}).get("target"))
        record = EvalRecord(
            run_id=run_id,
            baseline="GRPO_ADAPTIVE",
            dataset=str(row.get("data_source") or "unknown"),
            question_id=question_id,
            question=str(row.get("question") or ""),
            gold_answers=golds,
            normalized_gold_answers=[normalize_answer_for_metrics(gold) for gold in golds],
            model_path=str(args.checkpoint),
            retriever_url=args.retriever_url,
            timeout_seconds_per_example=600.0,
        )
        started = time.perf_counter()
        try:
            torch.cuda.reset_peak_memory_stats()
            result = evaluator.evaluate(row)
            for key in ("prediction", "retrieved_documents", "retrieval_rounds", "retrieval_stop_reason", "retrieval_latency_seconds", "generation_latency_seconds", "prompt_tokens", "generated_tokens"):
                setattr(record, key, result[key])
            record.normalized_prediction = normalize_answer_for_metrics(record.prediction)
            record.latency_seconds = time.perf_counter() - started
            record.peak_gpu_memory_gib = round(torch.cuda.max_memory_allocated() / 1024**3, 4)
            record.mark_success()
            out = record.to_dict()
            out.update({
                "valid_actions": result["valid_actions"],
                "valid_searches": result["valid_searches"],
                "raw_trajectory": result["raw_trajectory"],
                "strict_em": any(record.normalized_prediction == gold for gold in record.normalized_gold_answers),
                "token_f1": max((token_f1(record.prediction, gold) for gold in golds), default=0.0),
            })
            print(f"[PASS] {index}/{len(rows)} {record.dataset}/{question_id} rounds={record.retrieval_rounds} answer={record.prediction[:80]!r}", flush=True)
        except Exception as exc:
            record.latency_seconds = time.perf_counter() - started
            record.mark_error(exc)
            out = record.to_dict()
            print(f"[FAIL] {index}/{len(rows)} {record.dataset}/{question_id} {type(exc).__name__}: {exc}", flush=True)
        append_jsonl(args.output, type("Record", (), {"to_json": lambda self: json.dumps(out, ensure_ascii=False)})())

    written = [json.loads(line) for line in args.output.read_text(encoding="utf-8").splitlines() if line.strip()]
    report = summary(written)
    args.output.with_suffix(".summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
