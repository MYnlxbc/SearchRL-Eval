#!/usr/bin/env python3
"""Offline rescore QA outputs with correctly parsed aliases and Strict EM."""

from __future__ import annotations

import argparse
import ast
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from searchrl_eval.eval_utils import normalize_answer_for_metrics


def parse_aliases(value: Any) -> list[str]:
    if isinstance(value, str):
        # NumPy renders a string array as ``['alias one' 'alias two']``.
        # ``ast.literal_eval`` silently concatenates those adjacent literals,
        # so recover each quoted element before its fallback parser runs.
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
            return parse_aliases(ast.literal_eval(value))
        except (SyntaxError, ValueError):
            return [value]
    if isinstance(value, (list, tuple)):
        aliases: list[str] = []
        for item in value:
            aliases.extend(parse_aliases(item))
        return [alias for alias in aliases if alias]
    return [str(value)] if value is not None else []


def f1(left: str, right: str) -> float:
    left_tokens = normalize_answer_for_metrics(left).split()
    right_tokens = normalize_answer_for_metrics(right).split()
    if not left_tokens or not right_tokens:
        return float(left_tokens == right_tokens)
    common: dict[str, int] = {}
    for token in left_tokens:
        common[token] = common.get(token, 0) + 1
    overlap = 0
    for token in right_tokens:
        if common.get(token, 0):
            overlap += 1
            common[token] -= 1
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(left_tokens), overlap / len(right_tokens)
    return 2 * precision * recall / (precision + recall)


def score_record(record: dict[str, Any]) -> dict[str, Any]:
    prediction = str(record.get("prediction") or "")
    aliases = parse_aliases(record.get("gold_answers") or [])
    normalized_prediction = normalize_answer_for_metrics(prediction)
    normalized_aliases = [normalize_answer_for_metrics(alias) for alias in aliases]
    strict = any(normalized_prediction == alias for alias in normalized_aliases)
    return {
        "strict_em": strict,
        "token_f1": max((f1(prediction, alias) for alias in aliases), default=0.0),
        "gold_aliases": aliases,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    details: list[dict[str, Any]] = []
    for input_path in args.inputs:
        for line in input_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("status") != "success":
                continue
            score = score_record(record)
            baseline, dataset = str(record["baseline"]), str(record["dataset"])
            grouped[(baseline, dataset)].append(score)
            details.append({
                "baseline": baseline,
                "dataset": dataset,
                "question_id": record["question_id"],
                "strict_em": score["strict_em"],
                "token_f1": score["token_f1"],
                "gold_aliases": score["gold_aliases"],
            })

    metrics: dict[str, dict[str, Any]] = {}
    for (baseline, dataset), rows in sorted(grouped.items()):
        metrics.setdefault(dataset, {})[baseline] = {
            "n": len(rows),
            "strict_em_percent": round(100 * sum(row["strict_em"] for row in rows) / len(rows), 2),
            "token_f1_percent": round(100 * sum(row["token_f1"] for row in rows) / len(rows), 2),
        }
    overall: dict[str, Any] = {}
    for baseline in sorted({key[0] for key in grouped}):
        rows = [row for (name, _), values in grouped.items() if name == baseline for row in values]
        overall[baseline] = {
            "n": len(rows),
            "strict_em_percent": round(100 * sum(row["strict_em"] for row in rows) / len(rows), 2),
            "token_f1_percent": round(100 * sum(row["token_f1"] for row in rows) / len(rows), 2),
        }
    report = {
        "definition": "Strict EM: after QA normalization, the complete prediction must exactly equal at least one complete gold alias. NumPy-style serialized alias arrays are parsed as separate aliases.",
        "inputs": [str(path) for path in args.inputs],
        "metrics": metrics,
        "overall": overall,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    details_path = args.output.with_suffix(".details.jsonl")
    details_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in details), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"details={details_path}")


if __name__ == "__main__":
    main()
