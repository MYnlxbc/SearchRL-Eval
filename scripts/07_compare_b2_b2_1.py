#!/usr/bin/env python3
"""Compare paired B2 and B2.1 SearchRL-Eval smoke runs.

The script intentionally uses only Python's standard library. It discovers the
newest valid 50-row run for each (baseline, dataset) pair, checks that B2 and
B2.1 contain identical question IDs, and writes JSON/CSV/Markdown reports.

Strict HotpotQA supporting-fact recall is not computed because the converted
Search-R1 parquet used by this project does not retain supporting_facts labels.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import unicodedata
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


DATASETS = ("nq", "hotpotqa")
BASELINES = ("B2", "B2.1")


def normalize_answer(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    text = re.sub(r"</?answer>", "", text, flags=re.IGNORECASE)
    text = re.sub(
        r"^\s*(?:final\s+answer|answer)\s*:\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = "".join(
        character
        for character in text
        if not unicodedata.category(character).startswith("P")
    )
    text = re.sub(r"\b(?:a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction: Any, gold_answers: Sequence[Any]) -> float:
    pred = normalize_answer(prediction)
    golds = [normalize_answer(answer) for answer in gold_answers]
    return float(bool(pred) and any(pred == gold for gold in golds))


def token_f1_single(prediction: str, gold: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def token_f1(prediction: Any, gold_answers: Sequence[Any]) -> float:
    golds = [str(answer) for answer in gold_answers] or [""]
    return max(token_f1_single(str(prediction or ""), gold) for gold in golds)


def document_text(document: Mapping[str, Any]) -> str:
    return " ".join(
        str(document.get(key) or "") for key in ("title", "text")
    )


def answer_hit(record: Mapping[str, Any], limit: Optional[int]) -> float:
    documents = list(record.get("retrieved_documents") or [])
    documents.sort(
        key=lambda item: (
            int(item.get("round_index") or 0),
            int(item.get("rank") or 0),
        )
    )
    if limit is not None:
        documents = documents[:limit]
    evidence = normalize_answer(" ".join(document_text(doc) for doc in documents))
    golds = [normalize_answer(answer) for answer in record.get("gold_answers") or []]
    return float(any(gold and gold in evidence for gold in golds))


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSONL: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: row is not a JSON object")
            rows.append(value)
    return rows


def normalized_baseline(value: Any) -> str:
    return str(value or "").strip().upper()


def normalized_dataset(value: Any) -> str:
    value = str(value or "").strip().lower()
    if "hotpot" in value:
        return "hotpotqa"
    if value in {"nq", "natural_questions", "naturalquestions"} or "nq" == value:
        return "nq"
    return value


def successful(record: Mapping[str, Any]) -> bool:
    return str(record.get("status") or "success").lower() == "success"


def discover_runs(
    outputs_dir: Path,
    expected_count: int,
) -> Dict[Tuple[str, str], Tuple[Path, List[Dict[str, Any]]]]:
    candidates: Dict[
        Tuple[str, str], List[Tuple[float, Path, List[Dict[str, Any]]]]
    ] = {}
    for path in outputs_dir.glob("*.jsonl"):
        try:
            rows = load_jsonl(path)
        except (OSError, ValueError):
            continue
        if len(rows) != expected_count:
            continue
        baselines = {normalized_baseline(row.get("baseline")) for row in rows}
        datasets = {normalized_dataset(row.get("dataset")) for row in rows}
        if len(baselines) != 1 or len(datasets) != 1:
            continue
        baseline = next(iter(baselines))
        dataset = next(iter(datasets))
        if baseline not in BASELINES or dataset not in DATASETS:
            continue
        candidates.setdefault((baseline, dataset), []).append(
            (path.stat().st_mtime, path, rows)
        )

    selected: Dict[Tuple[str, str], Tuple[Path, List[Dict[str, Any]]]] = {}
    missing = []
    for baseline in BASELINES:
        for dataset in DATASETS:
            key = (baseline, dataset)
            if key not in candidates:
                missing.append(f"{baseline}/{dataset}")
                continue
            _, path, rows = max(candidates[key], key=lambda item: item[0])
            selected[key] = (path, rows)
    if missing:
        raise SystemExit(
            "Missing valid runs with exactly "
            f"{expected_count} rows: {', '.join(missing)}"
        )
    return selected


def safe_mean(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.fmean(values) if values else math.nan


def summarize(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    total = len(rows)
    em_values = [exact_match(row.get("prediction"), row.get("gold_answers") or []) for row in rows]
    f1_values = [token_f1(row.get("prediction"), row.get("gold_answers") or []) for row in rows]
    hit3_values = [answer_hit(row, 3) for row in rows]
    hit_all_values = [answer_hit(row, None) for row in rows]
    second_round = [int(row.get("retrieval_rounds") or 0) >= 2 for row in rows]
    stop_reasons = Counter(str(row.get("retrieval_stop_reason") or "unknown") for row in rows)
    return {
        "n": total,
        "success_count": sum(successful(row) for row in rows),
        "failure_count": sum(not successful(row) for row in rows),
        "timeout_count": sum(bool(row.get("timed_out")) for row in rows),
        "empty_prediction_count": sum(not normalize_answer(row.get("prediction")) for row in rows),
        "em_percent": 100 * safe_mean(em_values),
        "token_f1_percent": 100 * safe_mean(f1_values),
        "answer_hit_at_3_percent": 100 * safe_mean(hit3_values),
        "answer_hit_all_percent": 100 * safe_mean(hit_all_values),
        "avg_latency_seconds": safe_mean(float(row.get("latency_seconds") or 0.0) for row in rows),
        "avg_retrieval_rounds": safe_mean(float(row.get("retrieval_rounds") or 0.0) for row in rows),
        "second_round_count": sum(second_round),
        "second_round_rate_percent": 100 * safe_mean(float(value) for value in second_round),
        "duplicate_followup_query_count": stop_reasons.get("duplicate_followup_query", 0),
        "duplicate_documents_filtered": sum(int(row.get("duplicate_documents_filtered") or 0) for row in rows),
        "stop_reasons": dict(sorted(stop_reasons.items())),
        "support_recall_at_3_percent": None,
        "support_recall_all_percent": None,
        "support_recall_note": (
            "Unavailable: converted nq_hotpotqa_train/test.parquet does not retain "
            "HotpotQA supporting_facts labels."
        ),
    }


def pair_rows(
    b2_rows: Sequence[Mapping[str, Any]],
    b21_rows: Sequence[Mapping[str, Any]],
) -> List[Tuple[Mapping[str, Any], Mapping[str, Any]]]:
    b2 = {str(row.get("question_id")): row for row in b2_rows}
    b21 = {str(row.get("question_id")): row for row in b21_rows}
    if len(b2) != len(b2_rows) or len(b21) != len(b21_rows):
        raise SystemExit("Duplicate question_id detected in a run")
    if b2.keys() != b21.keys():
        missing_b21 = sorted(b2.keys() - b21.keys())
        missing_b2 = sorted(b21.keys() - b2.keys())
        raise SystemExit(
            "B2/B2.1 question IDs do not match. "
            f"missing_in_B2.1={missing_b21[:5]}, missing_in_B2={missing_b2[:5]}"
        )
    return [(b2[key], b21[key]) for key in sorted(b2)]


def paired_summary(pairs: Sequence[Tuple[Mapping[str, Any], Mapping[str, Any]]]) -> Dict[str, Any]:
    em_deltas = []
    f1_deltas = []
    prediction_changed = 0
    for left, right in pairs:
        left_em = exact_match(left.get("prediction"), left.get("gold_answers") or [])
        right_em = exact_match(right.get("prediction"), right.get("gold_answers") or [])
        left_f1 = token_f1(left.get("prediction"), left.get("gold_answers") or [])
        right_f1 = token_f1(right.get("prediction"), right.get("gold_answers") or [])
        em_deltas.append(right_em - left_em)
        f1_deltas.append(right_f1 - left_f1)
        prediction_changed += (
            normalize_answer(left.get("prediction"))
            != normalize_answer(right.get("prediction"))
        )
    eps = 1e-12
    return {
        "paired_n": len(pairs),
        "em_improved": sum(delta > eps for delta in em_deltas),
        "em_regressed": sum(delta < -eps for delta in em_deltas),
        "em_unchanged": sum(abs(delta) <= eps for delta in em_deltas),
        "token_f1_improved": sum(delta > eps for delta in f1_deltas),
        "token_f1_regressed": sum(delta < -eps for delta in f1_deltas),
        "token_f1_unchanged": sum(abs(delta) <= eps for delta in f1_deltas),
        "mean_token_f1_delta_points": 100 * safe_mean(f1_deltas),
        "prediction_changed_count": prediction_changed,
        "prediction_changed_rate_percent": 100 * prediction_changed / len(pairs),
    }


def fmt(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def build_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# B2 与 B2.1 Smoke50 配对比较",
        "",
        f"生成时间：{report['generated_at']}",
        "",
        "## 输入文件",
        "",
    ]
    for key, path in report["input_files"].items():
        lines.append(f"- {key}: `{path}`")
    lines.extend([
        "",
        "## 汇总指标",
        "",
        "| 数据集 | 基线 | EM | Token F1 | AnswerHit@3 | AnswerHit@All | 平均耗时(s) | 平均轮数 | 二轮率 | 重复查询拦截 | 重复文档过滤 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for dataset in DATASETS:
        for baseline in BASELINES:
            row = report["metrics"][dataset][baseline]
            lines.append(
                f"| {dataset} | {baseline} | {fmt(row['em_percent'])} | "
                f"{fmt(row['token_f1_percent'])} | {fmt(row['answer_hit_at_3_percent'])} | "
                f"{fmt(row['answer_hit_all_percent'])} | {fmt(row['avg_latency_seconds'])} | "
                f"{fmt(row['avg_retrieval_rounds'])} | {fmt(row['second_round_rate_percent'])}% | "
                f"{row['duplicate_followup_query_count']} | {row['duplicate_documents_filtered']} |"
            )
    lines.extend(["", "## 配对变化（B2.1 - B2）", ""])
    for dataset in DATASETS:
        row = report["paired"][dataset]
        lines.extend([
            f"### {dataset}",
            "",
            f"- 同 ID 样本：{row['paired_n']}",
            f"- EM：改善 {row['em_improved']}，退化 {row['em_regressed']}，不变 {row['em_unchanged']}",
            f"- Token F1：改善 {row['token_f1_improved']}，退化 {row['token_f1_regressed']}，不变 {row['token_f1_unchanged']}",
            f"- Token F1 平均变化：{row['mean_token_f1_delta_points']:.2f} 个百分点",
            f"- 归一化答案发生变化：{row['prediction_changed_count']}/{row['paired_n']} "
            f"({row['prediction_changed_rate_percent']:.2f}%)",
            "",
        ])
    lines.extend([
        "## 指标口径与限制",
        "",
        "- EM 与 Token F1 使用与项目推理端一致的 SQuAD 风格英文归一化。",
        "- AnswerHit@3 检查排序后的前三篇检索文档标题与正文是否包含任一标准答案；AnswerHit@All 检查全部返回文档。",
        "- 当前转换后的 test.parquet 未保留 HotpotQA supporting_facts，因此本报告不伪造 SupportRecall@3/All；若补回原始标签，可在不重跑推理的情况下追加计算。",
        "- Smoke50 仅用于回归门禁和方向判断，不用于宣称统计显著性。",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--expected-count", type=int, default=50)
    args = parser.parse_args()

    selected = discover_runs(args.outputs_dir, args.expected_count)
    report: Dict[str, Any] = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "expected_count": args.expected_count,
        "input_files": {},
        "metrics": {dataset: {} for dataset in DATASETS},
        "paired": {},
    }
    for dataset in DATASETS:
        for baseline in BASELINES:
            path, rows = selected[(baseline, dataset)]
            report["input_files"][f"{baseline}/{dataset}"] = str(path)
            report["metrics"][dataset][baseline] = summarize(rows)
        pairs = pair_rows(
            selected[("B2", dataset)][1],
            selected[("B2.1", dataset)][1],
        )
        report["paired"][dataset] = paired_summary(pairs)

    args.reports_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = args.reports_dir / f"b2_vs_b2_1_smoke50_{timestamp}"
    json_path = stem.with_suffix(".json")
    csv_path = stem.with_suffix(".csv")
    md_path = stem.with_suffix(".md")
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(build_markdown(report), encoding="utf-8")

    csv_fields = [
        "dataset", "baseline", "n", "em_percent", "token_f1_percent",
        "answer_hit_at_3_percent", "answer_hit_all_percent",
        "avg_latency_seconds", "avg_retrieval_rounds",
        "second_round_count", "second_round_rate_percent",
        "duplicate_followup_query_count", "duplicate_documents_filtered",
        "failure_count", "timeout_count", "empty_prediction_count",
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        for dataset in DATASETS:
            for baseline in BASELINES:
                row = report["metrics"][dataset][baseline]
                writer.writerow({"dataset": dataset, "baseline": baseline, **{key: row.get(key) for key in csv_fields[2:]}})

    print("Selected runs:")
    for key, path in report["input_files"].items():
        print(f"  {key}: {path}")
    print()
    print(build_markdown(report))
    print("Written:")
    print(f"  {json_path}")
    print(f"  {csv_path}")
    print(f"  {md_path}")


if __name__ == "__main__":
    main()
