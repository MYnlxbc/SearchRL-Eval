#!/usr/bin/env python3
"""Export paired B2/B2.1 changed cases for manual error analysis."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping


def load_metrics_module(script_path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("b2_compare_metrics", script_path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Cannot load metrics script: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def titles(record: Mapping[str, Any]) -> List[str]:
    result = []
    for document in record.get("retrieved_documents") or []:
        title = str(document.get("title") or "").strip()
        marker = f"R{document.get('round_index', '?')}.{document.get('rank', '?')}"
        result.append(f"{marker} {title}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--expected-count", type=int, default=50)
    args = parser.parse_args()

    metrics = load_metrics_module(
        Path(__file__).with_name("07_compare_b2_b2_1.py")
    )
    selected = metrics.discover_runs(args.outputs_dir, args.expected_count)
    details: List[Dict[str, Any]] = []

    for dataset in metrics.DATASETS:
        pairs = metrics.pair_rows(
            selected[("B2", dataset)][1],
            selected[("B2.1", dataset)][1],
        )
        for b2, b21 in pairs:
            b2_norm = metrics.normalize_answer(b2.get("prediction"))
            b21_norm = metrics.normalize_answer(b21.get("prediction"))
            if b2_norm == b21_norm:
                continue
            b2_em = metrics.exact_match(
                b2.get("prediction"), b2.get("gold_answers") or []
            )
            b21_em = metrics.exact_match(
                b21.get("prediction"), b21.get("gold_answers") or []
            )
            b2_f1 = metrics.token_f1(
                b2.get("prediction"), b2.get("gold_answers") or []
            )
            b21_f1 = metrics.token_f1(
                b21.get("prediction"), b21.get("gold_answers") or []
            )
            details.append({
                "dataset": dataset,
                "question_id": str(b2.get("question_id")),
                "question": str(b2.get("question") or ""),
                "gold_answers": b2.get("gold_answers") or [],
                "b2_prediction": str(b2.get("prediction") or ""),
                "b2_1_prediction": str(b21.get("prediction") or ""),
                "b2_em": b2_em,
                "b2_1_em": b21_em,
                "em_delta": b21_em - b2_em,
                "b2_token_f1": b2_f1,
                "b2_1_token_f1": b21_f1,
                "token_f1_delta": b21_f1 - b2_f1,
                "b2_rounds": int(b2.get("retrieval_rounds") or 0),
                "b2_1_rounds": int(b21.get("retrieval_rounds") or 0),
                "b2_stop_reason": b2.get("retrieval_stop_reason"),
                "b2_1_stop_reason": b21.get("retrieval_stop_reason"),
                "b2_followup_query": b2.get("followup_query"),
                "b2_1_followup_query": b21.get("followup_query"),
                "b2_titles": titles(b2),
                "b2_1_titles": titles(b21),
                "b2_1_duplicates_filtered": int(
                    b21.get("duplicate_documents_filtered") or 0
                ),
            })

    details.sort(
        key=lambda row: (
            float(row["token_f1_delta"]),
            str(row["dataset"]),
            str(row["question_id"]),
        )
    )
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = args.reports_dir / f"b2_vs_b2_1_changed_cases_{timestamp}"
    jsonl_path = stem.with_suffix(".jsonl")
    csv_path = stem.with_suffix(".csv")
    md_path = stem.with_suffix(".md")

    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in details:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    fields = [
        "dataset", "question_id", "question", "gold_answers",
        "b2_prediction", "b2_1_prediction", "b2_em", "b2_1_em",
        "em_delta", "b2_token_f1", "b2_1_token_f1", "token_f1_delta",
        "b2_rounds", "b2_1_rounds", "b2_stop_reason", "b2_1_stop_reason",
        "b2_followup_query", "b2_1_followup_query", "b2_titles",
        "b2_1_titles", "b2_1_duplicates_filtered",
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in details:
            writer.writerow({
                **row,
                "gold_answers": json.dumps(row["gold_answers"], ensure_ascii=False),
                "b2_titles": json.dumps(row["b2_titles"], ensure_ascii=False),
                "b2_1_titles": json.dumps(row["b2_1_titles"], ensure_ascii=False),
            })

    lines = [
        "# B2 与 B2.1 答案变化样本",
        "",
        f"共 {len(details)} 条；按 Token F1 变化从差到好排序。",
        "",
    ]
    for index, row in enumerate(details, start=1):
        lines.extend([
            f"## {index}. {row['dataset']} / {row['question_id']}",
            "",
            f"- 问题：{row['question']}",
            f"- 标准答案：`{row['gold_answers']}`",
            f"- B2：`{row['b2_prediction']}`",
            f"- B2.1：`{row['b2_1_prediction']}`",
            f"- EM：{row['b2_em']:.0f} → {row['b2_1_em']:.0f}",
            f"- Token F1：{row['b2_token_f1']:.4f} → {row['b2_1_token_f1']:.4f} "
            f"（Δ {row['token_f1_delta']:+.4f}）",
            f"- 轮数：{row['b2_rounds']} → {row['b2_1_rounds']}",
            f"- B2.1 停止原因：`{row['b2_1_stop_reason']}`",
            f"- B2.1 follow-up：`{row['b2_1_followup_query']}`",
            f"- B2 标题：`{row['b2_titles']}`",
            f"- B2.1 标题：`{row['b2_1_titles']}`",
            "",
        ])
    md_path.write_text("\n".join(lines), encoding="utf-8")

    regressions = sum(float(row["token_f1_delta"]) < -1e-12 for row in details)
    improvements = sum(float(row["token_f1_delta"]) > 1e-12 for row in details)
    print(f"changed_cases: {len(details)}")
    print(f"token_f1_regressions: {regressions}")
    print(f"token_f1_improvements: {improvements}")
    print("Written:")
    print(f"  {jsonl_path}")
    print(f"  {csv_path}")
    print(f"  {md_path}")


if __name__ == "__main__":
    main()
