#!/usr/bin/env python3
"""Validate one B2.1 JSONL result without loading model dependencies."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from searchrl_eval.eval_utils import document_identity  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--similarity-threshold", type=float, default=0.85)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with args.output.open(encoding="utf-8") as file:
        rows = [json.loads(line) for line in file if line.strip()]

    failures = [row for row in rows if row.get("status") != "success"]
    timeouts = [row for row in rows if row.get("timed_out")]
    empty_predictions = [
        row for row in rows
        if not str(row.get("normalized_prediction") or "").strip()
    ]

    round_counts = Counter(row.get("retrieval_rounds") for row in rows)
    stop_reasons = Counter(row.get("retrieval_stop_reason") for row in rows)

    duplicate_output_rows = []
    invalid_document_counts = []
    invalid_second_round_queries = []

    for row in rows:
        documents = row.get("retrieved_documents") or []
        identities = [document_identity(document) for document in documents]

        if len(identities) != len(set(identities)):
            duplicate_output_rows.append(row.get("question_id"))

        rounds = row.get("retrieval_rounds")
        if rounds not in {1, 2} or len(documents) > rounds * 3:
            invalid_document_counts.append(
                (
                    row.get("question_id"),
                    rounds,
                    len(documents),
                )
            )

        similarity = row.get("followup_query_similarity")
        if (
            rounds == 2
            and isinstance(similarity, (int, float))
            and similarity >= args.similarity_threshold
        ):
            invalid_second_round_queries.append(
                (row.get("question_id"), similarity)
            )

    latencies = [
        row["latency_seconds"]
        for row in rows
        if isinstance(row.get("latency_seconds"), (int, float))
    ]

    print("file:", args.output)
    print("total:", len(rows))
    print("failed:", len(failures))
    print("timeouts:", len(timeouts))
    print("empty_predictions:", len(empty_predictions))
    print("round_distribution:", dict(round_counts))
    print("stop_reasons:", dict(stop_reasons))
    print(
        "duplicates_filtered:",
        sum(row.get("duplicate_documents_filtered", 0) for row in rows),
    )
    print("duplicate_output_rows:", duplicate_output_rows)
    print("invalid_document_counts:", invalid_document_counts)
    print("invalid_second_round_queries:", invalid_second_round_queries)
    print(
        "avg_latency_seconds:",
        statistics.mean(latencies) if latencies else None,
    )

    assert len(rows) == args.expected_count
    assert not failures
    assert not timeouts
    assert not empty_predictions
    assert set(round_counts).issubset({1, 2})
    assert not duplicate_output_rows
    assert not invalid_document_counts
    assert not invalid_second_round_queries

    print("B2.1 validation: PASSED")


if __name__ == "__main__":
    main()
