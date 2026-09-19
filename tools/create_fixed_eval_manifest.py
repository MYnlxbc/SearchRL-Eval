#!/usr/bin/env python3
"""Create a reproducible, balanced evaluation manifest from a QA parquet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


DEFAULT_DATASETS = (
    "nq",
    "musique",
    "hotpotqa",
    "popqa",
    "2wikimultihopqa",
    "triviaqa",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a fixed per-dataset QA evaluation manifest."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-dataset", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DEFAULT_DATASETS),
        choices=DEFAULT_DATASETS,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.per_dataset <= 0:
        raise ValueError("--per-dataset must be positive")

    dataframe = pd.read_parquet(args.input)
    rows = []
    summary = {
        "source": str(args.input.resolve()),
        "seed": args.seed,
        "per_dataset": args.per_dataset,
        "datasets": {},
    }

    for dataset in args.datasets:
        subset = dataframe.loc[dataframe["data_source"] == dataset]
        if len(subset) < args.per_dataset:
            raise ValueError(
                f"{dataset} has only {len(subset)} rows; "
                f"cannot select {args.per_dataset}"
            )
        selected = subset.sample(
            n=args.per_dataset,
            random_state=args.seed,
        ).sort_index()
        rows.extend(selected.to_dict(orient="records"))
        summary["datasets"][dataset] = {
            "count": len(selected),
            "source_row_indices": [int(index) for index in selected.index],
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str))
            handle.write("\n")

    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
