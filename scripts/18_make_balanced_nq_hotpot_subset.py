#!/usr/bin/env python3
"""Create deterministic, source-balanced NQ/HotpotQA parquet subsets."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def balanced_sample(frame: pd.DataFrame, per_source: int, seed: int) -> pd.DataFrame:
    sources = ("nq", "hotpotqa")
    parts: list[pd.DataFrame] = []
    for offset, source in enumerate(sources):
        subset = frame.loc[frame["data_source"] == source]
        if len(subset) < per_source:
            raise ValueError(f"{source} has only {len(subset)} rows; need {per_source}")
        parts.append(subset.sample(n=per_source, random_state=seed + offset))
    # Interleave sources deterministically, so each prefix remains balanced.
    ordered: list[pd.DataFrame] = []
    for index in range(per_source):
        ordered.extend(part.iloc[[index]] for part in parts)
    return pd.concat(ordered, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-per-source", type=int, default=100)
    parser.add_argument("--val-per-source", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260821)
    args = parser.parse_args()

    train = pd.read_parquet(args.source_dir / "train.parquet")
    test = pd.read_parquet(args.source_dir / "test.parquet")
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    train_subset = balanced_sample(train, args.train_per_source, args.seed)
    val_subset = balanced_sample(test, args.val_per_source, args.seed + 10_000)
    train_subset.to_parquet(output_dir / "train.parquet", index=False)
    val_subset.to_parquet(output_dir / "test.parquet", index=False)
    print(f"train={len(train_subset)} {train_subset['data_source'].value_counts().to_dict()}")
    print(f"test={len(val_subset)} {val_subset['data_source'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
