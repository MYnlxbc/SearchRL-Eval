#!/usr/bin/env python3
"""Build a compressed IVF-PQ index from an existing sequential FAISS index.

The source index is kept unchanged.  Vectors are reconstructed and added in
order, preserving the document-row alignment expected by the corpus reader.
This program is CPU-only; do not run it while the full retriever or GRPO is
using memory on the same 120 GiB container.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import List

import faiss
import numpy as np


def reconstruct_block(index: faiss.Index, start: int, count: int) -> np.ndarray:
    vectors = np.empty((count, index.d), dtype=np.float32)
    index.reconstruct_n(start, count, vectors)
    return vectors


def evenly_spaced_training_vectors(
    index: faiss.Index,
    requested_count: int,
    block_size: int,
) -> np.ndarray:
    total = int(index.ntotal)
    count = min(requested_count, total)
    if count < 256 * 64:
        raise ValueError("Need at least 16,384 vectors to train IVF-PQ")

    blocks = max(1, math.ceil(count / block_size))
    per_block = math.ceil(count / blocks)
    starts = np.linspace(0, max(0, total - per_block), blocks, dtype=np.int64)
    pieces: List[np.ndarray] = []
    remaining = count
    for start in starts:
        take = min(per_block, remaining, total - int(start))
        if take <= 0:
            continue
        pieces.append(reconstruct_block(index, int(start), take))
        remaining -= take
        if remaining <= 0:
            break
    vectors = np.concatenate(pieces, axis=0)
    if len(vectors) < count:
        raise RuntimeError(f"Collected only {len(vectors)} of {count} training vectors")
    return np.ascontiguousarray(vectors[:count])


def verify_recall(
    source: faiss.Index,
    compressed: faiss.Index,
    query_count: int,
    topk: int,
) -> dict:
    total = int(source.ntotal)
    query_count = min(query_count, total)
    if query_count == 0:
        return {"query_count": 0, "topk": topk, "recall_at_k": None}

    ids = np.linspace(0, total - 1, query_count, dtype=np.int64)
    queries = np.concatenate(
        [reconstruct_block(source, int(i), 1) for i in ids], axis=0
    )
    _, exact_ids = source.search(queries, topk)
    _, compressed_ids = compressed.search(queries, topk)
    overlaps = [
        len(set(map(int, left)) & set(map(int, right))) / topk
        for left, right in zip(exact_ids, compressed_ids)
    ]
    top1 = np.mean(exact_ids[:, 0] == compressed_ids[:, 0])
    return {
        "query_count": int(query_count),
        "topk": int(topk),
        "recall_at_k": float(np.mean(overlaps)),
        "top1_agreement": float(top1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nlist", type=int, default=4096)
    parser.add_argument("--m", type=int, default=96, help="PQ subquantizers")
    parser.add_argument("--nbits", type=int, default=8)
    parser.add_argument("--nprobe", type=int, default=32)
    parser.add_argument("--train-size", type=int, default=200_000)
    parser.add_argument("--train-block-size", type=int, default=5_000)
    parser.add_argument("--add-block-size", type=int, default=50_000)
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--verify-queries", type=int, default=100)
    parser.add_argument("--verify-topk", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.source.is_file():
        raise FileNotFoundError(f"Source index not found: {args.source}")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {args.output}; use --overwrite")

    faiss.omp_set_num_threads(max(1, args.threads))
    started = time.time()
    print(f"[INFO] Reading source index: {args.source}", flush=True)
    source = faiss.read_index(str(args.source))
    total = int(source.ntotal)
    dimension = int(source.d)
    metric = int(source.metric_type)
    print(
        f"[INFO] Source type={type(source).__name__}, ntotal={total}, "
        f"d={dimension}, metric={metric}",
        flush=True,
    )
    if total == 0:
        raise ValueError("Source index is empty")
    if dimension % args.m != 0:
        raise ValueError(f"Dimension {dimension} is not divisible by m={args.m}")

    print("[INFO] Collecting evenly spaced vectors for IVF-PQ training", flush=True)
    train_vectors = evenly_spaced_training_vectors(
        source, args.train_size, args.train_block_size
    )

    if metric == faiss.METRIC_INNER_PRODUCT:
        quantizer: faiss.Index = faiss.IndexFlatIP(dimension)
    elif metric == faiss.METRIC_L2:
        quantizer = faiss.IndexFlatL2(dimension)
    else:
        raise ValueError(f"Unsupported FAISS metric type: {metric}")

    compressed = faiss.IndexIVFPQ(
        quantizer, dimension, args.nlist, args.m, args.nbits, metric
    )
    print(
        f"[INFO] Training IVF-PQ: nlist={args.nlist}, m={args.m}, "
        f"nbits={args.nbits}",
        flush=True,
    )
    compressed.train(train_vectors)
    del train_vectors

    print("[INFO] Adding reconstructed vectors in order", flush=True)
    for start in range(0, total, args.add_block_size):
        count = min(args.add_block_size, total - start)
        compressed.add(reconstruct_block(source, start, count))
        completed = start + count
        if completed == total or completed % (args.add_block_size * 20) == 0:
            print(f"[INFO] Added {completed}/{total} vectors", flush=True)

    compressed.nprobe = min(args.nprobe, args.nlist)
    if int(compressed.ntotal) != total:
        raise RuntimeError(
            f"Unexpected output count: {compressed.ntotal} != {total}"
        )

    verification = verify_recall(
        source, compressed, args.verify_queries, args.verify_topk
    )
    print(f"[INFO] Vector-neighbor verification: {verification}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    if temporary.exists():
        temporary.unlink()
    print(f"[INFO] Writing compressed index: {args.output}", flush=True)
    faiss.write_index(compressed, str(temporary))
    os.replace(temporary, args.output)

    metadata = {
        "source": str(args.source),
        "output": str(args.output),
        "source_type": type(source).__name__,
        "ntotal": total,
        "dimension": dimension,
        "metric_type": metric,
        "nlist": args.nlist,
        "m": args.m,
        "nbits": args.nbits,
        "nprobe": compressed.nprobe,
        "train_size": min(args.train_size, total),
        "add_block_size": args.add_block_size,
        "threads": args.threads,
        "verification": verification,
        "elapsed_seconds": round(time.time() - started, 3),
        "warning": (
            "IVF-PQ changes retrieval recall. This index must be evaluated "
            "before being used for any reported result."
        ),
    }
    metadata_path = args.output.with_suffix(args.output.suffix + ".meta.json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[INFO] Wrote metadata: {metadata_path}", flush=True)
    print(f"[INFO] Output size: {args.output.stat().st_size / 1024**3:.2f} GiB")


if __name__ == "__main__":
    main()
