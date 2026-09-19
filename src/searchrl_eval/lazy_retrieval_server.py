"""Search-R1 retrieval server with disk-light lazy JSONL corpus access."""

from __future__ import annotations

import argparse
import json
import mmap
import os
import struct
import sys
import threading
from array import array
from pathlib import Path
from typing import Any, Dict


class LazyJsonlCorpus:
    """Random-access JSONL corpus using a compact uint64 offset file."""

    def __init__(self, corpus_path: str) -> None:
        self.corpus_path = Path(corpus_path)
        self.offset_path = Path(f"{corpus_path}.offsets.u64")
        self._local = threading.local()

        if not self.corpus_path.is_file():
            raise FileNotFoundError(self.corpus_path)

        if not self.offset_path.is_file() or self.offset_path.stat().st_size == 0:
            self._build_offsets()

        offset_bytes = self.offset_path.stat().st_size

        if offset_bytes % 8 != 0:
            raise RuntimeError(
                f"Invalid offset file size: {offset_bytes}"
            )

        self._length = offset_bytes // 8
        self._offset_file = self.offset_path.open("rb")
        self._offset_map = mmap.mmap(
            self._offset_file.fileno(),
            length=0,
            access=mmap.ACCESS_READ,
        )

        print(
            f"[INFO] Lazy corpus ready: {self._length} rows, "
            f"offsets={self.offset_path}",
            flush=True,
        )

    def _build_offsets(self) -> None:
        temp_path = Path(f"{self.offset_path}.tmp")
        temp_path.unlink(missing_ok=True)

        print(
            f"[INFO] Building compact offsets for {self.corpus_path}",
            flush=True,
        )

        count = 0
        byte_offset = 0
        batch = array("Q")
        batch_limit = 1_000_000

        with self.corpus_path.open(
            "rb",
            buffering=8 * 1024 * 1024,
        ) as source, temp_path.open(
            "wb",
            buffering=8 * 1024 * 1024,
        ) as target:
            for line in source:
                if not line.strip():
                    byte_offset += len(line)
                    continue
                batch.append(byte_offset)
                byte_offset += len(line)
                count += 1

                if len(batch) >= batch_limit:
                    batch.tofile(target)
                    batch = array("Q")

                    print(
                        f"[INFO] Indexed {count:,} corpus rows",
                        flush=True,
                    )

            if batch:
                batch.tofile(target)

            target.flush()
            os.fsync(target.fileno())

        os.replace(temp_path, self.offset_path)

        print(
            f"[INFO] Offset build complete: {count:,} rows, "
            f"{self.offset_path.stat().st_size / 1024**2:.1f} MiB",
            flush=True,
        )

    def __len__(self) -> int:
        return self._length

    def _corpus_handle(self):
        handle = getattr(self._local, "handle", None)

        if handle is None or handle.closed:
            handle = self.corpus_path.open(
                "rb",
                buffering=1024 * 1024,
            )
            self._local.handle = handle

        return handle

    def __getitem__(self, index: int) -> Dict[str, Any]:
        index = int(index)

        if index < 0:
            index += self._length

        if index < 0 or index >= self._length:
            raise IndexError(index)

        byte_offset = struct.unpack_from(
            "Q",
            self._offset_map,
            index * 8,
        )[0]

        handle = self._corpus_handle()
        handle.seek(byte_offset)
        line = handle.readline()

        if not line:
            raise RuntimeError(
                f"Cannot read corpus row {index} at offset {byte_offset}"
            )

        return json.loads(line.decode("utf-8", errors="replace"))


def adapt_batch_search_for_endpoint(batch_search):
    """Match Search-R1's HTTP endpoint contract without changing upstream code.

    The upstream ``/retrieve`` handler always unpacks ``(documents, scores)``,
    even when the client did not request scores.  ``DenseRetriever`` returns
    only documents in that case, so expose an empty score list per query for
    the HTTP layer while preserving the normal scored result unchanged.
    """

    def endpoint_batch_search(*args, **kwargs):
        result = batch_search(*args, **kwargs)
        return_score = kwargs.get("return_score", False)

        if return_score:
            return result

        return result, [[] for _ in result]

    return endpoint_batch_search


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_path", required=True)
    parser.add_argument("--corpus_path", required=True)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--retriever_name", default="e5")
    parser.add_argument("--retriever_model", required=True)
    parser.add_argument("--faiss_gpu", action="store_true")
    args = parser.parse_args()

    searchr1_code_dir = os.environ.get("SEARCHR1_CODE_DIR")

    if not searchr1_code_dir:
        raise RuntimeError("SEARCHR1_CODE_DIR is not configured")

    sys.path.insert(0, searchr1_code_dir)

    from search_r1.search import retrieval_server as server

    # DenseRetriever运行时查找模块全局的load_corpus，
    # 因此可以替换为懒加载实现，而不修改官方源码。
    server.load_corpus = LazyJsonlCorpus

    server.config = server.Config(
        retrieval_method=args.retriever_name,
        index_path=args.index_path,
        corpus_path=args.corpus_path,
        retrieval_topk=args.topk,
        faiss_gpu=args.faiss_gpu,
        retrieval_model_path=args.retriever_model,
        retrieval_pooling_method="mean",
        retrieval_query_max_length=256,
        retrieval_use_fp16=True,
        retrieval_batch_size=512,
    )

    server.retriever = server.get_retriever(server.config)

    # ``retrieval_server.py`` always expects a two-value result from
    # ``batch_search``. Keep this compatibility shim local to our launcher so
    # the bundled Search-R1 source remains untouched.
    server.retriever.batch_search = adapt_batch_search_for_endpoint(
        server.retriever.batch_search
    )

    corpus_rows = len(server.retriever.corpus)
    index_rows = int(server.retriever.index.ntotal)

    print(f"[INFO] Corpus rows: {corpus_rows:,}", flush=True)
    print(f"[INFO] FAISS rows: {index_rows:,}", flush=True)

    if corpus_rows != index_rows:
        raise RuntimeError(
            f"Corpus/index mismatch: corpus={corpus_rows}, "
            f"index={index_rows}"
        )

    server.uvicorn.run(
        server.app,
        host="0.0.0.0",
        port=8000,
    )


if __name__ == "__main__":
    main()
