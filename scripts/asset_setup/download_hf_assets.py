#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


ASSETS = {
    "core": [
        {
            "repo_id": "Qwen/Qwen2.5-3B-Instruct",
            "repo_type": "model",
            "relative_dir": "models/Qwen2.5-3B-Instruct",
            "allow_patterns": None,
        },
        {
            "repo_id": "intfloat/e5-base-v2",
            "repo_type": "model",
            "relative_dir": "models/e5-base-v2",
            "allow_patterns": None,
        },
        {
            "repo_id": "PeterJinGo/nq_hotpotqa_train",
            "repo_type": "dataset",
            "relative_dir": "datasets/raw/nq_hotpotqa_train",
            "allow_patterns": None,
        },
    ],
    "wiki": [
        {
            "repo_id": "PeterJinGo/wiki-18-e5-index",
            "repo_type": "dataset",
            "relative_dir": "retrieval/wiki-18-e5-index",
            "allow_patterns": ["part_aa", "part_ab"],
        },
        {
            "repo_id": "PeterJinGo/wiki-18-corpus",
            "repo_type": "dataset",
            "relative_dir": "retrieval/wiki-18-corpus",
            "allow_patterns": ["wiki-18.jsonl.gz"],
        },
    ],
}


def directory_size(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--group", required=True, choices=["core", "wiki", "all"])
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifests").mkdir(parents=True, exist_ok=True)

    groups = ["core", "wiki"] if args.group == "all" else [args.group]
    api = HfApi(endpoint=os.environ.get("HF_ENDPOINT"))
    records = []

    for group in groups:
        for asset in ASSETS[group]:
            repo_id = asset["repo_id"]
            repo_type = asset["repo_type"]
            local_dir = root / asset["relative_dir"]
            local_dir.mkdir(parents=True, exist_ok=True)

            print(f"[INFO] resolving {repo_type}:{repo_id}", flush=True)
            info = api.repo_info(repo_id=repo_id, repo_type=repo_type)
            revision = info.sha
            print(f"[INFO] revision={revision}", flush=True)

            snapshot_download(
                repo_id=repo_id,
                repo_type=repo_type,
                revision=revision,
                local_dir=str(local_dir),
                allow_patterns=asset["allow_patterns"],
                max_workers=args.workers,
            )

            size_bytes = directory_size(local_dir)
            records.append(
                {
                    "group": group,
                    "repo_id": repo_id,
                    "repo_type": repo_type,
                    "revision": revision,
                    "local_dir": str(local_dir),
                    "size_bytes": size_bytes,
                    "downloaded_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            print(f"[DONE] {repo_id} size_bytes={size_bytes}", flush=True)

    manifest = root / "manifests" / f"hf_assets_{args.group}.json"
    manifest.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[DONE] manifest={manifest}")


if __name__ == "__main__":
    main()

