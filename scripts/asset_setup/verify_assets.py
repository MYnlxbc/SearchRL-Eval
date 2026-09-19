#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def record(path: Path, required: bool = True) -> dict:
    exists = path.exists()
    size = path.stat().st_size if exists and path.is_file() else None
    return {
        "path": str(path),
        "required": required,
        "exists": exists,
        "size_bytes": size,
        "ok": exists and (not path.is_file() or size > 0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    root = args.root.expanduser().resolve()

    checks = [
        record(root / "models/Qwen2.5-3B-Instruct/config.json"),
        record(root / "models/Qwen2.5-3B-Instruct/tokenizer_config.json"),
        record(root / "models/e5-base-v2/config.json"),
        record(root / "datasets/raw/nq_hotpotqa_train"),
        record(root / "retrieval/e5_Flat.index", required=False),
        record(root / "retrieval/wiki-18.jsonl", required=False),
    ]

    corpus_check = {"checked": False, "ok": None, "error": None}
    corpus = root / "retrieval/wiki-18.jsonl"
    if corpus.exists():
        corpus_check["checked"] = True
        try:
            with corpus.open("r", encoding="utf-8") as handle:
                sample = json.loads(handle.readline())
            corpus_check["ok"] = "id" in sample and "contents" in sample
        except Exception as exc:
            corpus_check["ok"] = False
            corpus_check["error"] = repr(exc)

    required_ok = all(item["ok"] for item in checks if item["required"])
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "required_assets_ok": required_ok,
        "checks": checks,
        "corpus_sample_check": corpus_check,
        "note": "Wiki-18 checks are optional until 02_download_wiki18.sh has completed.",
    }

    output = root / "manifests/verification_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[DONE] report={output}")
    raise SystemExit(0 if required_ok else 2)


if __name__ == "__main__":
    main()

