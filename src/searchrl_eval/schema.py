"""Unified input and output schemas for B0/B1/B2 evaluation."""

from __future__ import annotations

import json
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class EvalExample:
    """A normalized QA example."""

    question_id: str
    dataset: str
    question: str
    gold_answers: List[str]
    source_split: str = "unknown"
    source_index: Optional[int] = None


@dataclass
class RetrievedDocument:
    """One retrieved document from one retrieval round."""

    round_index: int
    rank: int
    query: str
    document_id: Optional[str] = None
    title: str = ""
    text: str = ""
    score: Optional[float] = None


@dataclass
class EvalRecord:
    """One complete evaluation record shared by B0/B1/B2."""

    run_id: str
    baseline: str
    dataset: str
    question_id: str
    question: str
    gold_answers: List[str]

    prediction: str = ""
    retrieved_documents: List[Dict[str, Any]] = field(default_factory=list)
    retrieval_rounds: int = 0

    latency_seconds: float = 0.0
    retrieval_latency_seconds: float = 0.0
    generation_latency_seconds: float = 0.0

    prompt_tokens: int = 0
    generated_tokens: int = 0
    peak_gpu_memory_gib: Optional[float] = None

    model_path: str = ""
    retriever_url: Optional[str] = None

    status: str = "pending"
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    error_traceback: Optional[str] = None

    started_at_utc: str = field(default_factory=utc_now)
    finished_at_utc: Optional[str] = None

    def mark_success(self) -> None:
        self.status = "success"
        self.finished_at_utc = utc_now()

    def mark_error(self, exc: BaseException) -> None:
        self.status = "failed"
        self.error_type = type(exc).__name__
        self.error_message = str(exc)
        self.error_traceback = traceback.format_exc()
        self.finished_at_utc = utc_now()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
        )


def append_jsonl(path: Path, record: EvalRecord) -> None:
    """Append one record immediately for crash-safe evaluation."""

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as file:
        file.write(record.to_json())
        file.write("\n")
        file.flush()
