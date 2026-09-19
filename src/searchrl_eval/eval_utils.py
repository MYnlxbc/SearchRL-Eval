"""Lightweight evaluation helpers with no model/runtime dependencies."""

from __future__ import annotations

import hashlib
import re
import time
import unicodedata
from typing import Any, Dict, Iterable, List, Optional, Tuple


class EmptyPredictionError(RuntimeError):
    """Raised when generation contains no evaluable answer text."""


class ExampleTimeoutError(TimeoutError):
    """Raised when one example exceeds its configured deadline."""

    def __init__(self, stage: str, timeout_seconds: float) -> None:
        self.stage = stage
        self.timeout_seconds = timeout_seconds
        super().__init__(
            f"Example exceeded {timeout_seconds:.3f}s during {stage}"
        )


class ExampleDeadline:
    """Boundary-enforced per-example deadline.

    The deadline is checked before and after retrieval/generation stages. HTTP
    retrieval receives the remaining time as its request timeout. A running
    CUDA kernel is not forcefully interrupted; an overrun is recorded as soon
    as control returns to Python.
    """

    def __init__(self, timeout_seconds: Optional[float]) -> None:
        self.timeout_seconds = (
            float(timeout_seconds)
            if timeout_seconds is not None and timeout_seconds > 0
            else None
        )
        self.started = time.perf_counter()

    def remaining(self, stage: str) -> Optional[float]:
        if self.timeout_seconds is None:
            return None

        remaining = self.timeout_seconds - (
            time.perf_counter() - self.started
        )
        if remaining <= 0:
            raise ExampleTimeoutError(stage, self.timeout_seconds)
        return remaining

    def check(self, stage: str) -> None:
        self.remaining(stage)


def clean_answer_text(value: Any) -> str:
    """Normalize wrappers/whitespace while preserving answer semantics."""

    text = unicodedata.normalize("NFKC", str(value or ""))
    text = re.sub(r"</?answer>", "", text, flags=re.IGNORECASE)
    text = re.sub(
        r"^\s*(?:final\s+answer|answer)\s*:\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    return " ".join(text.split()).strip()


def normalize_answer_for_metrics(value: Any) -> str:
    """SQuAD-style English answer normalization for EM/Token-F1."""

    text = clean_answer_text(value).casefold()
    text = "".join(
        character
        for character in text
        if not unicodedata.category(character).startswith("P")
    )
    text = re.sub(r"\b(?:a|an|the)\b", " ", text)
    return " ".join(text.split())


def normalized_query_tokens(value: Any) -> List[str]:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return re.findall(r"[\w]+", text, flags=re.UNICODE)


def query_similarity(left: Any, right: Any) -> float:
    """Token-set Jaccard similarity used for duplicate-query stopping."""

    left_tokens = set(normalized_query_tokens(left))
    right_tokens = set(normalized_query_tokens(right))

    if not left_tokens and not right_tokens:
        return 1.0
    if not left_tokens or not right_tokens:
        return 0.0

    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def document_identity(document: Dict[str, Any]) -> str:
    document_id = document.get("document_id")
    if document_id is not None and str(document_id).strip():
        return f"id:{str(document_id).strip()}"

    title = " ".join(
        unicodedata.normalize(
            "NFKC",
            str(document.get("title") or ""),
        ).casefold().split()
    )
    if title:
        return f"title:{title}"

    text = unicodedata.normalize(
        "NFKC",
        str(document.get("text") or ""),
    ).strip()
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return f"text:{digest}"


def select_unique_documents(
    existing: Iterable[Dict[str, Any]],
    candidates: Iterable[Dict[str, Any]],
    limit: int,
) -> Tuple[List[Dict[str, Any]], int]:
    """Select up to ``limit`` new documents and count filtered duplicates."""

    seen = {document_identity(document) for document in existing}
    selected: List[Dict[str, Any]] = []
    duplicate_count = 0

    for candidate in candidates:
        identity = document_identity(candidate)
        if identity in seen:
            duplicate_count += 1
            continue

        seen.add(identity)
        selected.append(dict(candidate))

        if len(selected) >= limit:
            break

    return selected, duplicate_count
