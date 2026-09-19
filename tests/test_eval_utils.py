from __future__ import annotations

import unittest
from unittest.mock import patch

from searchrl_eval.eval_utils import (
    ExampleDeadline,
    ExampleTimeoutError,
    clean_answer_text,
    normalize_answer_for_metrics,
    query_similarity,
    select_unique_documents,
)


class AnswerNormalizationTests(unittest.TestCase):
    def test_clean_answer_wrappers_and_whitespace(self) -> None:
        self.assertEqual(
            clean_answer_text(" <answer>  New   York City </answer> "),
            "New York City",
        )
        self.assertEqual(
            clean_answer_text("Final answer: 20%"),
            "20%",
        )

    def test_metric_normalization(self) -> None:
        self.assertEqual(
            normalize_answer_for_metrics("The New-York City!"),
            "newyork city",
        )
        self.assertEqual(normalize_answer_for_metrics("<answer></answer>"), "")


class QueryAndDocumentTests(unittest.TestCase):
    def test_query_similarity(self) -> None:
        question = "Where do they grow hops in the US?"
        self.assertEqual(query_similarity(question, question.upper()), 1.0)
        self.assertGreaterEqual(
            query_similarity(
                question,
                "Where do they grow hops in the US today?",
            ),
            0.85,
        )
        self.assertLess(
            query_similarity(question, "Columbia University location"),
            0.2,
        )

    def test_document_deduplication(self) -> None:
        existing = [
            {"document_id": "1", "title": "Alpha", "text": "a"},
        ]
        candidates = [
            {"document_id": "1", "title": "Alpha duplicate", "text": "a"},
            {"document_id": None, "title": "Beta", "text": "b"},
            {"document_id": None, "title": " beta ", "text": "different"},
            {"document_id": "3", "title": "Gamma", "text": "c"},
        ]

        selected, duplicate_count = select_unique_documents(
            existing,
            candidates,
            limit=3,
        )

        self.assertEqual(duplicate_count, 2)
        self.assertEqual(
            [document["title"] for document in selected],
            ["Beta", "Gamma"],
        )


class DeadlineTests(unittest.TestCase):
    def test_deadline_records_stage(self) -> None:
        with patch(
            "searchrl_eval.eval_utils.time.perf_counter",
            side_effect=[100.0, 100.5, 101.1],
        ):
            deadline = ExampleDeadline(1.0)
            self.assertAlmostEqual(deadline.remaining("first"), 0.5)

            with self.assertRaises(ExampleTimeoutError) as context:
                deadline.check("answer_generation_end")

        self.assertEqual(
            context.exception.stage,
            "answer_generation_end",
        )


if __name__ == "__main__":
    unittest.main()
