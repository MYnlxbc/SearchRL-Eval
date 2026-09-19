from __future__ import annotations

import sys
import types
import unittest


def install_lightweight_import_stubs() -> None:
    torch_stub = types.ModuleType("torch")
    sys.modules["torch"] = torch_stub

    datasets_stub = types.ModuleType("datasets")
    datasets_stub.load_dataset = None
    sys.modules["datasets"] = datasets_stub

    transformers_stub = types.ModuleType("transformers")
    transformers_stub.AutoModelForCausalLM = object
    transformers_stub.AutoTokenizer = object
    sys.modules["transformers"] = transformers_stub


install_lightweight_import_stubs()

from searchrl_eval.eval_utils import ExampleDeadline  # noqa: E402
from searchrl_eval.evaluate import run_example  # noqa: E402
from searchrl_eval.schema import EvalExample  # noqa: E402


class FakeGenerator:
    def __init__(self, responses):
        self.responses = list(responses)

    def generate(self, messages, max_new_tokens=None):
        del messages, max_new_tokens
        text = self.responses.pop(0)
        return text, 10, 2, 0.01


class FakeRetriever:
    def __init__(self, rounds):
        self.rounds = rounds
        self.calls = []

    def retrieve(
        self,
        query,
        round_index,
        topk=None,
        timeout_seconds=None,
    ):
        self.calls.append(
            {
                "query": query,
                "round_index": round_index,
                "topk": topk,
                "timeout_seconds": timeout_seconds,
            }
        )
        return [dict(document) for document in self.rounds[round_index]], 0.02


def document(document_id, title, round_index):
    return {
        "round_index": round_index,
        "rank": 1,
        "query": "query",
        "document_id": str(document_id),
        "title": title,
        "text": f"Text for {title}",
        "score": 1.0,
    }


def config():
    return {
        "retrieval": {
            "topk": 3,
            "deduplicate_documents": True,
            "followup_similarity_threshold": 0.85,
            "second_round_candidate_multiplier": 2,
        },
        "generation": {
            "max_new_tokens": 64,
            "query_max_new_tokens": 32,
        },
    }


EXAMPLE = EvalExample(
    question_id="dev_0",
    dataset="hotpotqa",
    question="Where do they grow hops in the US?",
    gold_answers=["Washington"],
)


class RunExampleTests(unittest.TestCase):
    def test_duplicate_followup_query_stops_before_second_retrieval(self):
        retriever = FakeRetriever(
            {
                1: [
                    document(1, "Hops", 1),
                    document(2, "Washington", 1),
                    document(3, "Oregon", 1),
                ]
            }
        )
        generator = FakeGenerator(
            [
                "Where do they grow hops in the US today?",
                "Final answer: Washington",
            ]
        )

        result = run_example(
            EXAMPLE,
            "B2",
            config(),
            generator,
            retriever,
            ExampleDeadline(10),
        )

        self.assertEqual(len(retriever.calls), 1)
        self.assertEqual(result["retrieval_rounds"], 1)
        self.assertEqual(
            result["retrieval_stop_reason"],
            "duplicate_followup_query",
        )
        self.assertEqual(result["prediction"], "Washington")
        self.assertEqual(result["normalized_prediction"], "washington")

    def test_second_round_filters_duplicates_and_keeps_three_new_docs(self):
        retriever = FakeRetriever(
            {
                1: [
                    document(1, "Ralph Hefferline", 1),
                    document(2, "Psychology", 1),
                    document(3, "Columbia", 1),
                ],
                2: [
                    document(2, "Psychology duplicate", 2),
                    document(3, "Columbia duplicate", 2),
                    document(4, "New York City", 2),
                    document(5, "Manhattan", 2),
                    document(6, "New York", 2),
                    document(7, "Unused", 2),
                ],
            }
        )
        generator = FakeGenerator(
            ["Columbia University location", "New York City"]
        )

        result = run_example(
            EXAMPLE,
            "B2",
            config(),
            generator,
            retriever,
            ExampleDeadline(10),
        )

        self.assertEqual(len(retriever.calls), 2)
        self.assertEqual(retriever.calls[1]["topk"], 6)
        self.assertEqual(result["retrieval_rounds"], 2)
        self.assertEqual(result["duplicate_documents_filtered"], 2)
        self.assertEqual(len(result["retrieved_documents"]), 6)
        self.assertEqual(
            [
                item["document_id"]
                for item in result["retrieved_documents"][3:]
            ],
            ["4", "5", "6"],
        )


if __name__ == "__main__":
    unittest.main()
