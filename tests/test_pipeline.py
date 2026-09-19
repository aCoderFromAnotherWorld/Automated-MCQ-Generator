"""Focused orchestration tests for M13."""

import json
import unittest
from unittest.mock import patch

from config import CONFIG
from src.pipeline import generate_mcqs


class _FakeGenerator:
    def generate_question(self, context: str, answer: str, *, chunk_id: int | None = None) -> dict[str, object]:
        return {
            "chunk_id": chunk_id,
            "answer": answer,
            "model_input": f"fake context: {context} answer: {answer}",
            "backend": "test",
            "model_name": "fake-qg",
            "raw_output": "Which protocol provides reliable delivery?",
            "candidate_questions": ["Which protocol provides reliable delivery?"],
            "selected_question": "Which protocol provides reliable delivery?",
        }


class PipelineTests(unittest.TestCase):
    def test_pipeline_returns_full_schema_and_valid_mcq(self) -> None:
        candidates = [
            {"text": "TCP", "chunk_id": 1, "rake_score": 10.0, "is_noun_phrase": True, "is_named_entity": False, "entity_label": None},
            {"text": "UDP", "chunk_id": 1, "rake_score": 9.0, "is_noun_phrase": True, "is_named_entity": False, "entity_label": None},
            {"text": "IP", "chunk_id": 1, "rake_score": 8.0, "is_noun_phrase": True, "is_named_entity": False, "entity_label": None},
            {"text": "ARP", "chunk_id": 1, "rake_score": 7.0, "is_noun_phrase": True, "is_named_entity": False, "entity_label": None},
        ]
        def similarity(left: str, right: str) -> float:
            if "reliable delivery" in right.lower() and left.lower() in {"udp", "ip", "arp"}:
                return 0.1
            return 0.9

        with patch("src.pipeline.extract_candidates", return_value=candidates), patch(
            "src.pipeline.add_tfidf_scores", side_effect=lambda chunks, values: [dict(value, tfidf_score=1.0) for value in values]
        ), patch("src.pipeline.rank_candidates", return_value=candidates):
            result = generate_mcqs(
                "TCP provides reliable delivery. UDP is connectionless. IP routes packets. ARP resolves addresses.",
                num_questions=1,
                config=CONFIG,
                question_generator=_FakeGenerator(),
                similarity=similarity,
            )

        self.assertEqual(len(result["questions"]), 1)
        self.assertEqual(result["questions"][0]["answer"], "TCP")
        self.assertEqual(len(result["questions"][0]["distractors"]), 3)
        self.assertEqual(result["validation"]["summary"]["valid"], 1)
        self.assertEqual(len(result["generation_records"]), 1)
        self.assertEqual(len(result["distractor_records"]), 1)
        self.assertTrue(result["stats"]["concept_coverage"])
        json.dumps(result)

    def test_pipeline_rejects_invalid_question_without_fabricating_mcq(self) -> None:
        class EmptyGenerator:
            def generate_question(self, context: str, answer: str, *, chunk_id: int | None = None) -> dict[str, object]:
                return {"chunk_id": chunk_id, "selected_question": "", "candidate_questions": []}

        result = generate_mcqs(
            "TCP provides reliable delivery.",
            num_questions=1,
            config=CONFIG,
            question_generator=EmptyGenerator(),
        )
        self.assertEqual(result["questions"], [])
        self.assertEqual(result["generation_records"][0]["selected_question"], "")
        self.assertEqual(result["validation"]["per_question"][0]["reasons"], ["question_empty"])


if __name__ == "__main__":
    unittest.main()
