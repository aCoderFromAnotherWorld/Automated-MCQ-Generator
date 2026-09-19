"""Focused tests for M12 MCQ assembly and critical-failure rejection."""

import json
import unittest

from src.mcq_validator import assemble_mcq


class McqValidatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.question = "Which protocol provides reliable delivery?"
        self.answer = "TCP"
        self.context = "TCP provides reliable delivery. UDP provides connectionless delivery."
        self.distractors = {
            "candidate_pool": ["UDP", "IP", "ARP"],
            "selected": ["UDP", "IP", "ARP"],
            "scored_candidates": [],
            "rejected": [],
            "valid": True,
            "reasons": [],
        }

    def test_valid_mcq_contains_required_provenance_and_is_json_safe(self) -> None:
        result = assemble_mcq(
            self.question,
            self.answer,
            self.context,
            self.distractors,
            chunk_id=4,
            source_page=2,
            generator="local",
            similarity=lambda left, right: 0.9,
        )
        self.assertTrue(result["valid"])
        self.assertEqual(result["mcq"]["correct_answer"], "TCP")
        self.assertEqual(result["mcq"]["source_context"], self.context)
        json.dumps(result)

    def test_unsupported_answer_is_critical_failure(self) -> None:
        result = assemble_mcq(
            self.question,
            "QUIC",
            self.context,
            self.distractors,
            similarity=lambda left, right: 0.9,
        )
        self.assertFalse(result["valid"])
        self.assertIn("answer_not_supported_by_context", result["validation"]["critical_failures"])
        self.assertIsNone(result["mcq"])

    def test_fewer_than_three_distractors_is_critical_failure(self) -> None:
        distractors = {**self.distractors, "selected": ["UDP", "IP"], "valid": False}
        result = assemble_mcq(
            self.question,
            self.answer,
            self.context,
            distractors,
            similarity=lambda left, right: 0.9,
        )
        self.assertFalse(result["valid"])
        self.assertIn("fewer_than_three_valid_distractors", result["validation"]["critical_failures"])

    def test_duplicate_options_are_rejected(self) -> None:
        distractors = {**self.distractors, "selected": ["UDP", "tcp", "ARP"]}
        result = assemble_mcq(
            self.question,
            self.answer,
            self.context,
            distractors,
            similarity=lambda left, right: 0.9,
        )
        self.assertFalse(result["valid"])
        self.assertIn("duplicate_options", result["validation"]["critical_failures"])

    def test_contextually_supported_distractor_is_rejected(self) -> None:
        distractors = {
            **self.distractors,
            "selected": ["UDP", "IP", "ARP"],
            "rejected": [{
                "text": "UDP",
                "reasons": ["contextually_supported_critical_failure"],
            }],
        }
        result = assemble_mcq(
            self.question,
            self.answer,
            self.context,
            distractors,
            similarity=lambda left, right: 0.9,
        )
        self.assertFalse(result["valid"])
        self.assertIn("contextually_supported_distractor", result["validation"]["critical_failures"])

    def test_semantic_duplicate_question_is_rejected(self) -> None:
        result = assemble_mcq(
            self.question,
            self.answer,
            self.context,
            self.distractors,
            previous_questions=["Which protocol gives reliable delivery?"],
            similarity=lambda left, right: 0.95,
        )
        self.assertFalse(result["valid"])
        self.assertIn("duplicate_question", result["validation"]["reasons"])


if __name__ == "__main__":
    unittest.main()
