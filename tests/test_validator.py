"""Focused tests for M10 question validation."""

import unittest

from src.validator import validate_question


class ValidatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.question = "Which protocol provides reliable delivery?"
        self.answer = "TCP"
        self.context = "TCP provides reliable delivery."

    def test_valid_question_passes_all_checks(self) -> None:
        result = validate_question(
            self.question,
            self.answer,
            self.context,
            similarity=lambda left, right: 0.9,
        )
        self.assertTrue(result["valid"])
        self.assertTrue(all(result["checks"].values()))
        self.assertEqual(result["reasons"], [])

    def test_empty_question_is_rejected(self) -> None:
        result = validate_question("", self.answer, self.context, similarity=lambda left, right: 0.9)
        self.assertFalse(result["valid"])
        self.assertIn("question_empty", result["reasons"])

    def test_question_length_is_checked(self) -> None:
        result = validate_question("Why?", self.answer, self.context, similarity=lambda left, right: 0.9)
        self.assertFalse(result["valid"])
        self.assertIn("question_length_invalid", result["reasons"])

    def test_exact_and_semantic_duplicates_are_rejected(self) -> None:
        exact = validate_question(
            self.question,
            self.answer,
            self.context,
            previous_questions=[self.question],
            similarity=lambda left, right: 0.1,
        )
        semantic = validate_question(
            self.question,
            self.answer,
            self.context,
            previous_questions=["Which protocol gives reliable delivery?"],
            similarity=lambda left, right: 0.95,
        )
        self.assertIn("duplicate_question", exact["reasons"])
        self.assertIn("duplicate_question", semantic["reasons"])
        self.assertGreaterEqual(semantic["duplicate_score"], 0.95)

    def test_answer_must_be_supported_by_context(self) -> None:
        result = validate_question(
            self.question,
            "UDP",
            self.context,
            similarity=lambda left, right: 0.9,
        )
        self.assertFalse(result["valid"])
        self.assertIn("answer_not_supported_by_context", result["reasons"])
        self.assertFalse(result["checks"]["answer_present"])

    def test_context_relevance_threshold_is_enforced(self) -> None:
        result = validate_question(
            self.question,
            self.answer,
            self.context,
            similarity=lambda left, right: 0.1,
            relevance_threshold=0.25,
        )
        self.assertFalse(result["valid"])
        self.assertIn("question_not_relevant_to_context", result["reasons"])
        self.assertEqual(result["scores"]["context_relevance"], 0.1)

    def test_unsupported_information_is_reported(self) -> None:
        result = validate_question(
            "Which protocol encrypts packets?",
            self.answer,
            self.context,
            similarity=lambda left, right: 0.9,
        )
        self.assertFalse(result["valid"])
        self.assertIn("unsupported_information", result["reasons"])
        self.assertIn("encrypts", result["unsupported_words"])

    def test_function_and_paraphrase_words_are_not_treated_as_unsupported(self) -> None:
        # "did" is a question-frame word; "provide" is a stem match for "provides".
        result = validate_question(
            "What did that provide for reliable delivery?",
            self.answer,
            self.context,
            similarity=lambda left, right: 0.9,
        )
        self.assertEqual(result["unsupported_words"], [])
        self.assertTrue(result["checks"]["supported_information"])


if __name__ == "__main__":
    unittest.main()
