"""Tests for M01 and M02 contracts/input handling."""

import json
import unittest

from config import CONFIG
from src.api import generate_mcqs
from src.contracts import RESULT_KEYS, ensure_result_shape, new_result
from src.input_handler import InputValidationError, text_to_result, validate_text


class ContractsAndInputTests(unittest.TestCase):
    def test_new_result_contains_the_complete_json_safe_schema(self) -> None:
        result = new_result(source="TCP is reliable.", input_mode="text", config=CONFIG)
        self.assertEqual(set(result), set(RESULT_KEYS))
        json.dumps(result)

    def test_shape_fills_skipped_stages(self) -> None:
        result = ensure_result_shape({"source": "Example", "input_mode": "text"}, CONFIG)
        self.assertEqual(result["questions"], [])
        self.assertIn("validation", result)

    def test_text_input_rejects_empty_text(self) -> None:
        with self.assertRaises(InputValidationError):
            validate_text("   ")

    def test_text_input_creates_a_source_artifact(self) -> None:
        result = text_to_result(" TCP is reliable. ", CONFIG)
        self.assertEqual(result["source"], "TCP is reliable.")
        self.assertEqual(result["input_mode"], "text")

    def test_pre_generation_api_returns_ranked_candidates_not_fake_questions(self) -> None:
        result = generate_mcqs(
            "Transmission Control Protocol provides reliable delivery. "
            "User Datagram Protocol provides connectionless delivery.",
            num_questions=2,
        )
        self.assertEqual(result["questions"], [])
        self.assertIn("ranked_candidates", result)
        self.assertEqual(result["validation"]["summary"]["status"], "awaiting_embeddings_and_generation")
