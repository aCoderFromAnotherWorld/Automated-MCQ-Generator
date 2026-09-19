"""Tests for local trained-checkpoint integration without loading a model."""

import unittest

from src.question_generator import LocalQuestionGenerator, QuestionGenerationError, build_prompt


class QuestionGeneratorTests(unittest.TestCase):
    def test_prompt_is_normalized_and_answer_aware(self) -> None:
        prompt = build_prompt(" TCP   is reliable. ", " TCP ")
        self.assertEqual(prompt, "generate question: context: TCP is reliable. answer: TCP")

    def test_remote_backend_is_rejected_by_local_checkpoint_adapter(self) -> None:
        with self.assertRaises(QuestionGenerationError):
            LocalQuestionGenerator.from_config({"question_model_backend": "hf_api"})
