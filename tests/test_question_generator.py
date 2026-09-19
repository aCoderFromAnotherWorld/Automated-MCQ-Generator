"""Tests for local trained-checkpoint integration without loading a model."""

import os
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from src.question_generator import (
    ApiGeneratorSettings,
    HuggingFaceApiQuestionGenerator,
    LocalGeneratorSettings,
    LocalQuestionGenerator,
    QuestionGenerationError,
    build_prompt,
)


class _FakeTokenizer:
    def __call__(self, prompt: str, **kwargs: object) -> dict[str, torch.Tensor]:
        return {"input_ids": torch.tensor([[1, 2]])}

    def batch_decode(self, generated: torch.Tensor, **kwargs: object) -> list[str]:
        return ["Question: Which protocol is reliable?", "2. Which protocol is reliable?"]


class _FakeModel:
    def generate(self, **kwargs: object) -> torch.Tensor:
        return torch.tensor([[1, 2], [3, 4]])


class _FakeApiClient:
    def __init__(self, responses: list[object]) -> None:
        self.responses = responses
        self.calls = 0

    def text_generation(self, prompt: str, **kwargs: object) -> object:
        response = self.responses[self.calls]
        self.calls += 1
        if isinstance(response, Exception):
            raise response
        return response


class QuestionGeneratorTests(unittest.TestCase):
    def test_prompt_is_normalized_and_answer_aware(self) -> None:
        prompt = build_prompt(" TCP   is reliable. ", " TCP ")
        self.assertEqual(prompt, "generate question: context: TCP is reliable. answer: TCP")

    def test_remote_backend_is_rejected_by_local_checkpoint_adapter(self) -> None:
        with self.assertRaises(QuestionGenerationError):
            LocalQuestionGenerator.from_config({"question_model_backend": "hf_api"})

    def test_local_generation_returns_multiple_cleaned_candidates(self) -> None:
        generator = LocalQuestionGenerator(
            LocalGeneratorSettings(Path("checkpoint"), "google/flan-t5-base", num_return_sequences=2)
        )
        with patch("src.question_generator._load_checkpoint", return_value=(_FakeTokenizer(), _FakeModel(), torch.device("cpu"))):
            record = generator.generate_question("TCP is reliable.", "TCP", chunk_id=3)
        self.assertEqual(record["backend"], "local")
        self.assertEqual(record["chunk_id"], 3)
        self.assertEqual(record["candidate_questions"], ["Which protocol is reliable?"])
        self.assertEqual(record["selected_question"], "Which protocol is reliable?")

    def test_api_backend_uses_same_traceable_record_shape(self) -> None:
        generator = HuggingFaceApiQuestionGenerator(
            ApiGeneratorSettings("google/flan-t5-base", "test-token", retries=0),
            client=_FakeApiClient([{"generated_text": "Which protocol is reliable?"}]),
        )
        record = generator.generate_question("TCP is reliable.", "TCP", chunk_id=4)
        self.assertEqual(
            set(record),
            {"chunk_id", "answer", "model_input", "backend", "model_name", "checkpoint", "raw_output", "candidate_questions", "selected_question"},
        )
        self.assertEqual(record["backend"], "hf_api")
        self.assertEqual(record["candidate_questions"], ["Which protocol is reliable?"])

    def test_api_factory_requires_token_without_fallback(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(QuestionGenerationError, "HUGGINGFACE_API_TOKEN"):
                HuggingFaceApiQuestionGenerator.from_config({"question_model_backend": "hf_api"})

    def test_api_backend_retries_then_returns_record(self) -> None:
        client = _FakeApiClient([TimeoutError("temporary"), "Which protocol is reliable?"])
        generator = HuggingFaceApiQuestionGenerator(
            ApiGeneratorSettings("google/flan-t5-base", "test-token", retries=1, backoff_seconds=0),
            client=client,
        )
        with patch("src.question_generator.time.sleep") as sleep:
            record = generator.generate_question("TCP is reliable.", "TCP")
        self.assertEqual(record["selected_question"], "Which protocol is reliable?")
        self.assertEqual(client.calls, 2)
        sleep.assert_called_once()
