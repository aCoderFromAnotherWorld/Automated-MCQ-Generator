"""Local inference for the SQuADv2-fine-tuned FLAN-T5 checkpoint.

This module is deliberately independent from Streamlit and pipeline
orchestration. It loads only a checkpoint produced by ``training/train_qg.py``
and returns a traceable generation record for later validation stages.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path
import time
from typing import Any, Mapping


class QuestionGenerationError(RuntimeError):
    """Raised when the local trained checkpoint cannot generate a question."""


@dataclass(frozen=True)
class LocalGeneratorSettings:
    checkpoint: Path
    base_model: str
    max_new_tokens: int = 64
    num_beams: int = 4
    num_return_sequences: int = 3


@dataclass(frozen=True)
class ApiGeneratorSettings:
    model_name: str
    token: str
    max_new_tokens: int = 64
    num_return_sequences: int = 3
    timeout_seconds: float = 30.0
    retries: int = 3
    backoff_seconds: float = 1.0


def build_prompt(context: str, answer: str) -> str:
    """Create the exact answer-aware prompt used during local fine-tuning."""

    cleaned_context = " ".join(context.split())
    cleaned_answer = " ".join(answer.split())
    if not cleaned_context or not cleaned_answer:
        raise ValueError("Both non-empty context and answer are required.")
    return f"generate question: context: {cleaned_context} answer: {cleaned_answer}"


def _clean_question(value: Any) -> str:
    """Normalize common text-generation wrappers without inventing content."""

    text = " ".join(str(value).split()).strip()
    for prefix in ("question:", "question -", "question -"):
        if text.lower().startswith(prefix):
            text = text[len(prefix) :].strip()
    if len(text) > 2 and text[0].isdigit() and text[1:3] in {". ", ") "}:
        text = text[3:].strip()
    return text


def _record(
    *,
    chunk_id: int | None,
    answer: str,
    prompt: str,
    backend: str,
    model_name: str,
    raw_outputs: list[str],
    checkpoint: str | None = None,
) -> dict[str, Any]:
    questions = [_clean_question(output) for output in raw_outputs]
    questions = list(dict.fromkeys(question for question in questions if question))
    return {
        "chunk_id": chunk_id,
        "answer": " ".join(answer.split()),
        "model_input": prompt,
        "backend": backend,
        "model_name": model_name,
        "checkpoint": checkpoint,
        "raw_output": raw_outputs[0] if raw_outputs else "",
        "candidate_questions": questions,
        "selected_question": questions[0] if questions else "",
    }


@lru_cache(maxsize=2)
def _load_checkpoint(checkpoint_text: str) -> tuple[Any, Any, Any]:
    """Load and cache a local model/tokenizer pair by checkpoint directory."""

    checkpoint = Path(checkpoint_text)
    if not (checkpoint / "config.json").is_file():
        raise QuestionGenerationError(
            f"Local trained checkpoint is unavailable at '{checkpoint}'. "
            "Run training/train_qg.py first, then set question_model_checkpoint."
        )
    try:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except ModuleNotFoundError as error:
        raise QuestionGenerationError(
            "Local generation requires torch and transformers. Install requirements.txt."
        ) from error

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device


class LocalQuestionGenerator:
    """Generate questions using the locally trained checkpoint only."""

    def __init__(self, settings: LocalGeneratorSettings) -> None:
        self.settings = settings

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "LocalQuestionGenerator":
        if config.get("question_model_backend", "local") != "local":
            raise QuestionGenerationError(
                "The trained-checkpoint integration supports only the local backend."
            )
        checkpoint_value = str(config.get("question_model_checkpoint", "")).strip()
        if not checkpoint_value:
            raise QuestionGenerationError("question_model_checkpoint is required for local generation.")
        checkpoint = Path(checkpoint_value)
        return cls(
            LocalGeneratorSettings(
                checkpoint=checkpoint,
                base_model=str(config.get("question_model", "google/flan-t5-base")),
                max_new_tokens=int(config.get("generation_max_new_tokens", 64)),
                num_beams=int(config.get("generation_num_beams", 4)),
                num_return_sequences=max(1, int(config.get("generation_num_return_sequences", 3))),
            )
        )

    def generate_question(self, context: str, answer: str, *, chunk_id: int | None = None) -> dict[str, Any]:
        """Generate one cleaned question and return an auditable record."""

        prompt = build_prompt(context, answer)
        tokenizer, model, device = _load_checkpoint(str(self.settings.checkpoint))
        try:
            import torch

            encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            encoded = {name: value.to(device) for name, value in encoded.items()}
            with torch.no_grad():
                generated = model.generate(
                    **encoded,
                    max_new_tokens=self.settings.max_new_tokens,
                    num_beams=self.settings.num_beams,
                    num_return_sequences=self.settings.num_return_sequences,
                    early_stopping=True,
                )
            raw_outputs = tokenizer.batch_decode(generated, skip_special_tokens=True)
        except Exception as error:
            raise QuestionGenerationError(
                f"Local question generation failed: {error}. "
                "Switch backend and retry if appropriate."
            ) from error

        return _record(
            chunk_id=chunk_id,
            answer=answer,
            prompt=prompt,
            backend="local",
            model_name=self.settings.base_model,
            raw_outputs=list(raw_outputs),
            checkpoint=str(self.settings.checkpoint),
        )


class HuggingFaceApiQuestionGenerator:
    """Generate questions through the explicit Hugging Face API backend."""

    def __init__(self, settings: ApiGeneratorSettings, *, client: Any | None = None) -> None:
        self.settings = settings
        self._client = client

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "HuggingFaceApiQuestionGenerator":
        if config.get("question_model_backend", "local") != "hf_api":
            raise QuestionGenerationError(
                "The Hugging Face API generator requires question_model_backend='hf_api'."
            )
        token = os.getenv("HUGGINGFACE_API_TOKEN", "").strip()
        if not token:
            raise QuestionGenerationError(
                "HUGGINGFACE_API_TOKEN is required for the hf_api backend. "
                "Set it, or switch backend and retry."
            )
        return cls(
            ApiGeneratorSettings(
                model_name=str(config.get("question_model", "google/flan-t5-base")),
                token=token,
                max_new_tokens=int(config.get("generation_max_new_tokens", 64)),
                num_return_sequences=max(1, int(config.get("generation_num_return_sequences", 3))),
                timeout_seconds=float(config.get("generation_timeout_seconds", 30.0)),
                retries=max(0, int(config.get("generation_retries", 3))),
                backoff_seconds=float(config.get("generation_backoff_seconds", 1.0)),
            )
        )

    def _client_instance(self) -> Any:
        if self._client is None:
            try:
                from huggingface_hub import InferenceClient
            except ModuleNotFoundError as error:
                raise QuestionGenerationError(
                    "The hf_api backend requires huggingface-hub. Install requirements.txt."
                ) from error
            self._client = InferenceClient(
                model=self.settings.model_name,
                token=self.settings.token,
                timeout=self.settings.timeout_seconds,
            )
        return self._client

    @staticmethod
    def _outputs(response: Any) -> list[str]:
        if isinstance(response, str):
            return [response]
        if isinstance(response, Mapping):
            return [str(response.get("generated_text", response.get("text", "")))]
        if isinstance(response, (list, tuple)):
            outputs: list[str] = []
            for item in response:
                if isinstance(item, Mapping):
                    outputs.append(str(item.get("generated_text", item.get("text", ""))))
                else:
                    outputs.append(str(item))
            return outputs
        return [str(response)]

    def generate_question(self, context: str, answer: str, *, chunk_id: int | None = None) -> dict[str, Any]:
        prompt = build_prompt(context, answer)
        last_error: Exception | None = None
        for attempt in range(self.settings.retries + 1):
            try:
                response = self._client_instance().text_generation(
                    prompt,
                    max_new_tokens=self.settings.max_new_tokens,
                    num_return_sequences=self.settings.num_return_sequences,
                    return_full_text=False,
                )
                return _record(
                    chunk_id=chunk_id,
                    answer=answer,
                    prompt=prompt,
                    backend="hf_api",
                    model_name=self.settings.model_name,
                    raw_outputs=self._outputs(response),
                )
            except Exception as error:
                last_error = error
                if attempt < self.settings.retries:
                    time.sleep(self.settings.backoff_seconds * (2**attempt))
        raise QuestionGenerationError(
            f"Hugging Face API question generation failed after {self.settings.retries + 1} "
            f"attempts: {last_error}. Switch backend and retry."
        ) from last_error


def generator_from_config(config: Mapping[str, Any]) -> LocalQuestionGenerator | HuggingFaceApiQuestionGenerator:
    """Build the explicitly selected backend without automatic fallback."""

    backend = str(config.get("question_model_backend", "local"))
    if backend == "local":
        return LocalQuestionGenerator.from_config(config)
    if backend == "hf_api":
        return HuggingFaceApiQuestionGenerator.from_config(config)
    raise QuestionGenerationError(
        f"Unsupported question_model_backend '{backend}'. Choose 'local' or 'hf_api'."
    )


def generate_question(
    context: str,
    answer: str,
    *,
    config: Mapping[str, Any],
    chunk_id: int | None = None,
) -> dict[str, Any]:
    """Stable backend-independent generation entry point."""

    return generator_from_config(config).generate_question(context, answer, chunk_id=chunk_id)
