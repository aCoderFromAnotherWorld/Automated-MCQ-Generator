"""Local inference for the SQuADv2-fine-tuned FLAN-T5 checkpoint.

This module is deliberately independent from Streamlit and pipeline
orchestration. It loads only a checkpoint produced by ``training/train_qg.py``
and returns a traceable generation record for later validation stages.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping


class QuestionGenerationError(RuntimeError):
    """Raised when the local trained checkpoint cannot generate a question."""


@dataclass(frozen=True)
class LocalGeneratorSettings:
    checkpoint: Path
    base_model: str
    max_new_tokens: int = 64
    num_beams: int = 4


def build_prompt(context: str, answer: str) -> str:
    """Create the exact answer-aware prompt used during local fine-tuning."""

    cleaned_context = " ".join(context.split())
    cleaned_answer = " ".join(answer.split())
    if not cleaned_context or not cleaned_answer:
        raise ValueError("Both non-empty context and answer are required.")
    return f"generate question: context: {cleaned_context} answer: {cleaned_answer}"


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
                    early_stopping=True,
                )
            raw_output = tokenizer.decode(generated[0], skip_special_tokens=True)
        except Exception as error:
            raise QuestionGenerationError(f"Local question generation failed: {error}") from error

        question = " ".join(raw_output.split())
        return {
            "chunk_id": chunk_id,
            "answer": " ".join(answer.split()),
            "model_input": prompt,
            "backend": "local",
            "model_name": self.settings.base_model,
            "checkpoint": str(self.settings.checkpoint),
            "raw_output": raw_output,
            "candidate_questions": [question] if question else [],
            "selected_question": question,
        }
