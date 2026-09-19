"""Raw-text input validation and source-artifact creation (M02)."""

from __future__ import annotations

from typing import Any, Mapping

from src.contracts import new_result


class InputValidationError(ValueError):
    """Raised for text that cannot enter the NLP pipeline."""


def validate_text(text: str, *, min_characters: int = 1) -> str:
    if not isinstance(text, str):
        raise InputValidationError("Text input must be a string.")
    normalized = text.strip()
    if not normalized:
        raise InputValidationError("Text input cannot be empty.")
    if len(normalized) < min_characters:
        raise InputValidationError(f"Text input must contain at least {min_characters} characters.")
    return normalized


def text_to_result(text: str, config: Mapping[str, Any]) -> dict[str, Any]:
    """Return a complete, unprocessed result artifact for raw text."""

    source = validate_text(text, min_characters=1)
    return new_result(source=source, input_mode="text", config=config)
