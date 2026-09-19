"""Conservative textbook-text cleaning (M04)."""

from __future__ import annotations

import re
from typing import Any, Mapping


def clean_text(text: str) -> str:
    """Normalize PDF artifacts while preserving sentence punctuation and case."""

    if not isinstance(text, str):
        raise TypeError("Text to preprocess must be a string.")
    cleaned = text.replace("\u00ad", "")  # soft hyphen
    cleaned = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", cleaned)  # line-break hyphenation
    cleaned = re.sub(r"[\t\r\f\v]+", " ", cleaned)
    cleaned = re.sub(r"\s*\n\s*", " ", cleaned)
    cleaned = re.sub(r" {2,}", " ", cleaned)
    return cleaned.strip()


def preprocess_result(result: Mapping[str, Any]) -> dict[str, Any]:
    """Copy a result artifact and attach cleaned text without losing raw text."""

    updated = dict(result)
    raw_text = str(updated.get("raw_text", updated.get("source", "")))
    updated["raw_text"] = raw_text
    updated["cleaned_text"] = clean_text(raw_text)
    return updated
