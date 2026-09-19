"""Conservative textbook-text cleaning (M04)."""

from __future__ import annotations

import re
from typing import Any, Mapping

# Wikipedia-style reference markers: [a], [b], [9], [15], [note 1], ...
_CITATION_RE = re.compile(r"\s*\[(?:[a-z]{1,2}|\d{1,4}|note\s+[a-z0-9]+)\]", re.IGNORECASE)


def clean_text(text: str) -> str:
    """Normalize PDF artifacts while preserving sentence punctuation and case.

    Citation markers (``[a]``, ``[9]``, ``[note 1]``) are stripped, soft line
    wraps are joined into single spaces, and blank-line paragraph breaks are
    preserved as ``\\n\\n`` so downstream chunking can keep chunks topically
    coherent.
    """

    if not isinstance(text, str):
        raise TypeError("Text to preprocess must be a string.")
    cleaned = text.replace("\u00ad", "")  # soft hyphen
    cleaned = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", cleaned)  # line-break hyphenation
    cleaned = _CITATION_RE.sub("", cleaned)
    cleaned = re.sub(r"[\t\r\f\v]+", " ", cleaned)
    # Split into paragraphs on blank lines, flatten soft line wraps inside each
    # paragraph, then re-join with a preserved paragraph separator.
    paragraphs = [re.sub(r"\s*\n\s*", " ", part) for part in re.split(r"\n\s*\n", cleaned)]
    cleaned = "\n\n".join(part.strip() for part in paragraphs if part.strip())
    cleaned = re.sub(r" {2,}", " ", cleaned)
    return cleaned.strip()


def preprocess_result(result: Mapping[str, Any]) -> dict[str, Any]:
    """Copy a result artifact and attach cleaned text without losing raw text."""

    updated = dict(result)
    raw_text = str(updated.get("raw_text", updated.get("source", "")))
    updated["raw_text"] = raw_text
    updated["cleaned_text"] = clean_text(raw_text)
    return updated
