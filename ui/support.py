"""Non-NLP helpers used by the Streamlit presentation layer."""

from __future__ import annotations

from io import BytesIO
from typing import Any, Mapping


RESULT_KEYS = (
    "config",
    "source",
    "pages",
    "cleaned_text",
    "sentences",
    "chunks",
    "candidates",
    "ranked_candidates",
    "generation_records",
    "distractor_records",
    "validation",
    "stats",
    "questions",
)


def extract_pdf_pages(file_bytes: bytes) -> list[str]:
    """Extract page text for UI preview; OCR is intentionally out of scope."""

    from pypdf import PdfReader

    reader = PdfReader(BytesIO(file_bytes))
    return [(page.extract_text() or "").strip() for page in reader.pages]


def normalize_result(result: Mapping[str, Any] | None, config: Mapping[str, Any]) -> dict[str, Any]:
    """Make an eventual pipeline result safe for rendering during development."""

    value = dict(result or {})
    defaults: dict[str, Any] = {
        "config": dict(config),
        "source": "",
        "pages": [],
        "cleaned_text": "",
        "sentences": [],
        "chunks": [],
        "candidates": [],
        "ranked_candidates": [],
        "generation_records": [],
        "distractor_records": [],
        "validation": {},
        "stats": {},
        "questions": [],
    }
    for key in RESULT_KEYS:
        value.setdefault(key, defaults[key])
    return value
