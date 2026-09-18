"""Small, non-NLP helpers used by the Streamlit interface.

This module intentionally contains presentation/input-preview concerns only.
It must not grow question-generation, ranking, or validation logic.
"""

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
    """Return extracted page text for a user-uploaded text PDF.

    ``pypdf`` is imported lazily so importing UI helpers does not require the
    optional PDF dependency.  Scanned/image-only PDFs correctly return empty
    page text; OCR is outside the web-interface phase.
    """

    from pypdf import PdfReader

    reader = PdfReader(BytesIO(file_bytes))
    return [(page.extract_text() or "").strip() for page in reader.pages]


def normalize_result(result: Mapping[str, Any] | None, config: Mapping[str, Any]) -> dict[str, Any]:
    """Make an eventual pipeline response safe for UI rendering.

    Pipeline implementations remain responsible for producing the documented
    schema. This defensive normalization lets the UI show partial/failure
    results without key errors while the project is developed incrementally.
    """

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
