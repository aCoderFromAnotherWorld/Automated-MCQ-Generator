"""Deterministic sentence segmentation with optional page provenance (M05)."""

from __future__ import annotations

import re
from typing import Any, Iterable, Mapping

_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def segment_text(
    text: str,
    *,
    page_number: int | None = None,
    start_id: int = 1,
) -> list[dict[str, Any]]:
    """Split text into stable, JSON-serializable sentence records."""

    if not isinstance(text, str):
        raise TypeError("Text to segment must be a string.")
    normalized = " ".join(text.split())
    if not normalized:
        return []
    pieces = [piece.strip() for piece in _BOUNDARY.split(normalized) if piece.strip()]
    return [
        {
            "sentence_id": start_id + index,
            "text": sentence,
            "page_number": page_number,
        }
        for index, sentence in enumerate(pieces)
    ]


# Kept as the public name used by the existing chunking API.
def segment_sentences(
    text: str,
    *,
    page_number: int | None = None,
    start_id: int = 1,
) -> list[dict[str, Any]]:
    return segment_text(text, page_number=page_number, start_id=start_id)


def segment_pages(pages: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Segment pages independently while preserving their page numbers."""

    sentences: list[dict[str, Any]] = []
    next_id = 1
    for page in pages:
        page_sentences = segment_text(
            str(page.get("text", "")),
            page_number=page.get("page_number"),
            start_id=next_id,
        )
        sentences.extend(page_sentences)
        next_id += len(page_sentences)
    return sentences
