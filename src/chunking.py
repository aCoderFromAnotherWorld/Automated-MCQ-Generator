"""Sentence segmentation and overlapping, source-mapped chunks (M05)."""

from __future__ import annotations

import re
from typing import Any, Iterable, Mapping


_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def segment_sentences(text: str, *, page_number: int | None = None, start_id: int = 1) -> list[dict[str, Any]]:
    """Use a deterministic lightweight segmenter suitable before spaCy setup."""

    normal = " ".join(text.split())
    if not normal:
        return []
    pieces = [piece.strip() for piece in _BOUNDARY.split(normal) if piece.strip()]
    return [
        {"sentence_id": start_id + index, "text": sentence, "page_number": page_number}
        for index, sentence in enumerate(pieces)
    ]


def segment_pages(pages: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Segment pages individually so every sentence retains its source page."""

    sentences: list[dict[str, Any]] = []
    next_id = 1
    for page in pages:
        page_sentences = segment_sentences(
            str(page.get("text", "")), page_number=page.get("page_number"), start_id=next_id
        )
        sentences.extend(page_sentences)
        next_id += len(page_sentences)
    return sentences


def chunk_sentences(sentences: list[Mapping[str, Any]], *, chunk_size: int, overlap: int) -> list[dict[str, Any]]:
    """Create overlapping chunks measured in sentences, with stable source IDs."""

    if chunk_size < 1:
        raise ValueError("chunk_size must be at least one sentence.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size.")
    chunks: list[dict[str, Any]] = []
    step = chunk_size - overlap
    for start in range(0, len(sentences), step):
        group = sentences[start : start + chunk_size]
        if not group:
            break
        page_numbers = [item.get("page_number") for item in group if item.get("page_number") is not None]
        chunks.append(
            {
                "chunk_id": len(chunks) + 1,
                "text": " ".join(str(item["text"]) for item in group),
                "sentence_start": group[0]["sentence_id"],
                "sentence_end": group[-1]["sentence_id"],
                "page_start": min(page_numbers) if page_numbers else None,
                "page_end": max(page_numbers) if page_numbers else None,
            }
        )
        if start + chunk_size >= len(sentences):
            break
    return chunks


def chunk_result(result: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    """Attach sentence and chunk artifacts to a preprocessed result."""

    updated = dict(result)
    pages = updated.get("pages", [])
    sentences = segment_pages(pages) if pages else segment_sentences(str(updated.get("cleaned_text", "")))
    updated["sentences"] = sentences
    updated["chunks"] = chunk_sentences(
        sentences,
        chunk_size=int(config.get("chunk_size_sentences", 8)),
        overlap=int(config.get("chunk_overlap_sentences", 2)),
    )
    return updated
