"""Sentence segmentation and overlapping, source-mapped chunks (M05)."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from src.segmentation import segment_pages, segment_sentences


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
