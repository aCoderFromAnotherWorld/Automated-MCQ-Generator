"""Sentence segmentation and overlapping, source-mapped chunks (M05)."""

from __future__ import annotations

import re
from typing import Any, Iterable, Mapping

from src.segmentation import segment_pages, segment_sentences, segment_text


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


def _paragraph_texts(cleaned_text: str) -> list[str]:
    """Split cleaned text on preserved blank-line paragraph breaks."""

    return [part.strip() for part in re.split(r"\n\s*\n", cleaned_text) if part.strip()]


def _sentence_groups(cleaned_text: str, pages: Iterable[Mapping[str, Any]]) -> list[list[dict[str, Any]]]:
    """Build sentence groups that must not be merged into the same chunk.

    Each PDF page, and each paragraph of plain pasted text, becomes its own
    group so chunks never mix unrelated topics.
    """

    page_list = list(pages)
    if page_list:
        groups: list[list[dict[str, Any]]] = []
        next_id = 1
        for page in page_list:
            page_sentences = segment_text(
                str(page.get("text", "")),
                page_number=page.get("page_number"),
                start_id=next_id,
            )
            if page_sentences:
                groups.append(page_sentences)
                next_id += len(page_sentences)
        return groups

    paragraphs = _paragraph_texts(cleaned_text)
    if len(paragraphs) <= 1:
        sentences = segment_sentences(cleaned_text)
        return [sentences] if sentences else []

    groups = []
    next_id = 1
    for paragraph in paragraphs:
        paragraph_sentences = segment_text(paragraph, start_id=next_id)
        if paragraph_sentences:
            groups.append(paragraph_sentences)
            next_id += len(paragraph_sentences)
    return groups


def chunk_result(result: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    """Attach sentence and chunk artifacts to a preprocessed result.

    Chunks are built inside page/paragraph boundaries, so a long multi-topic
    passage yields topically coherent chunks instead of blended ones.
    """

    updated = dict(result)
    groups = _sentence_groups(str(updated.get("cleaned_text", "")), updated.get("pages", []))
    sentences = [sentence for group in groups for sentence in group]
    chunk_size = int(config.get("chunk_size_sentences", 8))
    overlap = int(config.get("chunk_overlap_sentences", 2))
    chunks: list[dict[str, Any]] = []
    for group in groups:
        for chunk in chunk_sentences(group, chunk_size=chunk_size, overlap=overlap):
            chunk["chunk_id"] = len(chunks) + 1
            chunks.append(chunk)
    updated["sentences"] = sentences
    updated["chunks"] = chunks
    return updated
