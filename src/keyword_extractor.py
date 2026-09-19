"""Replaceable RAKE and optional spaCy candidate extraction (M06)."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any, Iterable, Mapping


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "is", "it", "of",
    "on", "or", "that", "the", "this", "to", "was", "were", "with", "which", "who", "will",
}
_WORD = re.compile(r"[A-Za-z][A-Za-z0-9'-]*")


def normalize_candidate(text: str) -> str:
    return " ".join(_WORD.findall(text.lower()))


def _valid_phrase(text: str, *, min_chars: int, max_words: int) -> bool:
    normalized = normalize_candidate(text)
    words = normalized.split()
    return bool(normalized) and len(normalized) >= min_chars and len(words) <= max_words and not all(word in STOPWORDS for word in words)


def rake_phrases(text: str, *, min_chars: int = 3, max_words: int = 6) -> dict[str, float]:
    """Small dependency-free RAKE implementation returning phrase scores."""

    candidate_words: list[list[str]] = []
    current: list[str] = []
    for token in re.findall(r"[A-Za-z][A-Za-z0-9'-]*|[.!?,;:()]", text.lower()):
        if token in STOPWORDS or not _WORD.fullmatch(token):
            if current:
                candidate_words.append(current)
                current = []
        else:
            current.append(token)
    if current:
        candidate_words.append(current)

    frequency: Counter[str] = Counter()
    degree: Counter[str] = Counter()
    for phrase in candidate_words:
        length = len(phrase)
        for word in phrase:
            frequency[word] += 1
            degree[word] += length - 1
    word_score = {word: (degree[word] + frequency[word]) / frequency[word] for word in frequency}
    phrases: dict[str, float] = {}
    for words in candidate_words:
        phrase = " ".join(words)
        if _valid_phrase(phrase, min_chars=min_chars, max_words=max_words):
            phrases[phrase] = max(phrases.get(phrase, 0.0), sum(word_score[word] for word in words))
    return phrases


def _spacy_candidates(text: str, nlp: Any) -> list[dict[str, Any]]:
    doc = nlp(text)
    records: list[dict[str, Any]] = []
    for chunk in getattr(doc, "noun_chunks", []):
        records.append(
            {
                "text": chunk.text,
                "source": "spacy_noun_chunk",
                "rake_score": 0.0,
                "is_noun_phrase": True,
                "is_named_entity": False,
                "entity_label": None,
            }
        )
    for entity in getattr(doc, "ents", []):
        records.append(
            {
                "text": entity.text,
                "source": "spacy_ner",
                "rake_score": 0.0,
                "is_noun_phrase": False,
                "is_named_entity": True,
                "entity_label": entity.label_,
            }
        )
    return records


def extract_candidates(chunks: Iterable[Mapping[str, Any]], config: Mapping[str, Any], *, nlp: Any | None = None) -> list[dict[str, Any]]:
    """Return the fixed candidate dictionary contract, never bare strings."""

    min_chars = int(config.get("min_candidate_chars", 3))
    max_words = int(config.get("max_candidate_words", 6))
    merged: dict[tuple[int | None, str], dict[str, Any]] = {}
    for chunk in chunks:
        chunk_id = chunk.get("chunk_id")
        records = [
            {
                "text": phrase,
                "source": "rake",
                "rake_score": score,
                "is_noun_phrase": False,
                "is_named_entity": False,
                "entity_label": None,
            }
            for phrase, score in rake_phrases(str(chunk.get("text", "")), min_chars=min_chars, max_words=max_words).items()
        ]
        if nlp is not None:
            records.extend(_spacy_candidates(str(chunk.get("text", "")), nlp))
        for record in records:
            if not _valid_phrase(record["text"], min_chars=min_chars, max_words=max_words):
                continue
            key = (chunk_id, normalize_candidate(record["text"]))
            existing = merged.get(key)
            if existing is None:
                merged[key] = {**record, "text": " ".join(record["text"].split()), "chunk_id": chunk_id}
                continue
            existing["rake_score"] = max(float(existing["rake_score"]), float(record["rake_score"]))
            existing["is_noun_phrase"] = bool(existing["is_noun_phrase"] or record["is_noun_phrase"])
            existing["is_named_entity"] = bool(existing["is_named_entity"] or record["is_named_entity"])
            existing["entity_label"] = existing["entity_label"] or record["entity_label"]
            if existing["source"] != "rake" and record["source"] == "rake":
                existing["source"] = "rake"
    return sorted(merged.values(), key=lambda item: (item["chunk_id"] or 0, item["text"].lower()))
