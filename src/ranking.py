"""Explainable candidate ranking and diversity filtering (M07)."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Iterable, Mapping

from src.keyword_extractor import STOPWORDS, normalize_candidate


def _normalise(values: list[float]) -> list[float]:
    if not values:
        return []
    low, high = min(values), max(values)
    if low == high:
        return [1.0 if high else 0.0 for _ in values]
    return [(value - low) / (high - low) for value in values]


def _token_overlap(left: str, right: str) -> float:
    left_tokens = set(normalize_candidate(left).split())
    right_tokens = set(normalize_candidate(right).split())
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / min(len(left_tokens), len(right_tokens))


def _hard_filter(candidate: Mapping[str, Any], config: Mapping[str, Any]) -> bool:
    text = normalize_candidate(str(candidate.get("text", "")))
    words = text.split()
    return bool(text) and len(text) >= int(config.get("min_candidate_chars", 3)) and len(words) <= int(config.get("max_candidate_words", 6)) and not all(word in STOPWORDS for word in words)


def rank_candidates(candidates: Iterable[Mapping[str, Any]], chunks: Iterable[Mapping[str, Any]], config: Mapping[str, Any], *, limit: int | None = None) -> list[dict[str, Any]]:
    """Rank candidates with documented features, then enforce token diversity."""

    chunk_text = " ".join(str(chunk.get("text", "")).lower() for chunk in chunks)
    unique: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        if not _hard_filter(candidate, config):
            continue
        key = normalize_candidate(str(candidate["text"]))
        existing = unique.get(key)
        if existing is None or float(candidate.get("rake_score", 0.0)) > float(existing.get("rake_score", 0.0)):
            unique[key] = dict(candidate)

    records = list(unique.values())
    if not records:
        return []
    for record in records:
        pattern = re.compile(rf"\b{re.escape(normalize_candidate(str(record['text'])))}\b")
        record["frequency"] = len(pattern.findall(chunk_text))
        first_index = chunk_text.find(normalize_candidate(str(record["text"])))
        record["position"] = 1.0 / (1 + first_index) if first_index >= 0 else 0.0
        record.setdefault("tfidf_score", 0.0)  # populated by M08 without changing this contract

    feature_names = ("tfidf_score", "rake_score", "frequency", "is_noun_phrase", "is_named_entity", "position")
    normalised = {
        name: _normalise([float(record[name]) for record in records])
        for name in feature_names
    }
    weights = dict(config.get("ranking_weights", {}))
    defaults = {"tfidf_score": 0.35, "rake_score": 0.20, "frequency": 0.15, "is_noun_phrase": 0.10, "is_named_entity": 0.10, "position": 0.10}
    defaults.update(weights)
    for index, record in enumerate(records):
        record["final_score"] = round(sum(defaults[name] * normalised[name][index] for name in feature_names), 6)

    overlap_limit = float(config.get("max_candidate_overlap", 0.6))
    selected: list[dict[str, Any]] = []
    for record in sorted(records, key=lambda item: (-item["final_score"], item["text"].lower())):
        if all(_token_overlap(record["text"], chosen["text"]) < overlap_limit for chosen in selected):
            selected.append(record)
        if limit is not None and len(selected) >= limit:
            break
    return selected
