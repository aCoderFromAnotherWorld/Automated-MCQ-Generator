"""Textbook-grounded distractor generation and validation (M11)."""

from __future__ import annotations

import re
from typing import Any, Callable, Iterable, Mapping

from src.embeddings import pairwise_cosine_similarity, semantic_similarity


def _normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def _candidate_text(candidate: Any) -> str:
    return str(candidate.get("text", "")) if isinstance(candidate, Mapping) else str(candidate)


def _type_match(candidate: Mapping[str, Any], answer_record: Mapping[str, Any] | None) -> bool:
    if not answer_record:
        return False
    if answer_record.get("is_named_entity"):
        return bool(candidate.get("is_named_entity")) and candidate.get("entity_label") == answer_record.get("entity_label")
    if answer_record.get("is_noun_phrase"):
        return bool(candidate.get("is_noun_phrase"))
    return False


def generate_distractors(
    question: str,
    answer: str,
    context: str,
    candidates: Iterable[Any],
    *,
    answer_record: Mapping[str, Any] | None = None,
    model: Any | None = None,
    similarity: Callable[[str, str], float] | None = None,
    context_similarity: Callable[[str, str], float] | None = None,
    context_support_threshold: float = 0.75,
    limit: int = 3,
) -> dict[str, Any]:
    """Select up to three plausible, source-grounded incorrect concepts.

    The returned record keeps every rejection reason so a caller can reject an
    MCQ when the required three valid distractors are unavailable.
    """

    rejected: list[dict[str, Any]] = []
    scored: list[dict[str, Any]] = []
    seen: set[str] = set()
    answer_key = _normalise(answer)

    def score(left: str, right: str) -> float:
        if similarity is not None:
            return float(similarity(left, right))
        return float(semantic_similarity(left, right, model=model))

    def support_score(candidate_text: str) -> float:
        scorer = context_similarity or similarity
        if scorer is not None:
            return float(scorer(candidate_text, question))
        return float(semantic_similarity(candidate_text, question, model=model))

    # Pre-filter candidates first, then compute the expensive semantic scores in
    # two batched passes (against the answer and against the question) instead of
    # one pairwise model call per candidate.
    filtered: list[dict[str, Any]] = []
    normalized_context = _normalise(context)
    for candidate in candidates:
        text = " ".join(_candidate_text(candidate).split())
        key = _normalise(text)
        reasons: list[str] = []
        if not key or key == answer_key:
            reasons.append("equals_answer")
        elif key in seen:
            reasons.append("duplicate_candidate")
        seen.add(key)
        needs_support = (not reasons) and (key in normalized_context)
        if reasons:
            rejected.append({"text": text, "reasons": reasons})
            continue
        filtered.append(
            {
                "text": text,
                "key": key,
                "candidate": candidate,
                "needs_support": needs_support,
            }
        )

    texts = [item["text"] for item in filtered]
    if similarity is None and texts:
        answer_scores = pairwise_cosine_similarity(answer, texts, model=model)
        question_scores = pairwise_cosine_similarity(question, texts, model=model)
    else:
        answer_scores = [score(text, answer) for text in texts]
        question_scores = [score(text, question) for text in texts]

    for item, answer_score, question_score in zip(filtered, answer_scores, question_scores):
        text = item["text"]
        context_score: float | None = None
        reasons = []
        if item["needs_support"]:
            context_score = question_score if context_similarity is None else support_score(text)
            if context_score >= context_support_threshold:
                reasons.append("contextually_supported_critical_failure")
        if reasons:
            rejection = {"text": text, "reasons": reasons}
            if context_score is not None:
                rejection["context_similarity"] = round(context_score, 6)
            rejected.append(rejection)
            continue
        candidate_score = 0.5 * answer_score + 0.5 * question_score
        type_bonus = 0.05 if isinstance(item["candidate"], Mapping) and _type_match(item["candidate"], answer_record) else 0.0
        scored.append(
            {
                "text": text,
                "score": round(candidate_score + type_bonus, 6),
                "semantic_score": round(candidate_score, 6),
                "context_similarity": round(context_score, 6) if context_score is not None else None,
                "type_match": bool(type_bonus),
                "candidate": dict(item["candidate"]) if isinstance(item["candidate"], Mapping) else {"text": text},
            }
        )

    selected = sorted(scored, key=lambda item: (-item["score"], item["text"].lower()))[:limit]
    return {
        "question": question,
        "answer": answer,
        "candidate_pool": [item["text"] for item in scored] + [item["text"] for item in rejected],
        "scored_candidates": sorted(scored, key=lambda item: (-item["score"], item["text"].lower())),
        "selected": [item["text"] for item in selected],
        "rejected": rejected,
        "valid": len(selected) == limit,
        "reasons": [] if len(selected) == limit else ["fewer_than_three_valid_distractors"],
    }


select_distractors = generate_distractors
