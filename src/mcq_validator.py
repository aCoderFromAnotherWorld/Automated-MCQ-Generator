"""MCQ assembly and final validation (M12)."""

from __future__ import annotations

import re
from typing import Any, Callable, Iterable, Mapping

from src.embeddings import pairwise_cosine_similarity
from src.validator import validate_question


def _normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def _clean(text: Any) -> str:
    return " ".join(str(text).split())


def _selected_distractors(record: Mapping[str, Any]) -> list[str]:
    selected = record.get("selected", record.get("distractors", []))
    return [_clean(value) for value in selected]


def _semantic_duplicate(
    question: str,
    previous_questions: Iterable[str],
    similarity: Callable[[str, str], float] | None,
    threshold: float,
) -> tuple[bool, float | None]:
    normalized = _normalise(question)
    previous_texts = [_clean(previous) for previous in previous_questions]
    if normalized and any(normalized == _normalise(previous_text) for previous_text in previous_texts):
        return True, 1.0
    if not previous_texts:
        return False, None
    if similarity is not None:
        scores = [float(similarity(question, previous_text)) for previous_text in previous_texts]
    else:
        scores = pairwise_cosine_similarity(question, previous_texts)
    for score in scores:
        if score >= threshold:
            return True, score
    return False, None


def _token_overlap(left: str, right: str) -> float:
    """Token F1 between two answer strings (used by round-trip verification)."""

    left_tokens = set(_normalise(left).split())
    right_tokens = set(_normalise(right).split())
    if not left_tokens or not right_tokens:
        return 0.0
    shared = len(left_tokens & right_tokens)
    return 2 * shared / (len(left_tokens) + len(right_tokens))


def _answer_matches(intended: str, predicted: str, threshold: float) -> bool:
    """True when a predicted answer reproduces the intended answer."""

    intended_tokens = set(_normalise(intended).split())
    predicted_tokens = set(_normalise(predicted).split())
    if not intended_tokens or not predicted_tokens:
        return False
    if intended_tokens <= predicted_tokens or predicted_tokens <= intended_tokens:
        return True
    return _token_overlap(intended, predicted) >= threshold


def validate_mcq(
    question: str,
    answer: str,
    context: str,
    distractor_record: Mapping[str, Any],
    *,
    chunk_id: int | None = None,
    source_page: int | None = None,
    generator: str | None = None,
    previous_questions: Iterable[str] = (),
    similarity: Callable[[str, str], float] | None = None,
    duplicate_threshold: float = 0.85,
    question_min_words: int = 3,
    question_max_words: int = 40,
    predicted_answer: str | None = None,
    answer_match_threshold: float = 0.5,
) -> dict[str, Any]:
    """Validate one generated MCQ and return a final JSON-safe record.

    Critical failures are rejecting conditions: unsupported answers, fewer than
    three distractors, contextually supported distractors, and duplicate
    options never produce a valid ``mcq``.
    """

    question_text = _clean(question)
    answer_text = _clean(answer)
    distractors = _selected_distractors(distractor_record)
    reasons: list[str] = []

    question_validation = validate_question(
        question_text,
        answer_text,
        context,
        previous_questions=(),
        similarity=similarity,
        min_words=question_min_words,
        max_words=question_max_words,
    )
    if not question_validation["valid"]:
        reasons.extend(f"question_{reason}" for reason in question_validation["reasons"])
    if not question_validation["checks"].get("answer_present", False):
        reasons.append("answer_not_supported_by_context")

    duplicate, duplicate_score = _semantic_duplicate(
        question_text, previous_questions, similarity, duplicate_threshold
    )
    if duplicate:
        reasons.append("duplicate_question")

    option_keys = [_normalise(answer_text)] + [_normalise(item) for item in distractors]
    duplicate_options = len(option_keys) != len(set(option_keys)) or not all(option_keys)
    if duplicate_options:
        reasons.append("duplicate_options")

    # A question whose stem already contains the answer is a giveaway.
    answer_tokens = set(_normalise(answer_text).split())
    question_tokens = set(_normalise(question_text).split())
    answer_in_question = bool(answer_tokens) and answer_tokens <= question_tokens
    if answer_in_question:
        reasons.append("answer_appears_in_question")

    # Round-trip verification: asking the question back must reproduce the answer.
    answer_verified: bool | None = None
    if predicted_answer is not None:
        answer_verified = _answer_matches(answer_text, str(predicted_answer), answer_match_threshold)
        if not answer_verified:
            reasons.append("answer_not_verified")

    enough_distractors = len(distractors) == 3 and bool(distractor_record.get("valid", True))
    if not enough_distractors:
        reasons.append("fewer_than_three_valid_distractors")

    critical_rejections = [
        rejection
        for rejection in distractor_record.get("rejected", [])
        if "contextually_supported_critical_failure" in rejection.get("reasons", [])
    ]
    if critical_rejections:
        reasons.append("contextually_supported_distractor")

    checks = {
        "question_valid": question_validation["valid"],
        "answer_supported": question_validation["checks"].get("answer_present", False),
        "source_relevant": question_validation["checks"].get("context_relevance", False),
        "semantic_duplicate": duplicate,
        "unique_options": not duplicate_options,
        "three_valid_distractors": enough_distractors,
        "distractors_contextually_incorrect": not bool(critical_rejections),
        "answer_not_in_question": not answer_in_question,
        "answer_verified": answer_verified,
    }
    validation = {
        "valid": not reasons,
        "reasons": reasons,
        "checks": checks,
        "question": question_validation,
        "duplicate_score": duplicate_score,
        "predicted_answer": predicted_answer,
        "critical_failures": [
            reason
            for reason in reasons
            if reason
            in {
                "answer_not_supported_by_context",
                "fewer_than_three_valid_distractors",
                "contextually_supported_distractor",
                "duplicate_options",
                "answer_appears_in_question",
                "answer_not_verified",
            }
        ],
    }

    result: dict[str, Any] = {"valid": not reasons, "validation": validation, "mcq": None}
    if not reasons:
        result["mcq"] = {
            "question": question_text,
            "answer": answer_text,
            "correct_answer": answer_text,
            "distractors": distractors,
            "context": _clean(context),
            "source_context": _clean(context),
            "chunk_id": chunk_id,
            "source_page": source_page,
            "generator": generator,
        }
    return result


def assemble_mcq(
    question: str,
    answer: str,
    context: str,
    distractor_record: Mapping[str, Any],
    **options: Any,
) -> dict[str, Any]:
    """Assemble and validate one MCQ using the M10/M11 result contracts."""

    return validate_mcq(question, answer, context, distractor_record, **options)
