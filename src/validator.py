"""Question-level validation before MCQ assembly (M10)."""

from __future__ import annotations

import re
from typing import Any, Callable, Iterable

from src.embeddings import semantic_similarity
from src.keyword_extractor import STOPWORDS as _LEXICAL_STOPWORDS


def _normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _stem(word: str) -> str:
    """Light suffix stemmer so 'provides'/'provide' and 'prevents'/'prevent' match.

    Only strips a suffix when at least four characters remain, so short words
    and unrelated words are never collapsed into false matches.
    """

    if word.endswith("ies") and len(word) - 3 >= 4:
        return word[:-3] + "y"
    if word.endswith("ing") and len(word) - 3 >= 4:
        return word[:-3]
    if word.endswith("ed") and len(word) - 2 >= 4:
        return word[:-2]
    if word.endswith(("ses", "xes", "zes", "ches", "shes")) and len(word) - 2 >= 4:
        return word[:-2]
    if word.endswith("s") and len(word) - 1 >= 4:
        return word[:-1]
    return word


def validate_question(
    question: str,
    answer: str,
    context: str,
    *,
    previous_questions: Iterable[str] = (),
    similarity: Callable[[str, str], float] | None = None,
    min_words: int = 3,
    max_words: int = 40,
    relevance_threshold: float = 0.25,
    duplicate_threshold: float = 0.85,
    semantic_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> dict[str, Any]:
    """Return validity, named checks, reasons, and measured relevance."""

    reasons: list[str] = []
    checks: dict[str, bool] = {}
    question_text = " ".join(str(question).split())
    answer_text = " ".join(str(answer).split())
    context_text = " ".join(str(context).split())

    checks["non_empty"] = bool(question_text)
    if not checks["non_empty"]:
        reasons.append("question_empty")
    word_count = len(question_text.split())
    checks["length"] = min_words <= word_count <= max_words
    if not checks["length"]:
        reasons.append("question_length_invalid")
    checks["answer_present"] = bool(answer_text) and _normalise(answer_text) in _normalise(context_text)
    if not checks["answer_present"]:
        reasons.append("answer_not_supported_by_context")

    normalized_question = _normalise(question_text)
    duplicate_score: float | None = None
    duplicate = False
    for previous in previous_questions:
        previous_text = " ".join(str(previous).split())
        if normalized_question and normalized_question == _normalise(previous_text):
            duplicate = True
            duplicate_score = 1.0
            break
        if question_text and previous_text:
            score = float(
                similarity(question_text, previous_text)
                if similarity is not None
                else semantic_similarity(question_text, previous_text, model_name=semantic_model_name)
            )
            if score >= duplicate_threshold:
                duplicate = True
                duplicate_score = score
                break
    checks["not_duplicate"] = bool(normalized_question) and not duplicate
    if not checks["not_duplicate"]:
        reasons.append("duplicate_question")

    relevance_score: float | None = None
    if question_text and context_text:
        if similarity is None:
            relevance_score = semantic_similarity(
                question_text,
                context_text,
                model_name=semantic_model_name,
            )
        else:
            relevance_score = float(similarity(question_text, context_text))
    checks["context_relevance"] = relevance_score is not None and relevance_score >= relevance_threshold
    if not checks["context_relevance"]:
        reasons.append("question_not_relevant_to_context")

    question_content = set(re.findall(r"[a-z0-9]+", question_text.lower())) - _QUESTION_STOPWORDS
    context_words = set(re.findall(r"[a-z0-9]+", context_text.lower()))
    answer_words = set(re.findall(r"[a-z0-9]+", answer_text.lower()))
    # Compare on light stems so morphological variants (provides/provide,
    # prevents/prevent, processes/process) count as supported information.
    context_stems = {_stem(word) for word in context_words}
    answer_stems = {_stem(word) for word in answer_words}
    unsupported_words = sorted(word for word in question_content if _stem(word) not in context_stems and _stem(word) not in answer_stems)
    checks["supported_information"] = not unsupported_words
    if unsupported_words:
        reasons.append("unsupported_information")

    return {
        "valid": not reasons,
        "reasons": reasons,
        "checks": checks,
        "scores": {"context_relevance": relevance_score},
        "duplicate_score": duplicate_score,
        "unsupported_words": unsupported_words,
        "question": question_text,
        "answer": answer_text,
    }


def _lexical_relevance(question: str, context: str) -> float:
    question_words = set(re.findall(r"[a-z0-9]+", question.lower()))
    context_words = set(re.findall(r"[a-z0-9]+", context.lower()))
    return len(question_words & context_words) / len(question_words) if question_words else 0.0


_QUESTION_STOPWORDS = {
    "a", "an", "and", "are", "be", "does", "do", "did", "how", "is", "of", "the",
    "protocol", "to", "what", "when", "where", "which", "who", "whom", "whose", "why",
    # Generic question-frame words that carry no verifiable content.
    "name", "type", "kind", "form", "purpose", "role", "called", "known", "main",
    "becomes", "become", "became",
    # Common irregular verbs (the light stemmer cannot normalise these).
    "come", "came", "go", "went", "give", "gave", "take", "took", "make", "made",
    "get", "got", "say", "said", "see", "saw", "know", "knew", "lead", "led",
    "fight", "fought", "begin", "began", "bring", "brought", "hold", "held",
    "win", "won", "rise", "rose", "run", "ran", "write", "wrote", "speak", "spoke",
} | set(_LEXICAL_STOPWORDS)


# Explicit alias for callers that use the module's task terminology.
validate = validate_question
