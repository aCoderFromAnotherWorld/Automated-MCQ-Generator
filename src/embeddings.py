"""TF-IDF and semantic representation utilities used by ranking and validation.

TF-IDF is the sparse, explainable representation for candidate importance and
ranking. Sentence Transformer vectors are the dense representation for
question-context relevance, distractor similarity, and duplicate detection.
These utilities support selection and validation; they do not replace the
FLAN-T5 question-generation model or its local training process.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


class EmbeddingError(RuntimeError):
    """Raised when an optional semantic embedding backend is unavailable."""


def _text(value: Any) -> str:
    if isinstance(value, Mapping):
        return str(value.get("text", ""))
    return str(value)


def fit_tfidf(texts: Iterable[Any]) -> tuple[Any, Any]:
    """Fit a TF-IDF vectorizer and return ``(vectorizer, matrix)``."""

    from sklearn.feature_extraction.text import TfidfVectorizer

    documents = [_text(value) for value in texts]
    if not any(document.strip() for document in documents):
        raise ValueError("At least one non-empty document is required for TF-IDF.")
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=True)
    return vectorizer, vectorizer.fit_transform(documents)


def candidate_tfidf_scores(
    chunks: Iterable[Mapping[str, Any]],
    candidates: Iterable[Mapping[str, Any]],
) -> list[float]:
    """Return each candidate's maximum TF-IDF term weight across chunk documents."""

    chunk_list = list(chunks)
    candidate_list = list(candidates)
    if not candidate_list:
        return []
    if not chunk_list:
        return [0.0] * len(candidate_list)
    vectorizer, matrix = fit_tfidf(chunk_list)
    scores: list[float] = []
    for candidate in candidate_list:
        phrase = _text(candidate)
        query = vectorizer.transform([phrase])
        scores.append(float(query.max()) if query.nnz else 0.0)
    return scores


def add_tfidf_scores(
    chunks: Iterable[Mapping[str, Any]],
    candidates: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Copy candidate records and attach the ranking contract's TF-IDF feature."""

    records = [dict(candidate) for candidate in candidates]
    for record, score in zip(records, candidate_tfidf_scores(chunks, records)):
        record["tfidf_score"] = round(score, 6)
    return records


@lru_cache(maxsize=2)
def load_sentence_transformer(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> Any:
    """Load the semantic encoder once, only when semantic encoding is requested."""

    try:
        from sentence_transformers import SentenceTransformer
    except ModuleNotFoundError as error:
        raise EmbeddingError(
            "Semantic similarity requires sentence-transformers. Install requirements.txt."
        ) from error
    try:
        return SentenceTransformer(model_name)
    except Exception as error:
        raise EmbeddingError(f"Could not load semantic embedding model '{model_name}'.") from error


def encode(
    texts: Sequence[str],
    *,
    model: Any | None = None,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> np.ndarray:
    """Encode texts using an injected model or the cached Sentence Transformer."""

    if not texts:
        return np.empty((0, 0), dtype=float)
    encoder = model or load_sentence_transformer(model_name)
    vectors = encoder.encode(list(texts), convert_to_numpy=True)
    return np.asarray(vectors, dtype=float)


def cosine_similarity(left: Any, right: Any) -> float:
    """Return cosine similarity, safely handling empty or zero vectors."""

    left_array = np.asarray(left, dtype=float).reshape(-1)
    right_array = np.asarray(right, dtype=float).reshape(-1)
    if left_array.size == 0 or right_array.size == 0 or left_array.size != right_array.size:
        return 0.0
    denominator = float(np.linalg.norm(left_array) * np.linalg.norm(right_array))
    return float(np.dot(left_array, right_array) / denominator) if denominator else 0.0


def semantic_similarity(
    left: str,
    right: str,
    *,
    model: Any | None = None,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> float:
    """Encode two texts and return their cosine similarity."""

    vectors = encode([left, right], model=model, model_name=model_name)
    return cosine_similarity(vectors[0], vectors[1])
