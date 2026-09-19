"""Focused tests for M08 representations and similarity utilities."""

import unittest

import numpy as np

from src.embeddings import (
    add_tfidf_scores,
    candidate_tfidf_scores,
    cosine_similarity,
    encode,
    fit_tfidf,
    load_sentence_transformer,
    semantic_similarity,
)


class _FakeEncoder:
    def encode(self, texts: list[str], *, convert_to_numpy: bool) -> np.ndarray:
        vectors = []
        for text in texts:
            normalized = text.lower()
            vectors.append(
                [
                    float("tcp" in normalized),
                    float("udp" in normalized),
                    float("reliable" in normalized),
                ]
            )
        return np.asarray(vectors, dtype=float)


class EmbeddingsTests(unittest.TestCase):
    def test_tfidf_fits_known_terms_and_scores_candidates(self) -> None:
        chunks = [
            {"text": "TCP provides reliable delivery."},
            {"text": "UDP provides connectionless delivery."},
        ]
        vectorizer, matrix = fit_tfidf(chunks)
        self.assertEqual(matrix.shape[0], 2)
        self.assertIn("tcp", vectorizer.vocabulary_)
        self.assertIn("udp", vectorizer.vocabulary_)

        scores = candidate_tfidf_scores(
            chunks,
            [{"text": "TCP"}, {"text": "UDP"}, {"text": "unknown-term"}],
        )
        self.assertEqual(len(scores), 3)
        self.assertGreater(scores[0], 0.0)
        self.assertGreater(scores[1], 0.0)
        self.assertEqual(scores[2], 0.0)

    def test_add_tfidf_scores_preserves_candidate_records(self) -> None:
        candidates = [{"text": "TCP", "chunk_id": 1, "rake_score": 2.0}]
        scored = add_tfidf_scores([{"text": "TCP is reliable."}], candidates)
        self.assertEqual(scored[0]["text"], "TCP")
        self.assertEqual(scored[0]["chunk_id"], 1)
        self.assertIn("tfidf_score", scored[0])
        self.assertNotIn("tfidf_score", candidates[0])

    def test_semantic_encoding_and_similarity_are_injectable(self) -> None:
        model = _FakeEncoder()
        vectors = encode(["TCP is reliable", "UDP is different"], model=model)
        self.assertEqual(vectors.shape, (2, 3))
        self.assertAlmostEqual(semantic_similarity("TCP", "TCP", model=model), 1.0)
        self.assertEqual(cosine_similarity([1, 0], [0, 1]), 0.0)
        self.assertGreater(semantic_similarity("TCP is reliable", "TCP", model=model), 0.5)

    def test_empty_encoding_does_not_load_a_model(self) -> None:
        self.assertEqual(encode([]).shape, (0, 0))

    def test_sentence_transformer_loader_is_cached(self) -> None:
        self.assertEqual(load_sentence_transformer.cache_info().maxsize, 2)


if __name__ == "__main__":
    unittest.main()
