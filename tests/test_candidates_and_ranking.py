"""Tests for M06 candidate extraction and M07 ranking/diversity."""

import unittest

from config import CONFIG
from src.keyword_extractor import extract_candidates
from src.ranking import rank_candidates


class CandidateAndRankingTests(unittest.TestCase):
    def test_extractor_returns_fixed_candidate_dictionary_shape(self) -> None:
        candidates = extract_candidates(
            [{"chunk_id": 1, "text": "Transmission Control Protocol provides reliable delivery."}],
            CONFIG,
        )
        self.assertTrue(candidates)
        required = {"text", "source", "rake_score", "chunk_id", "is_noun_phrase", "is_named_entity", "entity_label"}
        self.assertTrue(required.issubset(candidates[0]))
        self.assertFalse(isinstance(candidates[0], str))

    def test_ranking_removes_near_duplicate_candidates(self) -> None:
        candidates = [
            {"text": "Transmission Control Protocol", "rake_score": 10, "chunk_id": 1, "is_noun_phrase": True, "is_named_entity": False},
            {"text": "Control Protocol", "rake_score": 9, "chunk_id": 1, "is_noun_phrase": True, "is_named_entity": False},
            {"text": "User Datagram Protocol", "rake_score": 8, "chunk_id": 1, "is_noun_phrase": True, "is_named_entity": False},
        ]
        ranked = rank_candidates(candidates, [{"text": "Transmission Control Protocol and User Datagram Protocol."}], CONFIG)
        labels = [candidate["text"] for candidate in ranked]
        self.assertIn("Transmission Control Protocol", labels)
        self.assertIn("User Datagram Protocol", labels)
        self.assertNotIn("Control Protocol", labels)
        self.assertTrue(all("final_score" in candidate for candidate in ranked))

    def test_capitalised_entity_is_kept_intact_instead_of_fragments(self) -> None:
        candidates = extract_candidates(
            [{"chunk_id": 1, "text": "It has a coastline along the Bay of Bengal to its south."}],
            CONFIG,
        )
        labels = [candidate["text"].lower() for candidate in candidates]
        self.assertIn("bay of bengal", labels)
        # Fragments that only occur inside the entity phrase are dropped.
        self.assertNotIn("bay", labels)
        phrase = next(candidate for candidate in candidates if candidate["text"].lower() == "bay of bengal")
        self.assertTrue(phrase["is_named_entity"])
        self.assertEqual(phrase["entity_label"], "PROPER")

    def test_standalone_word_survives_when_it_also_occurs_outside_a_phrase(self) -> None:
        candidates = extract_candidates(
            [{"chunk_id": 1, "text": "Bengal became a commercial centre. The Bay of Bengal is to the south."}],
            CONFIG,
        )
        labels = [candidate["text"].lower() for candidate in candidates]
        self.assertIn("bay of bengal", labels)
        self.assertIn("bengal", labels)
