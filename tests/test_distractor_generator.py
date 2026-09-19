"""Focused tests for M11 textbook-grounded distractor generation."""

import json
import unittest

from src.distractor_generator import generate_distractors


class DistractorGeneratorTests(unittest.TestCase):
    def test_removes_answer_deduplicates_and_selects_three(self) -> None:
        question = "Which protocol is reliable?"
        candidates = [
            {"text": "TCP", "is_noun_phrase": True},
            {"text": "UDP", "is_noun_phrase": True},
            {"text": "udp", "is_noun_phrase": True},
            {"text": "IP", "is_noun_phrase": True},
            {"text": "ARP", "is_noun_phrase": True},
        ]

        def similarity(left: str, right: str) -> float:
            if right == "TCP":
                return {"UDP": 0.9, "IP": 0.7, "ARP": 0.5}.get(left, 0.1)
            return 0.1

        result = generate_distractors(
            question,
            "TCP",
            "TCP is reliable. UDP is connectionless. IP routes packets. ARP maps addresses.",
            candidates,
            answer_record={"is_noun_phrase": True},
            similarity=similarity,
        )
        self.assertTrue(result["valid"])
        self.assertEqual(len(result["selected"]), 3)
        self.assertNotIn("TCP", result["selected"])
        rejected = {item["text"]: item["reasons"] for item in result["rejected"]}
        self.assertIn("equals_answer", rejected["TCP"])
        self.assertIn("duplicate_candidate", rejected["udp"])
        json.dumps(result)

    def test_records_required_combined_semantic_score(self) -> None:
        result = generate_distractors(
            "Which protocol is reliable?",
            "TCP",
            "TCP is reliable. UDP is connectionless.",
            ["UDP", "IP", "ARP"],
            similarity=lambda left, right: 0.8 if right == "TCP" else 0.4,
        )
        self.assertAlmostEqual(result["scored_candidates"][0]["semantic_score"], 0.6)
        self.assertIn("score", result["scored_candidates"][0])

    def test_type_match_is_preferred_and_recorded(self) -> None:
        candidates = [
            {"text": "UDP", "is_noun_phrase": True},
            {"text": "routing table", "is_noun_phrase": False},
            {"text": "IP", "is_noun_phrase": True},
            {"text": "ARP", "is_noun_phrase": True},
        ]
        result = generate_distractors(
            "Which protocol is reliable?",
            "TCP",
            "TCP is reliable. UDP is connectionless. IP routes packets. ARP maps addresses.",
            candidates,
            answer_record={"is_noun_phrase": True},
            similarity=lambda left, right: 0.5,
        )
        typed = {item["text"]: item["type_match"] for item in result["scored_candidates"]}
        self.assertTrue(typed["UDP"])
        self.assertFalse(typed["routing table"])

    def test_contextually_supported_candidate_is_rejected_as_critical(self) -> None:
        question = "Which protocol is connectionless?"
        result = generate_distractors(
            question,
            "TCP",
            "TCP is reliable. UDP is connectionless. IP routes packets. ARP maps addresses.",
            ["UDP", "IP", "ARP", "HTTP"],
            context_similarity=lambda left, right: 0.9 if left == "UDP" else 0.1,
            similarity=lambda left, right: 0.4,
        )
        rejected = next(item for item in result["rejected"] if item["text"] == "UDP")
        self.assertIn("contextually_supported_critical_failure", rejected["reasons"])
        self.assertEqual(rejected["context_similarity"], 0.9)

    def test_fewer_than_three_qualifying_candidates_invalidates_pool(self) -> None:
        result = generate_distractors(
            "Which protocol is reliable?",
            "TCP",
            "TCP is reliable. UDP is connectionless.",
            ["UDP", "IP"],
            similarity=lambda left, right: 0.5,
        )
        self.assertFalse(result["valid"])
        self.assertIn("fewer_than_three_valid_distractors", result["reasons"])


if __name__ == "__main__":
    unittest.main()
