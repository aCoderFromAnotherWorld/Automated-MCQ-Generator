"""Tests for SQuAD-style training-record preparation."""

import unittest

from training.prepare_squad import iter_json_examples


class PrepareSquadTests(unittest.TestCase):
    def test_answerable_examples_become_qg_records(self) -> None:
        payload = {
            "data": [
                {
                    "paragraphs": [
                        {
                            "context": "TCP provides reliable delivery.",
                            "qas": [
                                {
                                    "id": "answerable",
                                    "question": "What provides reliable delivery?",
                                    "answers": [{"text": "TCP", "answer_start": 0}],
                                },
                                {
                                    "id": "impossible",
                                    "question": "What is absent?",
                                    "is_impossible": True,
                                    "answers": [],
                                },
                            ],
                        }
                    ]
                }
            ]
        }
        records = list(iter_json_examples(payload))
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["answer"], "TCP")
        self.assertIn("context:", records[0]["input_text"])
