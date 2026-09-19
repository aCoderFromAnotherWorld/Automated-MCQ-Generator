"""Smoke tests for repository-scaffold imports and UI result contracts."""

import unittest

from config import CONFIG
from ui.support import normalize_result


class ScaffoldTests(unittest.TestCase):
    def test_config_has_local_training_checkpoint(self) -> None:
        self.assertEqual(CONFIG["question_model_training_mode"], "local")
        self.assertTrue(CONFIG["question_model_checkpoint"])

    def test_ui_result_has_every_required_key(self) -> None:
        result = normalize_result({}, CONFIG)
        self.assertIn("questions", result)
        self.assertIn("generation_records", result)
        self.assertEqual(result["config"]["question_model"], CONFIG["question_model"])
