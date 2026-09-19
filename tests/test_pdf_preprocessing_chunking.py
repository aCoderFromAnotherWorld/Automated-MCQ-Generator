"""Tests for M03, M04, and M05 without requiring a real uploaded document."""

import unittest
from unittest.mock import patch

from config import CONFIG
from src.chunking import chunk_result, chunk_sentences, segment_sentences
from src.pdf_processor import PDFExtractionError, extract_pages
from src.preprocessing import clean_text, preprocess_result


class _Page:
    def __init__(self, text: str) -> None:
        self.text = text

    def extract_text(self) -> str:
        return self.text


class _Reader:
    pages = [_Page("TCP is reliable."), _Page("UDP is connectionless.")]


class PdfPreprocessChunkTests(unittest.TestCase):
    def test_pdf_extraction_returns_page_number_and_text(self) -> None:
        with patch("pypdf.PdfReader", return_value=_Reader()):
            pages = extract_pages(b"pdf-bytes")
        self.assertEqual(pages[0], {"page_number": 1, "text": "TCP is reliable."})
        self.assertEqual(pages[1]["page_number"], 2)

    def test_invalid_pdf_is_a_clear_error(self) -> None:
        with self.assertRaises(PDFExtractionError):
            extract_pages(b"not a pdf")

    def test_cleaning_keeps_sentence_punctuation_and_preserves_raw_text(self) -> None:
        result = preprocess_result({"raw_text": "reli-\nable.  TCP works!"})
        self.assertEqual(result["raw_text"], "reli-\nable.  TCP works!")
        self.assertEqual(result["cleaned_text"], "reliable. TCP works!")

    def test_chunking_has_overlap_and_metadata(self) -> None:
        sentences = segment_sentences("One. Two. Three. Four. Five.")
        chunks = chunk_sentences(sentences, chunk_size=3, overlap=1)
        self.assertEqual(chunks[0]["sentence_end"], 3)
        self.assertEqual(chunks[1]["sentence_start"], 3)
        self.assertEqual(chunks[0]["chunk_id"], 1)

    def test_result_chunking_uses_configured_defaults(self) -> None:
        result = chunk_result({"cleaned_text": "One. Two. Three."}, CONFIG)
        self.assertEqual(len(result["sentences"]), 3)
        self.assertEqual(result["chunks"][0]["page_start"], None)
