"""PDF extraction with page metadata; no preprocessing or NLP occurs here."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any, Mapping

from src.contracts import new_result


class PDFExtractionError(ValueError):
    """Raised for unreadable PDFs or PDFs without extractable text."""


def extract_pages(file_bytes: bytes) -> list[dict[str, Any]]:
    """Extract ordered, JSON-safe page records from a text-based PDF."""

    if not file_bytes:
        raise PDFExtractionError("PDF input is empty.")
    try:
        from pypdf import PdfReader

        reader = PdfReader(BytesIO(file_bytes))
        pages = [
            {"page_number": number, "text": (page.extract_text() or "").strip()}
            for number, page in enumerate(reader.pages, start=1)
        ]
    except Exception as error:
        raise PDFExtractionError("The uploaded file is not a readable PDF.") from error
    if not pages:
        raise PDFExtractionError("PDF contains no pages.")
    if not any(page["text"] for page in pages):
        raise PDFExtractionError("PDF has no extractable text; OCR is not supported.")
    return pages


def pdf_to_result(file_bytes: bytes, config: Mapping[str, Any], *, source_name: str = "uploaded.pdf") -> dict[str, Any]:
    """Create a complete source artifact from an uploaded PDF."""

    pages = extract_pages(file_bytes)
    source = "\n\n".join(page["text"] for page in pages if page["text"])
    result = new_result(source=source, input_mode="pdf", config=config, pages=pages)
    result["source_name"] = source_name
    return result


def path_to_result(path: str | Path, config: Mapping[str, Any]) -> dict[str, Any]:
    source_path = Path(path)
    return pdf_to_result(source_path.read_bytes(), config, source_name=source_path.name)
