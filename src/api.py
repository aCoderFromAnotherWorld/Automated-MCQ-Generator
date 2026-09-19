"""Pre-generation public API available after M01–M07.

It intentionally stops after candidate ranking.  The future M13 pipeline will
extend the same result schema with embeddings, generated questions,
distractors, and final validation.
"""

from __future__ import annotations

from typing import Any, Mapping

from config import CONFIG
from src.chunking import chunk_result
from src.input_handler import text_to_result
from src.keyword_extractor import extract_candidates
from src.preprocessing import preprocess_result
from src.ranking import rank_candidates


def generate_mcqs(text: str, num_questions: int = 5, **options: Any) -> dict[str, Any]:
    """Prepare ranked answer candidates using the stable future pipeline API.

    This does not fabricate questions. Until M08–M13 are implemented, the
    returned ``questions`` list is intentionally empty and the status explains
    the earliest unfinished stage.
    """

    if num_questions < 1:
        raise ValueError("num_questions must be at least one.")
    run_config: dict[str, Any] = {**CONFIG, **options}
    result = text_to_result(text, run_config)
    result = preprocess_result(result)
    result = chunk_result(result, run_config)
    result["candidates"] = extract_candidates(result["chunks"], run_config)
    result["ranked_candidates"] = rank_candidates(
        result["candidates"], result["chunks"], run_config, limit=num_questions
    )
    result["stats"] = {
        "sentences": len(result["sentences"]),
        "chunks": len(result["chunks"]),
        "candidates": len(result["candidates"]),
        "ranked_candidates": len(result["ranked_candidates"]),
    }
    result["validation"] = {
        "summary": {"status": "awaiting_embeddings_and_generation"},
        "per_question": [],
    }
    return result
