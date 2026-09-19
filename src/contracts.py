"""Stable, JSON-serializable contracts shared by pipeline modules.

No pipeline orchestration belongs here.  These helpers let independently built
modules return compatible artifacts before ``src.pipeline`` exists.
"""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any, Mapping


RESULT_KEYS = (
    "config",
    "source",
    "input_mode",
    "pages",
    "raw_text",
    "cleaned_text",
    "sentences",
    "chunks",
    "candidates",
    "ranked_candidates",
    "generation_records",
    "distractor_records",
    "validation",
    "stats",
    "questions",
)


def new_result(*, source: str, input_mode: str, config: Mapping[str, Any], pages: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Create the complete result shape required by the future pipeline/UI."""

    result: dict[str, Any] = {
        "config": deepcopy(dict(config)),
        "source": source,
        "input_mode": input_mode,
        "pages": pages or [],
        "raw_text": source,
        "cleaned_text": "",
        "sentences": [],
        "chunks": [],
        "candidates": [],
        "ranked_candidates": [],
        "generation_records": [],
        "distractor_records": [],
        "validation": {"summary": {}, "per_question": []},
        "stats": {},
        "questions": [],
    }
    assert_json_serializable(result)
    return result


def ensure_result_shape(result: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    """Fill skipped-stage fields without silently changing supplied values."""

    base = new_result(
        source=str(result.get("source", result.get("raw_text", ""))),
        input_mode=str(result.get("input_mode", "text")),
        config=result.get("config", config),
        pages=list(result.get("pages", [])),
    )
    base.update(dict(result))
    for key in RESULT_KEYS:
        if key not in base:
            raise ValueError(f"Missing required result key: {key}")
    assert_json_serializable(base)
    return base


def assert_json_serializable(value: Any) -> None:
    """Fail early if a module leaks an NLP/PDF library object into its output."""

    try:
        json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError) as error:
        raise TypeError("Module output must be JSON-serializable.") from error
