"""Prepare answerable SQuADv2 records for local answer-aware QG fine-tuning.

Usage:
    python -m training.prepare_squad --input data/raw/train-v2.0.json \
        --output data/training/squadv2_train.jsonl

Input may be official SQuAD/SQuADv2 JSON or a Hugging Face-style Parquet split.
The output is JSON Lines with ``input_text`` (context + answer) and
``target_text`` (question). Unanswerable examples are excluded because the
project's generation pipeline always supplies an answer candidate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Iterator


def _record(context: str, question: str, answers: list[dict[str, Any]], example_id: str) -> dict[str, Any] | None:
    """Build one QG record from an answerable QA example."""

    normalized_context = " ".join(context.split())
    normalized_question = " ".join(question.split())
    if not normalized_context or not normalized_question or not answers:
        return None
    answer = str(answers[0].get("text", "")).strip()
    if not answer:
        return None
    return {
        "id": example_id,
        "input_text": f"generate question: context: {normalized_context} answer: {answer}",
        "target_text": normalized_question,
        "context": normalized_context,
        "answer": answer,
        "answer_start": answers[0].get("answer_start"),
    }


def iter_json_examples(payload: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Yield records from official SQuAD/SQuADv2 JSON."""

    for article in payload.get("data", []):
        for paragraph in article.get("paragraphs", []):
            context = paragraph.get("context", "")
            for item in paragraph.get("qas", []):
                answers = item.get("answers") or []
                if item.get("is_impossible"):
                    continue
                record = _record(context, item.get("question", ""), answers, item.get("id", ""))
                if record is not None:
                    yield record


def iter_parquet_examples(input_path: Path) -> Iterator[dict[str, Any]]:
    """Yield records from a Hugging Face Parquet QA split."""

    try:
        import pyarrow.parquet as parquet
    except ModuleNotFoundError as error:
        raise RuntimeError("Parquet input requires pyarrow. Install requirements.txt first.") from error

    table = parquet.read_table(input_path, columns=["id", "context", "question", "answers"])
    for row in table.to_pylist():
        answers = row.get("answers") or {}
        texts = answers.get("text") or []
        starts = answers.get("answer_start") or []
        answer_items = [
            {"text": text, "answer_start": starts[index] if index < len(starts) else None}
            for index, text in enumerate(texts)
        ]
        record = _record(row.get("context", ""), row.get("question", ""), answer_items, row.get("id", ""))
        if record is not None:
            yield record


def iter_source_examples(input_path: Path) -> Iterable[dict[str, Any]]:
    suffix = input_path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(input_path.read_text(encoding="utf-8"))
        return iter_json_examples(payload)
    if suffix == ".parquet":
        return iter_parquet_examples(input_path)
    raise ValueError("Input must be an official SQuAD JSON file or a Parquet QA split.")


def prepare(input_path: Path, output_path: Path, split: str) -> dict[str, Any]:
    if not input_path.is_file():
        raise FileNotFoundError(f"Dataset split not found: {input_path}")
    records = list(iter_source_examples(input_path))
    if not records:
        raise ValueError("No answerable records found; check the dataset format and split.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {
        "split": split,
        "source": str(input_path),
        "source_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "output": str(output_path),
        "accepted_examples": len(records),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation"), required=True)
    args = parser.parse_args()
    stats = prepare(args.input, args.output, args.split)
    metadata_path = args.output.with_suffix(args.output.suffix + ".metadata.json")
    metadata_path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
