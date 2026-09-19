"""Fine-tune FLAN-T5 locally on prepared SQuADv2 answer-aware QG data.

This script performs training on the local machine. It does not call a hosted
inference API. The base checkpoint is downloaded only if it is not already in
the local Hugging Face cache.

Usage:
    python -m training.train_qg \
        --train data/training/squadv2_train.jsonl \
        --validation data/training/squadv2_validation.jsonl \
        --output models/question_generation/flan-t5-squadv2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def train(
    train_path: Path,
    validation_path: Path,
    output_dir: Path,
    model_name: str = "google/flan-t5-base",
    epochs: float = 3.0,
    batch_size: int = 4,
    max_input_length: int = 512,
    max_target_length: int = 64,
    seed: int = 42,
) -> dict[str, Any]:
    """Fine-tune a sequence-to-sequence model on local prepared files only."""

    if not train_path.is_file() or not validation_path.is_file():
        raise FileNotFoundError("Both prepared train and validation JSONL files are required.")
    if train_path.resolve() == validation_path.resolve():
        raise ValueError("Train and validation paths must be different files.")
    from datasets import load_dataset
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        set_seed,
    )

    set_seed(seed)
    dataset = load_dataset(
        "json",
        data_files={"train": str(train_path), "validation": str(validation_path)},
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def tokenize(batch: dict[str, list[str]]) -> dict[str, list[list[int]]]:
        inputs = tokenizer(
            batch["input_text"], max_length=max_input_length, truncation=True
        )
        labels = tokenizer(
            text_target=batch["target_text"],
            max_length=max_target_length,
            truncation=True,
        )
        inputs["labels"] = labels["input_ids"]
        return inputs

    if len(dataset["train"]) == 0 or len(dataset["validation"]) == 0:
        raise ValueError("Prepared train and validation splits must both contain examples.")
    tokenized = dataset.map(
        tokenize,
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing local question-generation data",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        predict_with_generate=True,
        report_to=[],
        load_best_model_at_end=True,
        save_total_limit=2,
        seed=seed,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
    )
    train_result = trainer.train()
    metrics = trainer.evaluate()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    run_record = {
        "model_name": model_name,
        "training_mode": "local",
        "train_file": str(train_path),
        "validation_file": str(validation_path),
        "train_examples": len(dataset["train"]),
        "validation_examples": len(dataset["validation"]),
        "epochs": epochs,
        "batch_size": batch_size,
        "max_input_length": max_input_length,
        "max_target_length": max_target_length,
        "seed": seed,
        "train_metrics": train_result.metrics,
        "validation_metrics": metrics,
    }
    (output_dir / "training_run.json").write_text(
        json.dumps(run_record, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return run_record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-name", default="google/flan-t5-base")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-input-length", type=int, default=512)
    parser.add_argument("--max-target-length", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    record = train(
        args.train,
        args.validation,
        args.output,
        args.model_name,
        args.epochs,
        args.batch_size,
        args.max_input_length,
        args.max_target_length,
        args.seed,
    )
    print(json.dumps(record, indent=2, default=str))


if __name__ == "__main__":
    main()
