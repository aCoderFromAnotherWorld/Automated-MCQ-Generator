# Local training workflow

The question-generation model is adapted locally using answerable SQuADv2
examples. SQuADv2 supplies `(context, answer) -> question` supervision; it does
not supply MCQ distractors. Distractors remain a separate textbook-grounded
pipeline task.

1. Place separate SQuADv2 train and validation splits in `data/raw/`. Official
   JSON and Hugging Face-style Parquet splits are supported.
2. Run `python -m training.prepare_squad --split train` and
   `python -m training.prepare_squad --split validation` for the corresponding
   files. Never use one split for both roles.
3. Run `python -m training.train_qg` on the two generated JSONL files.
4. Store the resulting checkpoint under `models/question_generation/`.
5. Record model, dataset version, seed, hyperparameters, and file counts.

Training executes on the local machine through Transformers and PyTorch. The
Hugging Face inference backend is not used by this training workflow.

Example with Parquet input:

```powershell
python -m training.prepare_squad --split train --input data/raw/train.parquet --output data/training/squadv2_train.jsonl
python -m training.prepare_squad --split validation --input data/raw/validation.parquet --output data/training/squadv2_validation.jsonl
```
