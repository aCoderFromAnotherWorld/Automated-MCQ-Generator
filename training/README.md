# Training workflow

The question-generation model is adapted using answerable SQuADv2 examples.
SQuADv2 supplies `(context, answer) -> question` supervision; it does
not supply MCQ distractors. Distractors remain a separate textbook-grounded
pipeline task.

## Deployed checkpoint (current)

`notebooks/train_final.ipynb` on Kaggle (Tesla T4): `google/flan-t5-small` on
84,821 answerable SQuADv2 records (86,821 prepared minus a 2,000 validation carve),
1 epoch, Adafactor lr 1e-3, batch 32, **fp32**, seed 42, ~25 min. The output was
zipped and unzipped into `models/question_generation/flan-t5-squadv2/`.

- `train_loss` 2.61, `eval_loss` 1.66 (perplexity 5.27) on 1,000 held-out samples
- BLEU-4 ≈ 20.9, ROUGE-1 0.49 (1,000 held-out samples, beam 4)
- Provenance: `models/question_generation/flan-t5-squadv2/training_run.json`
- This checkpoint replaced the local 256-example smoke checkpoint (eval_loss 1.49)
  and an earlier diverged fp16 run (train_loss 27.5, degenerate output).

## Precision rule (learned the hard way)

T5 **must not be fine-tuned in fp16 on T4-class GPUs** (Turing, no native bf16):
activations overflow and training diverges into repeated-token garbage.
Always train in **fp32** on T4 (or bf16 only on Ampere+ GPUs). `train_final.ipynb`
hard-codes `PRECISION = 'fp32'`; `train_kaggle_fast.ipynb` defaults to `auto` with
a loss-NaN guard that refuses to save a diverged run.

## Retraining on Kaggle

1. Upload `data/training/squadv2_train.jsonl` (+ optional
   `data/training/squadv2_validation.jsonl`) as a Kaggle dataset.
2. Run `notebooks/train_final.ipynb` end to end (GPU T4). Expect ~25 min for the
   `fast` preset on the full training set.
3. Check the printed sample generations and `training_run.json`: `eval_loss`
   should be ~1–3 (perplexity ~3–20), and the samples must be real questions.
4. Download `flan-t5-squadv2.zip` from the Output panel and unzip it **directly**
   into `models/question_generation/flan-t5-squadv2/` (no nested folder), replacing
   the current checkpoint.

Notebook lineage:
- `train_final.ipynb` — canonical, fp32-hardened recipe that produced the deployed model
  (includes a pre-flight loss check that aborts before training on a NaN loss).
- `train_kaggle_fast.ipynb` — same recipe with `PRECISION = 'auto'` (fp16 on T4 with
  a NaN guard). Prefer `train_final.ipynb`.
- `train.ipynb` / `train_gemini.ipynb` — earlier experiments (one diverged under fp16);
  superseded.
- `train_qg_colab.ipynb` — legacy Colab sketch; superseded by the Kaggle notebooks.

## Local training (no GPU)

1. Place separate SQuADv2 train and validation splits in `data/raw/`. Official
   JSON and Hugging Face-style Parquet splits are supported.
2. Run `python -m training.train_qg` directly on Parquet splits when possible;
   this avoids creating large duplicate JSONL artifacts. Never use one split
   for both roles.
3. Run `python -m training.prepare_squad --split train` and
   `python -m training.prepare_squad --split validation` only when prepared
   JSONL output is specifically needed, then train on those generated files.
4. Store the resulting checkpoint under `models/question_generation/`.
5. Record model, dataset version, seed, hyperparameters, and file counts
   (the script writes `training_run.json` automatically).

Training executes on the local machine through Transformers and PyTorch. The
Hugging Face inference backend is not used by this training workflow.

Example with Parquet input:

```powershell
python -m training.train_qg --train data/training/train-00000-of-00001.parquet --validation data/evaluation/validation-00000-of-00001.parquet --output models/question_generation/flan-t5-squadv2
```

Example for generating optional JSONL artifacts:

```powershell
python -m training.prepare_squad --split train --input data/raw/train.parquet --output data/training/squadv2_train.jsonl
python -m training.prepare_squad --split validation --input data/raw/validation.parquet --output data/training/squadv2_validation.jsonl
```
