# Automated MCQ Generator

The project is split into three independently explainable layers:

```text
data/       datasets, prepared records, and evaluation material
training/   local SQuADv2 preparation and FLAN-T5 fine-tuning
src/        NLP runtime pipeline: extraction, preprocessing, embeddings, QG, validation
ui/         Streamlit presentation helpers and UI-only concerns
app.py      Streamlit entry point and presentation composition
models/     locally saved checkpoints and embedding artifacts
```

The required question-generation model adaptation is performed locally using
answerable SQuADv2 records. SQuADv2 provides `(context, answer) -> question`
supervision; it does not provide distractors, so distractor generation remains
a separate textbook-grounded runtime stage.

## Run the interface

```powershell
python -m pip install -r requirements.txt
streamlit run app.py
```

The interface can preview raw text and text-based PDFs immediately. It consumes
`src.pipeline.generate_mcqs(...)` once the NLP pipeline is implemented.

## Run local training

Place the official SQuADv2 JSON files or local Hugging Face-style Parquet split
files in `data/raw/`, then prepare answer-aware JSONL files and fine-tune
locally. For the included dataset, use the compact Parquet splits directly:

```powershell
python -m training.train_qg --train data/training/train-00000-of-00001.parquet --validation data/evaluation/validation-00000-of-00001.parquet --output models/question_generation/flan-t5-squadv2
```

The JSONL preparation commands remain available when an answer-aware JSONL
artifact is specifically needed, but they are not required for local training.

Parquet splits can be passed directly to the trainer; raw SQuAD-style Parquet
rows are normalized automatically:

```powershell
python -m training.train_qg --train data/raw/train.parquet --validation data/raw/validation.parquet --output models/question_generation/flan-t5-squadv2
```

For a quick local smoke run, limit the data and optimizer steps:

```powershell
python -m training.train_qg --train data/training/train-00000-of-00001.parquet --validation data/evaluation/validation-00000-of-00001.parquet --output models/question_generation/flan-t5-smoke --max-train-examples 256 --max-validation-examples 64 --max-steps 50 --batch-size 2
```

For a quick local smoke run, limit the examples and optimizer steps:

```powershell
python -m training.train_qg --train data/training/train-00000-of-00001.parquet --validation data/evaluation/validation-00000-of-00001.parquet --output models/question_generation/flan-t5-smoke --max-train-examples 1000 --max-validation-examples 200 --epochs 1 --batch-size 8 --max-steps 50
```

The full modular plan, contracts, and task IDs are maintained in
`TODO.md`. The detailed raw-to-output explanation is maintained in
`ProjectDetails.md`, especially Sections 5, 14, 17, and 18.
