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

Place the official SQuADv2 JSON files in `data/raw/`, then prepare answer-aware
JSONL files and fine-tune locally:

```powershell
python -m training.prepare_squad --split train --input data/raw/train-v2.0.json --output data/training/squadv2_train.jsonl
python -m training.prepare_squad --split validation --input data/raw/dev-v2.0.json --output data/training/squadv2_validation.jsonl
python -m training.train_qg --train data/training/squadv2_train.jsonl --validation data/training/squadv2_validation.jsonl --output models/question_generation/flan-t5-squadv2
```

The full modular plan, contracts, and task IDs are maintained in
`TODO.md`. The detailed raw-to-output explanation is maintained in
`ProjectDetails.md`, especially Sections 5, 14, 17, and 18.
