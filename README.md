# Automated MCQ Generator

Generate validated multiple-choice questions (MCQs) from textbook passages. Paste text
or upload a text-based PDF, and the Streamlit interface returns source-grounded MCQs
with full pipeline transparency: candidates, generated questions, distractors,
validation reasons, and a playable quiz mode.

## What it does

`text or PDF → validated MCQs + audit trail + quiz`

- **Input:** raw pasted text or a text-based PDF (OCR-free; image-only PDFs are rejected).
- **Processing:** cleaning (incl. citation-marker stripping) → sentence segmentation →
  paragraph-aware chunking → answer-candidate extraction (RAKE + proper-noun phrases) →
  explainable ranking → FLAN-T5 question generation → textbook-grounded distractor
  generation → strict validation assembly.
- **Output:** MCQs with answer, 3 distractors, source context, model/backend provenance —
  plus every intermediate artifact (`generation_records`, `distractor_records`,
  per-question validation) for inspection. Invalid MCQs are discarded with reasons
  instead of being shown.
- **Quiz mode:** answer the generated MCQs, get feedback per question, and a final score.

Distractor generation is textbook-grounded (pool = the passage's own ranked concepts) —
deliberately never WordNet or external knowledge by default, so every option is
source-traceable.

## Project layout

```text
app.py          Streamlit entry point (no NLP logic; calls src.pipeline, renders results)
config.py       Single CONFIG shared by the UI and the pipeline — tune everything here
src/            NLP runtime: input, preprocessing, segmentation, chunking, candidates,
                ranking, embeddings, question generation, distractors, validation,
                and the end-to-end pipeline (src/pipeline.py :: generate_mcqs)
ui/             Streamlit presentation helpers only
training/       SQuADv2 preparation (prepare_squad) and local fine-tuning (train_qg)
notebooks/      Kaggle/Colab training recipes (see training/README.md)
data/           raw / training / evaluation datasets (never commit large files)
models/         locally saved checkpoints and embedding artifacts (git-ignored)
evaluation/     evaluation harness placeholder (Phase 8 — not yet implemented)
tests/          57 unit/integration tests covering every module
```

## Quickstart

```powershell
# 1. Create and activate a virtual environment (recommended)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install dependencies
python -m pip install -r requirements.txt

# 3. (First run on Windows) make sure the fine-tuned checkpoint is in place:
#    models/question_generation/flan-t5-squadv2/
#    ...containing model.safetensors, tokenizer.*, config.json, training_run.json.
#    See "Training" below for how it was produced on Kaggle.

# 4. Run the interface
streamlit run app.py

# 5. Run the tests
python -m pytest tests/ -q
```

On first generation the app loads the FLAN-T5 checkpoint (~300 MB) once and the
Sentence-Transformer model once; both are cached for the session, so the first run is
slower and later runs are faster.

## How generation works (short version)

1. The passage is cleaned (soft hyphens, citation refs like `[9]`/`[a]`, line wraps)
   while preserving blank-line paragraph breaks.
2. Sentences are segmented and grouped into **paragraph-bounded chunks** (8 sentences,
   2-sentence overlap) so chunks never mix unrelated topics.
3. Answer candidates are extracted with RAKE plus capitalised proper-noun phrases
   ("Bay of Bengal", "Battle of Plassey"), then ranked with explainable features
   (TF-IDF, RAKE score, frequency, noun-phrase/NER status, position, multi-word bonus).
4. The fine-tuned **FLAN-T5-small** checkpoint generates one question per top candidate
   (answer-aware prompt: `generate question: context: … answer: …`).
5. Distractors are chosen from the passage's own ranked candidates by semantic
   similarity (`0.5·cos(candidate, answer) + 0.5·cos(candidate, question)`), reusing
   no option more than twice per run and never an option that repeats the stem.
6. Validation rejects: empty/length-bad questions, answers not in the context,
   duplicate questions, duplicate options, fewer than 3 valid distractors,
   contextually-supported distractors, answers appearing in the question stem —
   **and round-trip verification**: the base QA model must independently reproduce
   the intended answer when asked the generated question. Rejected MCQs carry named
   reasons instead of being shown.

## Training

The deployed checkpoint (`models/question_generation/flan-t5-squadv2/`) was fine-tuned
on Kaggle (see `notebooks/train_final.ipynb`): flan-t5-small on 84,821 answerable
SQuADv2 records, 1 epoch, Adafactor lr 1e-3, **fp32** (T5 diverges in fp16 on T4-class
GPUs — a diverged fp16 run was discarded), seed 42, ~25 min on a Tesla T4. Result:
`train_loss` 2.61, `eval_loss` 1.66 (perplexity 5.27), BLEU-4 ≈ 20.9, ROUGE-1 0.49.
Full provenance lives in the checkpoint's `training_run.json`.

To retrain locally instead, use the commands in `training/README.md`.
SQuADv2 supplies `(context, answer) → question` supervision for answer-aware question
generation — SQuAD is a reading-comprehension dataset, **not** an MCQ dataset, and it
supplies no distractors.

## Key settings (`config.py`)

| Key | Default | Effect |
|---|---|---|
| `chunk_size_sentences` / `chunk_overlap_sentences` | 8 / 2 | Chunk shape; chunks never cross paragraphs/pages |
| `max_distractor_pool` | 12 | Only the top-N candidates are semantically scored per question |
| `generation_attempt_factor` | 5 | Max generation attempts = `num_questions × factor` |
| `context_support_threshold` | 0.8 | Distractor similarity to the question above this ⇒ "actually correct" ⇒ rejected |
| `answer_verification` / `answer_model` / `answer_match_threshold` | True / `google/flan-t5-small` / 0.5 | Round-trip QA check gating correctness |
| `max_distractor_reuse` | 2 | No distractor may appear more than N times per run |
| `generation_max_new_tokens` / `generation_num_beams` | 32 / 1 | Greedy, fast CPU generation |

## Known limitations

- CPU-only inference by default; long multi-topic texts need ~30–70 s for many questions.
- The small QG model (flan-t5-small, 1 epoch) occasionally produces awkward grammar; the
  validators are biased toward **rejecting** questionable MCQs rather than showing them.
- Proper-noun detection is heuristic (capitalisation-based); no spaCy NER model is
  installed by default, so `is_named_entity` flags rely on the `PROPER` detector.
- `evaluation/` (M14: automatic + human evaluation harness) and the final report (M17)
  are still outstanding — see `TODO.md`.
- Project folders live under OneDrive: keep large `models/`/`data/` synced or ignored,
  and rely on git commits — OneDrive sync has previously reverted working-tree files.

## Repository hygiene

- `models/`, `data/*.jsonl|*.parquet|*.zip`, `outputs/`, `__pycache__/` are git-ignored
  (`.gitignore`). Never commit checkpoints, datasets, or the `.zip` training artifacts.
- `HUGGINGFACE_API_TOKEN` (only needed for the optional `hf_api` backend) must stay in
  the environment, never in the repo.

The full modular plan, contracts, and task IDs are maintained in
`TODO.md`. The detailed raw-to-output explanation is maintained in
`ProjectDetails.md`, especially Sections 5, 14, 17, and 18.
