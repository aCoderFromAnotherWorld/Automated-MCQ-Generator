# TODO — Automated MCQ Generator (Development Tracker)

> Living progress tracker for the NLP project described in `ProjectDetails.md`.
> **Principle:** build incrementally, module-by-module. Keep the NLP pipeline
> independent from the web UI. Every phase must leave a working version.

**Legend:** `[ ]` todo · `[~]` in progress · `[x]` done · `[!]` blocked

---

## 0. Project Conventions & Ground Rules

- [ ] **Document roles (keep the two files from diverging):**
      `ProjectDetails.md` = *what / why / how* (the specification).
      `TODO.md` = *exactly what to implement and check off* (the tracker).
      This file never restates the rationale — it links to `ProjectDetails.md`
      section numbers instead.
- [ ] **Phase numbers come from the canonical phase map in `ProjectDetails.md` §7.1.**
      No other numbering may be invented.
- [ ] Keep the NLP pipeline (`src/`) fully independent from Streamlit (`app.py`).
- [ ] One module at a time — implement, run, inspect, test, then move on.
- [ ] Keep dependencies minimal; pin versions in `requirements.txt`.
- [ ] Never fabricate evaluation numbers — all metrics must come from real runs.
- [ ] Preserve original extracted text separately from cleaned text (debuggability).
- [ ] Keep configuration in a single `CONFIG` (separate from core code).
- [ ] Commit a working version to Git after each completed phase.
- [ ] Document model names, versions, seeds, and hyperparameters for reproducibility.
- [ ] **Use the resolved embedding roles (ProjectDetails.md §5.2 / §57.1) — do not
      mix them up:**
      TF-IDF → candidate importance + ranking;
      Sentence Transformer (`all-MiniLM-L6-v2`) → semantic similarity, distractor
      similarity, duplicate detection;
      Word2Vec → optional word-level experiment only.
- [ ] **Keep the two corpora strictly separate (§35.1 vs §35.2.1):** SQuAD trains/
      adapts the QG model; the Bonaventure chapter evaluates the system. Never
      evaluate on anything the model was trained on.

### Modular implementation protocol

Each future implementation prompt should target one task below. A task may be
marked complete only after its module tests pass and its public contract remains
compatible with the preceding tasks. Do not implement a downstream task by
duplicating logic in an upstream module or in `app.py`.

**Status legend:** `[ ]` not started · `[~]` partial/in progress · `[x]` implemented
and verified · `[!]` blocked. Dataset/model tasks stay unchecked until their
artifacts are supplied and the required local run is completed. A checked
shell task can still have an unchecked integration dependency; the row's
handoff contract states exactly what is available now.

| Done | ID | Implementable task | Owns | Depends on | Safe handoff contract |
|---|---|---|---|---|---|
| [ ] | M00 | Repository scaffolding | `src/__init__.py`, config, dependencies, test layout | None | Imports work; tests can run |
| [ ] | M01 | Pipeline result contract | schema tests and typed/documented result shape | M00 | `generate_mcqs(...)` returns all required top-level keys |
| [ ] | M02 | Raw-text input adapter | `src/input_handler.py` | M01 | Text input becomes a validated source artifact |
| [ ] | M03 | PDF extraction adapter | `src/pdf_processor.py` | M01 | PDF input becomes ordered page text plus metadata |
| [ ] | M04 | Preprocessing | `src/preprocessing.py` | M02 or M03 | Raw text remains preserved; cleaned text is separate |
| [ ] | M05 | Segmentation and chunking | `src/segmentation.py`, `src/chunking.py` | M04 | Sentences/chunks have stable IDs and source mapping |
| [ ] | M06 | Candidate extraction | `src/keyword_extractor.py` | M05 | Returns the fixed candidate dictionary shape |
| [ ] | M07 | Candidate ranking | `src/ranking.py` | M06 | Returns ranked candidates without changing candidate schema |
| [ ] | M08 | Representations and similarity | `src/embeddings.py` | M05, M07 | TF-IDF and semantic utilities are independently testable |
| [ ] | M09 | Question generation | `src/question_generator.py` | M05, M07, M08 | Returns traceable generation records; backend is explicit |
| [ ] | M10 | Question validation | `src/validator.py` | M09, M08 | Returns validity plus detailed reasons |
| [ ] | M11 | Distractor generation | `src/distractor_generator.py` | M07, M08, M10 | Returns candidates, scores, selected options, and rejection reasons |
| [ ] | M12 | MCQ assembly and validation | `src/mcq_validator.py` | M10, M11 | Rejects critical failures and returns validated MCQs only |
| [ ] | M13 | Pipeline orchestration | `src/pipeline.py` | M02–M12 | Connects modules without owning their internal algorithms |
| [ ] | M14 | Evaluation | `evaluation/` | M13 | Reports measured results without changing generation behavior |
| [x] | M15 | Streamlit integration shell | `app.py`, `ui/` | M01, M13 | UI shell consumes the result contract when available; no NLP logic in the UI |
| [x] | M16 | Quiz mode shell | UI quiz state/tests | M15, M13 | Quiz operates on validated MCQs when the pipeline returns them |
| [ ] | M17 | Documentation and release | README, report, reproducibility records | M13–M16 | Setup, tests, limitations, and known gaps are current |
| [ ] | M18 | SQuADv2 preparation | `training/prepare_squad.py` | M00 | Answerable context-answer-question JSONL |
| [ ] | M19 | Local QG fine-tuning | `training/train_qg.py` | M18 | Locally saved FLAN-T5 checkpoint and tokenizer |
| [ ] | M20 | Trained-checkpoint integration | `src/question_generator.py`, config | M09, M19 | Inference uses the recorded local checkpoint |

#### Safe task rules

- One prompt should normally implement one `M##` task; explicitly name the ID
  in the prompt and update only its checklist items.
- Before starting a task, inspect its dependency rows and the current public
  contract. If a dependency is incomplete, add a small adapter or test seam;
  do not silently implement the dependency as part of the new task.
- Every task must include focused unit tests and preserve the existing tests.
- Changes to shared schemas require updating the contract, all consumers, and
  schema tests in the same task.
- Keep unfinished optional work visibly unchecked. Do not mark a task complete
  because a stub or UI placeholder exists.
- The current completed scope is M15's UI shell and M16's quiz interaction;
  their NLP-pipeline integration remains unchecked until M13 is implemented.
- Local SQuADv2 training is mandatory: M18, M19, and M20 must not be marked
  optional or skipped because the runtime UI exists.

#### Prompt handoff template

Use this structure for future implementation requests:

```text
Implement task M##: <task name>.
Read the task row, its dependencies, and its safe handoff contract first.
Do not implement downstream tasks.
Add/update focused tests, preserve existing contracts, and update TODO.md only
for work actually completed.
```

---

## 1. Repository Scaffolding  *(Phase 0)*

- [ ] Create folder structure (per `ProjectDetails.md` §47):
  - [ ] `src/` (Python package with `__init__.py`)
  - [ ] `data/raw/`, `data/processed/`, `data/training/`, `data/evaluation/`
  - [ ] `models/embeddings/`, `models/question_generation/`, `models/distractor/`
  - [ ] `training/` (local SQuADv2 preparation and fine-tuning scripts)
  - [ ] `ui/` (Streamlit presentation helpers only)
  - [ ] `notebooks/`
  - [ ] `evaluation/results/`
  - [ ] `outputs/generated_mcqs/`, `outputs/logs/`
- [ ] Create `requirements.txt` (start minimal, grow per phase).
- [ ] Create `.gitignore` (ignore `data/`, `models/`, `outputs/`, `__pycache__`, `.ipynb_checkpoints`, venv).
- [ ] Create `config.py` with the base `CONFIG` dict (§54).
- [ ] Keep UI imports under `ui/` and NLP/runtime imports under `src/`.
- [ ] Expand `README.md` with setup + run instructions (currently only a title).
- [ ] Add `tests/` directory for unit tests.

**Deliverable:** clean, runnable skeleton committed to Git.

---

## 2. Input Handling — Raw Text  *(Phase 1)*

- [ ] Define the pipeline entry API: `generate_mcqs(text, num_questions, **opts)` (§49).
- [ ] Implement raw-text input path (accept a string, validate non-empty).
- [ ] Define the structured result schema returned by the pipeline
      (**authoritative schema: `ProjectDetails.md` §49.1** — must expose every
      intermediate stage the UI needs, not just the final questions):
  ```python
  {
    "config": {...},              # exact CONFIG used for this run (§54)
    "source": "...",              # original raw input
    "pages": [...],               # per-page text (PDF input only; else empty)
    "cleaned_text": "...",        # after preprocessing (§9)
    "sentences": [...],           # after segmentation (§10)
    "chunks": [...],              # chunk_id / page_start / page_end (§11)
    "candidates": [...],          # all extracted candidates + scores (§12)
    "ranked_candidates": [...],   # after ranking + diversity (§13)
    "generation_records": [...],  # one per attempt: model_input, backend,
                                  #   model_name, candidate_questions,
                                  #   selected_question (§16, §49.1)
    "distractor_records": [...],  # candidate_pool, selected, rejected+reason (§30)
    "validation": {...},          # summary counts + per-question reasons (§43)
    "stats": {...},               # counts + concept coverage (§43–§44)
    "questions": [{"question", "answer", "distractors", "context",
                   "chunk_id", "source_page", "generator"}]
  }
  ```
- [ ] Ensure every key is always present (empty value if a stage was skipped),
      so the UI never has to guess (§49.1).
- [ ] Keep the result **plain JSON-serializable** so it can be written to
      `outputs/` and reused by the evaluation scripts (§49.2).
- [ ] Handle empty-input error message (§52).

**Deliverable:** `generate_mcqs()` accepts raw text and returns a (stub) structured result.

---

## 3. PDF Extraction  *(Phase 2)*

- [ ] `src/pdf_processor.py` — load PDF, extract text per page, preserve page numbers.
- [ ] Return structured page-level text (page number + text).
- [ ] Error handling: invalid PDF, empty PDF, PDF with no extractable text (§52).
- [ ] Preserve original extracted text separately for inspection (§9).
- [ ] Unit tests: valid PDF, invalid PDF, text-less PDF.

**Deliverable:** `PDF → structured page text`.

---

## 4. Preprocessing  *(Phase 3)*

- [ ] `src/preprocessing.py` — clean text, normalize whitespace, remove obvious artifacts.
- [ ] Do **not** aggressively strip punctuation (§9).
- [ ] Keep original vs. cleaned text separate.
- [ ] Unit tests for cleaning edge cases.

**Deliverable:** `raw text → cleaned text`.

---

## 5. Sentence Segmentation & Chunking  *(Phase 3)*

- [ ] `src/chunking.py` — sentence segmentation (spaCy or NLTK).
- [ ] Create **overlapping** chunks (§11).
- [ ] Attach chunk metadata: `{chunk_id, text, page_start, page_end}`.
- [ ] Make chunk size / overlap configurable via `CONFIG`.
- [ ] Unit tests: segmentation, overlap correctness, metadata integrity.

**Deliverable:** `clean text → list of chunks with metadata`.

---

## 6. Concept / Answer-Candidate Extraction  *(Phase 4)*

- [ ] `src/keyword_extractor.py` — RAKE extraction.
- [ ] spaCy noun-phrase + named-entity extraction.
- [ ] Merge candidates; remove short/duplicate/generic/stopword candidates (§12).
- [ ] Keep module replaceable (independent interface).
- [ ] Unit tests for candidate filtering.

**Deliverable:** `chunk → candidate phrases with scores + chunk IDs`.

---

## 7. Candidate Answer Ranking  *(Phase 4)*

- [ ] `src/ranking.py` — score candidates using features (§13):
  - [ ] keyword score, frequency, position, noun-phrase status, NER status, semantic relevance.
- [ ] Select top candidates while maintaining **diversity** (avoid same-concept duplicates).
- [ ] Unit tests for ranking + diversity.

**Deliverable:** `candidates → ranked, diverse answer candidates`.

---

## 8. Text Representation  *(Phase 4)*

> Roles are fixed by `ProjectDetails.md` §5.2 / §57.1. Do not substitute one
> representation for another.

- [ ] **`src/embeddings.py` — TF-IDF (core).**
  - [ ] Fit `TfidfVectorizer` on the document; expose per-candidate scores.
  - [ ] Unit tests: known corpus → expected top terms.
- [ ] **`src/embeddings.py` — Sentence Transformer (core).**
  - [ ] Load `all-MiniLM-L6-v2` (small, CPU-friendly); load once and cache.
  - [ ] `encode(texts) -> vectors` and `cosine_similarity(a, b)`.
  - [ ] Used for: question↔context relevance, distractor↔answer similarity,
        duplicate-question detection (§46).
  - [ ] Unit tests: identical sentences → similarity ≈ 1.0; unrelated → low.
- [ ] **Word2Vec (OPTIONAL experiment only — not required for the pipeline).**
  - [ ] Train on the corpus as a word-level side experiment (§39 Exp 5) if time
        permits; report separately. Must never be a required import for the
        main pipeline to run.
- [ ] Document the representation choices explicitly (§14).

**Deliverable:** `text → TF-IDF scores + Sentence Transformer vectors` + similarity utilities.

### Explainable embedding and training flow

- [ ] Explain raw text/PDF extraction before any representation is built.
- [ ] Explain TF-IDF as sparse term-weight vectors for candidate importance and ranking.
- [ ] Explain Sentence Transformer output as dense sentence/phrase vectors for semantic similarity.
- [ ] Explain why these embeddings support selection and validation but do not replace FLAN-T5 training.
- [ ] Explain SQuADv2 tokenization: context-plus-answer input, question target, token IDs, masks, and labels.
- [ ] Explain local forward pass, loss calculation, backpropagation, optimizer update, and checkpoint saving.
- [ ] Explain inference: textbook context plus selected answer → fine-tuned model → question text → validation.
- [ ] Show one complete trace in the report and Streamlit intermediate panels.

---

## 9. Question Generation  *(Phase 5)*

> Model is **fixed** (§16.0): `google/flan-t5-base` primary,
> `google/flan-t5-small` documented low-memory fallback. Backend is **explicit**
> (§16.1) — never auto-switched mid-run.
> After M19, the local backend must load the derived checkpoint recorded in
> `CONFIG["question_model_checkpoint"]`.

- [ ] `src/question_generator.py` — `generate_question(context, answer)`.
- [ ] **`question_model_name` fixed to `google/flan-t5-base`** (fallback
      `google/flan-t5-small` only when explicitly documented as such).
- [ ] Keep **model loading separate** from generation (load once, cache).
- [ ] Backend is chosen by `CONFIG["question_model_backend"]` (`"local"` | `"hf_api"`);
      both backends use the identical prompt and return identical structure.
- [ ] Build model input: `context + answer` prompt (§16).
- [ ] Generate multiple candidate questions; clean/trim output.
- [ ] Return structured results **including a `generation_records` entry**
      (model name, backend, prompt, raw output, cleaned question, chunk_id) so
      every question is traceable (§49.1).
- [ ] Do **not** add automatic backend fallback. On failure, surface a clear
      error plus a user-triggered "switch backend and retry" control (§16.1).
- [ ] Unit tests with a tiny model / mocked generation (no heavy calls in CI).

**Deliverable:** `context + answer → candidate question(s)` + traceable generation record.

---

## 10. Question Validation  *(Phase 5)*

- [ ] `src/validator.py` — question checks (§19):
  - [ ] empty output, length, duplicate, answer presence, context relevance, unsupported info.
- [ ] Return boolean validity + detailed reasons.
- [ ] Unit tests for each check.

**Deliverable:** `question → valid/invalid + reasons`.

---

## 11. Distractor Generation  *(Phase 6)*

> **Fixed pipeline (§21.1)** — implement exactly these stages, in this order.
> WordNet is an *optional* source (§21.2), not part of the default path.

- [ ] `src/distractor_generator.py` — fixed pipeline:
  - [ ] **1. Candidate source:** textbook/corpus concepts (ranked candidates from
        §13 + chapter vocabulary) — this is the required pool.
  - [ ] **2. Remove the correct answer** (case- and punctuation-insensitive).
  - [ ] **3. Semantic embedding similarity** (Sentence Transformer, §14.2):
        `score = 0.5·cos(cand, answer) + 0.5·cos(cand, question)`.
  - [ ] **4. Type / domain filter** — prefer distractors of the same type/domain
        as the answer (§23 Check 5).
  - [ ] **5. Contextual incorrectness check** — reject any candidate that is
        actually correct per the source context (§23 Check 4). This is a
        **critical** check; a correct distractor is a critical failure.
  - [ ] **6. Select top 3** after de-duplication; if fewer than 3 qualify, mark
        the MCQ invalid — never pad.
- [ ] Optional (only if the pool is too small): WordNet fallback (§21.2),
      recorded as such.
- [ ] Distractor validation (§23): not equal to answer, unique, plausible,
      contextually incorrect, type-appropriate.
- [ ] Unit tests for selection + each validation check.

**Deliverable:** `question + answer + context → 3 plausible distractors`.

---

## 12. MCQ Validation Layer  *(Phase 7)*

- [ ] Assemble full MCQ: question + answer + 3 distractors + source context.
- [ ] Final validation pipeline (§24): question → answer → distractors → source relevance.
- [ ] **Critical-failure rules (must reject the MCQ outright, never "warn"):**
  - [ ] a distractor that is *actually correct* per the source context (§23 Check 4),
  - [ ] the correct answer not supported by the source context (§19 Check 5),
  - [ ] fewer than 3 valid distractors,
  - [ ] duplicate options.
- [ ] Regenerate or discard invalid MCQs.
- [ ] Semantic duplicate-question filtering (§46) using **Sentence Transformer**
      embeddings + cosine similarity (`all-MiniLM-L6-v2`, §14).
- [ ] Unit + integration tests (including one test per critical-failure rule).

**Deliverable:** `validated MCQs` + per-MCQ validation reasons.

---

## 13. Pipeline Integration  *(Phase 10 — after Phases 5–9 modules are working)*

- [ ] `src/pipeline.py` — connect all modules end-to-end.
- [ ] Enforce source-grounding / faithfulness (§25): store `source_context` + `chunk_id` per MCQ.
- [ ] Concept coverage tracking (§44) — avoid over-generating from one concept.
- [ ] Performance: load models once, cache, generate only requested count (§53).
- [ ] **Return the full intermediate-result schema (§49.1)** — the Streamlit UI
      (§28–§30) cannot show intermediate stages the pipeline does not return:
  - [ ] `source`, `input_mode`, `pages`, `raw_text`, `cleaned_text`
  - [ ] `sentences`, `chunks`
  - [ ] `candidates`, `ranked_candidates`
  - [ ] `generation_records` (per candidate: model input, raw output, cleaned question, backend used)
  - [ ] `distractor_records` (per question: candidate pool + scores + selected 3)
  - [ ] `questions` (validated MCQs), `validation` (per-MCQ checks + reasons)
  - [ ] `stats` (counts + per-concept coverage, §43–§44)
  - [ ] `config` (the exact `CONFIG` snapshot for reproducibility, §54)
- [ ] Integration test: `text → final MCQs`.

**Deliverable:** `generate_mcqs()` returns complete, validated, grounded MCQs **plus** all intermediate artifacts.

---

## 14. Baseline System  *(cross-cutting comparison aid — not a phase, see §7.1)*

> **Fixed specification — see §38.1.** Do not improvise templates; implement the
> four rules exactly as written so the comparison is reproducible.

- [ ] Template question generation (§38.1 Step 2) — 4 ordered rules; store
      `baseline_template` with each question.
- [ ] Answer selection = ranked candidates from §13 (no new logic).
- [ ] Distractors = next-3 highest-ranked candidates from the same chunk; mark
      invalid if fewer than 3 (never pad).
- [ ] Reuse the **same** validators as the Transformer system (§19, §23, §24).
- [ ] Keep baseline separate so it can be compared against the Transformer.

**Deliverable:** baseline MCQ generator for experimental comparison.

---

## 15. Evaluation  *(Phase 8)*

> **Evaluation philosophy (§40–§42):** human evaluation + source-grounded
> correctness are the **primary** evidence. BLEU/ROUGE are **secondary/supporting**
> and must never be presented as the main measure of question quality.

- [ ] **Primary — human evaluation** (§41): relevance, grammar, correctness,
      distractor plausibility, overall (1–5), from real evaluators.
- [ ] **Primary — source-grounded correctness**: does the source context support
      the question + correct answer? Is the answer unique? Record as pass/fail
      per MCQ.
- [ ] **Primary — distractor quality** (§42): plausibility, incorrectness,
      similarity, diversity — with incorrectness treated as critical.
- [ ] `evaluation/metrics.py` — automatic metrics:
  - [ ] BLEU, ROUGE (**secondary**, only where reference questions exist).
- [ ] Validation metrics (§43): generated / valid / rejected / regenerated / duplicates / avg score.
- [ ] Human-evaluation CSV template (§41) → `evaluation/human_evaluation.csv` + `evaluation/results/`.
- [ ] Report the **back-end and model actually used** for each run (§16.1).
- [ ] Document dataset splits (§37) — never evaluate on training data.

**Deliverable:** evaluation scripts + results (real numbers only).

---

## 16. Experimental Comparisons  *(Phase 8, optional/if feasible)*

- [ ] Exp 1 — Keyword extraction: RAKE vs. spaCy noun phrases (§39).
- [ ] Exp 2 — Representation: TF-IDF vs. Sentence Transformer (Word2Vec optional).
- [ ] Exp 3 — Question generation: deterministic baseline (§38) vs. base FLAN-T5 vs. fine-tuned FLAN-T5.
- [ ] Exp 4 — Distractors: fixed embedding pipeline (§21.1) vs. WordNet-only fallback (§21.2).
- [ ] Hold **all** other settings fixed while varying one factor; log the exact
      `config` snapshot for each run.
- [ ] Record results in report tables (§58) — replace every `Actual` with measured values.

**Deliverable:** comparison tables with real measured results.

---

## 17. Required Local SQuADv2 Training  *(Cross-cutting; required before final release)*

> **Wording matters (§17):** SQuAD is a *reading-comprehension* dataset, not a
> dedicated MCQ/question-generation dataset. It is used here only for
> **answer-aware question generation** (context + answer → question). Say exactly
> this in the report; do not call it an MCQ dataset.

- [ ] Download/place the official SQuADv2 train and validation JSON files under `data/raw/`.
- [ ] Run `training/prepare_squad.py`; exclude `is_impossible` records and write JSONL pairs.
- [ ] Run `training/train_qg.py` locally with PyTorch and Transformers; do not use the HF inference API for training.
- [ ] Save the tokenizer and fine-tuned checkpoint under `models/question_generation/`.
- [ ] Compare base vs. fine-tuned experimentally (§17) on the **held-out** fixed chapter (§35.2.1).
- [ ] Document hyperparameters, seed, dataset version, hardware, package versions, and record counts.
- [ ] Keep the fixed evaluation chapter (§35.2.1) completely separate from all
      SQuAD training/validation data — never evaluate on training examples (§37).

**Deliverable:** fine-tuned model + comparison results.

---

## 18. Interactive Web Interface (Streamlit)  *(Phase 9)*

### Implementation status (2026-09-19)

- [x] Added the runnable Streamlit interface in `app.py`; it contains no NLP logic.
- [x] Added text/PDF input, question-count, model, backend, and generate controls.
- [x] Added renderers for the full pipeline-result schema, final MCQs, source context, and JSON export.
- [x] Added UI error states for empty/short input, unreadable PDFs, generation errors, and pending pipeline integration.
- [x] Added stateful quiz controls: select, submit, feedback, next, restart, and score.
- [ ] Connect `app.py` to the Phase-10 `src.pipeline.generate_mcqs` implementation when it exists.
- [ ] Add pipeline-owned model caching during pipeline integration; the UI currently caches PDF preview extraction only.

- [ ] `app.py` — import pipeline from `src/`; **no NLP logic in `app.py`** (§49, §73).
- [ ] Input controls (§27, §34):
  - [ ] PDF upload / text area toggle,
  - [ ] number of questions,
  - [ ] **backend selector** `CONFIG["question_model_backend"]` → `"local"` | `"hf_api"` (§16.1),
  - [ ] model selector (fixed set: `google/flan-t5-base`, `google/flan-t5-small`),
  - [ ] Generate button.
- [ ] Display **which model + backend actually produced each question** (§16.1) —
      required for reproducibility and viva.
- [ ] Intermediate visualization (§28–30) via `st.expander()` / `st.tabs()`,
      driven by the full pipeline result schema (§49.1):
  - [ ] original input, extracted text (per page), cleaned text, sentences, chunks,
  - [ ] candidate concepts, ranked candidates,
  - [ ] model input + raw model output + cleaned question (per `generation_records`),
  - [ ] distractor candidate pool + scores + selected 3 (per `distractor_records`),
  - [ ] validation results with reasons, coverage stats.
- [ ] Final MCQ view (§31) with source-context display.
- [ ] Error handling UI (§52): empty text, invalid PDF, no text, too little text,
      model failure (**with explicit "switch backend and retry"**, §16.1), no distractors.
- [ ] Performance: `st.cache` model loading (§53).

**Deliverable:** working Streamlit app over the tested pipeline.

---

## 19. Interactive Quiz Mode  *(Phase 9)*

- [ ] Quiz state: `current_question`, `selected_answer`, `score` (§51).
- [ ] Per-question flow: options → select → submit → correct/incorrect → explanation + source context.
- [x] Final score display (`4 / 5`).
- [x] Keep quiz logic separate from NLP generation logic.
- [x] Optional: restart quiz, next question.

**Deliverable:** interactive quiz over generated MCQs.

---

## 20. Export / Extras (Optional)

- [ ] Download MCQs as TXT/PDF (§34).
- [ ] Difficulty selection / classification (§45) — only if evaluated, no false claims.
- [ ] Question-type selection.
- [ ] Debug/demo mode for distractor candidates (§30).

---

## 21. Testing  *(Phase 10)*

- [ ] Unit tests: PDF extraction, preprocessing, chunking, candidate extraction, validation.
- [ ] Integration test: `PDF → extraction → preprocessing → generation`.
- [ ] UI tests: upload, enter text, generate, view, select answer, submit quiz.
- [ ] **Critical-failure tests (§24):** a distractor that is actually correct must
      be rejected; an answer unsupported by the context must be rejected; fewer
      than 3 distractors must invalidate the MCQ; duplicate options must be rejected.
- [ ] **Backend tests (§16.1):** the local and `hf_api` backends return the same
      structure; a missing/invalid token produces a clear error (mocked, no real calls).
- [ ] Edge cases (§66): empty, very short, very long, invalid PDF, text-less PDF, duplicate concepts, no distractors, model failure.

**Deliverable:** passing test suite.

---

## 22. Documentation & Deliverables  *(Phase 10)*

- [ ] Final `README.md` with setup + usage instructions.
- [ ] `requirements.txt` finalized and pinned.
- [ ] Dataset/corpus documentation (§36): source, count, domain, format, method, cleaning, pages, sentences, chunks, splits.
- [ ] `data/evaluation/README.md` citing the fixed chapter **exactly** (author,
      title, chapter, CC BY 3.0 license, source URL, retrieval date) per §75.2.
- [ ] References section (§75) — include the method references for FLAN-T5,
      Sentence Transformers, RAKE, and BLEU/ROUGE (§75.1), not only the original
      proposal references.
- [ ] Reproducibility record (§54): model/version, dataset version, seed, hyperparameters, embedding, chunk size/overlap, thresholds.
- [ ] Limitations section (§64) — honest discussion.
- [ ] Security/privacy notes (§65) — file validation, temp cleanup, no external uploads without consent.
- [ ] Project report.
- [ ] Presentation slides.
- [ ] Viva preparation notes (§63).

---

## 23. Final Success Criteria  *(§74)*

A user can:
- [ ] 1. Open the web application.
- [ ] 2. Upload a textbook PDF or enter text.
- [ ] 3. See the extracted text.
- [ ] 4. See preprocessing/chunking results.
- [ ] 5. See important candidate concepts.
- [ ] 6. Request a chosen number of questions.
- [ ] 7. Observe the model input.
- [ ] 8. Observe generated questions.
- [ ] 9. Observe distractor candidates.
- [ ] 10. Observe validation results.
- [ ] 11. Receive complete MCQs.
- [ ] 12. Inspect the source context for each question.
- [ ] 13. Interactively answer the generated MCQs.
- [ ] 14. Receive a final score.
- [ ] 15. Repeat the process with another document.

---

## 24. Milestone Checklist (High-Level)

- [ ] **M1 — Skeleton:** repo structure, config, requirements, README.
- [ ] **M2 — Input:** raw text + PDF extraction working.
- [ ] **M3 — Preprocess:** cleaning, segmentation, chunking.
- [ ] **M4 — Extraction:** candidates + ranking + embeddings.
- [ ] **M5 — Generation:** question generation + validation.
- [ ] **M6 — Distractors:** fixed distractor pipeline (§21.1) + validation.
- [ ] **M7 — Pipeline:** end-to-end `generate_mcqs()`.
- [ ] **M8 — Evaluation:** automatic + human evaluation.
- [ ] **M9 — UI:** Streamlit app + quiz mode.
- [ ] **M10 — Finalize:** tests, docs, report, presentation.

---

## 25. Open Questions / Decisions

- [ ] Which sentence segmenter: spaCy vs. NLTK?
- [x] Which QG model? **Resolved (§16.0):** primary
      `google/flan-t5-base`; documented low-memory fallback
      `google/flan-t5-small`. Backend is separate from model choice (§16.1).
- [x] Embedding: TF-IDF only, or include Word2Vec experiments? **Resolved
      (§5.2/§14):** TF-IDF (candidate importance/ranking) + Sentence Transformer
      `all-MiniLM-L6-v2` (semantic similarity / duplicates / distractor ranking)
      are core; Word2Vec is an **optional word-level experiment only**.
- [x] Is local fine-tuning required? **Resolved:** yes; train/adapt FLAN-T5
      locally on answerable SQuADv2 records before final release.
- [x] Which evaluation textbook corpus to use (license/source)? **Resolved
      (§35.2.1):** *Computer Networking: Principles, Protocols and Practice*
      by Olivier Bonaventure (CC BY 3.0), Chapter 3 — The Transport Layer
      (UDP + TCP). Source: https://github.com/obonaventure/cnp3.
- [ ] Chunk size / overlap defaults?
- [x] Local vs. remote question-generation execution? **Resolved (§16.1):**
      dual backend selectable via `CONFIG["question_model_backend"]`
      (`"local"` | `"hf_api"`), chosen **explicitly per run and never
      auto-switched mid-experiment** so results stay reproducible.

---

## 26. Fixed Baseline Target Text  *(from §35.2.1 — do this before Phase 2 testing)*

- [ ] Download Chapter 3 (Transport Layer) of *Computer Networking:
      Principles, Protocols and Practice* (Olivier Bonaventure, CC BY 3.0)
      as PDF and store it in `data/evaluation/`.
- [ ] Record title, author, edition, license, source URL, retrieval date
      in the dataset documentation (§36).
- [ ] Record page/sentence/chunk counts after preprocessing.
- [ ] Use this exact chapter for every baseline run before testing on any
      other material.
- [ ] Keep the backup-target rule in mind if this source becomes unavailable.

---

## 27. Question-Generation Backend Strategy  *(from §16.1 — apply during Phase 5)*

- [ ] Implement `generate_question(context, answer)` in
      `src/question_generator.py` with a pluggable backend.
- [ ] **Backend A (local):** load the fixed checkpoint
      (`google/flan-t5-base`, §16.0) once, cache it, use GPU if available.
      `google/flan-t5-small` only as an explicitly recorded low-memory swap.
- [ ] **Backend B (remote):** call the Hugging Face Inference API for the
      **same fixed model** using a token from an environment variable
      (`HUGGINGFACE_API_TOKEN`, never committed to Git); add retries/backoff
      and timeout handling.
- [ ] Make backend selectable via `CONFIG["question_model_backend"]`
      (`"local"` | `"hf_api"`).
- [ ] **Do not auto-switch backends mid-run** — a run must use exactly the
      backend recorded in its config, or the results are not reproducible.
- [ ] Record which backend + model were used per run (§54.1), and store them in
      `generation_records` for every question (§49.1).
- [ ] Document the privacy implication of sending text to the Inference
      API (see §65) and the network-dependence limitation (§64).

---

*Last updated: added Hugging Face Inference API fallback (§16.1), fixed baseline
target text (§35.2.1), tightened embedding roles, QG model choice, distractor
pipeline, backend selection, baseline spec, and evaluation methodology after
review; added method + corpus references (§75).*
