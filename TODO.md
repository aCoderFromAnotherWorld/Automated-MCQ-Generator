# TODO — Automated MCQ Generator (Development Tracker)

> Living progress tracker for the NLP project described in `ProjectDetails.md`.
> **Principle:** build incrementally, module-by-module. Keep the NLP pipeline
> independent from the web UI. Every phase must leave a working version.

**Legend:** `[ ]` todo · `[~]` in progress · `[x]` done · `[!]` blocked

---

## 0. Project Conventions & Ground Rules

- [ ] Keep the NLP pipeline (`src/`) fully independent from Streamlit (`app.py`).
- [ ] One module at a time — implement, run, inspect, test, then move on.
- [ ] Keep dependencies minimal; pin versions in `requirements.txt`.
- [ ] Never fabricate evaluation numbers — all metrics must come from real runs.
- [ ] Preserve original extracted text separately from cleaned text (debuggability).
- [ ] Keep configuration in a single `CONFIG` (separate from core code).
- [ ] Commit a working version to Git after each completed phase.
- [ ] Document model names, versions, seeds, and hyperparameters for reproducibility.

---

## 1. Repository Scaffolding  *(Phase 0)*

- [ ] Create folder structure (per `ProjectDetails.md` §47):
  - [ ] `src/` (Python package with `__init__.py`)
  - [ ] `data/raw/`, `data/processed/`, `data/training/`, `data/evaluation/`
  - [ ] `models/embeddings/`, `models/question_generation/`, `models/distractor/`
  - [ ] `notebooks/`
  - [ ] `evaluation/results/`
  - [ ] `outputs/generated_mcqs/`, `outputs/logs/`
- [ ] Create `requirements.txt` (start minimal, grow per phase).
- [ ] Create `.gitignore` (ignore `data/`, `models/`, `outputs/`, `__pycache__`, `.ipynb_checkpoints`, venv).
- [ ] Create `config.py` with the base `CONFIG` dict (§54).
- [ ] Expand `README.md` with setup + run instructions (placeholder now).
- [ ] Add `tests/` directory for unit tests.

**Deliverable:** clean, runnable skeleton committed to Git.

---

## 2. Input Handling — Raw Text  *(Phase 1)*

- [ ] Define the pipeline entry API: `generate_mcqs(text, num_questions, **opts)` (§49).
- [ ] Implement raw-text input path (accept a string, validate non-empty).
- [ ] Define the structured result schema returned by the pipeline (§49):
  ```python
  {
    "source": "...", "chunks": [...], "candidates": [...],
    "questions": [{"question","answer","distractors","context","chunk_id"}]
  }
  ```
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

## 8. Embeddings & Text Representation  *(Phase 4)*

- [ ] `src/embeddings.py` — TF-IDF representation (start here).
- [ ] Pretrained Word2Vec loading (gensim).
- [ ] Custom Word2Vec training on the textbook corpus (optional/if corpus is large).
- [ ] Functions: get word vector, cosine similarity.
- [ ] Document the embedding process explicitly (§14).
- [ ] Unit tests: vector retrieval, similarity sanity check.

**Deliverable:** `text → vectors` + similarity utilities.

---

## 9. Question Generation  *(Phase 5)*

- [ ] `src/question_generator.py` — load T5/FLAN-T5 (Hugging Face).
- [ ] Keep **model loading separate** from generation.
- [ ] Build model input: `context + answer` prompt (§16).
- [ ] Generate multiple candidate questions; clean/trim output.
- [ ] Return structured results.
- [ ] Unit tests with a tiny model / mocked generation.

**Deliverable:** `context + answer → candidate question(s)`.

---

## 10. Question Validation  *(Phase 5/7)*

- [ ] `src/validator.py` — question checks (§19):
  - [ ] empty output, length, duplicate, answer presence, context relevance, unsupported info.
- [ ] Return boolean validity + detailed reasons.
- [ ] Unit tests for each check.

**Deliverable:** `question → valid/invalid + reasons`.

---

## 11. Distractor Generation  *(Phase 6)*

- [ ] `src/distractor_generator.py` — hybrid strategy (§21–22):
  - [ ] corpus/textbook-grounded candidates (preferred),
  - [ ] WordNet lexical relations,
  - [ ] embedding cosine similarity.
- [ ] Remove correct answer + duplicates; rank candidates; select top 3.
- [ ] Distractor validation (§23): not equal to answer, unique, plausible, contextually incorrect, type-appropriate.
- [ ] Unit tests for selection + validation.

**Deliverable:** `question + answer + context → 3 plausible distractors`.

---

## 12. MCQ Validation Layer  *(Phase 7)*

- [ ] Assemble full MCQ: question + answer + 3 distractors + source context.
- [ ] Final validation pipeline (§24): question → answer → distractors → source relevance.
- [ ] Regenerate or discard invalid MCQs.
- [ ] Semantic duplicate-question filtering (§46) using sentence embeddings/cosine.
- [ ] Unit + integration tests.

**Deliverable:** `validated MCQs`.

---

## 13. Pipeline Integration  *(Phase 10)*

- [ ] `src/pipeline.py` — connect all modules end-to-end.
- [ ] Enforce source-grounding / faithfulness (§25): store `source_context` + `chunk_id` per MCQ.
- [ ] Concept coverage tracking (§44) — avoid over-generating from one concept.
- [ ] Performance: load models once, cache, generate only requested count (§53).
- [ ] Integration test: `text → final MCQs`.

**Deliverable:** `generate_mcqs()` returns complete, validated, grounded MCQs.

---

## 14. Baseline System  *(Phase 5, for comparison)*

- [ ] Template-based question generation (§38): `"What is {answer}?"`, etc.
- [ ] Embedding-based distractors.
- [ ] Keep baseline separate so it can be compared against the Transformer.

**Deliverable:** baseline MCQ generator for experimental comparison.

---

## 15. Evaluation  *(Phase 8)*

- [ ] `evaluation/metrics.py` — automatic metrics:
  - [ ] BLEU, ROUGE (where reference questions exist).
- [ ] Validation metrics (§43): generated / valid / rejected / regenerated / duplicates / avg score.
- [ ] Distractor evaluation (§42): plausibility, incorrectness, similarity, diversity.
- [ ] Human-evaluation CSV template (§41): relevance, grammar, correctness, distractor plausibility, overall (1–5).
- [ ] `evaluation/human_evaluation.csv` + `evaluation/results/`.
- [ ] Document dataset splits (§37) — never evaluate on training data.

**Deliverable:** evaluation scripts + results (real numbers only).

---

## 16. Experimental Comparisons  *(Phase 8, optional/if feasible)*

- [ ] Exp 1 — Keyword extraction: RAKE vs. spaCy noun phrases (§39).
- [ ] Exp 2 — Representation: TF-IDF vs. pretrained Word2Vec vs. custom Word2Vec.
- [ ] Exp 3 — Question generation: base T5 vs. fine-tuned T5.
- [ ] Exp 4 — Distractors: WordNet vs. embedding similarity vs. hybrid.
- [ ] Record results in report tables (§58) — replace every `Actual` with measured values.

**Deliverable:** comparison tables with real measured results.

---

## 17. Fine-Tuning (Optional / Strong Version)  *(Phase 7)*

- [ ] Prepare SQuAD / SQuAD 2.0 data (§35.1).
- [ ] Fine-tune T5/FLAN-T5 for question generation.
- [ ] Compare base vs. fine-tuned experimentally (§17).
- [ ] Document hyperparameters, seed, dataset version.

**Deliverable:** fine-tuned model + comparison results.

---

## 18. Interactive Web Interface (Streamlit)  *(Phase 9)*

- [ ] `app.py` — import pipeline from `src/`; **no NLP logic in `app.py`** (§49, §73).
- [ ] Input controls (§27, §34):
  - [ ] PDF upload / text area toggle,
  - [ ] number of questions,
  - [ ] model + embedding selection,
  - [ ] Generate button.
- [ ] Intermediate visualization (§28–30) via `st.expander()` / `st.tabs()`:
  - [ ] original input, extracted text, cleaned text, chunks,
  - [ ] candidate concepts, model input, generated questions,
  - [ ] distractor candidates + selected distractors, validation results.
- [ ] Final MCQ view (§31) with source-context display.
- [ ] Error handling UI (§52): empty text, invalid PDF, no text, too little text, model failure, no distractors.
- [ ] Performance: `st.cache` model loading (§53).

**Deliverable:** working Streamlit app over the tested pipeline.

---

## 19. Interactive Quiz Mode  *(Phase 9)*

- [ ] Quiz state: `current_question`, `selected_answer`, `score` (§51).
- [ ] Per-question flow: options → select → submit → correct/incorrect → explanation + source context.
- [ ] Final score display (`4 / 5`).
- [ ] Keep quiz logic separate from NLP generation logic.
- [ ] Optional: restart quiz, next question.

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
- [ ] Edge cases (§66): empty, very short, very long, invalid PDF, text-less PDF, duplicate concepts, no distractors, model failure.

**Deliverable:** passing test suite.

---

## 22. Documentation & Deliverables  *(Phase 10)*

- [ ] Final `README.md` with setup + usage instructions.
- [ ] `requirements.txt` finalized and pinned.
- [ ] Dataset/corpus documentation (§36): source, count, domain, format, method, cleaning, pages, sentences, chunks, splits.
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
- [ ] **M6 — Distractors:** hybrid distractor generation + validation.
- [ ] **M7 — Pipeline:** end-to-end `generate_mcqs()`.
- [ ] **M8 — Evaluation:** automatic + human evaluation.
- [ ] **M9 — UI:** Streamlit app + quiz mode.
- [ ] **M10 — Finalize:** tests, docs, report, presentation.

---

## 25. Open Questions / Decisions

- [ ] Which sentence segmenter: spaCy vs. NLTK?
- [ ] Which QG model: T5 vs. FLAN-T5 (size trade-off)?
- [ ] Embedding: TF-IDF only, or include Word2Vec experiments?
- [ ] Is fine-tuning feasible given compute/time?
- [x] Which evaluation textbook corpus to use (license/source)? **Resolved
      (§35.2.1):** *Computer Networking: Principles, Protocols and Practice*
      by Olivier Bonaventure (CC BY 3.0), Chapter 3 — The Transport Layer
      (UDP + TCP). Source: https://github.com/obonaventure/cnp3.
- [ ] Chunk size / overlap defaults?
- [x] Local vs. remote question-generation execution? **Resolved (§16.1):**
      dual backend — local small model (`t5-small`/`flan-t5-small`) by
      default, with Hugging Face Inference API as an explicit fallback for
      hardware bottlenecks, selectable via `CONFIG["question_model_backend"]`.

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
- [ ] **Backend A (local):** load a small checkpoint (`t5-small` /
      `flan-t5-small`/`flan-t5-base`) once, cache it, use GPU if available.
- [ ] **Backend B (remote fallback):** call the Hugging Face Inference API
      using a token from an environment variable (`HUGGINGFACE_API_TOKEN`,
      never committed to Git); add retries/backoff and timeout handling.
- [ ] Make backend selectable via `CONFIG["question_model_backend"]`
      (`"local"` | `"hf_api"`).
- [ ] Record which backend was used per run for reproducibility.
- [ ] Document the privacy implication of sending text to the Inference
      API (see §65) and the network-dependence limitation (§64).

---

*Last updated: added Hugging Face Inference API fallback (§16.1) and fixed
baseline target text (§35.2.1) from `ProjectDetails.md`.*