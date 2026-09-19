# NLP runtime layer

`src/` implements the full MCQ pipeline end to end — it is importable from tests,
notebooks, training utilities, and command-line scripts, and must never import
Streamlit.

## Pipeline order

`input (text/PDF) → preprocessing → segmentation → chunking → candidate extraction
→ ranking → embeddings (TF-IDF + Sentence Transformer) → question generation →
round-trip answer verification → distractor generation → MCQ validation`

Entry point: `src/pipeline.py :: generate_mcqs(text, num_questions, *, config,
question_generator, similarity, answer_verifier)` — all three generator seams are
injectable so tests can run without real models.

## Modules

- `input_handler.py`, `pdf_processor.py` — validated source artifacts (text / PDF pages).
- `preprocessing.py` — conservative cleaning: soft hyphens, citation refs (`[a]`,
  `[9]`, `[note 1]`), line-wrap joining; preserves paragraph breaks (`\n\n`).
- `segmentation.py`, `chunking.py` — regex sentence IDs, then overlapping chunks built
  *inside* page/paragraph boundaries (never cross topics).
- `keyword_extractor.py` — dependency-free RAKE plus capitalised proper-noun phrases
  ("Bay of Bengal"); ~130-word stopword set; single-word fragments kept only when they
  also occur outside the phrase.
- `ranking.py` — weighted TF-IDF/RAKE/frequency/noun-phrase/NE/position/multi-word
  features, min-max normalised, with token-diversity filtering.
- `embeddings.py` — TF-IDF utilities, cached Sentence-Transformer encoder, batched
  `pairwise_cosine_similarity` for the selection stages.
- `question_generator.py` — answer-aware FLAN-T5 generation from the trained
  checkpoint (`generate question: context: … answer: …`), dual `local`/`hf_api`
  backends (explicit, never auto-switched), plus `answer_question` QA support on the
  base model for round-trip verification.
- `validator.py` — empty/length/duplicate/answer-presence/relevance checks plus a
  stem-aware, stopword-tolerant `unsupported_information` check.
- `distractor_generator.py` — textbook-grounded selection: score, type-match bonus,
  contextual-incorrectness critical check, stem/proper-option rejection, reuse cap.
- `mcq_validator.py` — final assembly with named reasons, critical failures
  (`answer_not_supported_by_context`, `fewer_than_three_valid_distractors`,
  `contextually_supported_distractor`, `duplicate_options`,
  `answer_appears_in_question`, `answer_not_verified`), and per-MCQ provenance.
- `contracts.py` — the JSON-serializable result schema every stage must honour.
- `api.py` — legacy pre-generation API (ranked candidates only), kept for history.

Training scripts belong in `training/`; presentation code belongs in `ui/` and
`app.py`.
