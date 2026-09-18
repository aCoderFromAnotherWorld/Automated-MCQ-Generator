# NLP Project --- Complete Implementation Plan

## Automated MCQ Generator from Textbook Using Natural Language Processing (NLP)

**Course:** CSE4122 --- Natural Language Processing Laboratory\
**Project Type:** NLP application with supervised model adaptation,
semantic processing, generation, validation, and interactive web
interface

------------------------------------------------------------------------

## 1. Project Overview

The project will develop an NLP-based system that automatically
generates multiple-choice questions (MCQs) from textbook text or PDF
documents.

The system will accept a textbook chapter, selected textbook text, or a
PDF as input. It will extract and preprocess the content, identify
important concepts or answer candidates, generate questions, create
plausible distractors, validate the generated MCQs against the source
material, and present the results through an **interactive web
interface**.

Each generated MCQ will contain:

-   One question
-   One correct answer
-   Three plausible distractors
-   The source context used to generate the question
-   The identified answer
-   Validation information where appropriate

The project will emphasize **faithfulness to the supplied textbook
content**. The system should avoid generating questions that depend on
information that is not supported by the input document.

------------------------------------------------------------------------

# 2. Main Project Goal

The main goal is:

> **To build an NLP system that can automatically generate reliable,
> textbook-grounded multiple-choice questions from text or PDF documents
> and visualize the complete input-to-output process through an
> interactive web interface.**

The project should demonstrate the major NLP stages rather than simply
calling an external question-generation API.

------------------------------------------------------------------------

# 3. Specific Objectives

1.  To extract textual content from textbook PDFs.
2.  To preprocess and segment textbook text into usable passages.
3.  To identify important concepts and candidate answers.
4.  To represent text using appropriate NLP representations such as
    TF-IDF and/or Word2Vec.
5.  To generate questions from selected contexts and answers.
6.  To generate three plausible distractors for each question.
7.  To validate generated questions and options against the source
    context.
8.  To compare suitable NLP/modeling approaches where feasible.
9.  To evaluate question and distractor quality using automatic and
    human evaluation.
10. To develop an interactive web interface for entering/uploading input
    and viewing generated MCQs.
11. To display relevant intermediate information so that the NLP
    pipeline can be demonstrated clearly.
12. To document the complete data collection, preprocessing, training,
    evaluation, and system-development process.

------------------------------------------------------------------------

# 4. Project Scope

## 4.1 In Scope

The system will support:

-   Raw text input
-   Textbook PDF input
-   Text extraction from PDF
-   Text cleaning
-   Sentence segmentation
-   Passage/chunk creation
-   Keyword/concept extraction
-   Candidate answer extraction
-   Question generation
-   Distractor generation
-   MCQ validation
-   Duplicate-question filtering
-   Interactive visualization
-   Question-count selection
-   Generated MCQ display
-   Source-context display
-   Model/evaluation information
-   Optional download/print of generated MCQs

## 4.2 Out of Scope

The initial version will not require:

-   A mobile application
-   A sophisticated production-grade GUI
-   OCR for scanned PDFs unless time permits
-   Training a large language model from scratch
-   A fully autonomous educational assessment platform
-   Perfect human-level question generation
-   External factual verification unrelated to the supplied textbook

------------------------------------------------------------------------

# 5. Core System Architecture

The overall system will follow this pipeline:

``` text
                 ┌──────────────────────┐
                 │   User / Web Client   │
                 └──────────┬───────────┘
                            │
                   Text / PDF Upload
                            │
                            ▼
                 ┌──────────────────────┐
                 │   Input Processing    │
                 │ PDF/Text Extraction   │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │  Text Preprocessing   │
                 │ Cleaning + Sentence   │
                 │ Segmentation + Chunk  │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Concept / Answer      │
                 │ Candidate Extraction  │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Question Generation   │
                 │ Transformer Model     │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Distractor Generation │
                 │ Semantic + NLP Method │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │    MCQ Validation     │
                 │ Relevance + Answer +  │
                 │ Duplicate Checking    │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Interactive Web UI    │
                 │ Input → Process → MCQ │
                 └──────────────────────┘
```
## 5.1 Detailed Implementation Pipeline (Canonical)

The diagram above is a **simplified high-level overview**. The diagram below is
the **canonical pipeline that the code in `src/` must implement**, and it is the
reference for module boundaries, ordering, and responsibility.

``` text
                 PDF / TEXT
                     │
                     ▼
              PDF Extraction
                     │
                     ▼
             Preprocessing
                     │
                     ▼
          Sentence Segmentation
                     │
                     ▼
                Chunking
                     │
                     ▼
       ┌─────────────────────────┐
       │ Candidate Extraction    │
       │ RAKE + spaCy            │
       └────────────┬────────────┘
                    ▼
          Candidate Ranking
          TF-IDF + features
                    │
                    ▼
            Answer Candidate
                    │
                    ▼
        ┌───────────────────────┐
        │ Question Generation   │
        │ FLAN-T5               │
        └───────────┬───────────┘
                    ▼
           Question Validation
                    │
                    ▼
        ┌───────────────────────┐
        │ Distractor Candidates │
        │ textbook concepts     │
        └───────────┬───────────┘
                    ▼
          Semantic Embeddings
          (Sentence Transformer)
                    │
                    ▼
          Type / Domain Filter
                    │
                    ▼
       Contextual Incorrectness
                    │
                    ▼
            Top 3 Distractors
                    │
                    ▼
             Final Validation
                    │
                    ▼
        Semantic Duplicate Check
                    │
                    ▼
              FINAL MCQ
                    │
                    ▼
             Streamlit / Quiz
```

## 5.2 Embedding Roles (Resolved: Which Embedding Is Used Where?)

Earlier drafts of this plan left the question *"which embedding are we actually
using?"* ambiguous. That ambiguity is now resolved: **each pipeline stage has
exactly one primary representation**, and each representation has one clearly
stated role.

  Purpose                        Recommended method                Required?
  ------------------------------ --------------------------------- -----------
  Candidate importance           TF-IDF                            Yes (core)
  Candidate ranking              TF-IDF + linguistic features      Yes (core)
  Semantic similarity            Sentence Transformer              Yes (core)
  Distractor similarity          Sentence Transformer              Yes (core)
  Duplicate-question detection   Sentence Transformer              Yes (core)
  Word-level experiment          Word2Vec                          No (optional)

Consequences for the implementation:

-   **TF-IDF is the core representation for candidate extraction and ranking.**
    It is cheap, deterministic, and fully explainable in a viva.
-   **Sentence Transformer embeddings are the core representation for all
    semantic work:** question-to-source relevance, distractor ranking, and
    duplicate-question detection. A small, fast model such as
    `all-MiniLM-L6-v2` is sufficient and runs comfortably on CPU.
-   **Word2Vec is demoted to an optional experiment**, not a required core
    component. It survives only as a comparison point in §14 and §39.

------------------------------------------------------------------------

# 6. Recommended Technology Stack

A practical implementation can use:

  Component                  Recommended Technology
  -------------------------- -------------------------------------------
  Programming Language       Python
  NLP                        spaCy
  Candidate Representation   TF-IDF (core)
  Semantic Embeddings        Sentence Transformer (`all-MiniLM-L6-v2`)
  Keyword Extraction         RAKE and/or spaCy
  Question Generation        FLAN-T5 (`google/flan-t5-base` primary,
                             `google/flan-t5-small` fallback)
  Distractor Generation      Textbook concepts + Sentence Transformer
                             similarity (WordNet optional)
  ML Classifier/Ranking      Scikit-learn
  PDF Extraction             PyMuPDF
  Similarity                 Cosine similarity
  Web Interface              Streamlit
  Visualization              Streamlit components + Plotly if required
  Model/Dataset Management   Hugging Face Transformers/Datasets
  Experiment Tracking        CSV/JSON/MLflow optional
  Development                Jupyter Notebook + Python modules
  Version Control            Git/GitHub

The stack can be simplified if computational resources are limited.

------------------------------------------------------------------------

# 7. Development Strategy

Do not build the complete system at once.

Build it incrementally:

``` text
Phase 0 → Repository scaffolding
Phase 1 → Basic text input
Phase 2 → PDF extraction
Phase 3 → Preprocessing, segmentation, and chunking
Phase 4 → Candidate extraction, ranking, and embeddings
Phase 5 → Question generation and validation
Phase 6 → Distractor generation and validation
Phase 7 → MCQ validation and duplicate filtering
Phase 8 → Evaluation
Phase 9 → Interactive web interface and quiz mode
Phase 10 → Integration, testing, and documentation
```

At every phase, keep a working version.

## 7.1 Canonical Phase Map (Single Source of Truth for Numbering)

Earlier drafts of this plan numbered phases inconsistently (for example,
"pipeline integration" appeared as Phase 7 in one section and Phase 10 in
another). The table below is the **single source of truth for phase numbering**.
`TODO.md` must use these exact phase numbers, and no section may invent its own.

  Phase   Name                                    Deliverable
  ------- --------------------------------------- --------------------------------
  Phase 0 Repository scaffolding                  Runnable skeleton + config
  Phase 1 Basic text input                        `generate_mcqs()` accepts raw text
  Phase 2 PDF extraction                          PDF → structured page text
  Phase 3 Preprocessing + segmentation + chunking  Clean text → overlapping chunks
  Phase 4 Candidate extraction, ranking,          Ranked, diverse answer candidates
          embeddings                              (+ representations)
  Phase 5 Question generation + question          Context + answer → valid question
          validation
  Phase 6 Distractor generation + distractor      Question + answer → 3 distractors
          validation
  Phase 7 MCQ validation + duplicate filtering    Validated, de-duplicated MCQs
  Phase 8 Evaluation                              Automatic + human evaluation
  Phase 9 Interactive web interface + quiz mode   Working Streamlit application
  Phase 10 Integration, testing, documentation    End-to-end tested + documented
                                                  project

**Cross-cutting work:** local SQuADv2 question-generation training (§17) is a
required workstream, not an optional add-on. Experimental comparisons (§39) and
export extras (§34) remain optional and are scheduled only after the relevant
core phase is working.

------------------------------------------------------------------------

# 8. Input Handling

The system should support two input modes.

## 8.1 Raw Text

The user can paste textbook content into a text area.

Example:

``` text
TCP is a connection-oriented transport layer protocol.
It provides reliable and ordered delivery of data.
```

## 8.2 PDF

The user can upload a textbook chapter in PDF format.

Recommended library:

``` text
PyMuPDF / fitz
```

Basic workflow:

``` text
PDF
 ↓
Read each page
 ↓
Extract text
 ↓
Combine pages
 ↓
Clean formatting artifacts
 ↓
Send to NLP pipeline
```

------------------------------------------------------------------------

# 9. PDF Preprocessing

PDF text often contains unwanted elements.

Possible problems:

-   Page numbers
-   Repeated headers
-   Repeated footers
-   Broken words
-   Multiple spaces
-   Line breaks in the middle of sentences
-   Special symbols
-   Table artifacts

A cleaning function should normalize these where possible.

Example:

``` python
def clean_text(text):
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()
```

Do not aggressively remove punctuation because punctuation can be useful
for NLP processing.

For every PDF, preserve the original extracted text separately so that
preprocessing can be inspected and debugged.

------------------------------------------------------------------------

# 10. Sentence Segmentation

After extraction, divide the document into sentences.

Possible tools:

-   spaCy
-   NLTK

Example:

``` text
Input:
TCP is reliable. UDP is connectionless.

Output:
1. TCP is reliable.
2. UDP is connectionless.
```

Sentence boundaries are important because answer candidates and question
contexts should normally be based on meaningful passages.

------------------------------------------------------------------------

# 11. Text Chunking

Transformer models have maximum input lengths.

Therefore, long textbook chapters should not be passed directly to the
model.

Use overlapping chunks.

Example:

``` text
Chunk 1:
Sentence 1 → Sentence 8

Chunk 2:
Sentence 7 → Sentence 14

Chunk 3:
Sentence 13 → Sentence 20
```

The overlap helps prevent important information from being lost at chunk
boundaries.

The exact chunk size should depend on the selected model.

Store metadata for every chunk:

``` python
{
    "chunk_id": 3,
    "text": "...",
    "page_start": 4,
    "page_end": 5
}
```

This will allow the web interface to show where a generated question
came from.

------------------------------------------------------------------------

# 12. Important Concept Extraction

The system needs to identify important concepts from each passage.

Possible approaches:

### Approach A --- RAKE

RAKE can extract important phrases based on word frequency and
co-occurrence.

Example:

``` text
Input:
TCP provides reliable data transmission using acknowledgments.

Keywords:
TCP
reliable data transmission
acknowledgments
```

### Approach B --- spaCy

spaCy can identify noun phrases and named entities.

Example:

``` text
"Transmission Control Protocol provides reliable delivery."

Candidates:
Transmission Control Protocol
reliable delivery
```

### Recommended Implementation

Start with:

``` text
RAKE + spaCy noun phrases
```

Then remove:

-   Very short candidates
-   Duplicate candidates
-   Generic stopwords
-   Candidates that are not meaningful

Keep the candidate extraction module independent so it can be replaced
later.

### Module Output Contract

To keep the module replaceable, `src/keyword_extractor.py` must return a **list
of dictionaries** with a fixed shape:

``` python
{
    "text": "Transmission Control Protocol",   # the candidate phrase
    "source": "rake" | "spacy_noun_chunk" | "spacy_ner",
    "rake_score": 8.4,                         # 0.0 if not from RAKE
    "chunk_id": 3,
    "is_noun_phrase": True,
    "is_named_entity": False,
    "entity_label": None                       # e.g. "ORG" when NER
}
```

Every downstream module (§13 ranking, §21 distractors) consumes **this exact
shape** and nothing else. Candidate extraction must never return bare strings, and
ranking must never reach back into spaCy objects.

------------------------------------------------------------------------

# 13. Candidate Answer Selection

Not every extracted keyword should become an answer.

Create a ranking mechanism.

Possible features:

-   Keyword score
-   Frequency
-   Position in paragraph
-   Noun-phrase status
-   Named-entity status
-   Semantic relevance to the chunk

Example:

``` text
Candidate                 Score
--------------------------------
Transmission Control      0.92
Acknowledgment            0.81
Reliable delivery         0.76
Protocol                  0.43
```

## Ranking Features (Concrete)

Each candidate is scored with a **weighted sum of normalized features**. All
features are min–max normalized to `[0, 1]` across the candidate pool before
combination, so weights are directly interpretable:

  Feature           Symbol           Default weight   Notes
  ----------------- ---------------- ---------------- --------------------------
  TF-IDF score      `tfidf_score`    0.35             core representation (§14.1)
  RAKE score        `rake_score`     0.20             0.0 if from spaCy only
  Frequency         `frequency`      0.15             occurrences in the chapter
  Noun-phrase       `is_noun_phrase` 0.10             1.0 or 0.0
  Named entity      `is_named_entity`0.10             1.0 or 0.0
  Position          `position`       0.10             earlier in chunk ranks higher

`final_score = Σ (weight × normalized_feature)`

Default weights live in `CONFIG["ranking_weights"]` so the ranking method can be
experimented with (§39 Experiment 1) without editing code.

## Hard Filters (Applied Before Ranking)

A candidate is discarded outright if it:

-   is shorter than `CONFIG["min_candidate_chars"]` (default 3),
-   is longer than `CONFIG["max_candidate_words"]` (default 6),
-   is composed entirely of stopwords,
-   is composed entirely of digits/punctuation,
-   duplicates another candidate after lowercasing, lemmatizing, and stripping
    punctuation.

## Diversity Rule (Concrete)

Selecting the top-N by score alone produces near-duplicate answers. The selection
step therefore enforces:

1.  Sort candidates by `final_score` descending.
2.  Accept a candidate only if its **normalized token overlap** with every
    already-accepted candidate is below `CONFIG["max_candidate_overlap"]`
    (default 0.6).
3.  Stop when `num_questions` distinct candidates are accepted, or when the
    candidate pool is exhausted (then generate fewer MCQs rather than padding).

Example of what requirement 2 prevents: `Transmission Control Protocol` and
`TCP` must not both become answers.

------------------------------------------------------------------------

# 14. Embedding Strategy

Because the project needs to explain how text representation works, the
implementation must **explicitly document which representation is used, where,
and why**.

This is now **decided, not exploratory.** Per §5.2, the project uses two core
representations plus one optional experiment:

1.  **TF-IDF** — candidate importance and candidate ranking (core, Phase 4)
2.  **Sentence Transformer embeddings** — all semantic similarity (core,
    Phases 5–7)
3.  **Word2Vec** — optional word-level comparison experiment only

## 14.1 TF-IDF (Core — Candidate Importance and Ranking)

TF-IDF weights each term/phrase by term frequency and inverse document frequency.

Usage in this project:

-   Chunks act as the "documents"; candidate n-grams act as the "terms".
-   A candidate frequent in one chunk but rare across the chapter scores higher,
    i.e. it is locally important and discriminative.
-   This score becomes the **`tfidf_score` feature** in candidate ranking (§13).

Why TF-IDF is the core choice:

-   No training needed, and fully deterministic — important for reproducibility.
-   Works on a single chapter; no large corpus required.
-   Directly explainable in a viva: the score can be inspected term by term.

## 14.2 Sentence Transformer Embeddings (Core — Semantic Similarity)

A Sentence Transformer encodes a **whole sentence or phrase** into one dense
vector, so semantic similarity between a question and a passage can be compared
directly with cosine similarity.

Recommended model: `sentence-transformers/all-MiniLM-L6-v2` (small, fast,
CPU-friendly, 384-dimensional).

Used for:

-   **Question ↔ source-context relevance** (question validation, §19 Check 5).
-   **Distractor candidate ranking** by relatedness to the answer and question
    (§21).
-   **Duplicate-question detection** by comparing each new question against
    previously accepted ones (§46).

``` text
Question / Passage text
        ↓
Sentence Transformer encoder
        ↓
Fixed-length dense vector (e.g. 384-dim)
        ↓
Cosine similarity
```

## 14.3 Word2Vec (Optional Experiment Only)

Word2Vec is **not required by the core system.** It is retained only as an
optional word-level representation experiment and may be dropped entirely without
affecting any core functionality.

If attempted, the two variants are:

### Pretrained Word2Vec

Use a pretrained Word2Vec or similar embedding.

Advantages: better with a small corpus, faster, no training required.

Disadvantages: the vectors were learned outside the project corpus, so the
embedding process is less concrete.

### Custom Word2Vec

Train Word2Vec on the collected textbook corpus.

``` text
Textbook Corpus
      ↓
Tokenization
      ↓
Training Word2Vec
      ↓
Word → Vector
```

Example:

``` text
"protocol" → [0.12, -0.31, 0.45, ...]
```

Word-level vectors must be **averaged** to compare phrases, and they capture
phrase similarity worse than sentence embeddings — which is precisely why
Word2Vec is optional here, not core.

## 14.4 Representation Summary

  Stage                              Representation              Core?
  ---------------------------------- --------------------------- -------
  Candidate importance               TF-IDF                      Yes
  Candidate ranking features         TF-IDF + linguistic feats   Yes
  Question ↔ context relevance       Sentence Transformer        Yes
  Distractor ↔ answer relatedness    Sentence Transformer        Yes
  Duplicate question detection       Sentence Transformer        Yes
  Word-level similarity experiment   Word2Vec                    No

------------------------------------------------------------------------

# 15. Question Generation

Question generation is the central component.

The model receives:

``` text
Context + Answer
```

and produces:

``` text
Question
```

Example:

``` text
Context:
TCP is a connection-oriented protocol that provides
reliable and ordered delivery.

Answer:
TCP

Generated question:
Which protocol provides reliable and ordered delivery?
```

------------------------------------------------------------------------

# 16. Question Generation Model

The question-generation component is a **sequence-to-sequence Transformer**. The
model choice is now **fixed** rather than left open, because leaving several
options open makes both the implementation and the experiments ambiguous.

## 16.0 Fixed Model Choice

  Role        Model                 Notes
  ----------- --------------------- ------------------------------------------
  Primary     `google/flan-t5-base` Used when hardware permits; better quality
                                    of the two.
  Fallback    `google/flan-t5-small`Used when `flan-t5-base` does not fit in
                                    available RAM/VRAM or is too slow.

Rules:

-   **Exactly these two base checkpoints are allowed.** Do not add another base
    architecture mid-project without recording the change.
-   The base model is set by `CONFIG["question_model"]`, which must be one of
    `"google/flan-t5-base"` or `"google/flan-t5-small"`.
-   Local inference after M19 uses the derived checkpoint at
    `CONFIG["question_model_checkpoint"]`; this is a trained copy of the
    selected base model, not a third base architecture.
-   The model choice and the execution backend (§16.1) are **independent
    settings.** Choosing the HF API backend does not change which model is
    requested; it only changes *where* it runs.

A question-generation prompt may follow:

``` text
generate question:
context: TCP is a connection-oriented protocol that provides
reliable and ordered delivery.
answer: TCP
```

Expected output:

``` text
Which protocol provides reliable and ordered delivery?
```

The exact prompt should be adjusted based on the selected model.

## 16.1 Execution Strategy (Local vs. Remote)

T5 and FLAN-T5 are sequence-to-sequence Transformers and can be
**exceptionally heavy to run locally**. Model size, RAM/VRAM limits, CPU-only
inference speed, and first-time download size can all become hardware
bottlenecks, especially on modest student machines.

To keep development unblocked, the project must support **two execution
backends** behind a single, stable interface. The question-generation module
(`src/question_generator.py`) should define one function --- for example
`generate_question(context, answer)` --- while the backend is selected by
configuration. Swapping backends must **not** require changing any calling code.

### Backend A --- Local Model (required for the trained final model)

-   Load the locally trained checkpoint from
    `CONFIG["question_model_checkpoint"]` after M19; use the selected base
    checkpoint only for pre-training smoke tests or before the local artifact
    exists.
-   Load the model **once** and cache it; never reload per request.
-   Use GPU if available, otherwise fall back to CPU gracefully.
-   Suitable when a lightweight model runs acceptably on the local machine.

### Backend B --- Hugging Face Inference API (fallback for hardware bottlenecks)

If local execution is too slow or runs out of memory, call the **Hugging Face
Inference API** instead of running the model locally.

-   Access the **same model chosen in §16.0** (`google/flan-t5-base` or
    `google/flan-t5-small`) over HTTPS; no local GPU or large download required.
-   Requires a Hugging Face access token, supplied via an environment variable
    (e.g. `HUGGINGFACE_API_TOKEN`). The token must **never** be hard-coded or
    committed to Git.
-   Use the `huggingface_hub` / `requests` client to send the same prompt that
    the local backend would use, and parse the returned text identically.
-   Apply retries with backoff and a clear timeout; surface a readable error if
    the API is unreachable, rate-limited, or the token is missing/invalid.

### Backend Selection Rules (Explicit, Never Automatic)

Automatic runtime fallback conflicts with reproducibility: two runs of the same
experiment could silently use different backends (and therefore different model
execution paths) without that difference ever being recorded. The backend is
therefore **chosen explicitly before a run and never switched mid-run.**

``` text
CONFIG["question_model_backend"]
    ├── "local"    → Backend A (local model)
    └── "hf_api"   → Backend B (Hugging Face Inference API)
```

-   The backend is set explicitly by `CONFIG["question_model_backend"]`; there is
    **no implicit switching** during generation.
-   If the selected backend fails (OOM, timeout, missing/invalid token), the run
    **stops with a clear error** instructing the user to change the setting. It
    does **not** silently continue on the other backend.
-   Record **which backend** was used for every run in the results and the report.
-   Keep both backends interchangeable so the rest of the pipeline (validation,
    distractors, scoring) is unaffected by the choice.
-   Manual convenience is still allowed: the Streamlit UI may offer a
    "switch backend and retry" control. That is a deliberate user action, not an
    automatic fallback.
-   A run must record **both** `question_model` and `question_model_backend`.

### Practical Notes

-   During development and unit testing, use a tiny local model or a mocked
    generation response to avoid repeated heavy calls.
-   Privacy: sending text to the Inference API transmits document content to an
    external service. Do not send private or licensed textbook content without
    the user's explicit permission (see §65).
-   Network dependence: the API backend requires internet access and is subject
    to external rate limits; document this as a known limitation.
-   If neither backend is available, the pipeline should degrade gracefully
    (clear error message and retry option) rather than crash.

------------------------------------------------------------------------

# 17. Required Local SQuADv2 Training Strategy

The project must train/adapt its question-generation model on the local
machine using the selected dataset: **SQuAD 2.0**. The base checkpoint is
FLAN-T5, and the training script is maintained under `training/train_qg.py`.
Hosted inference may be used only as an explicitly selected runtime backend;
it does not replace local training.

The workflow is split into two reproducible scripts:

1. `training/prepare_squad.py` filters out unanswerable records and converts
   SQuADv2 into JSONL records with `input_text` and `target_text`.
2. `training/train_qg.py` loads those records and fine-tunes FLAN-T5 locally
   with Transformers, PyTorch, and the local machine's CPU/GPU.

The resulting checkpoint is saved under
`models/question_generation/flan-t5-squadv2/` and is later loaded by
`src/question_generator.py` for inference.

### Important wording for the report

SQuAD is a **reading-comprehension dataset, not a dedicated question-generation
or MCQ dataset.** It was created for *answer extraction* (span selection), not for
generating questions. It contains:

``` text
Context
Question
Answer   (a span inside the context)
```

It is nevertheless **useful here for answer-aware question generation**, because
the `(context, answer) → question` relationship it contains is exactly the input
format this project uses (§15). What SQuAD does *not* provide is:

-   distractor options (SQuAD has no MCQ options),
-   deliberately designed wrong answers,
-   multiple plausible options per question.

So the report must state clearly that:

> SQuAD/SQuAD 2.0 is used as a source of **answer-aware question-generation**
> training signal (`context + answer → question`). It is **not** an MCQ dataset
> and provides **no distractor supervision**; distractor quality is therefore
> evaluated separately (§42) using the textbook corpus.

For SQuAD 2.0 specifically, note that it adds **unanswerable** questions
(§75 refs 4–5). These must be filtered out before training, because this project
always supplies a valid answer candidate.

### Corpus separation (critical)

The fine-tuning data and the evaluation data must never overlap:

``` text
SQuAD / SQuAD 2.0
   ↓
Question-generation model adaptation / training

Bonaventure, Chapter 3 (Transport Layer)   ← §35.2.1
   ↓
This system's evaluation / demonstration
```

If the model is fine-tuned on SQuAD, the evaluation chapter must remain
completely separate from it. Do not mix the two, and never evaluate on training
examples (§37).

The experiment can compare:

``` text
Base pretrained model
        vs
Fine-tuned model
```

This comparison is useful for the report because it demonstrates how
adaptation affects generation quality.

Do not assume that fine-tuning will always produce better results;
measure the difference experimentally.

### 17.1 Raw text to trained model to generated question

The complete explainable path is:

```text
SQuADv2 JSON
  → filter unanswerable records
  → normalize context and answer spans
  → input: context + answer
  → FLAN-T5 tokenizer
  → token IDs and attention masks
  → local gradient training / checkpoint saving
  → fine-tuned FLAN-T5 checkpoint

Textbook PDF or raw text
  → extraction and cleaning
  → sentence segmentation and overlapping chunks
  → RAKE/spaCy candidate extraction
  → TF-IDF candidate ranking
  → selected context + answer
  → Sentence Transformer embeddings for semantic checks
  → fine-tuned FLAN-T5 question generation
  → question validation
  → textbook concept distractor ranking
  → final MCQ validation
  → Streamlit display and quiz
```

The embedding layer has two deliberately separate roles. TF-IDF is a sparse,
interpretable representation used for candidate importance and ranking.
Sentence Transformer embeddings are dense semantic vectors used for
question-to-context relevance, distractor similarity, and duplicate-question
detection. The embedding layer does not train FLAN-T5; it supports selection,
similarity, and validation around the trained generator.

------------------------------------------------------------------------

# 18. Question Generation Pipeline

For every selected answer candidate:

``` text
Context
   +
Answer Candidate
   ↓
Question Generation Model
   ↓
Candidate Question
   ↓
Question Cleaning
   ↓
Question Validation
```

Generate multiple candidate questions if necessary.

For example:

``` text
Answer: TCP

Candidate 1:
Which protocol provides reliable delivery?

Candidate 2:
Which protocol provides reliable and ordered data transmission?

Candidate 3:
Which transport protocol is connection-oriented?
```

A ranking/validation stage can then select the most suitable one.

------------------------------------------------------------------------

# 19. Question Validation

Each question should be checked before proceeding.

### Check 1 --- Empty output

Reject if no meaningful question is generated.

### Check 2 --- Length

Reject extremely short or excessively long questions.

### Check 3 --- Duplicate

Compare against previously generated questions.

### Check 4 --- Answer presence

The question should be compatible with the selected answer.

### Check 5 --- Context relevance

The question should be supported by the source passage.

### Check 6 --- Unsupported information

Reject questions that introduce facts not contained in the supplied
context.

------------------------------------------------------------------------

# 20. Distractor Generation

After generating a question and correct answer, create three incorrect
options.

Example:

``` text
Question:
Which protocol provides reliable data delivery?

Correct:
TCP

Distractors:
UDP
IP
ARP
```

The challenge is not merely generating incorrect words. The distractors
should be plausible.

------------------------------------------------------------------------

# 21. Distractor Generation Strategy

Earlier drafts described this method as
"WordNet/Sense2Vec + semantic similarity + corpus candidates", which is too broad
to implement reproducibly. **The final method is now fixed** to the pipeline
below, and this pipeline is the only one required for the core system.

``` text
Textbook / corpus candidate concepts   (pool from §12–§13)
              ↓
Candidate set = all ranked concepts except the correct answer
              ↓
Semantic embedding similarity          (Sentence Transformer, §14.2)
   score = 0.5·cos(cand, answer) + 0.5·cos(cand, question)
              ↓
Type / domain filter                   (prefer same entity label / phrase type)
              ↓
Contextual incorrectness check         (reject anything the context supports)
              ↓
Rank remaining candidates by combined similarity score
              ↓
Select top 3
```

## 21.1 Stage Definitions (Concrete)

1.  **Candidate source — textbook/corpus concepts (required).** The candidate pool
    is the ranked concept list from §12–§13, together with all accepted answer
    candidates from the whole chapter. This keeps distractors grounded in the
    supplied textbook.
2.  **Remove the correct answer.** Case-insensitive, punctuation-insensitive
    comparison against the answer.
3.  **Semantic embedding similarity (required).** Using the Sentence Transformer
    from §14.2, compute
    `score = 0.5 · cos(cand, answer) + 0.5 · cos(cand, question)`.
    Distractors must be *near* the answer in meaning — that is what makes them
    plausible.
4.  **Type / domain filter.** If the answer is a named entity with label `L`,
    prefer candidates whose label is also `L`. If the answer is a noun phrase,
    prefer noun phrases. This satisfies §23 Check 5.
5.  **Contextual incorrectness check.** Reject a candidate if the source context
    actually supports it as the answer to this question (checked with the same
    embedding model against the question). This satisfies §23 Check 4.
6.  **Rank and select top 3.** Take the three highest-scoring survivors. If fewer
    than three survive, mark the MCQ invalid (§52) rather than padding with
    unrelated words.

## 21.2 Optional Sources (Experiments / Fallbacks Only)

These sources are **not core**. Use them only if the textbook-derived pool is too
small, or as an ablation experiment (§39 Experiment 4).

### WordNet

English lexical relationships (synonyms, hypernyms, hyponyms). Useful as a
fallback when the chapter yields too few in-domain concepts. Risky on its own
because WordNet relations are *lexical*, so it can produce distractors that are
semantically *equivalent* to the answer. Any WordNet distractor must still pass
the contextual incorrectness check (stage 5).

### Sense2Vec

Retrieves contextually related terms. Optional; it adds a heavy dependency and is
therefore not part of the core pipeline.

### Word-embedding similarity (Word2Vec)

Only as part of the optional Word2Vec experiment (§14.3, §39 Experiment 4).

### Corpus candidates (primary)

Concepts extracted from the same textbook — this is the **primary** source per
stage 1, because distractors then belong to the same conceptual domain.

------------------------------------------------------------------------

# 22. Recommended Distractor Strategy

> **The final method is fixed in §21.1.** This section only illustrates *why*
> textbook-grounded distractors are preferred. If §21.1 and this section ever
> appear to disagree, **§21.1 is authoritative.**

Prefer textbook-grounded distractors.

Example:

``` text
Context contains:
TCP
UDP
IP
HTTP
FTP
```

If the correct answer is:

``` text
TCP
```

candidate distractors can be:

``` text
UDP
IP
FTP
```

This produces options that belong to the same conceptual domain.

Avoid unrelated distractors such as:

``` text
Banana
Computer
Dhaka
```

even if they are technically incorrect.

------------------------------------------------------------------------

# 23. Distractor Validation

Each distractor should pass several checks. Checks 1, 2, 4, and 5 map directly
onto stages of the fixed pipeline (§21.1); Check 3 is produced by the scoring
formula in §21.1 stage 3.

## Check 1 --- Not equal to answer

``` python
distractor.lower() != answer.lower()
```

(Case- and punctuation-insensitive; this is stage 2 of §21.1.)

## Check 2 --- No duplicate distractors

All four options should be unique.

## Check 3 --- Semantic plausibility

The distractor should be related to the answer/question. This is measured
numerically by the §21.1 stage-3 score and reported as a raw number (§42.1), not
as a 1--5 rating.

## Check 4 --- Contextual incorrectness (CRITICAL)

The distractor should not actually be the correct answer to the question, judged
**against the source context** — not by intuition.

> A distractor that the source context actually supports as an answer is a
> **critical failure** (§42.1). It must be counted separately from ordinary
> low-quality distractors, and any MCQ containing one must be rejected.

## Check 5 --- Appropriate type

If the answer is a protocol, distractors should preferably also be
protocols.

If the answer is a person, distractors should preferably be people.

If the answer is a location, distractors should preferably be locations.

This is enforced by the type/domain filter (stage 4 of §21.1).

------------------------------------------------------------------------

# 24. MCQ Validation Layer

The complete MCQ should pass a final validation stage.

``` text
Question
Correct Answer
Distractor 1
Distractor 2
Distractor 3
Source Context
```

Validation:

``` text
               ┌─────────────────┐
               │ Generated MCQ   │
               └────────┬────────┘
                        ↓
             ┌─────────────────────┐
             │ Question Validation │
             └──────────┬──────────┘
                        ↓
             ┌─────────────────────┐
             │ Answer Validation   │
             └──────────┬──────────┘
                        ↓
             ┌─────────────────────┐
             │ Distractor Checks   │
             └──────────┬──────────┘
                        ↓
             ┌─────────────────────┐
             │ Source Relevance    │
             └──────────┬──────────┘
                        ↓
                   Final MCQ
```

Invalid MCQs should either be regenerated or discarded.

------------------------------------------------------------------------

# 25. Source-Grounding / Faithfulness

This is one of the most important project constraints.

The generated MCQ should be answerable from the supplied textbook
context.

For every MCQ store:

``` python
{
    "question": "...",
    "correct_answer": "...",
    "distractors": ["...", "...", "..."],
    "source_context": "...",
    "chunk_id": 5
}
```

The web interface should allow the user to inspect the source context.

This makes the system more transparent and helps demonstrate that the
generated question is grounded in the input.

------------------------------------------------------------------------

# 26. Interactive Web Interface Requirement

The project **must include an interactive web interface**.

A fancy GUI is not necessary. However, the user must be able to interact
with the system and observe the model's input and output.

Recommended framework:

> **Streamlit**

The interface should provide a clear:

``` text
USER INPUT
    ↓
NLP PROCESSING
    ↓
MODEL INPUT
    ↓
MODEL OUTPUT
    ↓
VALIDATION
    ↓
FINAL MCQ
```

workflow.

------------------------------------------------------------------------

# 27. Web Interface --- Main Page

Suggested layout:

``` text
-------------------------------------------------
        AUTOMATED MCQ GENERATOR
-------------------------------------------------

Input Method:
( ) Upload PDF
( ) Enter Text

[Upload PDF / Text Area]

Number of Questions:
[ 5 ]

Semantic Embedding:
[ Sentence Transformer (all-MiniLM-L6-v2) ]

Question Model:
[ FLAN-T5 (google/flan-t5-base) ]

Execution Backend:
( ) Local   ( ) Hugging Face Inference API

[ Generate MCQs ]

-------------------------------------------------
```

------------------------------------------------------------------------

# 28. Interactive Input Visualization

After the user provides input, show:

### Original Input

``` text
Original textbook text / PDF
```

### Extracted Text

``` text
Extracted text from PDF
```

### Cleaned Text

``` text
Preprocessed text
```

### Chunks

``` text
Chunk 1
Chunk 2
Chunk 3
...
```

Use expandable sections such as:

``` text
▼ Extracted Text
▼ Preprocessed Text
▼ Text Chunks
```

This is useful for demonstrating the NLP pipeline.

------------------------------------------------------------------------

# 29. Model Input Visualization

The interface should explicitly show what is passed into the
question-generation model.

Example:

``` text
Model Input

Context:
TCP is a connection-oriented protocol that provides
reliable and ordered delivery.

Answer:
TCP
```

Then display:

``` text
Generated Question

Which protocol provides reliable and ordered delivery?
```

This makes the project much easier to demonstrate during evaluation.

------------------------------------------------------------------------

# 30. Distractor Visualization

Show:

``` text
Correct Answer:
TCP

Candidate Distractors:
1. UDP
2. IP
3. FTP
4. HTTP
5. ARP
```

Then show the selected options:

``` text
Selected Distractors:
UDP
IP
FTP
```

This is optional in the final user-facing mode but useful in a
**debug/demo mode**.

------------------------------------------------------------------------

# 31. Final Interactive MCQ View

The final result should look like:

``` text
Question 1

Which protocol provides reliable and ordered delivery?

○ UDP
○ TCP
○ IP
○ ARP

[ Show Answer ]

Source Context
▼
TCP is a connection-oriented protocol...
```

The user should be able to interact with the options.

Possible functionality:

-   Select an answer
-   Submit answer
-   Show whether the selected answer is correct
-   Show correct answer
-   Move to next question
-   Restart quiz

This transforms the output from a static text result into an interactive
NLP demonstration.

------------------------------------------------------------------------

# 32. Recommended Web Interface Modes

Use two modes.

## Mode 1 --- Generator Mode

For instructors/project evaluators:

``` text
Upload → Generate → Inspect → Export
```

Shows:

-   Input
-   Extracted text
-   Keywords
-   Candidate answers
-   Model input
-   Generated questions
-   Distractors
-   Validation
-   Final MCQs

## Mode 2 --- Quiz Mode

For students:

``` text
Generated MCQs
       ↓
Answer questions
       ↓
Submit
       ↓
Score
```

The quiz mode is optional but strongly improves the interactive aspect
of the project.

------------------------------------------------------------------------

# 33. Web Interface Pages/Sections

A simple single-page Streamlit application is sufficient.

Suggested sections:

``` text
1. Input
2. Preprocessing
3. Candidate Extraction
4. Question Generation
5. Distractor Generation
6. Validation
7. Final MCQs
8. Interactive Quiz
9. Evaluation / Model Information
```

Avoid creating unnecessary complex navigation.

------------------------------------------------------------------------

# 34. User Controls

The interface can provide:

### Required

-   PDF upload
-   Text input
-   Number of questions
-   Generate button
-   Generated MCQ display

### Recommended

-   Model selection
-   Show/hide intermediate processing
-   Show source context
-   Regenerate question
-   Regenerate distractors
-   Quiz mode

### Optional

-   Download MCQs as TXT/PDF
-   Difficulty selection
-   Question type selection

Do not add controls that the underlying implementation does not actually
support.

------------------------------------------------------------------------

# 35. Project Data

There are two different datasets/corpora to consider.

## 35.1 Model Training/Adaptation Dataset

For question generation:

``` text
SQuAD / SQuAD 2.0
```

It provides:

``` text
Context
Question
Answer
```

This is the required local training/adaptation dataset for the question-generation
model. The training workflow is implemented in `training/` and runs on the
developer's machine.

> **Wording caveat (see §17):** SQuAD is a **reading-comprehension** dataset, not a
> dedicated question-generation or MCQ dataset. It is used here only as a source
> of **answer-aware** question-generation signal (`context + answer → question`).
> It provides **no distractor supervision**, so distractor quality is evaluated
> separately (§42). For SQuAD 2.0, filter out the unanswerable questions before
> training. This dataset must never overlap with the evaluation chapter (§35.2.1).

## 35.2 Evaluation Textbook Corpus

The final system should be tested on textbook material that is separate
from the question-generation training data.

Possible sources:

-   Open educational textbooks
-   Open course materials
-   Public-domain textbooks
-   Open-access educational documents

Document the source and license/usage conditions.

### 35.2.1 Named Baseline Target Text (Fixed Now, Not Later)

To avoid hunting for text data when the testing phases arrive, the
**primary evaluation target is fixed now**. All baseline testing and the
MVP demonstration should be performed on a single, specific chapter:

> **Primary target:** *"Computer Networking: Principles, Protocols and
> Practice"* by **Olivier Bonaventure** (open textbook, licensed under
> **Creative Commons Attribution 3.0 Unported (CC BY 3.0)**, source:
> https://github.com/obonaventure/cnp3, hosted at
> https://inl.info.ucl.ac.be/cnp3) --- **Chapter 3: The Transport Layer**
> (the chapter covering UDP and TCP).

Rationale:

-   It is an **openly licensed** textbook, so it can be used, stored, and
    redistributed in `data/evaluation/` without legal concerns.
-   Its transport-layer chapter is a natural match for the TCP/UDP
    examples used throughout this plan (§8, §59), so the expected
    questions and distractors are easy to sanity-check by hand.
-   It is a bounded, self-contained unit of text --- ideal for a
    reproducible baseline.

Concrete setup tasks:

-   [ ] Download the chapter as PDF (or clean text) and store it in
      `data/evaluation/`.
-   [ ] Record the exact title, author, edition/version, license, source
      URL, and retrieval date.
-   [ ] Record the number of pages, sentences, and resulting chunks after
      preprocessing.
-   [ ] Use this single chapter for the Step-by-Step baseline run before
      testing on any other material.

**Backup target (if the primary is unavailable):** any single openly
licensed computer-networking chapter that covers TCP and UDP (for
example, a chapter from another CC-licensed networking textbook or an
open course chapter). Keep the same rule: **one specific chapter, named
and stored up front.**

> **Rule:** Never begin the generation/evaluation phase without a
> concrete, named chapter already sitting in `data/evaluation/`.

------------------------------------------------------------------------

# 36. Data Collection Documentation

If textbook material is collected manually or automatically, document:

1.  Source
2.  Number of documents
3.  Subject/domain
4.  File format
5.  Collection method
6.  Selection criteria
7.  Cleaning process
8.  Number of pages
9.  Number of sentences
10. Number of chunks
11. Train/validation/test split where applicable

Example:

``` text
Corpus:
Computer Networking textbook chapters

Documents:
10 chapters

Format:
PDF

Extraction:
PyMuPDF

Preprocessing:
Header/footer removal,
whitespace normalization,
sentence segmentation,
chunking
```

------------------------------------------------------------------------

# 37. Dataset Splitting

For supervised model experiments:

``` text
Training Set
Validation Set
Test Set
```

For example:

``` text
80% Training
10% Validation
10% Testing
```

The exact split should be documented rather than assumed.

Do not evaluate a fine-tuned model on examples used during training.

------------------------------------------------------------------------

# 38. Baseline System

Before implementing the full Transformer system, create a baseline. The baseline
must be **fully specified** here, because a vaguely described baseline cannot be
reproduced or fairly compared against the Transformer system.

## 38.1 Fixed Baseline Specification

The baseline is deliberately non-neural so that it is deterministic and cheap:

``` text
Ranked answer candidates (§13, top-N by TF-IDF + features)
      ↓
Template question generation (deterministic rules, below)
      ↓
Distractors = next-highest ranked candidates from the same chunk
      ↓
Same validation as the main system (§19, §23, §24)
```

**Step 1 — Answer selection.** Use the ranked candidate list from §13 with the
same diversity rule (§13: `max_candidate_overlap = 0.6`). No new logic.

**Step 2 — Question templates (exact rules).** Pick the template by the
candidate's surface form, in this fixed priority order:

``` text
1. Acronym / all-caps token (TCP, UDP, IP, HTTP)
     "Which of the following is {answer}?"
2. Candidate is a noun phrase and appears as the subject of a definitional
   sentence in the chunk ("X is ..." / "X refers to ..." / "X is called ...")
     "What is {answer}?"
3. Candidate is the object of a definitional sentence
   ("... is called {answer}" / "... is known as {answer}")
     "Which of the following is known as {answer}?"
4. Fallback (no definitional pattern matched)
     "Which of the following is related to {answer}?"
```

Templates are applied by **string rules only** — no model is involved. The
matched template name and rule number must be stored with the question
(`"baseline_template": "rule_2"`) so the comparison is auditable.

**Step 3 — Distractors.** Take the next three highest-ranked candidates from
§13 that (a) are not the answer, (b) are not duplicates of each other, and
(c) occur in the same chunk or document. If fewer than three qualify, mark that
baseline MCQ invalid (§52) — never pad.

**Step 4 — Validation.** Run the baseline through the *same* validators used by
the Transformer system (§19, §23, §24), so the comparison isolates the
generation method, not the validation.

## 38.2 What the Baseline Is For

The baseline exists only to answer one question in the report:

> Does the FLAN-T5 generator produce better questions than a deterministic
> template method, measured with the same evaluation protocol (§40–§43)?

Both systems are evaluated on the **same** fixed chapter (§35.2.1) with the same
`num_questions` and the same validation settings.

------------------------------------------------------------------------

# 39. Experimental Comparisons

A strong report can compare several components. **Only comparisons that can be
completed and evaluated honestly should be attempted** — a smaller set of
well-executed comparisons is worth more than a large set of half-finished ones.

## Experiment 1 --- Candidate Extraction (cheap, required)

``` text
RAKE only
vs
spaCy noun phrases only
vs
RAKE + spaCy combined (the proposed system)
```

Measure: number of candidates, overlap with a hand-labelled gold set drawn from
the fixed chapter (§35.2.1).

## Experiment 2 --- Candidate Ranking / Representation (cheap, required)

``` text
TF-IDF only
vs
TF-IDF + linguistic features (§13)
```

Measure: whether the top-ranked candidates match the hand-labelled important
concepts.

## Experiment 3 --- Question Generation (required, this is the core comparison)

``` text
Deterministic template baseline (§38)
vs
FLAN-T5 (fixed model per §16.0)
```

Measure: BLEU/ROUGE *and* human scores (§40–§41). The baseline is the reference
point; the Transformer must beat it on the human criteria to justify its cost.

## Experiment 4 --- Distractor Generation (recommended)

``` text
Embedding similarity only
vs
Full fixed pipeline (corpus candidates → embedding similarity → type filter
→ contextual incorrectness, §21.1)
```

Measure: plausibility, incorrectness, diversity (§42).

## Experiment 5 --- Word2Vec (optional, exploratory only)

Word2Vec is **not** a core component (§14). It may be trained on the corpus as a
word-level side experiment and reported as such, but it must not be required for
the main pipeline to run.

## Fine-Tuning Comparison (optional analysis; local training is required)

``` text
Base FLAN-T5
vs
FLAN-T5 fine-tuned on SQuAD-style answer-aware QG data
```

Compare the base and locally fine-tuned checkpoints when resources permit;
never evaluate on training data (§37).

------------------------------------------------------------------------

# 40. Evaluation of Question Generation

## 40.1 Primary Evaluation: Source-Grounded Correctness + Human Judgement

Because this project generates **textbook-grounded** MCQs, the central evaluation
is whether the generated question is **supported by the source context** and
**useful to a learner**. BLEU and ROUGE are secondary, supporting evidence only.

The primary protocol, applied to every generated MCQ:

``` text
1. Source support        Is the question answerable from the supplied context?
2. Answer correctness    Is the stored answer actually correct per the context?
3. Question relevance    Is the question about a meaningful concept in the chapter?
4. Well-formedness       Is the question grammatical and unambiguous?
5. Single correct option  Exactly one of the four options is correct?
6. Distractor quality    Plausible, incorrect, and mutually distinct? (§42)
```

Items 1, 2, and 5 are checked **automatically** where possible (string/embedding
matching against the context) and then **confirmed by human reviewers**. Items
3, 4, and 6 are human-only. A question is reported as *valid* only if all
automatic checks pass; human scores are reported separately and never mixed into
the automatic validity rate.

## 40.2 Secondary Evaluation: BLEU / ROUGE

Automatic reference-based metrics can also be reported:

### BLEU

Measures n-gram overlap between generated and reference questions.

### ROUGE

Measures overlap with reference text.

**Important caveats** — these must appear wherever BLEU/ROUGE are reported:

- Reference questions are required. For the fixed evaluation chapter (§35.2.1)
  there are **no reference questions**, so BLEU/ROUGE can only be computed on a
  dataset that has them (e.g. a held-out SQuAD slice, §17).
- A generated question may be perfectly valid while scoring near-zero BLEU,
  because there are many correct ways to phrase the same question.
- BLEU/ROUGE must therefore **never** be presented as the headline result or as
  a measure of question quality. They are diagnostics for wording similarity.

------------------------------------------------------------------------

# 41. Human Evaluation

Human evaluation is the **primary** quality measure for this project (§40.1),
not an optional extra.

Ask several evaluators to rate generated MCQs.

## 41.1 Rated Criteria

Suggested criteria:

  Criterion                   Score
  ------------------------- -------
  Question relevance           1--5
  Grammatical quality          1--5
  Correctness                  1--5
  Source support               1--5
  Distractor plausibility      1--5
  Distractor incorrectness     1--5
  Overall quality              1--5

Calculate average scores.

## 41.2 Reviewer Protocol

- **Minimum 3 reviewers**, each rating the **same fixed set** of MCQs generated
  from the fixed chapter (§35.2.1).
- Reviewers must be shown the **source context** alongside each MCQ, so that
  "source support" and "correctness" can actually be judged.
- Reviewers must **not** be told whether a question came from the baseline (§38)
  or from FLAN-T5. The generation method is hidden to avoid bias.
- Report **per-criterion mean and standard deviation**, plus the number of
  reviewers, so agreement/disagreement is visible.
- Store raw ratings in `evaluation/human_evaluation.csv`, one row per
  (reviewer, question), before computing any averages.
- Save the empty rating form to Git so the protocol itself is reproducible.

Example (illustrative format only — replace with real measured values):

``` text
Relevance              4.4 / 5
Grammar                4.5 / 5
Correctness            4.6 / 5
Distractor quality     3.9 / 5
Overall                4.2 / 5
```

These numbers must come from actual evaluation and must not be
fabricated.

------------------------------------------------------------------------

# 42. Distractor Evaluation

Distractors should be evaluated separately.

Measure:

### Plausibility

Does the option look reasonable?

### Incorrectness

Is it actually incorrect?

### Similarity

Is it sufficiently related to the question?

### Diversity

Are the three distractors meaningfully different?

A good distractor should not be obviously unrelated.

## 42.1 How Each Criterion Is Measured

| Criterion | How it is measured | Who measures |
| --------- | ------------------ | ------------ |
| Plausibility | Does the option look reasonable for this question? | Human (1--5) |
| Incorrectness | Is it actually wrong **per the source context**? Verified by re-reading the context, not by intuition. | Human (1--5) + automatic context check |
| Similarity | Cosine similarity between distractor and answer (Sentence Transformer, §14) | Automatic (reported as a number) |
| Diversity | Pairwise similarity among the three distractors must be below a threshold | Automatic (reported as a number) |

Two rules that must be respected:

1. **Incorrectness must be verified against the source context.** A distractor
   that also appears as a correct statement in the context is a **critical
   failure** and must be counted separately from ordinary low-quality ones.
2. **Similarity and Diversity are reported as raw numbers**, not converted into
   1--5 scores, so they cannot be confused with human ratings.

------------------------------------------------------------------------

# 43. Validation Metrics

The system can report:

``` text
Total generated questions
Valid questions
Rejected questions
Regenerated questions
Duplicate questions
Average validation score
```

Example:

``` text
Generated: 30
Valid: 25
Rejected: 5
```

These values should be generated automatically from actual system runs.

------------------------------------------------------------------------

# 44. Coverage

The system should avoid generating all questions from one concept.

Track:

``` text
Candidate concepts
        ↓
Selected concepts
        ↓
Generated questions
```

For example:

``` text
TCP       → 2 questions
UDP       → 1 question
Routing   → 2 questions
IP        → 1 question
```

This helps demonstrate concept diversity.

------------------------------------------------------------------------

# 45. Difficulty Level

Difficulty classification can be added as an optional feature.

Possible categories:

``` text
Easy
Medium
Hard
```

Possible heuristics:

-   Question length
-   Concept complexity
-   Distractor similarity
-   Number of reasoning steps
-   Vocabulary complexity

Do not claim scientifically validated difficulty prediction unless it
has been evaluated using labeled data.

------------------------------------------------------------------------

# 46. Question Diversity

Prevent repetitive questions.

For each new question:

``` text
New Question
     ↓
Compare with previous questions
     ↓
Semantic similarity
     ↓
If too similar → reject/regenerate
```

## 46.1 Resolved Method

- **Representation:** Sentence Transformer embeddings (§14.2) — the same model
  used for distractor ranking. No second embedding method is introduced.
- **Comparison:** cosine similarity between the new question and **every already
  accepted** question.
- **Threshold:** `CONFIG["duplicate_similarity_threshold"]` (default `0.85`, §54).
  If any pairwise similarity `>= threshold`, the new question is rejected.
- **On rejection:** attempt regeneration from a different answer candidate
  (§18); if no replacement is found, drop it rather than ship a duplicate.
- The similarity of every rejected question and the question it collided with is
  recorded in the validation log, so diversity filtering is auditable.

Candidate concepts should also be diversified *before* generation (§13), since
preventing duplicates upstream is cheaper than filtering them downstream.

------------------------------------------------------------------------

# 47. Project Folder Structure

Recommended structure:

``` text
mcq-generator/
│
├── app.py
│
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── training/
│   └── evaluation/
│
├── models/
│   ├── embeddings/
│   ├── question_generation/
│   └── distractor/
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_embeddings.ipynb
│   ├── 04_question_generation.ipynb
│   ├── 05_distractor_generation.ipynb
│   └── 06_evaluation.ipynb
│
├── src/
│   ├── __init__.py
│   ├── pdf_processor.py
│   ├── preprocessing.py
│   ├── chunking.py
│   ├── keyword_extractor.py
│   ├── embeddings.py
│   ├── question_generator.py
│   ├── distractor_generator.py
│   ├── validator.py
│   ├── ranking.py
│   └── pipeline.py
│
├── training/
│   ├── __init__.py
│   ├── prepare_squad.py
│   ├── train_qg.py
│   └── README.md
│
├── ui/
│   ├── __init__.py
│   └── support.py
│
├── evaluation/
│   ├── metrics.py
│   ├── human_evaluation.csv
│   └── results/
│
└── outputs/
    ├── generated_mcqs/
    └── logs/
```

------------------------------------------------------------------------

# 48. Module Responsibilities

## `pdf_processor.py`

Responsibilities:

-   Load PDF
-   Extract text
-   Preserve page information

## `preprocessing.py`

Responsibilities:

-   Clean text
-   Normalize whitespace
-   Remove obvious artifacts

## `chunking.py`

Responsibilities:

-   Sentence segmentation
-   Create chunks
-   Maintain chunk metadata

## `keyword_extractor.py`

Responsibilities:

-   RAKE
-   spaCy noun phrases
-   Candidate ranking

## `embeddings.py`

Responsibilities:

-   **TF-IDF** construction and scoring (core — candidate importance/ranking, §14.1)
-   **Sentence Transformer** loading and encoding (core — semantic similarity, §14.2)
-   Cosine similarity utilities
-   Single shared interface: `embed(text) -> vector`, `similarity(a, b) -> float`,
      so no module re-implements similarity
-   **Word2Vec** loading/training (optional experiment only, §14.3) — must never
      be required for the core pipeline to run

## `question_generator.py`

Responsibilities:

-   Build model input from context + answer (§16)
-   **Load the locally trained FLAN-T5 checkpoint once** and cache it (§16.0);
      model loading separated from generation. Use the base checkpoint only
      before M19 has produced the local artifact.
-   Ask for the workload without prescribing the backend
-   **Pluggable, explicitly configured backend** (§16.1): `local` or `hf_api`.
      The backend is fixed before the run and **never switched mid-run** (§54.1)
-   Generate candidate questions and decode/clean the output
-   Return structured results including `backend` and `model_name` (§49.1)

## `distractor_generator.py`

Responsibilities:

-   Implement the **fixed pipeline of §21.1, in order**: corpus candidates →
      remove answer → Sentence Transformer similarity → type/domain filter →
      contextual incorrectness check → top 3
-   Return the candidate pool and rejected candidates with reasons (§30, §49.1)
-   Never pad: if fewer than three distractors survive, report the MCQ as invalid
-   Optional WordNet fallback only when the textbook pool is exhausted (§21.2)

## `validator.py`

Responsibilities:

-   Check question
-   Check answer
-   Check distractors
-   Check source relevance
-   Remove duplicates

## `pipeline.py`

Responsibilities:

-   Connect all modules

## `training/prepare_squad.py`

Responsibilities:

-   Read official SQuADv2 JSON files
-   Exclude unanswerable questions
-   Normalize context and answer spans
-   Write reproducible answer-aware QG JSONL records

## `training/train_qg.py`

Responsibilities:

-   Load prepared local JSONL data
-   Tokenize context-plus-answer inputs and question targets
-   Fine-tune FLAN-T5 locally with PyTorch and Transformers
-   Save the tokenizer, model checkpoint, and training configuration

## `ui/` and `app.py`

Responsibilities:

-   Render controls and pipeline artifacts
-   Display final validated MCQs and source context
-   Manage quiz state and presentation-only PDF preview
-   Call `src.pipeline.generate_mcqs()` through the documented contract

The UI must not import training internals, implement embeddings, rank
candidates, generate questions, or validate MCQs. `src/` must remain usable
from scripts and tests without Streamlit.

------------------------------------------------------------------------

# 49. Pipeline API Design

Keep the core NLP pipeline independent from Streamlit.

For example:

``` python
result = generate_mcqs(
    text=text,
    num_questions=5
)
```

Return structured data.

## 49.1 Required Top-Level Keys

The Streamlit UI is required to display the **whole pipeline** (§28–§33), so the
result object must expose every intermediate stage, not just the final MCQs:

``` python
{
    "config": {...},            # the exact CONFIG used for this run (§54)
    "source": "...",            # original raw input (text or extracted PDF text)
    "pages": [...],             # per-page text when the input was a PDF (§8.2)
    "cleaned_text": "...",      # after preprocessing (§9)
    "sentences": [...],         # after sentence segmentation (§10)
    "chunks": [...],            # with chunk_id / page_start / page_end (§11)
    "candidates": [...],        # all extracted candidates with scores (§12)
    "ranked_candidates": [...], # after ranking + diversity filtering (§13)
    "generation_records": [     # one record per generation attempt
        {
            "chunk_id": 2,
            "answer": "TCP",
            "model_input": "generate question: context: ... answer: TCP",
            "backend": "local",         # or "hf_api" (§16.1)
            "model_name": "google/flan-t5-base",
            "candidate_questions": ["...", "..."],
            "selected_question": "..."
        }
    ],
    "distractor_records": [     # candidate + rejected distractors (§30)
        {
            "answer": "TCP",
            "candidate_pool": ["UDP", "IP", "ARP", "HTTP"],
            "selected": ["UDP", "IP", "ARP"],
            "rejected": [{"option": "HTTP", "reason": "duplicate"}]
        }
    ],
    "validation": {             # aggregate + per-question (§19, §24, §43)
        "summary": {"generated": 30, "valid": 25, "rejected": 5},
        "per_question": [...]
    },
    "stats": {                  # counts and concept-coverage metrics (§43, §44)
        "...": "..."
    },
    "questions": [              # the final, validated MCQs
        {
            "question": "...",
            "answer": "...",
            "distractors": ["...", "...", "..."],
            "context": "...",
            "chunk_id": 2,
            "source_page": 5,
            "generator": "flan-t5"      # or "baseline" (§38)
        }
    ]
}
```

## 49.2 Rules for This Interface

- **Every key above is produced by `src/pipeline.py`**, never assembled inside
  `app.py`. The UI only reads and displays.
- The same function signature is used by scripts, notebooks, and tests, so a run
  can be reproduced without Streamlit:
  `result = generate_mcqs(text=text, num_questions=5)`.
- Intermediate keys must be **plain, JSON-serializable data** (lists, dicts,
  strings, numbers) so results can be written to `outputs/` for inspection and
  for the evaluation scripts (§43).
- If a stage is skipped (e.g. `pages` for text input), the key is present with an
  empty value rather than omitted, so the UI code never has to guess.

The web interface should only visualize this result.

This separation makes the project easier to test and maintain.

------------------------------------------------------------------------

# 50. Streamlit Application Flow

Basic flow:

``` python
uploaded_file = st.file_uploader(...)

text_input = st.text_area(...)

num_questions = st.number_input(...)

if st.button("Generate MCQs"):
    result = generate_mcqs(...)
```

Then show:

``` text
Input
 ↓
Preprocessing
 ↓
Candidates
 ↓
Generated Questions
 ↓
Distractors
 ↓
Validation
 ↓
Final MCQs
```

Use:

-   `st.expander()`
-   `st.tabs()`
-   `st.progress()`
-   `st.metric()`
-   `st.radio()`
-   `st.button()`

where useful.

------------------------------------------------------------------------

# 51. Interactive Quiz Implementation

The quiz mode can maintain:

``` python
current_question
selected_answer
score
```

For each question:

``` text
Question
   ↓
Options
   ↓
User selects option
   ↓
Submit
   ↓
Correct / Incorrect
   ↓
Explanation + Source Context
```

At the end:

``` text
Final Score:
4 / 5
```

This is an application-level feature and should be kept separate from
the NLP generation logic.

------------------------------------------------------------------------

# 52. Error Handling

The application should handle:

### Empty text

Display:

``` text
Please provide textbook text or upload a PDF.
```

### Invalid PDF

Display a clear error.

### PDF with no extractable text

Display:

``` text
No selectable text was found. OCR support is not available in the current version.
```

### Too little text

Do not attempt to generate a large number of questions.

### Model failure

Display an informative message and allow retry.

### No suitable distractors

Regenerate or mark the question as invalid.

------------------------------------------------------------------------

# 53. Performance Considerations

Transformer generation can be slow.

To improve performance:

-   Load the model only once.
-   Cache models using Streamlit caching.
-   Process chunks selectively.
-   Generate only the requested number of questions.
-   Avoid unnecessary repeated model calls.
-   Use GPU if available.
-   Keep a lightweight fallback model for development.

During development, use a small number of chunks/questions.

------------------------------------------------------------------------

# 54. Reproducibility

Record:

-   Model name
-   Model version
-   Dataset version
-   Random seed
-   Hyperparameters
-   Embedding method
-   Chunk size
-   Chunk overlap
-   Number of generated candidates
-   Validation thresholds

Example configuration:

``` python
CONFIG = {
    # --- generation volume ---
    "num_questions": 5,
    "num_distractors": 3,

    # --- chunking (§11) ---
    "chunk_size": 8,
    "chunk_overlap": 2,

    # --- candidate extraction / ranking (§12, §13) ---
    "keyword_method": "rake+spacy",
    "ranking_method": "tfidf+features",

    # --- semantic similarity (§14) ---
    # Sentence Transformer is the single semantic representation used for
    # distractor similarity and duplicate-question detection.
    "sentence_embedding_model": "all-MiniLM-L6-v2",

    # --- question generation (§16) ---
    # Model is FIXED for the project (§16.0). Backend is EXPLICIT (§16.1).
    "question_model_name": "google/flan-t5-base",
    "question_model_backend": "local",          # "local" | "hf_api"
    "question_model_max_new_tokens": 64,

    # --- Word2Vec is an optional experiment only, never required (§14) ---
    "enable_word2vec_experiment": False,

    # --- validation thresholds (§19, §23, §24, §46) ---
    "min_question_words": 5,
    "max_question_words": 40,
    "duplicate_similarity_threshold": 0.85,
    "max_candidate_overlap": 0.6,

    # --- reproducibility ---
    "random_seed": 42,
}
```

## 54.1 Configuration Rules

- **`question_model_backend` must be set explicitly before a run.** The system
  must **never silently switch backends mid-run** — otherwise two runs labelled
  "local" and "hf_api" could produce results from different models and quietly
  break reproducibility. If the configured backend fails, the run *fails with a
  clear error* (§52) and the user chooses to switch, which changes the recorded
  config.
- **`question_model_name` is fixed** to `google/flan-t5-base`, with
  `google/flan-t5-small` available only as an explicitly documented fallback for
  low-memory machines (§16.0). Changing it changes the recorded config.
- **The backend record** (`backend`, `model_name`) is stored inside
  `generation_records` for every question (§49.1), so each MCQ is traceable to
  the exact model and execution path that produced it.
- **Keep configuration separate from core code** — `CONFIG` lives in one place
  (`config.py`) and is injected into the pipeline, never hard-coded inside
  modules.
- Every run's effective `CONFIG` is returned inside the result object
  (`"config"`, §49.1) and written to `outputs/logs/` so results can be audited.

------------------------------------------------------------------------

# 55. Suggested Development Order

> **Mapping note:** the "Step" numbers below are an *implementation ordering*, not
> the project phase numbers. Phases come only from §7.1. Approximate mapping:
> Step 1 → Phase 0 · Step 2 → Phase 2 · Step 3 → Phase 3 · Steps 4–5 → Phase 4 ·
> Steps 6–7 → Phase 5 (+ required local SQuADv2 training) · Step 8 → Phase 6 ·
> Step 9 → Phase 7 · Step 10 → Phase 8 · Steps 11–12 → Phase 9 ·
> Step 13 → Phase 10.

## Step 1 --- Create the repository

Create the folder structure and Git repository.

## Step 2 --- Build PDF extraction

Input:

``` text
PDF
```

Output:

``` text
Clean text
```

## Step 3 --- Build preprocessing

Implement:

``` text
Cleaning
Sentence segmentation
Chunking
```

## Step 4 --- Build candidate extraction

Implement:

``` text
RAKE
spaCy noun phrases
```

## Step 5 --- Implement representations

Start with:

``` text
TF-IDF
```

Then implement:

``` text
Sentence Transformer (all-MiniLM-L6-v2)   ← core, for all semantic similarity
```

Optional word-level experiment only:

``` text
Word2Vec
```

## Step 6 --- Build question generation

Start with the fixed pretrained Transformer (§16.0): `google/flan-t5-base`
(`google/flan-t5-small` as the documented low-memory fallback). Backend is
explicit (§16.1).

## Step 7 --- Experiment with fine-tuning

Only after the baseline works. SQuAD is a reading-comprehension dataset used
for **answer-aware question generation** (§17).

## Step 8 --- Build distractor generation

Implement the **fixed distractor pipeline** (§21.1): textbook/corpus candidates
→ semantic embedding similarity → type/domain filter → contextual incorrectness
→ top 3. WordNet is an optional fallback (§21.2).

## Step 9 --- Build validation

Reject poor MCQs, including the critical failures in §24.

## Step 10 --- Build evaluation scripts

Calculate automatic metrics and prepare human evaluation.

## Step 11 --- Build Streamlit UI

Connect the tested pipeline.

## Step 12 --- Add interactive quiz mode

Add answer selection and scoring.

## Step 13 --- Final integration

Test the entire workflow from PDF upload to quiz.

------------------------------------------------------------------------

# 56. Recommended Minimum Viable Product

If time is limited, the minimum complete project should contain:

``` text
PDF/Text Input
      ↓
Text Extraction
      ↓
Preprocessing
      ↓
Keyword/Answer Extraction
      ↓
Question Generation
      ↓
Distractor Generation
      ↓
Validation
      ↓
Interactive Web Interface
      ↓
Final MCQs
```

The following can be considered extensions:

-   Fine-tuning
-   Custom Word2Vec
-   Difficulty prediction
-   Human evaluation dashboard
-   Quiz mode
-   Downloadable output

------------------------------------------------------------------------

# 57. Recommended Strong Version

For a stronger academic implementation. This is the **concrete target
architecture** — every box names a specific, implemented method.

``` text
                 PDF / TEXT
                     │
                     ▼
              PDF Extraction              (§8.2, §9)
                     │
                     ▼
             Preprocessing                (§9)
                     │
                     ▼
          Sentence Segmentation           (§10)
                     │
                     ▼
                Chunking                  (§11, overlapping)
                     │
                     ▼
       ┌─────────────────────────┐
       │ Candidate Extraction    │        (§12)
       │ RAKE + spaCy            │
       └────────────┬────────────┘
                    ▼
          Candidate Ranking               (§13)
          TF-IDF + features
                    │
                    ▼
            Answer Candidate
                    │
                    ▼
        ┌───────────────────────┐
        │ Question Generation   │         (§16, FLAN-T5)
        └───────────┬───────────┘
                    ▼
           Question Validation            (§19)
                    │
                    ▼
        ┌───────────────────────┐
        │ Distractor Candidates │         (§21, textbook concepts)
        └───────────┬───────────┘
                    ▼
          Semantic Embeddings             (§14, Sentence Transformer)
                    │
                    ▼
          Type / Domain Filter            (§23 Check 5)
                    │
                    ▼
       Contextual Incorrectness           (§23 Check 4)
                    │
                    ▼
            Top 3 Distractors
                    │
                    ▼
             Final Validation             (§24)
                    │
                    ▼
        Semantic Duplicate Check          (§46)
                    │
                    ▼
              FINAL MCQ
                    │
                    ▼
             Streamlit / Quiz             (§26, §31, §51)
```

## 57.1 Embedding Roles Are Separated by Purpose

There is **one semantic representation** (Sentence Transformer) and **one
lexical/statistical representation** (TF-IDF). They are not interchangeable and
each is used only for its stated job:

| Purpose                      | Method                      | Section |
| ---------------------------- | --------------------------- | ------- |
| Candidate importance         | TF-IDF                      | §12, §13 |
| Candidate ranking            | TF-IDF + linguistic features | §13 |
| Semantic similarity          | Sentence Transformer        | §14, §46 |
| Distractor similarity        | Sentence Transformer        | §21, §42 |
| Duplicate-question detection | Sentence Transformer        | §46 |
| Word-level experiment        | Word2Vec (**optional**)     | §14, §39 Exp 5 |

**This table is the single source of truth for "which embedding are we using?"**
Word2Vec is deliberately excluded from the required pipeline (§14).

------------------------------------------------------------------------

# 58. Possible Experimental Table

The final report can contain tables such as:

### Question Generation

  Model                      BLEU    ROUGE   Source Support   Human Quality
  ------------------------- -------- -------- --------------- --------------
  Template Baseline (§38)    Actual   Actual   Actual              Actual
  FLAN-T5 base               Actual   Actual   Actual              Actual
  FLAN-T5 fine-tuned         Actual   Actual   Actual              Actual

### Distractor Generation

  Method                 Plausibility   Incorrectness   Diversity   Similarity
  --------------------- -------------- --------------- ----------- -----------
  Embedding similarity        Actual         Actual        Actual      Actual
  Full pipeline (§21.1)       Actual         Actual        Actual      Actual

### Candidate Extraction (Experiment 1)

  Method        Candidates   Gold Overlap   Precision   Recall
  ------------ ------------ ------------- ----------- --------
  RAKE             Actual        Actual        Actual    Actual
  spaCy NPs        Actual        Actual        Actual    Actual
  RAKE + spaCy     Actual        Actual        Actual    Actual

**Important:** Replace every `Actual` with measured results. Never
fabricate evaluation numbers.

**Reporting rules:**

- The **primary table** is human + source-grounded evaluation (§40.1, §41).
  BLEU/ROUGE are reported beside it as secondary diagnostics, never alone.
- WordNet and Word2Vec rows appear **only if** those optional experiments
  (§39 Exp 4/5) were actually run. Do not include rows for experiments that
  were not performed.
- Every row must state the model and backend used (`google/flan-t5-base`,
  `local` or `hf_api`) so the numbers are traceable (§54.1).

------------------------------------------------------------------------

# 59. Final Output Example

Suppose the textbook contains:

``` text
TCP is a connection-oriented transport layer protocol.
It provides reliable and ordered delivery of data.
UDP is a connectionless protocol and does not provide
the same reliability mechanisms as TCP.
```

The system may generate:

``` text
Question:
Which transport layer protocol provides reliable and ordered
delivery of data?

A. UDP
B. TCP
C. IP
D. ARP

Correct Answer:
B. TCP
```

The interface should additionally provide:

``` text
Source Context:
TCP is a connection-oriented transport layer protocol...

Answer Candidate:
TCP

Model Input:
Context + TCP

Generated Question:
Which transport layer protocol provides reliable and ordered
delivery of data?

Distractor Candidates:
UDP
IP
ARP
HTTP

Selected Distractors:
UDP
IP
ARP

Validation:
✓ Question generated
✓ Correct answer supported by context
✓ Options are unique
✓ Three distractors selected
✓ Source context available
```

------------------------------------------------------------------------

# 60. AI-Assisted Development Workflow

Because the project will be developed with help from AI tools, use AI as
a coding and research assistant rather than allowing it to generate the
entire project blindly.

Work module-by-module.

Recommended workflow:

``` text
Requirement
    ↓
Ask AI for design
    ↓
Implement
    ↓
Run code
    ↓
Inspect output
    ↓
Ask AI to debug
    ↓
Test
    ↓
Document
```

Do not ask an AI tool to generate the entire application in one prompt
initially.

------------------------------------------------------------------------

# 61. Useful AI Prompts for Development

## Prompt 1 --- PDF Extraction

``` text
Implement a Python module that extracts text from PDF files using
PyMuPDF. Preserve page numbers and return structured page-level
text. Add error handling for invalid PDFs and empty PDFs.
Do not build the web interface yet.
```

## Prompt 2 --- Preprocessing

``` text
Implement an NLP preprocessing module for textbook text.
It should normalize whitespace, remove obvious PDF artifacts,
segment sentences using spaCy, and create overlapping chunks.
Keep the original text and processed text separately.
Write test cases for the module.
```

## Prompt 3 --- Candidate Extraction

``` text
Implement an answer-candidate extraction module using RAKE and
spaCy noun phrases. Return candidate phrases with scores and
source chunk IDs. Remove duplicates and obviously irrelevant
candidates.
```

## Prompt 4 --- Word2Vec

``` text
Implement a Word2Vec training module using gensim. Explain each
step of training and provide functions for obtaining word vectors
and cosine similarity. Include a small example for testing.
```

## Prompt 5 --- Question Generation

``` text
Implement a question-generation module using a Hugging Face
T5-compatible model. The module should accept context and answer,
generate multiple candidate questions, clean the generated text,
and return structured results. Keep model loading separate from
generation.
```

## Prompt 6 --- Distractors

``` text
Implement a distractor-generation module that takes a correct
answer, question, source context, and candidate concepts. Rank
candidate distractors using semantic similarity, remove the
correct answer and duplicates, and return three plausible
distractors.
```

## Prompt 7 --- Validation

``` text
Implement an MCQ validation module. Check question quality,
answer uniqueness, distractor uniqueness, source support,
semantic relevance, and duplicate questions. Return both a
boolean validity result and detailed validation reasons.
```

## Prompt 8 --- Streamlit

``` text
Build a Streamlit interface for an existing Python MCQ generation
pipeline. Users must be able to upload a PDF or enter raw text,
select the number of questions, generate MCQs, inspect extracted
text and model inputs, and interact with the final questions.
Do not put NLP logic directly inside app.py; import it from src.
```

## Prompt 9 --- Debugging

``` text
Here is my current module and the error/output I received.
First explain the root cause, then propose the smallest reliable
fix. Do not rewrite unrelated parts of the project.
```

## Prompt 10 --- Evaluation

``` text
Design an evaluation script for my automatic MCQ generation
system. Evaluate question generation with BLEU and ROUGE where
reference questions are available, and create a human-evaluation
CSV template for relevance, grammar, correctness, distractor
plausibility, and overall quality.
```

------------------------------------------------------------------------

# 62. AI Coding Rules

When using AI to implement the project:

1.  Ask for one module at a time.
2.  Understand every important function before using it.
3.  Run generated code immediately.
4.  Keep dependencies minimal.
5.  Do not blindly copy undocumented code.
6.  Ask AI to explain unfamiliar NLP algorithms.
7.  Keep experiments reproducible.
8.  Save working versions with Git.
9.  Test edge cases.
10. Record which model and dataset versions were used.

------------------------------------------------------------------------

# 63. What Should Be Explained in the Viva

The project should allow you to explain:

### NLP Fundamentals

-   Tokenization
-   Sentence segmentation
-   Stopwords
-   Stemming/lemmatization where used
-   TF-IDF
-   Word embeddings
-   Word2Vec
-   Cosine similarity

### NLP Generation

-   Sequence-to-sequence learning
-   Transformer architecture
-   Encoder-decoder concept
-   T5
-   Attention
-   Fine-tuning

### Question Generation

-   Context-answer formulation
-   Candidate answer extraction
-   Generation
-   Validation

### Distractor Generation

-   Semantic similarity
-   Candidate selection
-   Plausibility
-   Incorrectness

### Evaluation

-   BLEU
-   ROUGE
-   Precision/recall where applicable
-   Human evaluation

### Software Engineering

-   Modular architecture
-   API/module separation
-   Streamlit
-   Input validation
-   Error handling

------------------------------------------------------------------------

# 64. Important Limitations

The final report should honestly discuss limitations.

Possible limitations include:

1.  PDF extraction may fail on scanned documents.
2.  Transformer models may generate grammatically correct but factually
    unsupported questions.
3.  Distractors may occasionally be too easy or accidentally correct.
4.  Automatic BLEU/ROUGE scores do not fully represent question quality.
5.  Small domain-specific corpora may limit custom embedding quality.
6.  Question generation can require significant computational resources.
7.  Some textbook layouts may produce noisy PDF extraction.
8.  Generated questions may focus disproportionately on frequently
    occurring concepts.
9.  Human evaluation is subjective.
10. English-focused models may perform differently across domains.

------------------------------------------------------------------------

# 65. Security and Privacy Considerations

If the application accepts uploaded documents:

-   Do not permanently store user documents unless necessary.
-   Avoid exposing uploaded files publicly.
-   Validate file type and size.
-   Avoid executing uploaded files.
-   Delete temporary files after processing where practical.
-   Do not send private textbook/document content to an external API
    without user permission.

------------------------------------------------------------------------

# 66. Testing Plan

Testing should occur at multiple levels.

## Unit Testing

Test individual modules:

``` text
PDF extraction
Preprocessing
Chunking
Candidate extraction
Validation
```

## Integration Testing

Test:

``` text
PDF → extraction → preprocessing → generation
```

## UI Testing

Test:

``` text
Upload PDF
Enter text
Generate
View results
Select answer
Submit quiz
```

## Edge Cases

Test:

-   Empty input
-   Very short input
-   Very long input
-   Invalid PDF
-   PDF without text
-   Duplicate concepts
-   No suitable distractors
-   Model generation failure

------------------------------------------------------------------------

# 67. Final Demonstration Scenario

For the final presentation:

### Step 1

Open the web application.

### Step 2

Upload a textbook chapter.

### Step 3

Show extracted text.

### Step 4

Show preprocessing/chunks.

### Step 5

Show extracted candidate concepts.

### Step 6

Click **Generate MCQs**.

### Step 7

Show:

``` text
Model Input
```

### Step 8

Show:

``` text
Generated Question
```

### Step 9

Show:

``` text
Distractor Candidates
```

### Step 10

Show validation results.

### Step 11

Display final MCQs.

### Step 12

Enter quiz mode and answer the generated questions.

### Step 13

Show final score.

This provides a complete demonstration of:

``` text
NLP Processing
+
Machine Learning
+
Text Generation
+
Semantic Processing
+
Interactive Application
```

------------------------------------------------------------------------

# 68. Suggested 10-Week Development Plan

## Week 1 --- Literature Review and System Design

Study:

-   Automatic question generation
-   MCQ generation
-   Distractor generation
-   T5
-   Word2Vec
-   Evaluation methods

Deliverables:

-   Literature review
-   Architecture
-   Technology selection

------------------------------------------------------------------------

## Week 2 --- Dataset and Corpus

Tasks:

-   Select textbook/evaluation corpus
-   Collect question-generation dataset
-   Document sources
-   Inspect corpus

Deliverables:

-   Dataset
-   Data collection documentation

------------------------------------------------------------------------

## Week 3 --- PDF Processing

Implement:

-   PDF extraction
-   Cleaning
-   Sentence segmentation
-   Chunking

Deliverable:

``` text
PDF → Clean chunks
```

------------------------------------------------------------------------

## Week 4 --- Candidate Extraction and Embeddings

Implement:

-   RAKE
-   spaCy
-   TF-IDF
-   Word2Vec

Deliverable:

``` text
Chunk → Ranked answer candidates
```

------------------------------------------------------------------------

## Week 5 --- Question Generation

Implement:

-   Base Transformer
-   Context + answer input
-   Candidate question generation

Deliverable:

``` text
Context + Answer → Question
```

------------------------------------------------------------------------

## Week 6 --- Distractor Generation

Implement:

-   Candidate collection
-   Semantic similarity
-   Ranking
-   Three distractors

Deliverable:

``` text
Question + Answer → Complete MCQ
```

------------------------------------------------------------------------

## Week 7 --- Validation and Evaluation

Implement:

-   Relevance checks
-   Duplicate checks
-   Answer checks
-   Distractor validation
-   BLEU/ROUGE
-   Human evaluation form

Deliverable:

``` text
Validated MCQs + evaluation results
```

------------------------------------------------------------------------

## Week 8 --- Interactive Web Interface

Implement:

-   PDF upload
-   Text input
-   Generate button
-   Input visualization
-   Intermediate processing visualization
-   Model input/output visualization

Deliverable:

``` text
Working Streamlit application
```

------------------------------------------------------------------------

## Week 9 --- Interactive Quiz and Integration

Implement:

-   MCQ selection
-   Answer submission
-   Score calculation
-   Source-context display
-   Regeneration if required
-   Error handling

Deliverable:

``` text
Complete end-to-end application
```

------------------------------------------------------------------------

## Week 10 --- Testing, Documentation and Presentation

Tasks:

-   Final experiments
-   Human evaluation
-   Performance analysis
-   Screenshots
-   Report writing
-   Presentation preparation
-   Viva preparation

Deliverables:

-   Final report
-   Source code
-   Dataset documentation
-   Results
-   Presentation
-   Working web demo

------------------------------------------------------------------------

# 69. Final System Requirements

The completed project should satisfy these requirements.

## Functional Requirements

### FR1 --- Text Input

The system shall accept raw textbook text.

### FR2 --- PDF Input

The system shall accept textbook PDF documents.

### FR3 --- Text Extraction

The system shall extract text from PDF documents.

### FR4 --- Preprocessing

The system shall preprocess extracted text.

### FR5 --- Candidate Extraction

The system shall identify important concepts/answer candidates.

### FR6 --- Question Generation

The system shall generate questions from textbook passages.

### FR7 --- Distractor Generation

The system shall generate three distractors.

### FR8 --- Validation

The system shall validate generated MCQs.

### FR9 --- Interactive Visualization

The system shall visualize the input, intermediate processing, model
input, generated output, and validation results through a web interface.

### FR10 --- Interactive Quiz

The system should allow users to answer generated MCQs and view their
score.

------------------------------------------------------------------------

# 70. Non-Functional Requirements

### NFR1 --- Usability

The interface should be simple enough for a user to upload a document
and generate MCQs without technical knowledge.

### NFR2 --- Explainability

The system should expose important intermediate processing stages for
demonstration.

### NFR3 --- Reliability

Invalid generated questions should be filtered or regenerated.

### NFR4 --- Reproducibility

Experiments should use documented configurations and random seeds where
applicable.

### NFR5 --- Performance

The system should avoid unnecessary model loading and repeated
computation.

### NFR6 --- Maintainability

NLP modules and UI code should remain separated.

------------------------------------------------------------------------

# 71. Final Deliverables

The project should produce:

``` text
1. Source code
2. Dataset/corpus documentation
3. Preprocessing pipeline
4. Embedding implementation/experiments
5. Question-generation implementation
6. Distractor-generation implementation
7. Validation module
8. Evaluation scripts
9. Human evaluation results
10. Interactive Streamlit web application
11. Interactive quiz mode
12. Project report
13. Presentation slides
14. README with setup instructions
15. Requirements file
```

------------------------------------------------------------------------

# 72. Final Project Workflow

The final system can be summarized as:

``` text
                    USER
                     │
                     ▼
          ┌────────────────────┐
          │ Upload PDF / Text  │
          └──────────┬─────────┘
                     ▼
          ┌────────────────────┐
          │ Text Extraction    │
          └──────────┬─────────┘
                     ▼
          ┌────────────────────┐
          │ Preprocessing      │
          │ + Sentence Split   │
          │ + Chunking         │
          └──────────┬─────────┘
                     ▼
          ┌────────────────────┐
          │ Concept Extraction │
          │ RAKE + spaCy       │
          └──────────┬─────────┘
                     ▼
          ┌────────────────────┐
          │ Candidate Ranking  │
          │ TF-IDF + features  │
          └──────────┬─────────┘
                     ▼
          ┌────────────────────┐
          │ Question Generator │
          │ FLAN-T5 (base/     │
          │ small; local/API)  │
          └──────────┬─────────┘
                     ▼
          ┌────────────────────┐
          │ Question Validation│
          └──────────┬─────────┘
                     ▼
          ┌────────────────────┐
          │ Distractor         │
          │ Candidates         │
          │ (textbook concepts)│
          └──────────┬─────────┘
                     ▼
          ┌────────────────────┐
          │ Semantic Embedding │
          │ Similarity         │
          │ (Sentence Transf.) │
          └──────────┬─────────┘
                     ▼
          ┌────────────────────┐
          │ Type / Domain +    │
          │ Incorrectness Check│
          └──────────┬─────────┘
                     ▼
          ┌────────────────────┐
          │ MCQ Validation +   │
          │ Duplicate Filtering│
          └──────────┬─────────┘
                     ▼
          ┌────────────────────┐
          │ Interactive Web UI │
          └──────────┬─────────┘
                     ▼
             ┌───────────────┐
             │ Final MCQs    │
             │ + Quiz Mode   │
             └───────────────┘
```

------------------------------------------------------------------------

# 73. Final Recommended Architecture

The recommended final implementation is:

``` text
                     FRONTEND
                  Streamlit UI  (app.py)
                         │
                         ▼
                   APPLICATION LAYER
                         │
                         ▼
                   NLP PIPELINE
                   (src/pipeline.py)
                         │
   ┌──────────────┬──────┴───────┬────────────────┐
   ▼              ▼              ▼                ▼
Preprocessing  Extraction    Generation       Validation
src/           src/          src/             src/
preprocessing  keyword_      question_        validator
chunking       extractor     generator
               ranking       distractor_
               embeddings    generator
   │              │              │                │
   └──────────────┴──────┬───────┴────────────────┘
                         ▼
                  Structured result dict
                  (§49, includes intermediate
                   stages for the UI)
                         │
                         ▼
                  Final MCQs + Quiz / Export
```

Representations used per stage (see §5.2):

``` text
Extraction + Ranking   →  TF-IDF + linguistic features
Generation input       →  FLAN-T5 prompt (context + answer)
Semantic similarity    →  Sentence Transformer (all-MiniLM-L6-v2)
Word-level experiment  →  Word2Vec (optional, not core)
```

The most important architectural principle is:

> **Keep the NLP pipeline independent from the web interface.**

The NLP modules should be testable from notebooks and Python scripts
without Streamlit. The Streamlit application should act as the
interactive presentation layer over the tested NLP pipeline.

------------------------------------------------------------------------

# 74. Final Success Criteria

The project can be considered successfully implemented when a user can:

1.  Open the web application.
2.  Upload a textbook PDF or enter text.
3.  See the extracted text.
4.  See preprocessing/chunking results.
5.  See important candidate concepts.
6.  Request a chosen number of questions.
7.  Observe the model input.
8.  Observe generated questions.
9.  Observe distractor candidates.
10. Observe validation results.
11. Receive complete MCQs.
12. Inspect the source context for each question.
13. Interactively answer the generated MCQs.
14. Receive a final score.
15. Repeat the process with another document.

The project should demonstrate not only that MCQs can be generated, but
also **how the NLP system transforms textbook content into validated
interactive MCQs**.

------------------------------------------------------------------------

# 75. References

The original proposal references should be retained and corrected to use
actual web URLs rather than `mailto:` links.

## 75.1 Core Method References (added)

These references support the concrete choices fixed in this plan (§5.2, §14, §16,
§21, §35.2.1, §38) and should be cited where those choices are justified.

- **FLAN-T5 / instruction-tuned T5** (question-generation model, §16.0):
  Chung, H. W., et al. (2022). *Scaling Instruction-Finetuned Language Models*.
  arXiv:2210.11416.
- **Sentence-BERT / Sentence Transformers** (semantic similarity backbone, §14.2):
  Reimers, N., & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using
  Siamese BERT-Networks*. EMNLP. The `all-MiniLM-L6-v2` model card is at
  `https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2`.
- **RAKE** (candidate extraction, §12):
  Rose, S., Engel, D., Cramer, N., & Cowley, W. (2010). *Automatic Keyword
  Extraction from Individual Documents*. In *Text Mining: Applications and
  Theory*.
- **Baseline MCQ generation from text** (template baseline rationale, §38):
  Nwafor, C. A., & Onyenwe, I. E. *An Automated Multiple-Choice Question
  Generation Using Natural Language Processing Techniques*. arXiv.
- **Evaluation of question generation** (BLEU/ROUGE as secondary metrics, §40):
  Papineni, K., et al. (2002). *BLEU: a Method for Automatic Evaluation of
  Machine Translation*. ACL. Lin, C.-Y. (2004). *ROUGE: A Package for Automatic
  Evaluation of Summaries*. Text Summarization Branches Out.

## 75.2 Fixed Evaluation Corpus Reference (added)

The baseline target text fixed in §35.2.1 must be cited exactly, with its
license, in the report and in `data/evaluation/README.md`:

> Bonaventure, O. *Computer Networking: Principles, Protocols and Practice*.
> Open textbook. **Chapter: The Transport Layer** (UDP and TCP).
> License: **Creative Commons Attribution 3.0 Unported (CC BY 3.0)**.
> Source: `https://github.com/obonaventure/cnp3` /
> `https://inl.info.ucl.ac.be/cnp3`.

Record the exact version/edition, the retrieval date, and the chapter's
page/sentence/chunk counts alongside the file (§36).

## 75.3 Original Proposal References

1.  Neupane, A., Chaudhari, S., & Shah, S. (2025). *An Automated MCQ
    Generator using NLP*. International Journal of Innovative Science
    and Research Technology.

2.  Nwafor, C. A., & Onyenwe, I. E. *An Automated Multiple-Choice
    Question Generation Using Natural Language Processing Techniques*.
    arXiv.

3.  Saturi, R., Anusha, G., & Sony, C. (2025). *AI-Driven MCQ Generation
    Using NLP*. International Journal of Innovative Research in
    Information Technology.

4.  Rajpurkar, P., Jia, R., & Liang, P. (2018). *Know What You Don't
    Know: Unanswerable Questions for SQuAD*. Proceedings of ACL.

5.  Rajpurkar, P., Zhang, S., Lopyrev, K., & Liang, P. (2018). *SQuAD
    2.0: 100,000+ Questions for Machine Comprehension of Text*. EMNLP.

6.  SQuAD / SQuAD 2.0 dataset documentation.

7.  Hugging Face Transformers documentation for T5/FLAN-T5.

8.  spaCy documentation.

9.  Gensim Word2Vec documentation.

10. PyMuPDF documentation.

------------------------------------------------------------------------

# 76. One-Sentence Project Description

> **An interactive NLP system that extracts knowledge from textbook text
> or PDFs and automatically generates, validates, and presents
> source-grounded multiple-choice questions through a web interface.**
