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

------------------------------------------------------------------------

# 6. Recommended Technology Stack

A practical implementation can use:

  Component                  Recommended Technology
  -------------------------- -------------------------------------------
  Programming Language       Python
  NLP                        spaCy, NLTK
  Embeddings                 Word2Vec / pretrained embeddings
  Keyword Extraction         RAKE and/or spaCy
  Question Generation        T5 / FLAN-T5
  Distractor Generation      WordNet/Sense2Vec + semantic similarity
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
Phase 1 → Basic text input
Phase 2 → PDF extraction
Phase 3 → Preprocessing
Phase 4 → Keyword/answer extraction
Phase 5 → Question generation
Phase 6 → Distractor generation
Phase 7 → Validation
Phase 8 → Evaluation
Phase 9 → Interactive web interface
Phase 10 → Integration and final testing
```

At every phase, keep a working version.

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

Select the top candidates while maintaining diversity.

Avoid selecting several candidates from exactly the same concept.

------------------------------------------------------------------------

# 14. Embedding Strategy

Because the project needs to explain how text representation works, the
implementation should explicitly document the embedding process.

Two approaches can be explored.

## 14.1 Pretrained Embedding

Use a pretrained Word2Vec or similar embedding.

Advantages:

-   Better with a small corpus
-   Faster
-   No need to train embeddings from scratch

Disadvantages:

-   The embedding process is less concrete
-   The vectors were learned outside the project corpus

## 14.2 Custom Word2Vec

Train Word2Vec on the collected textbook corpus.

Basic idea:

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

If the available corpus is sufficiently large, a custom embedding
experiment can be included.

## Recommended Comparison

If resources permit:

``` text
TF-IDF
   vs
Pretrained Word2Vec
   vs
Custom Word2Vec
```

The comparison does not need to be forced if the project timeline or
corpus size makes it impractical.

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

A sequence-to-sequence Transformer such as:

-   T5
-   FLAN-T5

can be used.

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

------------------------------------------------------------------------

# 17. Fine-Tuning Strategy

A pretrained question-generation model can first be tested directly.

Then, if computational resources permit, fine-tune it on a
question-generation dataset.

A suitable starting dataset is SQuAD/SQuAD 2.0 because it contains
context, question, and answer relationships.

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

Use a hybrid approach.

``` text
Correct Answer
      ↓
Find semantically related candidates
      ↓
Remove the correct answer
      ↓
Remove duplicates
      ↓
Check contextual incompatibility
      ↓
Rank candidates
      ↓
Select top 3 distractors
```

Possible sources:

### WordNet

Useful for English lexical relationships.

### Sense2Vec

Can retrieve contextually related terms.

### Word embeddings

Use cosine similarity to identify semantically similar candidates.

### Corpus candidates

Use concepts extracted from the same textbook.

The last approach is particularly useful because distractors can come
from concepts that actually appear in the textbook.

------------------------------------------------------------------------

# 22. Recommended Distractor Strategy

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

Each distractor should pass several checks.

## Check 1 --- Not equal to answer

``` python
distractor.lower() != answer.lower()
```

## Check 2 --- No duplicate distractors

All four options should be unique.

## Check 3 --- Semantic plausibility

The distractor should be related to the answer/question.

## Check 4 --- Contextual incorrectness

The distractor should not actually be the correct answer to the
question.

## Check 5 --- Appropriate type

If the answer is a protocol, distractors should preferably also be
protocols.

If the answer is a person, distractors should preferably be people.

If the answer is a location, distractors should preferably be locations.

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

Question Generation Model:
[ T5 ]

Embedding:
[ Word2Vec ]

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

This can be used for question-generation experimentation or fine-tuning.

## 35.2 Evaluation Textbook Corpus

The final system should be tested on textbook material that is separate
from the question-generation training data.

Possible sources:

-   Open educational textbooks
-   Open course materials
-   Public-domain textbooks
-   Open-access educational documents

Document the source and license/usage conditions.

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

Before implementing the full Transformer system, create a baseline.

Possible baseline:

``` text
Keyword extraction
      ↓
Template question generation
      ↓
Embedding-based distractors
```

Example template:

``` text
"What is {answer}?"
"Which of the following refers to {definition}?"
```

The baseline provides something to compare against.

------------------------------------------------------------------------

# 39. Experimental Comparisons

A strong report can compare several components.

## Experiment 1 --- Keyword Extraction

``` text
RAKE
vs
spaCy noun phrases
```

## Experiment 2 --- Text Representation

``` text
TF-IDF
vs
Pretrained Word2Vec
vs
Custom Word2Vec
```

## Experiment 3 --- Question Generation

``` text
Base T5
vs
Fine-tuned T5
```

## Experiment 4 --- Distractor Generation

``` text
WordNet
vs
Embedding similarity
vs
Hybrid method
```

Do not implement every experiment if the computational cost becomes
excessive. Select comparisons that can be completed and evaluated
properly.

------------------------------------------------------------------------

# 40. Evaluation of Question Generation

Automatic metrics can include:

### BLEU

Measures n-gram overlap between generated and reference questions.

### ROUGE

Measures overlap with reference text.

These metrics can be reported, but they should not be treated as the
only measure of question quality.

A generated question may be valid even if its wording differs
substantially from the reference question.

------------------------------------------------------------------------

# 41. Human Evaluation

Human evaluation is strongly recommended.

Ask several evaluators to rate generated MCQs.

Suggested criteria:

  Criterion                   Score
  ------------------------- -------
  Question relevance           1--5
  Grammatical quality          1--5
  Correctness                  1--5
  Distractor plausibility      1--5
  Overall quality              1--5

Calculate average scores.

Example:

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

Use cosine similarity with sentence embeddings or another semantic
representation.

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

-   Load pretrained embeddings
-   Train custom Word2Vec
-   Generate vectors
-   Calculate similarity

## `question_generator.py`

Responsibilities:

-   Load Transformer
-   Build model input
-   Generate candidate questions
-   Decode output

## `distractor_generator.py`

Responsibilities:

-   Generate candidate distractors
-   Rank candidates
-   Return three distractors

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

Return structured data:

``` python
{
    "source": "...",
    "chunks": [...],
    "candidates": [...],
    "questions": [
        {
            "question": "...",
            "answer": "...",
            "distractors": [...],
            "context": "...",
            "chunk_id": 2
        }
    ]
}
```

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
    "num_questions": 5,
    "chunk_size": 8,
    "chunk_overlap": 2,
    "embedding": "word2vec",
    "question_model": "t5",
    "num_distractors": 3
}
```

Keep configuration separate from core code.

------------------------------------------------------------------------

# 55. Suggested Development Order

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

## Step 5 --- Implement embeddings

Start with:

``` text
TF-IDF
```

Then implement:

``` text
Word2Vec
```

## Step 6 --- Build question generation

Start with a pretrained Transformer.

## Step 7 --- Experiment with fine-tuning

Only after the baseline works.

## Step 8 --- Build distractor generation

Start with corpus/embedding-based candidates.

## Step 9 --- Build validation

Reject poor MCQs.

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

For a stronger academic implementation:

``` text
PDF/Text
   ↓
Preprocessing
   ↓
Chunking
   ↓
RAKE + spaCy
   ↓
Candidate Answer Ranking
   ↓
Word2Vec / TF-IDF
   ↓
Base T5
   ↓
Fine-tuned T5
   ↓
Hybrid Distractor Generation
   ↓
MCQ Validation
   ↓
Semantic Duplicate Filtering
   ↓
Human + Automatic Evaluation
   ↓
Interactive Streamlit Application
```

------------------------------------------------------------------------

# 58. Possible Experimental Table

The final report can contain tables such as:

### Question Generation

  Model               BLEU    ROUGE   Human Quality
  --------------- -------- -------- ---------------
  Baseline          Actual   Actual          Actual
  Base T5           Actual   Actual          Actual
  Fine-tuned T5     Actual   Actual          Actual

### Distractor Generation

  Method       Plausibility   Correctness   Diversity
  ---------- -------------- ------------- -----------
  WordNet            Actual        Actual      Actual
  Word2Vec           Actual        Actual      Actual
  Hybrid             Actual        Actual      Actual

**Important:** Replace every `Actual` with measured results. Never
fabricate evaluation numbers.

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
          │ TF-IDF / Embedding │
          └──────────┬─────────┘
                     ▼
          ┌────────────────────┐
          │ Question Generator │
          │ T5 / FLAN-T5      │
          └──────────┬─────────┘
                     ▼
          ┌────────────────────┐
          │ Distractor         │
          │ Generation         │
          └──────────┬─────────┘
                     ▼
          ┌────────────────────┐
          │ MCQ Validation     │
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
              Streamlit UI
                    │
                    ▼
              APPLICATION
                 LAYER
                    │
                    ▼
             NLP PIPELINE
                    │
       ┌────────────┼────────────┐
       ▼            ▼            ▼
  Preprocessing  Extraction   Generation
       │            │            │
       │            │       ┌────┴─────┐
       │            │       ▼          ▼
       │            │    Questions  Distractors
       │            │       └────┬─────┘
       │            │            ▼
       └────────────┴────── Validation
                              │
                              ▼
                         Final MCQs
                              │
                              ▼
                         Quiz / Export
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
