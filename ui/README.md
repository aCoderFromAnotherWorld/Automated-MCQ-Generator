# Presentation layer

`ui/` contains Streamlit-facing helpers and presentation concerns only. It may
render the result schema produced by `src.pipeline`, maintain quiz state, and
preview uploaded files. It must not implement tokenization, embeddings,
candidate ranking, question generation, distractor selection, or validation.

## Current surface (`app.py`)

- Sidebar: input method (text/PDF), number of questions (1–20, default 3),
  question-model selector (`google/flan-t5-base` | `google/flan-t5-small`) and
  backend selector (`local` | `hf_api`).
- Generator tab: input/preprocessing expanders, per-stage pipeline artifacts
  (candidates, ranked candidates, generation records, distractor records,
  validation + stats), validated MCQs with per-question model/backend captions,
  source-context expanders, JSON download of the full run data.
- Quiz tab: shuffled options, submit → correct/incorrect feedback + source context,
  running score, final score (`N / M`), restart.
- Error states: empty input, too-short input, unreadable PDFs, generation failures
  (with a "switch backend and retry" hint).
- Model/backend caching lives in the pipeline layer (`lru_cache`); the UI caches
  PDF preview extraction only (`st.cache_data`).
