# NLP runtime layer

`src/` contains reusable NLP components and the future
`generate_mcqs(...)` orchestration API. It must remain importable from tests,
notebooks, training utilities, and command-line scripts without importing
Streamlit.

Planned order:

`input -> preprocessing -> segmentation/chunking -> candidates -> ranking -> embeddings -> question generation -> validation -> distractors -> final MCQ`

Training scripts belong in `training/`; presentation code belongs in `ui/` and
`app.py`.
