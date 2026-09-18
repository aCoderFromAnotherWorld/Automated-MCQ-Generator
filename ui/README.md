# Presentation layer

`ui/` contains Streamlit-facing helpers and presentation concerns only. It may
render the result schema produced by `src.pipeline`, maintain quiz state, and
preview uploaded files. It must not implement tokenization, embeddings,
candidate ranking, question generation, distractor selection, or validation.
