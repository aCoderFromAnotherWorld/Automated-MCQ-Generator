"""Streamlit web interface for the Automated MCQ Generator.

Run with: streamlit run app.py

The UI does not contain NLP logic. It renders the documented pipeline result
and calls ``src.pipeline.generate_mcqs`` when that Phase-10 integration point
is available.
"""

from __future__ import annotations

import json
import random
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import streamlit as st

# Some managed Python environments omit the working directory from ``sys.path``.
# Resolve local modules from this file's location so ``streamlit run app.py`` and
# direct execution behave consistently.
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import CONFIG, QUESTION_BACKENDS, QUESTION_MODELS
from ui.support import extract_pdf_pages, normalize_result


st.set_page_config(page_title="Automated MCQ Generator", page_icon="📝", layout="wide")


def _pipeline():
    """Return the future pipeline entry point without making the UI brittle."""

    try:
        from src.pipeline import generate_mcqs  # type: ignore[import-not-found]
    except ModuleNotFoundError as error:
        if error.name == "src.pipeline":
            return None
        raise
    return generate_mcqs


@st.cache_data(show_spinner=False)
def preview_pdf(file_bytes: bytes) -> list[str]:
    """Cache PDF preview extraction; model caching belongs to the pipeline."""

    return extract_pdf_pages(file_bytes)


def initialise_state() -> None:
    defaults: dict[str, Any] = {
        "result": None,
        "quiz_index": 0,
        "quiz_score": 0,
        "quiz_submitted": False,
        "quiz_options": [],
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def reset_quiz() -> None:
    st.session_state.quiz_index = 0
    st.session_state.quiz_score = 0
    st.session_state.quiz_submitted = False
    st.session_state.quiz_options = []


def render_input_artifacts(result: Mapping[str, Any]) -> None:
    with st.expander("Input and preprocessing", expanded=False):
        st.caption("Original input")
        st.text_area("Source text", result["source"], height=160, disabled=True, key="source_preview")

        pages = result["pages"]
        if pages:
            st.caption("Extracted PDF text")
            for index, page in enumerate(pages, start=1):
                with st.expander(f"Page {index}"):
                    st.text(page or "No extractable text on this page.")

        st.caption("Cleaned text")
        st.text(result["cleaned_text"] or "Waiting for preprocessing.")
        st.caption("Sentences and chunks")
        st.json({"sentences": result["sentences"], "chunks": result["chunks"]})


def render_pipeline_artifacts(result: Mapping[str, Any]) -> None:
    left, right = st.columns(2)
    with left:
        with st.expander("Candidate extraction and ranking"):
            st.json({"candidates": result["candidates"], "ranked_candidates": result["ranked_candidates"]})
        with st.expander("Question generation"):
            st.json(result["generation_records"])
    with right:
        with st.expander("Distractor generation"):
            st.json(result["distractor_records"])
        with st.expander("Validation and coverage"):
            st.json({"validation": result["validation"], "stats": result["stats"]})


def render_questions(result: Mapping[str, Any]) -> None:
    questions = result["questions"]
    if not questions:
        st.info("No validated MCQs are available yet.")
        return

    st.subheader("Validated MCQs")
    run_config = result["config"]
    for number, mcq in enumerate(questions, start=1):
        st.markdown(f"#### Question {number}")
        st.write(mcq.get("question", "Question text unavailable."))
        options = [mcq.get("answer", "")] + list(mcq.get("distractors", []))
        st.radio("Options", options, key=f"generator_option_{number}", index=None)
        st.caption(
            f"Model: {mcq.get('model_name', run_config.get('question_model', 'unknown'))} · "
            f"Backend: {mcq.get('backend', run_config.get('question_model_backend', 'unknown'))}"
        )
        with st.expander("Source context"):
            st.write(mcq.get("context", "Source context unavailable."))
        st.divider()


def render_quiz(result: Mapping[str, Any]) -> None:
    questions = result["questions"]
    st.subheader("Quiz mode")
    if not questions:
        st.info("Generate validated MCQs first; quiz mode uses only those results.")
        return

    index = st.session_state.quiz_index
    if index >= len(questions):
        st.success(f"Final score: {st.session_state.quiz_score} / {len(questions)}")
        if st.button("Restart quiz"):
            reset_quiz()
            st.rerun()
        return

    mcq = questions[index]
    if not st.session_state.quiz_options:
        options = [mcq.get("answer", "")] + list(mcq.get("distractors", []))
        random.shuffle(options)
        st.session_state.quiz_options = options

    st.caption(f"Question {index + 1} of {len(questions)} · Score: {st.session_state.quiz_score}")
    st.write(mcq.get("question", "Question text unavailable."))
    selected = st.radio("Choose one answer", st.session_state.quiz_options, index=None, key=f"quiz_choice_{index}")

    if not st.session_state.quiz_submitted:
        if st.button("Submit answer", type="primary", disabled=selected is None):
            st.session_state.quiz_submitted = True
            if selected == mcq.get("answer"):
                st.session_state.quiz_score += 1
            st.rerun()
        return

    if selected == mcq.get("answer"):
        st.success("Correct.")
    else:
        st.error(f"Incorrect. The correct answer is: {mcq.get('answer', '')}")
    with st.expander("Explanation and source context", expanded=True):
        st.write(mcq.get("context", "Source context unavailable."))
    if st.button("Next question", type="primary"):
        st.session_state.quiz_index += 1
        st.session_state.quiz_submitted = False
        st.session_state.quiz_options = []
        st.rerun()


def main() -> None:
    initialise_state()
    st.title("Automated MCQ Generator")
    st.caption("Textbook-grounded MCQs with transparent pipeline inspection")

    with st.sidebar:
        st.header("Generation settings")
        input_mode = st.radio("Input method", ("Enter text", "Upload PDF"))
        num_questions = st.slider("Number of questions", 1, CONFIG["max_questions"], 5)
        model = st.selectbox("Question model", QUESTION_MODELS, index=QUESTION_MODELS.index(CONFIG["question_model"]))
        backend = st.selectbox("Execution backend", QUESTION_BACKENDS, index=QUESTION_BACKENDS.index(CONFIG["question_model_backend"]))
        st.caption(f"Semantic similarity: {CONFIG['semantic_embedding_model']}")

    source_text = ""
    pages: list[str] = []
    if input_mode == "Enter text":
        source_text = st.text_area("Paste textbook text", height=220, placeholder="Paste a textbook passage or chapter excerpt here.")
    else:
        uploaded = st.file_uploader("Upload a text-based PDF", type=["pdf"])
        if uploaded is not None:
            try:
                pages = preview_pdf(uploaded.getvalue())
                source_text = "\n\n".join(page for page in pages if page)
                st.success(f"Read {len(pages)} page(s) from {uploaded.name}.")
            except Exception as error:
                st.error(f"The PDF could not be read: {error}")

    run_config = {**CONFIG, "question_model": model, "question_model_backend": backend}
    if st.button("Generate MCQs", type="primary", use_container_width=True):
        if not source_text.strip():
            st.error("Provide textbook text or a PDF with extractable text before generating.")
        elif len(source_text.strip()) < CONFIG["min_input_characters"]:
            st.warning(f"The input is very short. Provide at least {CONFIG['min_input_characters']} characters for reliable MCQs.")
        else:
            generator = _pipeline()
            if generator is None:
                st.warning("The interface is ready, but the NLP pipeline has not been implemented yet. Input preview is available below.")
                st.session_state.result = normalize_result(
                    {"source": source_text, "pages": pages, "config": run_config}, run_config
                )
            else:
                try:
                    with st.spinner("Generating and validating MCQs..."):
                        output = generator(text=source_text, num_questions=num_questions, config=run_config)
                    st.session_state.result = normalize_result(output, run_config)
                    reset_quiz()
                    st.success("Generation complete.")
                except Exception as error:
                    st.error(f"Generation failed: {error}")
                    st.info("Check the selected backend, then switch backend and retry if appropriate.")

    generator_tab, quiz_tab = st.tabs(("Generator", "Quiz"))
    result = st.session_state.result
    with generator_tab:
        if result is None:
            st.info("Choose an input method, provide source material, then select Generate MCQs.")
        else:
            render_input_artifacts(result)
            render_pipeline_artifacts(result)
            render_questions(result)
            st.download_button(
                "Download run data (JSON)",
                data=json.dumps(result, indent=2, ensure_ascii=False),
                file_name="mcq_run.json",
                mime="application/json",
            )
    with quiz_tab:
        if result is None:
            st.info("Quiz mode becomes available after the pipeline returns validated MCQs.")
        else:
            render_quiz(result)


if __name__ == "__main__":
    main()
