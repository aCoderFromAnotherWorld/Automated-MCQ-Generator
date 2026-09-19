"""End-to-end NLP orchestration for validated, source-grounded MCQs (M13)."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Mapping

from config import CONFIG
from src.contracts import assert_json_serializable, ensure_result_shape
from src.distractor_generator import generate_distractors
from src.embeddings import add_tfidf_scores
from src.input_handler import text_to_result
from src.keyword_extractor import extract_candidates
from src.mcq_validator import assemble_mcq
from src.preprocessing import preprocess_result
from src.question_generator import generator_from_config
from src.ranking import rank_candidates
from src.chunking import chunk_result


class PipelineError(RuntimeError):
    """Raised when orchestration cannot complete a generation stage."""


def _chunk_for_candidate(candidate: Mapping[str, Any], chunks: list[Mapping[str, Any]]) -> dict[str, Any] | None:
    chunk_id = candidate.get("chunk_id")
    return next((dict(chunk) for chunk in chunks if chunk.get("chunk_id") == chunk_id), None)


def _generate_record(generator: Any, context: str, answer: str, chunk_id: int | None) -> dict[str, Any]:
    try:
        if hasattr(generator, "generate_question"):
            record = generator.generate_question(context, answer, chunk_id=chunk_id)
        else:
            record = generator(context, answer, chunk_id=chunk_id)
    except Exception as error:
        raise PipelineError(f"Question generation failed for chunk {chunk_id}: {error}") from error
    if not isinstance(record, Mapping):
        raise PipelineError("Question generator must return a mapping record.")
    output = dict(record)
    output.setdefault("chunk_id", chunk_id)
    output.setdefault("answer", " ".join(answer.split()))
    output.setdefault("candidate_questions", [])
    output.setdefault("selected_question", "")
    return output


def _summary(*, attempted: int, valid: int, rejected: int, duplicate_questions: int) -> dict[str, Any]:
    return {
        "status": "complete",
        "generated": attempted,
        "valid": valid,
        "rejected": rejected,
        "duplicates": duplicate_questions,
    }


def generate_mcqs(
    text: str,
    num_questions: int = 5,
    *,
    config: Mapping[str, Any] | None = None,
    question_generator: Any | None = None,
    similarity: Callable[[str, str], float] | None = None,
    answer_verifier: Callable[[str, str], str] | None = None,
    **options: Any,
) -> dict[str, Any]:
    """Run the completed M00-M12 pipeline and return the full result contract.

    ``question_generator``, ``similarity``, and ``answer_verifier`` are injectable
    seams for tests and controlled experiments. Production calls use the
    configured local trained checkpoint and cached Sentence Transformer utilities.
    """

    if num_questions < 1:
        raise ValueError("num_questions must be at least one.")
    run_config: dict[str, Any] = deepcopy(dict(CONFIG))
    if config is not None:
        run_config.update(dict(config))
    run_config.update(options)
    similarity_fn = similarity
    max_attempts = num_questions * int(run_config.get("generation_attempt_factor", 3))
    distractor_pool_size = int(run_config.get("max_distractor_pool", 12))
    max_distractor_reuse = int(run_config.get("max_distractor_reuse", 2))
    answer_match_threshold = float(run_config.get("answer_match_threshold", 0.5))
    verify_answers = bool(run_config.get("answer_verification", True))
    distractor_usage: dict[str, int] = {}
    result = text_to_result(text, run_config)
    result = preprocess_result(result)
    result = chunk_result(result, run_config)

    result["candidates"] = extract_candidates(result["chunks"], run_config)
    scored_candidates = add_tfidf_scores(result["chunks"], result["candidates"])
    result["candidates"] = scored_candidates
    result["ranked_candidates"] = rank_candidates(
        scored_candidates,
        result["chunks"],
        run_config,
        limit=None,
    )

    generator = question_generator or generator_from_config(run_config)
    verifier = answer_verifier or getattr(generator, "answer_question", None)
    generation_records: list[dict[str, Any]] = []
    distractor_records: list[dict[str, Any]] = []
    validation_records: list[dict[str, Any]] = []
    questions: list[dict[str, Any]] = []
    accepted_questions: list[str] = []
    duplicate_questions = 0

    for candidate in result["ranked_candidates"]:
        if len(questions) >= num_questions:
            break
        if len(generation_records) >= max_attempts:
            break
        chunk = _chunk_for_candidate(candidate, result["chunks"])
        if chunk is None:
            validation_records.append({"valid": False, "reasons": ["missing_source_chunk"], "candidate": dict(candidate)})
            continue
        context = str(chunk.get("text", ""))
        answer = str(candidate.get("text", ""))
        generation = _generate_record(generator, context, answer, chunk.get("chunk_id"))
        generation_records.append(generation)
        question = str(generation.get("selected_question", "")).strip()
        if not question:
            validation = {"valid": False, "reasons": ["question_empty"], "candidate": dict(candidate)}
            validation_records.append(validation)
            continue

        distractors = generate_distractors(
            question,
            answer,
            context,
            result["ranked_candidates"][:distractor_pool_size],
            answer_record=candidate,
            similarity=similarity_fn,
            context_support_threshold=float(run_config.get("context_support_threshold", 0.8)),
            avoid_counts=distractor_usage,
            max_reuse=max_distractor_reuse,
        )
        distractor_records.append(distractors)

        predicted_answer: str | None = None
        if verify_answers and verifier is not None:
            try:
                predicted_answer = str(verifier(context, question) or "").strip() or None
            except Exception:
                predicted_answer = None

        assembled = assemble_mcq(
            question,
            answer,
            context,
            distractors,
            chunk_id=chunk.get("chunk_id"),
            source_page=chunk.get("page_start"),
            generator=str(generation.get("backend", generation.get("model_name", "unknown"))),
            previous_questions=accepted_questions,
            similarity=similarity_fn,
            duplicate_threshold=float(run_config.get("duplicate_question_threshold", 0.85)),
            predicted_answer=predicted_answer,
            answer_match_threshold=answer_match_threshold,
        )
        validation = dict(assembled["validation"])
        validation["candidate"] = dict(candidate)
        validation_records.append(validation)
        if not assembled["valid"]:
            if "duplicate_question" in validation.get("reasons", []):
                duplicate_questions += 1
            continue

        mcq = dict(assembled["mcq"])
        mcq["backend"] = generation.get("backend")
        mcq["model_name"] = generation.get("model_name")
        mcq["verified_answer"] = predicted_answer
        questions.append(mcq)
        accepted_questions.append(question)
        for distractor in distractors.get("selected", []):
            key = " ".join(str(distractor).lower().split())
            distractor_usage[key] = distractor_usage.get(key, 0) + 1

    result["generation_records"] = generation_records
    result["distractor_records"] = distractor_records
    result["questions"] = questions
    result["validation"] = {
        "summary": _summary(
            attempted=len(generation_records),
            valid=len(questions),
            rejected=len(validation_records) - len(questions),
            duplicate_questions=duplicate_questions,
        ),
        "per_question": validation_records,
    }
    result["stats"] = {
        "sentences": len(result["sentences"]),
        "chunks": len(result["chunks"]),
        "candidates": len(result["candidates"]),
        "ranked_candidates": len(result["ranked_candidates"]),
        "generation_attempts": len(generation_records),
        "valid_questions": len(questions),
        "rejected_questions": len(validation_records) - len(questions),
        "concept_coverage": {question["answer"]: 1 for question in questions},
    }
    result = ensure_result_shape(result, run_config)
    assert_json_serializable(result)
    return result
