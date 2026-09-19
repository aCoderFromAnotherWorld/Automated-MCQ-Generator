"""Shared application configuration.

Keep model and pipeline settings here so that the Streamlit interface never
hard-codes choices that later pipeline modules also need to use.
"""

CONFIG = {
    "question_model": "google/flan-t5-small",
    "question_model_checkpoint": "models/question_generation/flan-t5-squadv2",
    "question_model_training_dataset": "SQuADv2",
    "question_model_training_mode": "local",
    "question_model_backend": "local",
    "generation_max_new_tokens": 32,
    "generation_num_beams": 1,
    "generation_num_return_sequences": 1,
    "generation_timeout_seconds": 30.0,
    "generation_retries": 3,
    "generation_backoff_seconds": 1.0,
    "semantic_embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "min_input_characters": 120,
    "max_questions": 20,
    "default_questions": 3,
    "chunk_size_sentences": 8,
    "chunk_overlap_sentences": 2,
    "min_candidate_chars": 3,
    "max_candidate_words": 6,
    "max_candidate_overlap": 0.6,
    # Performance caps: bound the per-question distractor pool and the number of
    # generation attempts so a single run stays fast on CPU.
    "max_distractor_pool": 12,
    "generation_attempt_factor": 5,
    # Distractor contextual-incorrectness threshold: a candidate that appears in
    # the source context and scores at or above this similarity to the question
    # is treated as "actually correct" and rejected.
    "context_support_threshold": 0.8,
    # Round-trip answer verification: the base QA model must reproduce the
    # intended answer when asked the generated question.
    "answer_verification": True,
    "answer_model": "google/flan-t5-small",
    "answer_match_threshold": 0.5,
    # Diversity: a distractor may not be reused more than this many times per run.
    "max_distractor_reuse": 2,
    "ranking_weights": {
        "tfidf_score": 0.33,
        "rake_score": 0.18,
        "frequency": 0.12,
        "is_noun_phrase": 0.10,
        "is_named_entity": 0.12,
        "position": 0.10,
        "is_multi_word": 0.05,
    },
}

QUESTION_MODELS = ("google/flan-t5-base", "google/flan-t5-small")
QUESTION_BACKENDS = ("local", "hf_api")
