"""Shared application configuration.

Keep model and pipeline settings here so that the Streamlit interface never
hard-codes choices that later pipeline modules also need to use.
"""

CONFIG = {
    "question_model": "google/flan-t5-base",
    "question_model_checkpoint": "models/question_generation/flan-t5-squadv2",
    "question_model_training_dataset": "SQuADv2",
    "question_model_training_mode": "local",
    "question_model_backend": "local",
    "generation_max_new_tokens": 64,
    "generation_num_beams": 4,
    "semantic_embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "min_input_characters": 120,
    "max_questions": 20,
}

QUESTION_MODELS = ("google/flan-t5-base", "google/flan-t5-small")
QUESTION_BACKENDS = ("local", "hf_api")
