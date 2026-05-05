"""Constants for retrieval evaluation."""

from __future__ import annotations

from config import (
    BASE_DIR,
    DEFAULT_CHUNKER_STRATEGY as APP_DEFAULT_CHUNKER_STRATEGY,
    DEFAULT_EMBEDDING_PROVIDER,
    DEFAULT_ENSEMBLE_BM25_WEIGHT,
    DEFAULT_ENSEMBLE_CANDIDATE_K,
    DEFAULT_ENSEMBLE_USE_CHUNK_ID,
    DEFAULT_LOADER_STRATEGY as APP_DEFAULT_LOADER_STRATEGY,
    RETRIEVER_K,
)
from rag.pipeline.reranker import RERANKER_STRATEGIES
from rag.pipeline.retriever import RETRIEVAL_STRATEGIES


DEFAULT_TESTSET_PATH = (
    BASE_DIR / "data" / "testsets" / "langsmith" / "retrieval_eval.jsonl"
)
DEFAULT_MATRIX_PATH = BASE_DIR / "evaluation" / "retrieval_eval_matrix.json"
DEFAULT_OUTPUT_DIR = BASE_DIR / "evaluation" / "results" / "uncategorized"
DEFAULT_DATASET_PREFIX = "MDM retrieval testset"
DEFAULT_EXPERIMENT_PREFIX = "MDM retrieval eval"
DEFAULT_LOADER_STRATEGY = APP_DEFAULT_LOADER_STRATEGY
DEFAULT_CHUNKER_STRATEGY = APP_DEFAULT_CHUNKER_STRATEGY
DEFAULT_RETRIEVER_STRATEGY = "similarity"
DEFAULT_RERANKER_STRATEGY = "none"
DEFAULT_K = RETRIEVER_K
DEFAULT_RETRIEVAL_INPUT_MODE = "raw"
RETRIEVAL_INPUT_MODE_CHOICES = ("raw", "intake")
RETRIEVER_STRATEGY_CHOICES = tuple(RETRIEVAL_STRATEGIES)
RERANKER_STRATEGY_CHOICES = tuple(
    strategy for strategy in ("none", "cross-encoder", "flashrank", "llm-score")
    if strategy in RERANKER_STRATEGIES
)
RERANKER_STRATEGY_ALIASES = {
    "cross_encoder": "cross-encoder",
    "llm_score": "llm-score",
}
RERANKER_STRATEGY_INPUT_CHOICES = tuple(
    [*RERANKER_STRATEGY_CHOICES, *RERANKER_STRATEGY_ALIASES]
)
CONTENT_PREVIEW_CHARS = 500
EXAMPLE_METADATA_FIELDS = (
    "suite",
    "case_type_codes",
    "difficulty",
    "case_family",
    "inference_type",
    "query_style",
    "requires_diagram",
    "requires_table",
    "filter_risk",
    "candidate_k",
    "final_k",
)
