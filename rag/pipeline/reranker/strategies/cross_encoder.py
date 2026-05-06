"""HuggingFace CrossEncoder 리랭커 전략."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from rag.pipeline.reranker.strategies.common import build_scored_document
from rag.service.tracing import TraceContext


DEFAULT_CROSS_ENCODER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
# DEFAULT_CROSS_ENCODER_MODEL = "BAAI/bge-reranker-v2-m3"
DEFAULT_CROSS_ENCODER_MAX_CHARS = 4000
DEFAULT_CROSS_ENCODER_BATCH_SIZE = 4
DEFAULT_CROSS_ENCODER_TIMEOUT_SECONDS = 180


@dataclass(frozen=True)
class CrossEncoderRerankerConfig:
    """HuggingFace CrossEncoder 기반 리랭커 설정."""

    model_name: str = DEFAULT_CROSS_ENCODER_MODEL
    top_n: int | None = None
    max_chars: int = DEFAULT_CROSS_ENCODER_MAX_CHARS
    batch_size: int = DEFAULT_CROSS_ENCODER_BATCH_SIZE
    timeout_seconds: int = DEFAULT_CROSS_ENCODER_TIMEOUT_SECONDS
    use_subprocess: bool = True
    model: Any | None = None


def _configure_torch_runtime() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    try:
        import torch
    except ImportError:
        return

    if torch.get_num_threads() > 1:
        torch.set_num_threads(1)
    if torch.get_num_interop_threads() > 1:
        torch.set_num_interop_threads(1)


@lru_cache(maxsize=2)
def get_cross_encoder_model(model_name: str):
    """CrossEncoder 모델을 프로세스 단위로 캐시합니다."""
    _configure_torch_runtime()
    try:
        from langchain_community.cross_encoders import HuggingFaceCrossEncoder
    except ImportError as error:
        raise ImportError(
            "cross-encoder 리랭커를 사용하려면 `langchain-community`와 "
            "`sentence-transformers` 패키지가 설치되어 있어야 합니다."
        ) from error

    return HuggingFaceCrossEncoder(model_name=model_name)


def _score_with_cross_encoder(
    model: Any,
    pairs: list[tuple[str, str]],
    batch_size: int,
) -> list[float]:
    client = getattr(model, "client", None)
    if client is not None and hasattr(client, "predict"):
        scores = client.predict(
            pairs,
            batch_size=batch_size,
            show_progress_bar=False,
        )
    else:
        scores = model.score(pairs)

    if hasattr(scores, "tolist"):
        scores = scores.tolist()

    normalized_scores: list[float] = []
    for score in scores:
        if isinstance(score, (list, tuple)):
            normalized_scores.append(float(score[-1]))
        elif hasattr(score, "tolist"):
            score_value = score.tolist()
            if isinstance(score_value, list):
                normalized_scores.append(float(score_value[-1]))
            else:
                normalized_scores.append(float(score_value))
        else:
            normalized_scores.append(float(score))
    return normalized_scores


def _score_with_cross_encoder_subprocess(
    model_name: str,
    pairs: list[tuple[str, str]],
    batch_size: int,
    timeout_seconds: int,
) -> list[float]:
    with tempfile.TemporaryDirectory(prefix="mdm-cross-encoder-") as temp_dir:
        temp_path = Path(temp_dir)
        input_path = temp_path / "input.json"
        output_path = temp_path / "output.json"
        input_path.write_text(
            json.dumps(
                {
                    "model_name": model_name,
                    "pairs": pairs,
                    "batch_size": batch_size,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        env = {
            **os.environ,
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
        command = [
            sys.executable,
            "-m",
            "rag.pipeline.reranker.strategies.cross_encoder",
            str(input_path),
            str(output_path),
        ]
        completed = subprocess.run(
            command,
            cwd=str(Path(__file__).resolve().parents[4]),
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=max(timeout_seconds, 1),
            check=False,
        )
        if completed.returncode != 0:
            details = (completed.stderr or completed.stdout).strip()
            raise RuntimeError(
                "cross-encoder subprocess failed"
                + (f": {details}" if details else f" with exit code {completed.returncode}")
            )
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        scores = payload.get("scores")
        if not isinstance(scores, list):
            raise RuntimeError("cross-encoder subprocess returned no scores")
        return [float(score) for score in scores]


def rerank_with_cross_encoder(
    query: str,
    documents: list[Document],
    k: int,
    strategy_config: CrossEncoderRerankerConfig | None = None,
    trace_context: TraceContext | None = None,
) -> list[Document]:
    """HuggingFace CrossEncoder로 문서를 재정렬한 뒤 상위 k개를 반환합니다."""
    if not documents:
        return []

    del trace_context
    config = strategy_config or CrossEncoderRerankerConfig()
    max_chars = max(config.max_chars, 1)
    batch_size = max(config.batch_size, 1)
    pairs = [
        (query, document.page_content[:max_chars])
        for document in documents
    ]
    if config.model is not None:
        scores = _score_with_cross_encoder(config.model, pairs, batch_size)
    elif config.use_subprocess:
        scores = _score_with_cross_encoder_subprocess(
            config.model_name,
            pairs,
            batch_size,
            config.timeout_seconds,
        )
    else:
        model = get_cross_encoder_model(config.model_name)
        scores = _score_with_cross_encoder(model, pairs, batch_size)
    ranked_documents = sorted(
        zip(documents, scores),
        key=lambda item: item[1],
        reverse=True,
    )[: config.top_n or k]

    return [
        build_scored_document(document, score)
        for document, score in ranked_documents[:k]
    ]


def _main(argv: list[str]) -> int:
    if len(argv) != 3:
        print("usage: python -m rag.pipeline.reranker.strategies.cross_encoder INPUT OUTPUT", file=sys.stderr)
        return 2
    input_path = Path(argv[1])
    output_path = Path(argv[2])
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    model_name = str(payload["model_name"])
    raw_pairs = payload["pairs"]
    batch_size = int(payload["batch_size"])
    pairs = [(str(query), str(document)) for query, document in raw_pairs]
    model = get_cross_encoder_model(model_name)
    scores = _score_with_cross_encoder(model, pairs, batch_size)
    output_path.write_text(
        json.dumps({"scores": scores}, ensure_ascii=False),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv))
