"""intake를 끄고 로컬 RAG 응답을 RAGAS로 평가합니다."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from warnings import filterwarnings

filterwarnings("ignore", category=DeprecationWarning)

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import matplotlib.pyplot as plt
import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import BASE_DIR, EMBEDDING_MODEL, LLM_MODEL
from rag.service.conversation.app_service import answer_question_without_intake


DEFAULT_TESTSET_PATH = BASE_DIR / "data" / "testsets" / "accident_ragas_eval_dataset_ver_2.jsonl"
DEFAULT_OUTPUT_DIR = BASE_DIR / "data" / "eval_results"
METRIC_COLUMNS = ["context_precision", "context_recall", "faithfulness", "answer_relevancy"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="intake 없이 로컬 RAG를 실행하고 RAGAS 4개 메트릭으로 평가합니다.",
    )
    parser.add_argument(
        "--testset-path",
        type=Path,
        default=DEFAULT_TESTSET_PATH,
        help="평가용 JSONL 파일 경로입니다.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="평가 결과 CSV와 그래프를 저장할 폴더입니다.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=0,
        help="앞에서부터 N개만 평가합니다. 0이면 전체를 평가합니다.",
    )
    parser.add_argument(
        "--judge-model",
        default=LLM_MODEL,
        help="RAGAS 평가 LLM 모델입니다.",
    )
    parser.add_argument(
        "--embedding-model",
        default=EMBEDDING_MODEL,
        help="answer_relevancy 평가에 사용할 임베딩 모델입니다.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="RAGAS 평가 배치 크기입니다.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="막대그래프 파일을 만들지 않습니다.",
    )
    return parser.parse_args()


def load_jsonl(path: Path, max_examples: int) -> list[dict[str, Any]]:
    """JSONL 평가 데이터를 읽습니다."""
    if not path.exists():
        raise FileNotFoundError(f"평가 데이터셋을 찾을 수 없습니다: {path}")

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))

    if max_examples > 0:
        rows = rows[:max_examples]
    if not rows:
        raise RuntimeError("평가할 데이터가 없습니다.")
    return rows


def get_reference_text(reference: object) -> str:
    """원본 문자열 또는 구조화 reference에서 자연어 정답만 꺼냅니다."""
    if isinstance(reference, dict):
        return str(reference.get("response") or "")
    return str(reference or "")


def build_ragas_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """테스트셋 질문을 no-intake RAG에 넣고 RAGAS 입력 row로 변환합니다."""
    ragas_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        question = str(row.get("user_input") or "")
        if not question:
            continue

        print(f"[{index}/{len(rows)}] RAG 실행: {row.get('id', index)}")
        answer, contexts = answer_question_without_intake(question)
        ragas_rows.append(
            {
                "id": row.get("id", f"row-{index}"),
                "user_input": question,
                "response": answer,
                "retrieved_contexts": contexts,
                "reference": get_reference_text(row.get("reference")),
                "reference_contexts": row.get("reference_contexts") or [],
            }
        )

    if not ragas_rows:
        raise RuntimeError("user_input이 있는 평가 row가 없습니다.")
    return ragas_rows


def run_ragas_evaluation(
    ragas_rows: list[dict[str, Any]],
    judge_model: str,
    embedding_model: str,
    batch_size: int,
) -> pd.DataFrame:
    """RAGAS 평가를 실행하고 결과를 DataFrame으로 반환합니다."""
    dataset = Dataset.from_list(ragas_rows)
    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model=judge_model, temperature=0))
    evaluator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model=embedding_model))

    result = evaluate(
        dataset=dataset,
        metrics=[context_precision, context_recall, faithfulness, answer_relevancy],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        batch_size=batch_size,
    )
    return result.to_pandas()


def print_metric_summary(result_df: pd.DataFrame) -> None:
    """주요 RAGAS 메트릭 평균을 콘솔에 출력합니다."""
    metrics_df = result_df[METRIC_COLUMNS]

    print("=" * 80)
    print("RAGAS 평가 결과")
    print("=" * 80)

    print("\n평균 점수:")
    for metric in METRIC_COLUMNS:
        score = metrics_df[metric].mean()
        status = "OK" if score > 0.8 else "WARN" if score > 0.6 else "LOW"
        print(f"{status:4s} {metric:25s}: {score:.3f}")

    overall_score = metrics_df.mean().mean()
    print(f"\n{'=' * 80}")
    print(f"전체 평균 점수: {overall_score:.3f}")
    print(f"{'=' * 80}")


def save_outputs(result_df: pd.DataFrame, output_dir: Path, make_plot: bool) -> None:
    """평가 결과 CSV와 메트릭 막대그래프를 저장합니다."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    csv_path = output_dir / f"ragas_eval_{timestamp}.csv"
    result_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\nCSV 저장: {csv_path}")

    if not make_plot:
        return

    means = result_df[METRIC_COLUMNS].mean().sort_values(ascending=False)
    ax = means.plot(kind="bar", ylim=(0, 1), rot=25, figsize=(9, 5), color="#4C78A8")
    ax.set_title("RAGAS Metric Averages")
    ax.set_ylabel("Score")
    ax.grid(axis="y", alpha=0.25)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=3)
    plt.tight_layout()

    plot_path = output_dir / f"ragas_eval_{timestamp}.png"
    plt.savefig(plot_path, dpi=160)
    plt.close()
    print(f"그래프 저장: {plot_path}")


def main() -> None:
    load_dotenv()
    args = parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY가 필요합니다.")
    if args.max_examples < 0:
        raise ValueError("--max-examples는 0 이상이어야 합니다.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size는 1 이상이어야 합니다.")

    rows = load_jsonl(args.testset_path, args.max_examples)
    ragas_rows = build_ragas_rows(rows)
    result_df = run_ragas_evaluation(
        ragas_rows=ragas_rows,
        judge_model=args.judge_model,
        embedding_model=args.embedding_model,
        batch_size=args.batch_size,
    )
    print_metric_summary(result_df)
    save_outputs(result_df, args.output_dir, make_plot=not args.no_plot)


if __name__ == "__main__":
    main()
