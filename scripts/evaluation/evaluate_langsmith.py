"""Evaluate the local RAG app on a synthetic testset with LangSmith."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langsmith import Client, evaluate
from langsmith.schemas import Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import BASE_DIR
from rag.service.conversation.app_service import answer_question


DEFAULT_TESTSET_PATH = BASE_DIR / "data" / "testsets" / "accident_ragas_eval_dataset.jsonl"
DEFAULT_DATASET_PREFIX = "MDM synthetic RAGAS testset"
DEFAULT_EXPERIMENT_PREFIX = "MDM RAG baseline"
DEFAULT_JUDGE_MODEL = "gpt-4o-mini"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload a JSONL testset to LangSmith and evaluate the local RAG app.",
    )
    parser.add_argument(
        "--testset-path",
        type=Path,
        default=DEFAULT_TESTSET_PATH,
        help="JSONL testset created by scripts/generate_ragas_testset.py.",
    )
    parser.add_argument(
        "--dataset-name",
        default=None,
        help="LangSmith dataset name. Defaults to a timestamped name.",
    )
    parser.add_argument(
        "--experiment-prefix",
        default=DEFAULT_EXPERIMENT_PREFIX,
        help="LangSmith experiment prefix.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=0,
        help="Evaluate only the first N examples. Set to 0 for all examples.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=1,
        help="Number of examples to run concurrently.",
    )
    parser.add_argument(
        "--upload-only",
        action="store_true",
        help="Only create/upload the LangSmith dataset; do not run evaluation.",
    )
    parser.add_argument(
        "--llm-judge",
        action="store_true",
        help="Use an OpenAI LLM-as-Judge evaluator in addition to lexical metrics.",
    )
    parser.add_argument(
        "--judge-model",
        default=DEFAULT_JUDGE_MODEL,
        help="OpenAI model used for --llm-judge.",
    )
    return parser.parse_args()


def load_testset(path: Path, max_examples: int) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Testset file not found: {path}")

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))

    if max_examples > 0:
        rows = rows[:max_examples]
    if not rows:
        raise RuntimeError("No testset examples were loaded.")
    return rows


def make_dataset_name(path: Path) -> str:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{DEFAULT_DATASET_PREFIX} - {path.stem} - {timestamp}"


def upload_dataset(
    client: Client,
    dataset_name: str,
    rows: list[dict[str, Any]],
) -> Dataset:
    dataset = client.create_dataset(
        dataset_name=dataset_name,
        description="Synthetic RAG evaluation dataset generated with RAGAS.",
    )

    examples = []
    for index, row in enumerate(rows, start=1):
        question = row.get("user_input")
        reference = row.get("reference")
        reference_contexts = row.get("reference_contexts") or []
        if not question:
            continue

        examples.append(
            {
                "inputs": {"question": question},
                "outputs": {
                    "reference": reference or "",
                    "reference_contexts": reference_contexts,
                },
                "metadata": {
                    "row_index": index,
                    "synthesizer_name": row.get("synthesizer_name"),
                    "persona_name": row.get("persona_name"),
                    "query_style": row.get("query_style"),
                    "query_length": row.get("query_length"),
                },
            }
        )

    if not examples:
        raise RuntimeError("No valid examples with user_input were found.")

    client.create_examples(dataset_id=dataset.id, examples=examples)
    print(f"Uploaded examples: {len(examples)}")
    return dataset


def rag_target(inputs: dict[str, Any]) -> dict[str, Any]:
    answer, contexts = answer_question(inputs["question"])
    return {
        "answer": answer,
        "contexts": contexts,
    }


def tokenize(text: str) -> set[str]:
    return {
        token.lower()
        for token in re.findall(r"[0-9A-Za-z가-힣]{2,}", text or "")
    }


def reference_token_overlap(
    outputs: dict[str, Any],
    reference_outputs: dict[str, Any],
) -> dict[str, Any]:
    answer_tokens = tokenize(outputs.get("answer", ""))
    reference_tokens = tokenize(reference_outputs.get("reference", ""))

    if not reference_tokens:
        return {
            "key": "reference_token_overlap",
            "score": None,
            "comment": "Reference answer is empty.",
        }

    overlap = len(answer_tokens & reference_tokens) / len(reference_tokens)
    return {
        "key": "reference_token_overlap",
        "score": overlap,
        "comment": "Fraction of reference-answer tokens found in the generated answer.",
    }


def retrieved_context_overlap(
    outputs: dict[str, Any],
    reference_outputs: dict[str, Any],
) -> dict[str, Any]:
    retrieved_text = "\n\n".join(outputs.get("contexts") or [])
    reference_text = "\n\n".join(reference_outputs.get("reference_contexts") or [])
    retrieved_tokens = tokenize(retrieved_text)
    reference_tokens = tokenize(reference_text)

    if not reference_tokens:
        return {
            "key": "retrieved_context_overlap",
            "score": None,
            "comment": "Reference contexts are empty.",
        }

    overlap = len(retrieved_tokens & reference_tokens) / len(reference_tokens)
    return {
        "key": "retrieved_context_overlap",
        "score": overlap,
        "comment": "Fraction of reference-context tokens found in retrieved contexts.",
    }


def make_llm_correctness_judge(judge_model: str):
    judge_llm = ChatOpenAI(model=judge_model, temperature=0)

    def llm_correctness(
        inputs: dict[str, Any],
        outputs: dict[str, Any],
        reference_outputs: dict[str, Any],
    ) -> dict[str, Any]:
        prompt = f"""
You are grading a RAG answer against a reference answer.

Score from 0.0 to 1.0:
- 1.0: The answer is correct and fully supported by the reference.
- 0.5: The answer is partially correct but misses important details.
- 0.0: The answer is incorrect, unsupported, or refuses despite enough evidence.

Return only JSON with keys "score" and "comment".

Question:
{inputs.get("question", "")}

Reference answer:
{reference_outputs.get("reference", "")}

Generated answer:
{outputs.get("answer", "")}
""".strip()
        response = judge_llm.invoke(prompt).content
        try:
            parsed = json.loads(response)
            score = float(parsed["score"])
            comment = str(parsed.get("comment", ""))
        except Exception:
            score = None
            comment = f"Judge returned non-JSON output: {response}"

        return {
            "key": "llm_correctness",
            "score": score,
            "comment": comment,
        }

    return llm_correctness


def main() -> None:
    load_dotenv()
    args = parse_args()

    if not os.getenv("LANGSMITH_API_KEY"):
        raise RuntimeError("LANGSMITH_API_KEY is required to upload/evaluate in LangSmith.")
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required because the RAG app calls OpenAI.")
    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGSMITH_PROJECT", args.experiment_prefix)

    rows = load_testset(args.testset_path, args.max_examples)
    dataset_name = args.dataset_name or make_dataset_name(args.testset_path)

    client = Client()
    dataset = upload_dataset(
        client=client,
        dataset_name=dataset_name,
        rows=rows,
    )
    print(f"LangSmith dataset: {dataset.name}")

    if args.upload_only:
        print("Upload complete. Evaluation was skipped.")
        return

    evaluators = [
        reference_token_overlap,
        retrieved_context_overlap,
    ]
    if args.llm_judge:
        evaluators.append(make_llm_correctness_judge(args.judge_model))

    results = evaluate(
        rag_target,
        data=dataset.name,
        evaluators=evaluators,
        experiment_prefix=args.experiment_prefix,
        description="Baseline local RAG evaluation on the synthetic RAGAS testset.",
        max_concurrency=args.max_concurrency,
        client=client,
    )
    print(f"LangSmith experiment: {results.experiment_name}")


if __name__ == "__main__":
    main()
