"""Dataset and testset helpers for retrieval evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langsmith import Client
from langsmith.schemas import Dataset

from evaluation.retrieval_eval.constants import DEFAULT_DATASET_PREFIX, EXAMPLE_METADATA_FIELDS


def load_jsonl(path: Path, max_examples: int = 0) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Testset file not found: {path}")

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            if line.strip():
                rows.append(json.loads(line))

    if max_examples > 0:
        rows = rows[:max_examples]
    if not rows:
        raise RuntimeError("No examples were loaded.")
    return rows


def make_dataset_name(path: Path) -> str:
    return f"{DEFAULT_DATASET_PREFIX} - {path.stem}"


def build_examples(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for row in rows:
        question = row.get("question")
        if not question:
            continue

        inputs: dict[str, Any] = {"question": question}
        if row.get("candidate_k") is not None:
            inputs["candidate_k"] = row.get("candidate_k")
        if row.get("final_k") is not None:
            inputs["final_k"] = row.get("final_k")

        expected_keywords = row.get("expected_keywords")
        if expected_keywords is None:
            expected_keywords = row.get("expected_evidence_keywords", [])

        metadata = {
            "id": row.get("id"),
            "notes": row.get("notes"),
            "suite": row.get("suite") or infer_suite_from_row(row),
        }
        for field in EXAMPLE_METADATA_FIELDS:
            if field == "suite":
                continue
            if row.get(field) is not None:
                metadata[field] = row.get(field)

        examples.append(
            {
                "inputs": inputs,
                "outputs": {
                    "reference": row.get("reference", ""),
                    "expected_diagram_ids": row.get("expected_diagram_ids", []),
                    "acceptable_diagram_ids": row.get("acceptable_diagram_ids", []),
                    "near_miss_diagram_ids": row.get("near_miss_diagram_ids", []),
                    "expected_party_type": row.get("expected_party_type"),
                    "expected_location": row.get("expected_location"),
                    "expected_chunk_types": row.get("expected_chunk_types", []),
                    "expected_keywords": expected_keywords,
                    "requires_diagram": row.get("requires_diagram"),
                    "requires_table": row.get("requires_table"),
                },
                "metadata": metadata,
            }
        )

    if not examples:
        raise RuntimeError("No valid examples with question were found.")
    return examples


def infer_suite_from_row(row: dict[str, Any]) -> str:
    """Infer a suite label from testset IDs when rows omit an explicit suite."""

    row_id = str(row.get("id") or "")
    prefix_map = {
        "retrieval_": "retrieval",
        "reranker_": "reranker",
        "intake_": "intake",
        "router_": "router",
        "filter_": "metadata_filter",
        "mt_": "multiturn",
        "struct_": "structured_output",
    }
    for prefix, suite in prefix_map.items():
        if row_id.startswith(prefix):
            return suite
    return "retrieval"


def get_existing_dataset(client: Client, dataset_name: str) -> Dataset | None:
    return next(client.list_datasets(dataset_name=dataset_name, limit=1), None)


def dataset_example_count(client: Client, dataset_name: str) -> int:
    return sum(1 for _ in client.list_examples(dataset_name=dataset_name))


def get_or_create_dataset(
    client: Client,
    dataset_name: str,
    rows: list[dict[str, Any]],
) -> Dataset:
    existing_dataset = get_existing_dataset(client, dataset_name)
    if existing_dataset is not None:
        example_count = dataset_example_count(client, dataset_name)
        if example_count > 0:
            print(
                f"[INFO] reusing existing dataset: {dataset_name} "
                f"({example_count} examples)"
            )
            return existing_dataset

        examples = build_examples(rows)
        client.create_examples(dataset_id=existing_dataset.id, examples=examples)
        print(f"[INFO] added examples to empty dataset: {len(examples)}")
        return existing_dataset

    dataset = client.create_dataset(
        dataset_name=dataset_name,
        description="Retrieval evaluation dataset for accident fault-ratio RAG.",
    )

    examples = build_examples(rows)
    client.create_examples(dataset_id=dataset.id, examples=examples)
    print(f"[INFO] uploaded examples: {len(examples)}")
    return dataset
