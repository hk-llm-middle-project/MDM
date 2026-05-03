"""Evaluate intake slot extraction on a local JSONL testset."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import BASE_DIR
from rag.service.intake.filter_service import build_metadata_filters
from rag.service.intake.intake_service import evaluate_input_sufficiency
from rag.service.intake.query_normalizer import normalize_retrieval_query_terms


DEFAULT_TESTSET_PATH = BASE_DIR / "data" / "testsets" / "q2_fault_ratio" / "testset.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print intake query_slots and normalized retrieval queries for a JSONL testset.",
    )
    parser.add_argument(
        "--testset-path",
        type=Path,
        default=DEFAULT_TESTSET_PATH,
        help="JSONL file with question and expected_diagram_ids fields.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=0,
        help="Evaluate only the first N examples. Set to 0 for all examples.",
    )
    return parser.parse_args()


def load_jsonl(path: Path, max_examples: int = 0) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return rows[:max_examples] if max_examples > 0 else rows


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")
    os.environ.setdefault("LANGSMITH_TRACING", "false")
    args = parse_args()
    rows = load_jsonl(args.testset_path, args.max_examples)

    print(
        "idx\texpected\tparty_type\tlocation\tfilters\t"
        "road_control\trelation\ta_signal\tb_signal\ta_movement\tb_movement\t"
        "road_priority\tspecial_condition\tretrieval_query"
    )
    for index, row in enumerate(rows, start=1):
        question = row["question"]
        expected = ",".join(row.get("expected_diagram_ids", []))
        decision = evaluate_input_sufficiency(question)
        metadata = decision.search_metadata
        slots = metadata.query_slots
        retrieval_query = normalize_retrieval_query_terms(question, metadata)
        filters = build_metadata_filters(metadata)
        print(
            "\t".join(
                [
                    str(index),
                    expected,
                    metadata.party_type or "",
                    metadata.location or "",
                    json.dumps(filters, ensure_ascii=False, sort_keys=True),
                    slots.road_control or "",
                    slots.relation or "",
                    slots.a_signal or "",
                    slots.b_signal or "",
                    slots.a_movement or "",
                    slots.b_movement or "",
                    slots.road_priority or "",
                    slots.special_condition or "",
                    retrieval_query or "",
                ]
            )
        )


if __name__ == "__main__":
    main()
