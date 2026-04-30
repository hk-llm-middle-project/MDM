"""Normalize Upstage chunk location metadata to intake allowed values."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import UPSTAGE_FINAL_DOCUMENTS_PATH
from rag.service.intake.values import LOCATIONS


DEFAULT_INPUT_PATH = UPSTAGE_FINAL_DOCUMENTS_PATH

LOCATION_NORMALIZATION_MAP = {
    "횡단보도 내": "횡단보도 내",
    "횡단보도 부근": "횡단보도 부근",
    "횡단시설 부근[보20~보21]": "횡단보도 부근",
    "횡단보도 없음": "횡단보도 없음",
    "기타 사고유형 [보29~보36]": "기타",
    "양쪽 신호등 있는 교차로": "교차로 사고",
    "한쪽 신호등 있는 교차로": "교차로 사고",
    "한쪽 지시표지 있는 교차로": "교차로 사고",
    "교차로 노면 표시 위반 사고": "교차로 사고",
    "신호등 없는 교차로": "교차로 사고",
    "교차로 부근 동시 우회전 내지 좌회전 사고": "교차로 사고",
    "중앙선 침범 사고": "마주보는 방향 진행차량 상호 간의 사고",
    "중앙선 없거나 중앙선침범 미적용 도로에서 교행 사고": "마주보는 방향 진행차량 상호 간의 사고",
    "직진차와 유턴차 사이의 사고": "마주보는 방향 진행차량 상호 간의 사고",
    "안전거리미확보로 인한 추돌사고": "같은 방향 진행차량 상호간의 사고",
    "주정차 차량 추돌사고": "같은 방향 진행차량 상호간의 사고",
    "진로변경 사고": "같은 방향 진행차량 상호간의 사고",
    "도로로 진입하는 차와 직진차와의 사고": "같은 방향 진행차량 상호간의 사고",
    "앞지르기 금지 장소에서 추월사고": "같은 방향 진행차량 상호간의 사고",
    "선행 유턴 대 후행 유턴 사고": "같은 방향 진행차량 상호간의 사고",
    "정차 후 출발 대 직진 사고": "같은 방향 진행차량 상호간의 사고",
    "낙하물 사고": "같은 방향 진행차량 상호간의 사고",
    "주차장 사고": "기타",
    "문 열림 사고": "기타",
    "횡단보도 횡단 차량": "기타",
    "회전교차로 사고": "기타",
    "긴급자동차 사고": "기타",
    "동일차로 통행 중 사고": "교차로 사고",
    "자전거 도로횡단 사고": "기타",
    "자전거횡단도로 횡단사고": "기타",
    "자전거 도로 사고": "기타",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize metadata.location in an Upstage chunk JSON file.",
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Path to the Upstage chunk JSON file to normalize.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Output path. Defaults to overwriting --input-path.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned changes without writing a file.",
    )
    return parser.parse_args()


def load_chunks(input_path: Path) -> list[dict[str, Any]]:
    with input_path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON list: {input_path}")
    return payload


def normalize_locations(chunks: list[dict[str, Any]]) -> Counter[tuple[str, str]]:
    changed: Counter[tuple[str, str]] = Counter()
    allowed_locations = set(LOCATIONS)

    for chunk in chunks:
        metadata = chunk.get("metadata")
        if not isinstance(metadata, dict):
            continue

        current_location = metadata.get("location")
        if current_location is None:
            continue
        if current_location in allowed_locations:
            continue

        normalized_location = LOCATION_NORMALIZATION_MAP.get(current_location)
        if normalized_location is None:
            continue

        metadata["location"] = normalized_location
        changed[(current_location, normalized_location)] += 1

    return changed


def find_invalid_locations(chunks: list[dict[str, Any]]) -> Counter[str]:
    allowed_locations = set(LOCATIONS)
    invalid_locations: Counter[str] = Counter()
    for chunk in chunks:
        metadata = chunk.get("metadata")
        if not isinstance(metadata, dict):
            continue

        location = metadata.get("location")
        if location is not None and location not in allowed_locations:
            invalid_locations[str(location)] += 1
    return invalid_locations


def save_chunks(chunks: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(chunks, fp, ensure_ascii=False, indent=2)


def print_summary(
    input_path: Path,
    output_path: Path,
    changed: Counter[tuple[str, str]],
    invalid_locations: Counter[str],
    dry_run: bool,
) -> None:
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Dry run: {dry_run}")
    print(f"Changed chunks: {sum(changed.values())}")
    for (before, after), count in sorted(changed.items()):
        print(f"- {before} -> {after}: {count}")

    if invalid_locations:
        print("Unmapped invalid locations remain:")
        for location, count in sorted(invalid_locations.items()):
            print(f"- {location}: {count}")
    else:
        print("All non-null location values are in LOCATIONS.")


def main() -> None:
    args = parse_args()
    input_path = args.input_path
    output_path = args.output_path or input_path

    chunks = load_chunks(input_path)
    changed = normalize_locations(chunks)
    invalid_locations = find_invalid_locations(chunks)

    if not args.dry_run:
        save_chunks(chunks, output_path)

    print_summary(
        input_path=input_path,
        output_path=output_path,
        changed=changed,
        invalid_locations=invalid_locations,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
