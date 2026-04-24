"""저장된 파싱 결과 JSON을 점검하는 스크립트."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from parsing.loader import IMAGE_CATEGORIES, load_documents_json


JSON_PATH = Path("data/raw/230630_자동차사고 과실비율 인정기준_최종/parsed_documents.json")


def print_summary(docs: list) -> None:
    """카테고리와 페이지 분포를 출력한다."""
    category_counts = Counter(str(doc.metadata.get("category", "unknown")).lower() for doc in docs)
    page_counts = Counter(doc.metadata.get("page", "unknown") for doc in docs)
    numeric_pages = sorted(page for page in page_counts if isinstance(page, int))

    print("\n=== 페이지 파싱 요약 ===")
    if numeric_pages:
        print(f"파싱된 페이지 수: {len(numeric_pages)}")
        print(f"페이지 범위: {numeric_pages[0]} ~ {numeric_pages[-1]}")
    else:
        print("파싱된 페이지 정보를 찾지 못했습니다.")

    print("\n=== 카테고리별 개수 ===")
    for category, count in sorted(category_counts.items()):
        print(f"{category}: {count}")

    print("\n=== 페이지별 요소 수 ===")
    for page in numeric_pages:
        print(f"page {page}: {page_counts[page]}")

    for page, count in page_counts.items():
        if not isinstance(page, int):
            print(f"page {page}: {count}")


def print_text_samples(docs: list, limit: int = 5) -> None:
    """텍스트 요소 샘플을 출력한다."""
    print("\n=== 텍스트 샘플 ===")
    shown = 0
    for doc in docs:
        category = str(doc.metadata.get("category", "")).lower()
        if category in IMAGE_CATEGORIES:
            continue

        preview = " ".join(doc.page_content.split())[:200]
        print(f"[page={doc.metadata.get('page')}] [{category}] {preview}")
        shown += 1
        if shown >= limit:
            break


def print_image_samples(docs: list, limit: int = 5) -> None:
    """이미지 요소의 메타데이터 샘플을 출력한다."""
    print("\n=== 이미지 샘플 ===")
    shown = 0
    for doc in docs:
        category = str(doc.metadata.get("category", "")).lower()
        if category not in IMAGE_CATEGORIES:
            continue

        print(
            f"[page={doc.metadata.get('page')}] "
            f"[{category}] image_path={doc.metadata.get('image_path')} "
            f"coordinates={doc.metadata.get('coordinates')}"
        )
        preview = " ".join(doc.page_content.split())[:120]
        if preview:
            print(f"page_content: {preview}")
        shown += 1
        if shown >= limit:
            break


if __name__ == "__main__":
    if not JSON_PATH.exists():
        print(f"[ERROR] inspect_parsing - 저장된 JSON 파일이 없습니다: {JSON_PATH}")
        print("[INFO] 먼저 `uv run python parsing/loader.py`를 실행해 parsed_documents.json을 생성하세요.")
        raise SystemExit(1)

    docs = load_documents_json(JSON_PATH)
    print(f"\n파싱 요소 수: {len(docs)}")
    print_summary(docs)
    print_text_samples(docs)
    print_image_samples(docs)
