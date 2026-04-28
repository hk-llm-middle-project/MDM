"""Generate descriptions for parsed PDF image elements."""

from __future__ import annotations

import argparse
import base64
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from openai import OpenAI

from .storage import load_documents_json, save_documents_json


IMAGE_CATEGORIES = {"figure", "table"}
TEMPERATURE = 0
MAX_TOKENS = 1000
SAVE_INTERVAL = 50
OPENAI_MODEL = "gpt-4o-mini"

FIGURE_PROMPT = """다음은 PDF 문서의 [figure]입니다.
이미지에서 확인되는 사실만 기술하고 추정은 하지 마세요.
그림의 종류, 시각 요소, 핵심 내용을 4~6문장으로 한국어로 설명하세요.
자연어 검색 질의에 잘 매칭되도록 핵심 키워드를 포함하세요."""


def _category(doc: Document) -> str:
    return str(doc.metadata.get("category", "")).lower()


def _page(doc: Document) -> str:
    return str(doc.metadata.get("page", "unknown"))


def _has_description(doc: Document) -> bool:
    return bool(doc.metadata.get("description"))


def _image_path_exists(doc: Document) -> bool:
    image_path = doc.metadata.get("image_path")
    return bool(image_path) and Path(str(image_path)).exists()


def load_and_describe(json_path: str) -> None:
    """Load parsed documents, add descriptions, and overwrite the same JSON."""
    load_dotenv()

    docs = load_documents_json(json_path)
    image_docs_count = sum(1 for doc in docs if _category(doc) in IMAGE_CATEGORIES)

    print(f"[INFO] 처리 시작 - 전체 요소 수: {len(docs)}, IMAGE_CATEGORIES 요소 수: {image_docs_count}")
    generate_descriptions(docs, json_path)
    save_documents_json(docs, json_path)


def should_skip(doc: Document) -> bool:
    """Return True when this document should not be described."""
    category = _category(doc)
    if category not in IMAGE_CATEGORIES:
        return True
    if _has_description(doc):
        return True
    if category == "table":
        return False
    return not _image_path_exists(doc)


def encode_image(image_path: str) -> str:
    """Encode an image file as a base64 string."""
    with Path(image_path).open("rb") as fp:
        return base64.b64encode(fp.read()).decode("utf-8")


def build_prompt(category: str) -> str:
    """Return a prompt for the given image category."""
    normalized_category = category.lower()
    if normalized_category == "table":
        return ""
    return FIGURE_PROMPT.replace("[figure]", f"[{normalized_category}]")


def describe_image(image_path: str, category: str) -> str:
    """Call OpenAI vision model and return a Korean image description."""
    encoded_image = encode_image(image_path)
    prompt = build_prompt(category)
    client = OpenAI()

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{encoded_image}"},
                    },
                ],
            }
        ],
    )

    description = response.choices[0].message.content
    return description.strip() if description else ""


def generate_descriptions(docs: list[Document], json_path: str) -> None:
    """Generate descriptions with per-item logging and periodic saving."""
    success_count = 0
    fail_count = 0
    skip_count = 0
    processed_count = 0
    total_count = len(docs)

    for index, doc in enumerate(docs, start=1):
        category = _category(doc)
        page = _page(doc)

        if category not in IMAGE_CATEGORIES:
            skip_count += 1
            continue

        if _has_description(doc):
            skip_count += 1
            print(f"[INFO] 요소 {index}/{total_count}, category={category}, page={page}, 처리결과=스킵(기존 description)")
            continue

        try:
            if category == "table":
                doc.metadata["description"] = doc.page_content
                success_count += 1
                processed_count += 1
                print(f"[INFO] 요소 {index}/{total_count}, category={category}, page={page}, 처리결과=생성(page_content)")
            else:
                image_path = doc.metadata.get("image_path")
                if not image_path:
                    skip_count += 1
                    print(f"[INFO] 요소 {index}/{total_count}, category={category}, page={page}, 처리결과=스킵(image_path 없음)")
                    continue

                if not Path(str(image_path)).exists():
                    skip_count += 1
                    print(f"[INFO] 요소 {index}/{total_count}, category={category}, page={page}, 처리결과=스킵(이미지 파일 없음: {image_path})")
                    continue

                doc.metadata["description"] = describe_image(str(image_path), category)
                success_count += 1
                processed_count += 1
                print(f"[INFO] 요소 {index}/{total_count}, category={category}, page={page}, 처리결과=생성")

            if processed_count > 0 and processed_count % SAVE_INTERVAL == 0:
                save_documents_json(docs, json_path)
                print(f"[INFO] {processed_count}번째 처리 후 중간 저장")

        except Exception as exc:
            fail_count += 1
            print(
                f"[ERROR] 요소 {index}, category={category}, page={page}, "
                f"에러 타입={type(exc).__name__}, 에러 메시지={exc}"
            )
            continue

    print(f"[INFO] 처리 완료 - 성공: {success_count}, 실패: {fail_count}, 건너뜀: {skip_count}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add image descriptions to parsed_documents JSON.")
    parser.add_argument("json_path", help="Path to parsed_documents.json")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    load_and_describe(args.json_path)
