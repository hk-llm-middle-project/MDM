"""Upstage Document Parse 기반 PDF 파싱 유틸리티."""

from __future__ import annotations

import base64
import json
import tempfile
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_upstage import UpstageDocumentParseLoader
from pypdf import PdfReader, PdfWriter


IMAGE_CATEGORIES = {"figure", "chart", "table"}
TEXT_CATEGORIES = {"paragraph", "caption", "list"}
READ_TIMEOUT_SECONDS = 900


def _log_error(func_name: str, message: str, exc: Exception) -> None:
    """예외 정보를 일관된 형식으로 출력한다."""
    print(f"[ERROR] {func_name} - {message}: {type(exc).__name__} - {exc}")


def split_pdf_for_upstage(file_path: str, page_size: int = 100) -> list[dict]:
    """PDF를 page_size 단위 임시 PDF들로 분할한다."""
    pdf_path = Path(file_path)

    try:
        reader = PdfReader(str(pdf_path))
        split_files: list[dict] = []

        for start in range(0, len(reader.pages), page_size):
            end = min(start + page_size, len(reader.pages))
            writer = PdfWriter()

            # Upstage 100페이지 제한에 맞춰 페이지를 나눈다.
            for page_index in range(start, end):
                writer.add_page(reader.pages[page_index])

            temp_dir = Path(tempfile.mkdtemp(prefix="upstage_split_"))
            temp_path = temp_dir / f"pages_{start + 1}_{end}.pdf"
            with temp_path.open("wb") as fp:
                writer.write(fp)

            split_files.append({"path": str(temp_path), "page_offset": start})

        print(f"[INFO] split_pdf_for_upstage - {pdf_path.name} -> {len(split_files)}개 분할본 생성")
        return split_files
    except Exception as exc:
        _log_error("split_pdf_for_upstage", f"{pdf_path} 분할 실패", exc)
        raise


def load_documents(file_path: str) -> list[Document]:
    """Upstage Document Parse로 PDF를 element 단위 Document 목록으로 읽는다."""
    load_dotenv()

    pdf_path = Path(file_path)
    documents: list[Document] = []
    split_files = split_pdf_for_upstage(str(pdf_path), page_size=100)

    for split_info in split_files:
        split_path = Path(split_info["path"])
        page_offset = split_info["page_offset"]

        try:
            print(f"[INFO] load_documents - {split_path.name} 파싱 시작")
            print(
                "[INFO] load_documents - 현재 langchain-upstage 버전에서는 timeout 옵션을 직접 받지 않아 "
                f"로더 기본 타임아웃을 사용합니다. (목표 read timeout: {READ_TIMEOUT_SECONDS}초)"
            )
            loader = UpstageDocumentParseLoader(
                file_path=str(split_path),
                split="element",
                output_format="markdown",
                coordinates=True,
                base64_encoding=["figure", "chart", "table"],
                ocr="auto",
            )
            split_docs = loader.load()

            # 분할본 page 값을 원본 PDF page 번호로 복원한다.
            for doc in split_docs:
                page = doc.metadata.get("page")
                if isinstance(page, int):
                    doc.metadata["page"] = page + page_offset
                elif isinstance(page, str) and page.isdigit():
                    doc.metadata["page"] = int(page) + page_offset
                doc.metadata["source"] = str(pdf_path)

            documents.extend(split_docs)
            print(f"[INFO] load_documents - {split_path.name} 파싱 성공: {len(split_docs)}개 요소")
        except Exception as exc:
            _log_error("load_documents", f"분할본 {split_path.name} 파싱 실패", exc)
            raise

    print(f"[INFO] load_documents - 전체 파싱 완료: {len(documents)}개 요소")
    return documents


def save_base64_images(docs: list[Document], output_dir: Path) -> list[str]:
    """메타데이터의 base64 이미지를 PNG 파일로 저장하고 경로를 기록한다."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[str] = []
    counters: dict[tuple[int | str, str], int] = defaultdict(int)
    success_count = 0
    fail_count = 0

    for index, doc in enumerate(docs):
        metadata = doc.metadata
        category = str(metadata.get("category", "")).lower()
        encoded = metadata.get("base64_encoding")

        if category not in IMAGE_CATEGORIES or not encoded:
            continue

        try:
            page = metadata.get("page", "unknown")
            counters[(page, category)] += 1
            image_bytes = base64.b64decode(encoded)
            image_path = output_dir / f"page_{page}_{category}_{counters[(page, category)]}.png"

            # 후속 단계는 파일 경로만 쓰도록 base64를 이미지 파일로 내린다.
            image_path.write_bytes(image_bytes)
            metadata["image_path"] = str(image_path)
            metadata.pop("base64_encoding", None)

            saved_paths.append(str(image_path))
            success_count += 1
        except Exception as exc:
            fail_count += 1
            _log_error("save_base64_images", f"{index}번 요소 이미지 저장 실패", exc)
            metadata.pop("base64_encoding", None)

    print(f"[INFO] save_base64_images - 저장 성공 {success_count}건, 실패 {fail_count}건")
    return saved_paths


def save_documents_json(docs: list[Document], output_path: Path) -> Path:
    """Document 목록을 JSON 파일로 저장한다."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = [
        {
            "page_content": doc.page_content,
            "metadata": doc.metadata,
        }
        for doc in docs
    ]

    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)

    print(f"[INFO] save_documents_json - 저장 완료: {output_path}")
    return output_path


def load_documents_json(json_path: Path) -> list[Document]:
    """저장된 JSON 파일을 다시 Document 목록으로 복원한다."""
    json_path = Path(json_path)

    with json_path.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)

    docs = [
        Document(
            page_content=item.get("page_content", ""),
            metadata=item.get("metadata", {}),
        )
        for item in payload
    ]
    print(f"[INFO] load_documents_json - 불러오기 완료: {json_path}")
    return docs


def load_pdf(path: str) -> list[Document]:
    """기존 호출부 호환을 위해 load_documents를 감싼다."""
    return load_documents(path)


if __name__ == "__main__":
    load_dotenv()

    PDF_PATH = "data/raw/230630_자동차사고 과실비율 인정기준_최종.pdf"
    output_dir = Path(PDF_PATH).with_suffix("")
    docs = load_documents(PDF_PATH)
    saved = save_base64_images(docs, output_dir)
    json_path = save_documents_json(docs, output_dir / "parsed_documents.json")

    print(f"파싱 완료: {len(docs)}개 요소")
    print(f"이미지 저장 완료: {len(saved)}건")
    print(f"문서 JSON 저장 완료: {json_path}")
