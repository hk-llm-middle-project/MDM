"""Post-process chunked documents for vector ingestion."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


MIN_CONTENT_LENGTH = 10
HEADING_PREFIX = "# "
TABLE_SEPARATOR_PATTERN = "\n---\n"
PREFACE_CHUNK_TYPE = "preface"
IMAGE_PATH_KEY = "image_path"
IMAGE_DIRNAME = "img"
PARTY_TYPE_BY_PREFIX = {
    "\ubcf4": "\ubcf4\ud589\uc790",
    "\ucc28": "\uc790\ub3d9\ucc28",
    "\uac70": "\uc790\uc804\uac70",
}
LOCATION_CROSSWALK = "횡단보도 내"
LOCATION_CROSSWALK_NEAR = "횡단보도 부근"
LOCATION_NO_CROSSWALK = "횡단보도 없음"
LOCATION_INTERSECTION = "교차로 사고"
LOCATION_ETC = "기타"
LOCATION_OPPOSITE_DIRECTION = "마주보는 방향 진행차량 상호 간의 사고"
LOCATION_SAME_DIRECTION = "같은 방향 진행차량 상호간의 사고"
LOCATION_CAR_MOTORCYCLE = "자동차 대 이륜차 특수유형"
DIAGRAM_ID_PATTERN = re.compile(r"^(?P<prefix>[보차거])(?P<number>\d+)")


DocumentDict = dict[str, Any]


def clean_page_content(text: str) -> str:
    """Normalize chunk text content."""
    cleaned = text.replace(HEADING_PREFIX, "")
    cleaned = cleaned.replace(TABLE_SEPARATOR_PATTERN, "\n")
    return cleaned.strip()


def _party_type(diagram_id: Any) -> str | None:
    if not isinstance(diagram_id, str) or not diagram_id:
        return None

    for prefix, party_type in PARTY_TYPE_BY_PREFIX.items():
        if diagram_id.startswith(prefix):
            return party_type
    return None


def _location_from_diagram_id(diagram_id: Any) -> str | None:
    if not isinstance(diagram_id, str):
        return None

    match = DIAGRAM_ID_PATTERN.match(diagram_id)
    if not match:
        return None

    prefix = match.group("prefix")
    number = int(match.group("number"))

    if prefix == "보":
        if number <= 19:
            return LOCATION_CROSSWALK
        if number <= 21:
            return LOCATION_CROSSWALK_NEAR
        if number <= 28:
            return LOCATION_NO_CROSSWALK
        return LOCATION_ETC

    if prefix == "차":
        if number >= 61:
            return LOCATION_CAR_MOTORCYCLE

    return None


def _normalize_location(section: Any, subsection: Any, diagram_id: Any = None) -> str | None:
    """Map detailed section/subsection labels to the shared intake location values."""
    section_text = str(section or "")
    subsection_text = str(subsection or "")
    joined_text = f"{section_text} {subsection_text}"

    if "횡단보도 내" in joined_text:
        return LOCATION_CROSSWALK
    if "횡단보도 부근" in joined_text or "횡단시설 부근" in joined_text:
        return LOCATION_CROSSWALK_NEAR
    if "횡단보도 없음" in joined_text:
        return LOCATION_NO_CROSSWALK
    if "자동차 대 이륜차" in joined_text or "이륜차 특수유형" in joined_text:
        return LOCATION_CAR_MOTORCYCLE
    if "마주보는 방향" in joined_text:
        return LOCATION_OPPOSITE_DIRECTION
    if "같은 방향" in joined_text or "동일차로" in joined_text:
        return LOCATION_SAME_DIRECTION
    if "교차로" in joined_text:
        return LOCATION_INTERSECTION
    if subsection_text or section_text:
        return LOCATION_ETC
    return _location_from_diagram_id(diagram_id)


def add_metadata(doc: DocumentDict) -> DocumentDict:
    """Add derived metadata fields to a document dict."""
    metadata = doc.setdefault("metadata", {})
    diagram_id = metadata.get("diagram_id")
    metadata["party_type"] = _party_type(diagram_id)
    metadata["location"] = _normalize_location(
        metadata.get("section"),
        metadata.get("subsection"),
        diagram_id,
    )
    return doc


def normalize_image_path(doc: DocumentDict, output_path: str | Path) -> DocumentDict:
    """Rewrite image paths to the final output image directory when possible."""
    metadata = doc.get("metadata", {})
    image_path = metadata.get(IMAGE_PATH_KEY)
    if not isinstance(image_path, str) or not image_path:
        return doc

    candidate_path = Path(output_path).parent / IMAGE_DIRNAME / Path(image_path).name
    if candidate_path.exists():
        metadata[IMAGE_PATH_KEY] = str(candidate_path)
    return doc


def should_remove(doc: DocumentDict) -> bool:
    """Return True if a non-preface chunk is too short to keep."""
    metadata = doc.get("metadata", {})
    if metadata.get("chunk_type") == PREFACE_CHUNK_TYPE:
        return False
    return len(str(doc.get("page_content", "")).strip()) < MIN_CONTENT_LENGTH


def _load_json(input_path: str | Path) -> list[DocumentDict]:
    path = Path(input_path)
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def _save_json(docs: list[DocumentDict], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(docs, fp, ensure_ascii=False, indent=2)
    return path


def run_postprocess(input_path: str, output_path: str) -> None:
    """Load, clean, enrich, filter, and save chunk dicts."""
    docs = _load_json(input_path)
    print(f"[INFO] \ucc98\ub9ac \uc2dc\uc791 - \uc804\uccb4 \uccad\ud06c \uc218: {len(docs)}")

    processed_docs: list[DocumentDict] = []
    remove_count = 0

    for doc in docs:
        if should_remove(doc):
            remove_count += 1
            continue

        doc["page_content"] = clean_page_content(str(doc.get("page_content", "")))
        doc = add_metadata(doc)
        doc = normalize_image_path(doc, output_path)

        if should_remove(doc):
            remove_count += 1
            continue

        processed_docs.append(doc)

    distribution = Counter(doc.get("metadata", {}).get("chunk_type") for doc in processed_docs)
    _save_json(processed_docs, output_path)
    print(
        f"[INFO] \ucc98\ub9ac \uc644\ub8cc - \uc81c\uac70 \uc218: {remove_count}, "
        f"\ucd5c\uc885 \uccad\ud06c \uc218: {len(processed_docs)}, chunk_type\ubcc4 \ubd84\ud3ec: {dict(distribution)}"
    )
    print(f"[INFO] \uc800\uc7a5 \uc644\ub8cc: {output_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Post-process chunked documents.")
    parser.add_argument("input_path", help="Path to chunked_documents_v2.json")
    parser.add_argument("output_path", help="Path to chunked_documents_final.json")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_postprocess(args.input_path, args.output_path)
