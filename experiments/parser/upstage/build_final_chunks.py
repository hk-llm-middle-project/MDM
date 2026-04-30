"""Build versioned final chunks from Upstage raw parsed documents."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


RAW_INPUT = Path("data/upstage_output/main_pdf/raw/parsed_documents.json")
FINAL_OUTPUT = Path("data/upstage_output/main_pdf/final/chunked_documents_final.v2.json")
REPORT_OUTPUT = Path("data/upstage_output/main_pdf/final/chunked_documents_final.v2.report.json")
FINAL_IMAGE_DIR = Path("data/upstage_output/main_pdf/final/img")
EXCLUDED_IMAGE_FILENAMES = {
    "page_29_table_1.png",
    "page_455_table_1.png",
    "page_459_table_1.png",
    "page_464_table_1.png",
    "page_468_table_1.png",
    "page_471_table_1.png",
    "page_476_table_1.png",
    "page_479_table_1.png",
    "page_600_table_1.png",
}

# Pedestrian diagram ranges from the source document.
PEDESTRIAN_CROSSWALK_IN_MAX = 19
PEDESTRIAN_CROSSWALK_NEAR_MAX = 21
PEDESTRIAN_NO_CROSSWALK_MAX = 28

# Vehicle diagram ranges from the source document.
CAR_INTERSECTION_MAX = 19
CAR_MOTORCYCLE_SPECIAL_MIN = 61
BICYCLE_INTERSECTION_MAX = 21

PARTY_TYPES = ["보행자", "자동차", "자전거"]
LOCATIONS = [
    "횡단보도 내",
    "횡단보도 부근",
    "횡단보도 없음",
    "교차로 사고",
    "기타",
    "마주보는 방향 진행차량 상호 간의 사고",
    "같은 방향 진행차량 상호간의 사고",
    "자동차 대 이륜차 특수유형",
]

PREFACE = "preface"
GENERAL = "general"
PARENT = "parent"
CHILD = "child"

SKIP_CATEGORIES = {"header", "footer"}
TEXT_CATEGORIES = {"paragraph", "caption", "list", "heading1"}
TABLE_CATEGORY = "table"
INDEX_CATEGORY = "index"

DIAGRAM_RE = re.compile(r"(?P<prefix>[보차거])\s*(?P<num>\d+)(?:\s*[-–]\s*(?P<sub>\d+))?(?:\s*\((?P<var>[가-힣])\))?")
VARIANT_RE = re.compile(r"\(([가나다라])\)")
HEADING_PREFIX_RE = re.compile(r"^\s*#\s*")

CASE_HEADINGS = {
    "사고 상황",
    "기본 과실비율 해설",
    "수정요소(인과관계를 감안한 과실비율 조정) 해설",
    "활용시 참고 사항",
    "관련 법규",
    "참고 판례",
}

HEADING_ALIASES = {
    "수정요소 해설": "수정요소(인과관계를 감안한 과실비율 조정) 해설",
    "수정요소(인과관계를 감안한 과실비율 조정)의 해설": "수정요소(인과관계를 감안한 과실비율 조정) 해설",
}


@dataclass
class ChildDraft:
    page_content: str
    page: int | None
    source: str | None
    heading: str | None = None
    image_path: str | None = None


@dataclass
class ParentDraft:
    diagram_id: str
    title: str
    context_titles: list[str]
    page: int | None
    source: str | None
    party_type: str | None
    location: str | None
    image_path: str | None = None
    children: list[ChildDraft] = field(default_factory=list)

    def add_child(
        self,
        content: str,
        page: int | None,
        source: str | None,
        heading: str | None = None,
        image_path: str | None = None,
    ) -> None:
        cleaned = clean_text(content)
        if cleaned and cleaned != "-":
            if self.children and should_merge_child(self.children[-1].heading, heading):
                previous = self.children[-1]
                previous.page_content = f"{previous.page_content.rstrip()}\n\n{cleaned}"
                if image_path and previous.image_path is None:
                    previous.image_path = image_path
                return
            self.children.append(ChildDraft(cleaned, page, source, heading, image_path))


def should_merge_child(previous_heading: str | None, heading: str | None) -> bool:
    if previous_heading != heading:
        return False
    return heading in {
        "표",
        "사고 상황",
        "기본 과실비율 해설",
        "수정요소(인과관계를 감안한 과실비율 조정) 해설",
        "활용시 참고 사항",
        "관련 법규",
        "참고 판례",
    }


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == "---":
            continue
        lines.append(stripped)
    return "\n".join(line for line in lines if line).strip()


def normalize_diagram_id(value: str) -> str:
    def repl(match: re.Match[str]) -> str:
        base = f"{match.group('prefix')}{match.group('num')}"
        if match.group("sub"):
            base += f"-{match.group('sub')}"
        if match.group("var"):
            base += f"({match.group('var')})"
        return base

    return DIAGRAM_RE.sub(repl, value)


def find_diagram_ids(text: str) -> list[str]:
    ids: list[str] = []
    for match in DIAGRAM_RE.finditer(text):
        diagram_id = normalize_diagram_id(match.group(0))
        if diagram_id not in ids:
            ids.append(diagram_id)
    return ids


def first_markdown_cell(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("|"):
        return stripped.splitlines()[0] if stripped else ""
    first_line = stripped.splitlines()[0]
    cells = [cell.strip() for cell in first_line.strip("|").split("|")]
    return cells[0] if cells else ""


def extract_table_diagram_ids(text: str) -> list[str]:
    first_cell_ids = find_diagram_ids(first_markdown_cell(text))
    if not first_cell_ids or len(first_cell_ids) > 3:
        return []
    ids = first_cell_ids
    if not ids:
        return []

    if len(ids) == 1:
        variants = []
        for variant in VARIANT_RE.findall(text):
            if variant not in variants:
                variants.append(variant)
        if len(variants) >= 2:
            return [f"{ids[0]}({variant})" for variant in variants]
    return ids


def base_diagram_id(diagram_id: str) -> str:
    return re.sub(r"\([가나다라]\)$", "", diagram_id)


def party_type_for(diagram_id: str | None) -> str | None:
    if not diagram_id:
        return None
    if diagram_id.startswith("보"):
        return "보행자"
    if diagram_id.startswith("차"):
        return "자동차"
    if diagram_id.startswith("거"):
        return "자전거"
    return None


def diagram_number(diagram_id: str | None) -> int | None:
    if not diagram_id:
        return None
    match = DIAGRAM_RE.match(diagram_id)
    return int(match.group("num")) if match else None


def location_for(diagram_id: str | None, context: str = "") -> str | None:
    number = diagram_number(diagram_id)
    if not diagram_id or number is None:
        return None

    if diagram_id.startswith("보"):
        if number <= PEDESTRIAN_CROSSWALK_IN_MAX:
            return "횡단보도 내"
        if number <= PEDESTRIAN_CROSSWALK_NEAR_MAX:
            return "횡단보도 부근"
        if number <= PEDESTRIAN_NO_CROSSWALK_MAX:
            return "횡단보도 없음"
        return "기타"

    joined = context.replace(" ", "")
    if diagram_id.startswith("차"):
        if number >= CAR_MOTORCYCLE_SPECIAL_MIN:
            return "자동차 대 이륜차 특수유형"
        if any(token in joined for token in ["중앙선", "마주보는", "유턴"]):
            return "마주보는 방향 진행차량 상호 간의 사고"
        if any(token in joined for token in ["추돌", "진로변경", "동일차로", "정차후출발", "낙하물", "주정차", "앞지르기"]):
            return "같은 방향 진행차량 상호간의 사고"
        if "교차로" in joined or number <= CAR_INTERSECTION_MAX:
            return "교차로 사고"
        return "기타"

    if diagram_id.startswith("거"):
        if "동일차로" in joined:
            return "같은 방향 진행차량 상호간의 사고"
        if "교차로" in joined or number <= BICYCLE_INTERSECTION_MAX:
            return "교차로 사고"
        return "기타"
    return None


def normalize_heading(text: str) -> str | None:
    cleaned = clean_text(text)
    line = cleaned.splitlines()[0] if cleaned else ""
    line = HEADING_PREFIX_RE.sub("", line).strip()
    line = HEADING_ALIASES.get(line, line)
    return line if line in CASE_HEADINGS else None


def clean_heading_text(text: str) -> str:
    cleaned = clean_text(normalize_diagram_id(text))
    line = cleaned.splitlines()[0] if cleaned else ""
    return HEADING_PREFIX_RE.sub("", line).strip()


def clean_title_text(text: str) -> str:
    cleaned = clean_text(normalize_diagram_id(text))
    return HEADING_PREFIX_RE.sub("", cleaned).strip()


def heading_diagram_ids(text: str) -> list[str]:
    heading = clean_title_text(text)
    if not heading:
        return []
    bracketed = re.findall(r"\[([^\]]+)\]", heading)
    ids: list[str] = []
    for value in bracketed:
        for diagram_id in find_diagram_ids(value):
            if diagram_id not in ids:
                ids.append(diagram_id)
    return ids


def heading_has_diagram_range(text: str) -> bool:
    heading = clean_title_text(text)
    return bool(re.search(r"[보차거]\s*\d+\s*[~∼]\s*[보차거]?\s*\d+", heading))


def is_subsection_heading(text: str) -> bool:
    return bool(re.match(r"^\(\d+\)\s+", clean_heading_text(text)))


def is_context_heading(text: str) -> bool:
    heading = clean_heading_text(text)
    if not heading:
        return False
    if normalize_heading(text):
        return False
    if any(
        token in heading
        for token in [
            "도로교통법",
            "자동차관리법",
            "자전거 이용 활성화",
            "민법",
            "시행령",
            "시행규칙",
            "별표",
            "대법원",
            "법원",
            "판결",
        ]
    ):
        return False
    if re.match(r"^\d+\.\s*", heading):
        return False
    if heading_diagram_ids(text) and not heading_has_diagram_range(text):
        return False
    if heading.startswith(("⊙", "※")):
        return False
    if "세부유형별 과실비율 적용기준" in heading:
        return True
    if any(
        token in heading
        for token in [
            "횡단보도",
            "횡단시설",
            "교차로",
            "중앙선",
            "진로변경",
            "동일차로",
            "같은 방향",
            "마주보는 방향",
            "기타 사고유형",
            "특수유형",
        ]
    ):
        return True
    return bool(re.match(r"^(\(?\d+\)|\d+\)|[가-하]\.)\s+", heading))


def is_case_context_boundary(text: str) -> bool:
    heading = clean_heading_text(text)
    if not is_context_heading(text):
        return False
    if heading.startswith("(참고)") or heading.startswith("참고"):
        return False
    return True


def is_chapter_boundary(text: str) -> bool:
    heading = clean_heading_text(text)
    if not heading:
        return False
    if is_case_detail_root(text):
        return False
    if heading in {
        "과실비율 적용기준(사고유형별)",
        "자동차사고 과실비율 인정기준",
        "목 차",
    }:
        return True
    if re.match(r"^제\d+[편장]$", heading):
        return True
    return heading in {
        "1. 과실비율 인정기준의 필요성",
        "2. 과실과 과실상계",
        "3. 과실비율 인정기준의 기본원칙",
        "4. 과실비율 인정기준의 적용",
        "5. 인적 손해에서의 과실상계 별도적용기준",
        "1. 적용 범위",
        "2. 용어 정의",
    } or heading.startswith("3. 수정요소")


def is_case_detail_root(text: str) -> bool:
    heading = clean_heading_text(text)
    return heading.startswith("4. 세부유형별 과실비율 적용기준")


def is_flat_section_start(text: str) -> bool:
    heading = clean_heading_text(text)
    if not heading:
        return False
    if is_chapter_boundary(text):
        return True
    return bool(re.match(r"^\(\d+\)\s+", heading))


def title_from_table(diagram_id: str, table_content: str) -> str:
    lines = [line.strip() for line in clean_text(table_content).splitlines() if line.strip()]
    first_line = lines[0] if lines else diagram_id
    cells = [cell.strip() for cell in first_line.strip("|").split("|")] if first_line.startswith("|") else [first_line]
    cells = [cell for cell in cells if cell]
    if len(cells) >= 2:
        return f"{diagram_id} {cells[1]}"
    return cells[0] if cells else diagram_id


def render_child_for_parent(child: ChildDraft, previous_heading: str | None) -> tuple[str, str | None]:
    content = child.page_content.strip()
    if not content:
        return "", previous_heading

    parts: list[str] = []
    heading = child.heading
    if heading and heading != "표" and heading != previous_heading:
        parts.append(heading)
        previous_heading = heading

    if heading and heading in {"사고 상황", "기본 과실비율 해설"}:
        first_line = content.splitlines()[0] if content else ""
        if not first_line.lstrip().startswith(("⊙", "-", "•")):
            content = f"⊙ {content}"

    parts.append(content)
    return "\n".join(parts), previous_heading


def render_parent_content(parent: ParentDraft) -> str:
    parts = [title.strip() for title in parent.context_titles if title.strip()]
    parts.append(parent.title.strip())
    previous_heading: str | None = None
    for child in parent.children:
        rendered, previous_heading = render_child_for_parent(child, previous_heading)
        if rendered:
            parts.append(rendered)
    return "\n\n".join(part for part in parts if part).strip()


def render_child_content(child: ChildDraft) -> str:
    rendered, _ = render_child_for_parent(child, None)
    return rendered or child.page_content


def is_preface_element(doc: dict[str, Any]) -> bool:
    metadata = doc.get("metadata", {})
    page = metadata.get("page")
    category = str(metadata.get("category", "")).lower()
    return isinstance(page, int) and (page <= 9 or page >= 588 or (category == INDEX_CATEGORY and page < 30))


def metadata_source(doc: dict[str, Any], fallback: Path) -> str:
    return str(doc.get("metadata", {}).get("source") or fallback)


def metadata_page(doc: dict[str, Any]) -> int | None:
    page = doc.get("metadata", {}).get("page")
    if isinstance(page, int):
        return page
    if isinstance(page, str) and page.isdigit():
        return int(page)
    return None


def image_path_from(doc: dict[str, Any]) -> str | None:
    image_path = doc.get("metadata", {}).get("image_path")
    if not image_path:
        return None
    raw_candidate = Path(str(image_path))
    if raw_candidate.name in EXCLUDED_IMAGE_FILENAMES:
        return None
    raw_img_candidate = RAW_INPUT.parent / "img" / raw_candidate.name
    candidate = FINAL_IMAGE_DIR / Path(str(image_path)).name
    if candidate.exists():
        return str(candidate)
    source = None
    if raw_candidate.exists():
        source = raw_candidate
    elif raw_img_candidate.exists():
        source = raw_img_candidate
    if source:
        candidate.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, candidate)
        return str(candidate)
    return str(image_path)


def is_pre_heading_note(text: str) -> bool:
    cleaned = clean_text(text)
    cleaned = HEADING_PREFIX_RE.sub("", cleaned).strip()
    cleaned = re.sub(r"^[-*]\s*", "", cleaned).strip()
    return cleaned.startswith("※")


def split_labeled_blocks(text: str, active_ids: list[str]) -> tuple[dict[str, str], str | None]:
    cleaned = clean_text(normalize_diagram_id(text))
    if not cleaned:
        return {}, None

    markers: list[tuple[int, int, str]] = []
    for diagram_id in active_ids:
        candidates = [re.escape(diagram_id)]
        base = base_diagram_id(diagram_id)
        if base != diagram_id:
            candidates.append(re.escape(base) + r"\s*" + re.escape(diagram_id[len(base):]))
        pattern = re.compile(
            rf"(^|\n)\s*(?:[-*]|\u2022|\u2299)?\s*({'|'.join(candidates)})(?=\s|[:：]|$)"
        )
        for match in pattern.finditer(cleaned):
            markers.append((match.start(), match.end(), diagram_id))

    variant_ids = [diagram_id for diagram_id in active_ids if re.search(r"\([가나다라]\)$", diagram_id)]
    if variant_ids and len({base_diagram_id(diagram_id) for diagram_id in variant_ids}) == 1:
        for diagram_id in variant_ids:
            variant = diagram_id[-2:-1]
            pattern = re.compile(
                rf"(^|\n)\s*(?:[-*]|\u2022|\u2299)?\s*\({re.escape(variant)}\)(?=\s|[:：]|$)"
            )
            for match in pattern.finditer(cleaned):
                markers.append((match.start(), match.end(), diagram_id))

    if not markers:
        return {}, cleaned

    markers.sort(key=lambda item: item[0])
    blocks: dict[str, list[str]] = {diagram_id: [] for diagram_id in active_ids}
    prefix = cleaned[: markers[0][0]].strip()
    for idx, (start, _end, diagram_id) in enumerate(markers):
        next_start = markers[idx + 1][0] if idx + 1 < len(markers) else len(cleaned)
        block = (prefix + "\n" if prefix and idx == 0 else "") + cleaned[start:next_start].strip()
        blocks[diagram_id].append(block)

    return {key: "\n".join(parts).strip() for key, parts in blocks.items() if parts}, None


def make_flat_chunk(doc: dict[str, Any], chunk_type: str, raw_path: Path) -> dict[str, Any] | None:
    content = clean_text(str(doc.get("page_content", "")))
    if not content:
        return None
    return {
        "page_content": content,
        "metadata": {
            "chunk_id": None,
            "chunk_type": chunk_type,
            "diagram_id": None,
            "parent_id": None,
            "page": metadata_page(doc),
            "source": metadata_source(doc, raw_path),
            "party_type": None,
            "location": None,
            "image_path": image_path_from(doc),
        },
    }


def should_merge_flat_chunk(previous: dict[str, Any], current: dict[str, Any]) -> bool:
    previous_metadata = previous.get("metadata", {})
    current_metadata = current.get("metadata", {})
    previous_content = str(previous.get("page_content", ""))
    previous_is_section = is_flat_section_start(previous_content)
    previous_image_path = previous_metadata.get("image_path")
    current_image_path = current_metadata.get("image_path")
    has_different_image = bool(
        previous_image_path and current_image_path and previous_image_path != current_image_path
    )

    return (
        previous_metadata.get("chunk_type") == current_metadata.get("chunk_type")
        and current_metadata.get("chunk_type") in {GENERAL, PREFACE}
        and (previous_metadata.get("page") == current_metadata.get("page") or previous_is_section)
        and not has_different_image
    )


def materialize_chunks(items: list[dict[str, Any] | ParentDraft]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    next_id = 1

    for item in items:
        if isinstance(item, dict):
            item["metadata"]["chunk_id"] = next_id
            output.append(item)
            next_id += 1
            continue

        parent = item
        parent_id = next_id
        output.append(
            {
                "page_content": render_parent_content(parent),
                "metadata": {
                    "chunk_id": parent_id,
                    "chunk_type": PARENT,
                    "diagram_id": parent.diagram_id,
                    "parent_id": None,
                    "page": parent.page,
                    "source": parent.source,
                    "party_type": parent.party_type,
                    "location": parent.location,
                    "image_path": parent.image_path,
                },
            }
        )
        next_id += 1

        for child in parent.children:
            output.append(
                {
                    "page_content": render_child_content(child),
                    "metadata": {
                        "chunk_id": next_id,
                        "chunk_type": CHILD,
                        "diagram_id": parent.diagram_id,
                        "parent_id": parent_id,
                        "page": child.page,
                        "source": child.source or parent.source,
                        "party_type": parent.party_type,
                        "location": parent.location,
                        "image_path": child.image_path,
                    },
                }
            )
            next_id += 1

    return output


def build_chunks(raw_docs: list[dict[str, Any]], raw_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    items: list[dict[str, Any] | ParentDraft] = []
    parents: list[ParentDraft] = []
    active: list[ParentDraft] = []
    current_heading: str | None = None
    pending_titles: dict[str, tuple[str, list[str], int | None, str | None]] = {}
    context_root: str | None = None
    context_section: str | None = None
    context_case: str | None = None
    in_case_detail_mode = False
    warnings: list[str] = []

    def flush_active() -> None:
        nonlocal active, current_heading
        ready = [parent for parent in active if parent.children]
        parents.extend(ready)
        items.extend(ready)
        active = []
        current_heading = None

    def append_flat(chunk: dict[str, Any] | None) -> None:
        if not chunk:
            return

        content = chunk.get("page_content", "")
        metadata = chunk.get("metadata", {})

        current_is_section_start = is_flat_section_start(str(content))
        if current_is_section_start:
            items.append(chunk)
            return

        if items and isinstance(items[-1], dict):
            previous = items[-1]
            previous_metadata = previous.get("metadata", {})
            previous_image_path = previous_metadata.get("image_path")
            current_image_path = metadata.get("image_path")
            if should_merge_flat_chunk(previous, chunk):
                previous["page_content"] = f"{previous.get('page_content', '').rstrip()}\n\n{content}"
                if current_image_path and previous_image_path is None:
                    previous_metadata["image_path"] = current_image_path
                return

        if content.startswith("#"):
            items.append(chunk)
            return

        items.append(chunk)

    def current_context_titles() -> list[str]:
        return [title for title in [context_root, context_section, context_case] if title]

    def set_context_heading(title: str) -> None:
        nonlocal context_section, context_case
        if "세부유형별 과실비율 적용기준" in title:
            return
        context_section = title
        context_case = None

    for idx, doc in enumerate(raw_docs):
        metadata = doc.get("metadata", {})
        category = str(metadata.get("category", "")).lower()
        content = str(doc.get("page_content", ""))
        page = metadata_page(doc)
        source = metadata_source(doc, raw_path)

        if category in SKIP_CATEGORIES:
            continue

        if is_preface_element(doc):
            flush_active()
            in_case_detail_mode = False
            append_flat(make_flat_chunk(doc, PREFACE, raw_path))
            continue

        if is_case_detail_root(content):
            flush_active()
            in_case_detail_mode = True
            context_root = clean_heading_text(content)
            context_section = None
            context_case = None
            pending_titles.clear()
            continue

        if is_chapter_boundary(content):
            flush_active()
            in_case_detail_mode = False
            context_root = None
            context_section = None
            context_case = None
            pending_titles.clear()
            append_flat(make_flat_chunk(doc, GENERAL, raw_path))
            continue

        if (
            active
            and in_case_detail_mode
            and category == "heading1"
            and is_case_context_boundary(content)
            and not normalize_heading(content)
        ):
            flush_active()
            set_context_heading(clean_heading_text(content))
            continue

        case_title_ids = heading_diagram_ids(content) if category == "heading1" else []
        if case_title_ids:
            flush_active()
            title = clean_title_text(content)
            if heading_has_diagram_range(content):
                if is_subsection_heading(content):
                    set_context_heading(title)
                else:
                    context_case = title
            else:
                context_case = None
                for diagram_id in case_title_ids:
                    parent_context = current_context_titles()
                    pending_titles[diagram_id] = (title, parent_context, page, source)
            continue

        if in_case_detail_mode and category in TEXT_CATEGORIES:
            case_title_ids = heading_diagram_ids(content)
            if case_title_ids:
                flush_active()
                title = clean_title_text(content)
                if re.match(r"^[가-하]\.\s+", title):
                    title_lines = [line.strip() for line in title.splitlines() if line.strip()]
                    context_section = title_lines[0]
                    context_case = "\n".join(title_lines[1:]) if len(title_lines) > 1 else None
                elif heading_has_diagram_range(content):
                    context_case = title
                else:
                    context_case = title
                continue

        if category == TABLE_CATEGORY:
            table_ids = extract_table_diagram_ids(content)
            if not table_ids:
                if active:
                    for parent in active:
                        parent.add_child(content, page, source, current_heading or "표", image_path_from(doc))
                else:
                    append_flat(make_flat_chunk(doc, GENERAL, raw_path))
                    warnings.append(f"table without diagram_id at raw index {idx}, page {page}")
                continue

            if active and current_heading is not None:
                flush_active()

            context = clean_text(content)
            for diagram_id in table_ids:
                base_id = base_diagram_id(diagram_id)
                title_info = pending_titles.get(diagram_id) or pending_titles.get(base_id)
                title = title_info[0] if title_info else title_from_table(diagram_id, content)
                parent_context = title_info[1] if title_info else current_context_titles()
                title_page = title_info[2] if title_info else page
                title_source = title_info[3] if title_info else source
                parent = ParentDraft(
                    diagram_id=diagram_id,
                    title=title,
                    context_titles=parent_context,
                    page=title_page,
                    source=title_source,
                    party_type=party_type_for(diagram_id),
                    location=location_for(diagram_id, context),
                    image_path=image_path_from(doc),
                )
                parent.add_child(content, page, source, heading="표", image_path=image_path_from(doc))
                active.append(parent)
            context_root = None
            current_heading = None
            continue

        if (
            not active
            and category == "heading1"
            and is_context_heading(content)
            and not normalize_heading(content)
        ):
            flush_active()
            if not in_case_detail_mode or (isinstance(page, int) and page < 39):
                append_flat(make_flat_chunk(doc, GENERAL, raw_path))
                continue
            set_context_heading(clean_heading_text(content))
            continue

        heading = normalize_heading(content) if category == "heading1" else None
        if heading:
            if active:
                current_heading = heading
            else:
                if in_case_detail_mode and category == "heading1" and is_context_heading(content):
                    set_context_heading(clean_heading_text(content))
                append_flat(make_flat_chunk(doc, GENERAL, raw_path))
            continue

        if category in TEXT_CATEGORIES:
            inline_heading = normalize_heading(content)
            if inline_heading:
                if active:
                    current_heading = inline_heading
                else:
                    append_flat(make_flat_chunk(doc, GENERAL, raw_path))
                continue

            if active:
                if current_heading is None and is_pre_heading_note(content):
                    active[-1].add_child(content, page, source, "표")
                    continue

                active_ids = [parent.diagram_id for parent in active]
                targeted, common = split_labeled_blocks(content, active_ids)
                if targeted:
                    for parent in active:
                        block = targeted.get(parent.diagram_id)
                        if block:
                            parent.add_child(block, page, source, current_heading)
                    continue
                if common:
                    for parent in active:
                        parent.add_child(common, page, source, current_heading)
                    continue

            chunk_type = PREFACE if is_preface_element(doc) else GENERAL
            if in_case_detail_mode and category == "heading1" and is_context_heading(content):
                set_context_heading(clean_heading_text(content))
            append_flat(make_flat_chunk(doc, chunk_type, raw_path))
            continue

        append_flat(make_flat_chunk(doc, GENERAL, raw_path))

    flush_active()

    output = materialize_chunks(items)
    return output, validate_output(output, warnings)


def validate_output(chunks: list[dict[str, Any]], warnings: list[str]) -> dict[str, Any]:
    by_id = {chunk["metadata"]["chunk_id"]: chunk for chunk in chunks}
    parent_ids = {
        chunk["metadata"]["chunk_id"]
        for chunk in chunks
        if chunk.get("metadata", {}).get("chunk_type") == PARENT
    }
    broken_parent_refs = []
    child_image_path_count = 0
    invalid_locations = []
    invalid_party_types = []

    for chunk in chunks:
        metadata = chunk.get("metadata", {})
        chunk_type = metadata.get("chunk_type")
        if chunk_type == CHILD:
            if metadata.get("parent_id") not in parent_ids:
                broken_parent_refs.append(metadata.get("chunk_id"))
            if metadata.get("image_path") is not None:
                child_image_path_count += 1
        location = metadata.get("location")
        party_type = metadata.get("party_type")
        if location is not None and location not in LOCATIONS:
            invalid_locations.append({"chunk_id": metadata.get("chunk_id"), "location": location})
        if party_type is not None and party_type not in PARTY_TYPES:
            invalid_party_types.append({"chunk_id": metadata.get("chunk_id"), "party_type": party_type})

    distribution: dict[str, int] = {}
    diagram_ids: set[str] = set()
    for chunk in chunks:
        metadata = chunk.get("metadata", {})
        chunk_type = str(metadata.get("chunk_type"))
        distribution[chunk_type] = distribution.get(chunk_type, 0) + 1
        diagram_id = metadata.get("diagram_id")
        if diagram_id:
            diagram_ids.add(str(diagram_id))

    parent_text_by_diagram = {
        chunk["metadata"].get("diagram_id"): chunk.get("page_content", "")
        for chunk in chunks
        if chunk.get("metadata", {}).get("chunk_type") == PARENT
    }
    known_checks = {
        "bo1_does_not_include_application_scope": "1. 적용 범위" not in parent_text_by_diagram.get("보1", ""),
        "geo2_1_exists": "거2-1" in parent_text_by_diagram,
        "geo2_2_exists": "거2-2" in parent_text_by_diagram,
        "geo2_1_not_under_geo2_2": "거2-1" not in parent_text_by_diagram.get("거2-2", ""),
        "geo5_1_exists": "거5-1" in parent_text_by_diagram,
        "geo5_2_exists": "거5-2" in parent_text_by_diagram,
        "cha43_7_ga_exists": "차43-7(가)" in parent_text_by_diagram,
        "cha43_7_na_exists": "차43-7(나)" in parent_text_by_diagram,
        "child_parent_refs_valid": not broken_parent_refs,
        "locations_valid": not invalid_locations,
        "party_types_valid": not invalid_party_types,
    }

    return {
        "total_chunks": len(chunks),
        "chunk_type_distribution": distribution,
        "diagram_id_count": len(diagram_ids),
        "parent_count": len(parent_ids),
        "child_count": distribution.get(CHILD, 0),
        "child_image_path_count": child_image_path_count,
        "broken_parent_refs": broken_parent_refs,
        "invalid_locations": invalid_locations[:50],
        "invalid_party_types": invalid_party_types[:50],
        "known_checks": known_checks,
        "warnings": warnings[:100],
        "chunk_id_unique": len(by_id) == len(chunks),
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Upstage final v2 chunks.")
    parser.add_argument("--input", type=Path, default=RAW_INPUT)
    parser.add_argument("--output", type=Path, default=FINAL_OUTPUT)
    parser.add_argument("--report", type=Path, default=REPORT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.input.open("r", encoding="utf-8") as fp:
        raw_docs = json.load(fp)
    if not isinstance(raw_docs, list):
        raise ValueError(f"Expected a JSON list: {args.input}")

    chunks, report = build_chunks(raw_docs, args.input)
    write_json(args.output, chunks)
    write_json(args.report, report)

    print(f"[INFO] input: {args.input}")
    print(f"[INFO] output: {args.output}")
    print(f"[INFO] report: {args.report}")
    print(f"[INFO] chunks: {report['total_chunks']}")
    print(f"[INFO] distribution: {report['chunk_type_distribution']}")
    print(f"[INFO] known_checks: {report['known_checks']}")


if __name__ == "__main__":
    main()
