"""Case-boundary chunker for accident-ratio LlamaParse markdown."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Iterable

from langchain_core.documents import Document

from rag.chunkers.base import BaseChunker
from rag.chunkers.schema import Chunk


CASE_ID_PATTERN = re.compile(r"[차보거]\d+(?:-\d+)?")
VARIANT_LABEL_PATTERN = re.compile(r"\(\s*([가나다라마바사])\s*\)")
IMAGE_LINK_PATTERN = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
CHILD_HEADING_PATTERN = re.compile(
    r"^#{2,6}\s+(?:\*\*|<u>|<mark>)?\s*("
    r"사고\s*상황|"
    r"기본\s*과실비율\s*해설|"
    r"수정요소[^\n]*해설|"
    r"활용시\s*참고\s*사항|"
    r"관련\s*법규|"
    r"참고\s*판례"
    r")",
    re.IGNORECASE,
)
CHILD_HEADING_TITLE_PATTERN = re.compile(
    r"^(?:\*\*|<u>|</u>|<mark>|</mark>|\s)*("
    r"사고\s*상황|"
    r"기본\s*과실비율\s*해설|"
    r"수정요소[^\n]*해설|"
    r"활용시\s*참고\s*사항|"
    r"관련\s*법규|"
    r"참고\s*판례"
    r")",
    re.IGNORECASE,
)
GENERAL_SUBSECTION_PATTERN = re.compile(
    r"^#{2,6}\s+(?:\*\*|<u>|<mark>|`|_)?\s*\(?\s*\d+\s*\)"
)
RUNNING_TEXT_PATTERN = re.compile(
    r"^(?:자동차사고 과실비율 인정기준 \| |제[123]장\. )"
)
CHAPTER_NAV_PATTERN = re.compile(
    r"^제[123]장\.\s+자동차와\s+.+사고(?:\s*\(세로 텍스트\))?$"
)
PREFACE_KEYWORDS = (
    "발간사",
    "목차",
    "이용 안내",
    "이용안내",
    "문서 안내",
    "문서안내",
    "변경대비표",
    "별첨",
    "개정경과",
    "개정 경과",
)


@dataclass
class _Block:
    kind: str
    text: str
    page: int = 0
    source: str = ""
    case_ids: tuple[str, ...] = ()
    image_path: str | None = None


@dataclass
class _ActiveParent:
    case_id: str
    parent_index: int
    text_parts: list[str]


@dataclass
class _ActiveCase:
    parents: list[_ActiveParent]
    open_table_parent_indices: list[int]
    section_blocks: list[_Block] = field(default_factory=list)


@dataclass(frozen=True)
class CaseBoundaryChunker(BaseChunker):
    mode: str = "B"

    def chunk(self, parsed_input: str | Document | Iterable[Document]) -> list[Chunk]:
        chunks: list[Chunk] = []
        blocks = self._collect_blocks(self._documents_for(parsed_input))

        active: _ActiveCase | None = None
        idle_buffer: list[_Block] = []
        pending_case_ids: list[str] = []
        pending_context_blocks: list[_Block] = []
        # True only while pending_context_blocks holds an unconsumed (= no table
        # has used it yet) category heading. Set to False after a table consumes
        # the context, so the residual ancestor breadcrumbs are not flushed back
        # into idle (they're already represented in the parent chunk).
        pending_context_unconsumed = False
        for block in blocks:
            if block.kind == "table":
                effective_ids = list(block.case_ids)
                if not effective_ids and pending_case_ids:
                    effective_ids = self._apply_pending_variants(
                        pending_case_ids[0], block.text
                    )
                pending_case_ids = []
                if effective_ids:
                    self._finalize_idle(idle_buffer, chunks)
                    idle_buffer = []
                    context_blocks_for_table = pending_context_blocks
                    table_text = self._text_with_pending_context(
                        context_blocks_for_table, block.text
                    )
                    pending_context_blocks = self._ancestor_context_blocks(
                        context_blocks_for_table
                    )
                    pending_context_unconsumed = False
                    case_block = _Block(
                        kind=block.kind,
                        text=table_text,
                        page=block.page,
                        source=block.source,
                        case_ids=tuple(effective_ids),
                        image_path=block.image_path,
                    )
                    if active is not None and not self._has_body_section(active):
                        self._append_case_table(active, case_block, chunks)
                    else:
                        active = self._finalize_case(active, chunks)
                        active = self._start_case(case_block, chunks)
                    continue

            if active is not None:
                if block.kind == "image" and self._only_image_for_active_table(active, block):
                    self._attach_table_image(active, block, chunks)
                    continue
                if block.kind == "heading" and self._is_child_heading(block.text):
                    self._finalize_section(active, chunks)
                    active.section_blocks.append(block)
                    continue
                if block.kind == "heading" and self._is_case_context_heading(block.text):
                    if self._has_body_section(active):
                        active = self._finalize_case(active, chunks)
                    else:
                        self._attach_open_table_tail(active)
                    pending_context_blocks = self._context_blocks_after_heading(
                        pending_context_blocks, block
                    )
                    pending_case_ids = self._pending_ids_for_context_heading(block.text)
                    pending_context_unconsumed = True
                    continue
                active.section_blocks.append(block)
                continue

            if block.kind == "heading" and self._is_case_context_heading(block.text):
                # Keep case-title/group headings as breadcrumb context for the next table.
                pending_context_blocks = self._context_blocks_after_heading(
                    pending_context_blocks, block
                )
                pending_case_ids = self._pending_ids_for_context_heading(block.text)
                pending_context_unconsumed = True
                continue
            if block.kind == "heading" and self._is_top_level_heading(block.text):
                # Unconsumed category context never reached a table — recover its
                # content back into idle so it isn't silently dropped.
                if pending_context_unconsumed and pending_context_blocks:
                    idle_buffer.extend(pending_context_blocks)
                    pending_context_blocks = []
                    pending_case_ids = []
                    pending_context_unconsumed = False
                self._finalize_idle(idle_buffer, chunks)
                idle_buffer = [block]
                continue
            idle_buffer.append(block)

        if pending_context_unconsumed and pending_context_blocks:
            idle_buffer.extend(pending_context_blocks)
            pending_context_blocks = []
            pending_case_ids = []
            pending_context_unconsumed = False
        self._finalize_idle(idle_buffer, chunks)
        self._finalize_case(active, chunks)
        return chunks

    def _text_with_pending_context(
        self, pending_context_blocks: list[_Block], text: str
    ) -> str:
        context_text = "\n\n".join(
            block.text for block in pending_context_blocks if block.text
        ).strip()
        if not context_text:
            return text
        return f"{context_text}\n\n{text}".strip()

    def _extract_case_ids(self, text: str) -> list[str]:
        ids: list[str] = []
        for match in CASE_ID_PATTERN.finditer(text):
            case_id = match.group(0)
            if case_id not in ids:
                ids.append(case_id)
        return ids

    def _apply_pending_variants(self, base_id: str, table_text: str) -> list[str]:
        variants: list[str] = []
        for match in VARIANT_LABEL_PATTERN.finditer(table_text):
            label = match.group(1)
            if label not in variants:
                variants.append(label)
        if len(variants) >= 2:
            return [f"{base_id}({label})" for label in variants]
        return [base_id]

    def _is_top_level_heading(self, heading_text: str) -> bool:
        first_line = heading_text.splitlines()[0].strip() if heading_text else ""
        match = HEADING_PATTERN.match(first_line)
        return match is not None and len(match.group(1)) == 1

    def _is_case_title_heading(self, heading_text: str) -> bool:
        first_line = heading_text.splitlines()[0].strip() if heading_text else ""
        match = HEADING_PATTERN.match(first_line)
        if match is None:
            return False
        return CASE_ID_PATTERN.search(match.group(2)) is not None

    def _is_case_context_heading(self, heading_text: str) -> bool:
        return self._is_case_title_heading(heading_text) or self._is_case_category_heading(
            heading_text
        )

    def _pending_ids_for_context_heading(self, heading_text: str) -> list[str]:
        if self._is_case_group_heading(heading_text):
            return []
        return self._extract_case_ids(heading_text)

    def _is_case_group_heading(self, heading_text: str) -> bool:
        first_line = heading_text.splitlines()[0].strip() if heading_text else ""
        match = HEADING_PATTERN.match(first_line)
        if match is None:
            return False
        title = match.group(2)
        return "[" in title and "]" in title

    def _is_case_category_heading(self, heading_text: str) -> bool:
        first_line = heading_text.splitlines()[0].strip() if heading_text else ""
        match = HEADING_PATTERN.match(first_line)
        if match is None:
            return False
        level = len(match.group(1))
        title = match.group(2).strip()
        # Strip leading markdown emphasis so `**(1)…**` is treated like `(1)…`
        title = re.sub(r"^(?:\*\*|<u>|<mark>|`|_)\s*", "", title)
        return level <= 2 and re.match(r"^\(\s*\d+\s*\)", title) is not None

    def _context_blocks_after_heading(
        self, current_blocks: list[_Block], heading_block: _Block
    ) -> list[_Block]:
        first_line = heading_block.text.splitlines()[0].strip()
        match = HEADING_PATTERN.match(first_line)
        if match is None:
            return [heading_block]
        level = len(match.group(1))
        kept_blocks: list[_Block] = []
        for block in current_blocks:
            block_first_line = block.text.splitlines()[0].strip()
            block_match = HEADING_PATTERN.match(block_first_line)
            if block_match is not None and len(block_match.group(1)) < level:
                kept_blocks.append(block)
        kept_blocks.append(heading_block)
        return kept_blocks

    def _ancestor_context_blocks(self, current_blocks: list[_Block]) -> list[_Block]:
        return [
            block
            for block in current_blocks
            if self._is_case_group_heading(block.text)
            or self._is_case_category_heading(block.text)
        ]

    def _finalize_idle(self, buffer: list[_Block], chunks: list[Chunk]) -> None:
        if not buffer:
            return
        page = buffer[0].page
        source = buffer[0].source

        main_heading_text = ""
        for block in buffer:
            if block.kind == "heading":
                main_heading_text = block.text.splitlines()[0].strip()
                break

        if self._is_preface_heading(main_heading_text):
            preface_text = "\n\n".join(block.text for block in buffer if block.text).strip()
            if preface_text:
                chunks.append(
                    Chunk(
                        chunk_id=len(chunks),
                        text=preface_text,
                        chunk_type="preface",
                        page=page,
                        source=source,
                        diagram_id=None,
                        parent_id=None,
                        location=None,
                        party_type=None,
                        image_path=None,
                    )
                )
            return

        breadcrumb = main_heading_text
        sub_groups: list[list[_Block]] = []
        current: list[_Block] = []
        seen_subsection = False
        for block in buffer:
            is_sub = (
                block.kind == "heading"
                and GENERAL_SUBSECTION_PATTERN.match(
                    block.text.splitlines()[0].strip()
                )
                is not None
            )
            if is_sub:
                if current:
                    sub_groups.append(current)
                current = [block]
                seen_subsection = True
            else:
                current.append(block)
        if current:
            sub_groups.append(current)

        if not seen_subsection:
            general_text = "\n\n".join(block.text for block in buffer if block.text).strip()
            if general_text:
                chunks.append(
                    Chunk(
                        chunk_id=len(chunks),
                        text=general_text,
                        chunk_type="general",
                        page=page,
                        source=source,
                        diagram_id=None,
                        parent_id=None,
                        location=None,
                        party_type=None,
                        image_path=None,
                    )
                )
            return

        for group in sub_groups:
            if not any(
                block.kind == "heading"
                and GENERAL_SUBSECTION_PATTERN.match(block.text.splitlines()[0].strip())
                for block in group
            ):
                # Pre-subsection group: usually just the main heading line, but
                # _split_blocks may glue trailing prose under an H1 into the same
                # heading block. Emit as a standalone general chunk if it carries
                # substantive prose beyond the heading line itself.
                pre_text = "\n\n".join(block.text for block in group if block.text).strip()
                if not pre_text:
                    continue
                pre_body = "\n".join(pre_text.splitlines()[1:]).strip()
                if not pre_body:
                    continue
                pre_page = next((block.page for block in group if block.text), page)
                pre_source = next((block.source for block in group if block.text), source)
                chunks.append(
                    Chunk(
                        chunk_id=len(chunks),
                        text=pre_text,
                        chunk_type="general",
                        page=pre_page,
                        source=pre_source,
                        diagram_id=None,
                        parent_id=None,
                        location=None,
                        party_type=None,
                        image_path=None,
                    )
                )
                continue
            group_text = "\n\n".join(block.text for block in group if block.text).strip()
            if not group_text:
                continue
            if breadcrumb and not group_text.startswith(breadcrumb):
                group_text = f"{breadcrumb}\n\n{group_text}"
            for split_text in self._split_general_detail_examples(group_text):
                group_page = next((block.page for block in group if block.text), page)
                group_source = next((block.source for block in group if block.text), source)
                chunks.append(
                    Chunk(
                        chunk_id=len(chunks),
                        text=split_text,
                        chunk_type="general",
                        page=group_page,
                        source=group_source,
                        diagram_id=None,
                        parent_id=None,
                        location=None,
                        party_type=None,
                        image_path=None,
                    )
                )

    def _is_preface_heading(self, heading_text: str) -> bool:
        if not heading_text:
            return False
        normalized = heading_text.replace(" ", "")
        for keyword in PREFACE_KEYWORDS:
            if keyword.replace(" ", "") in normalized:
                return True
        return False

    def _start_case(self, table_block: _Block, chunks: list[Chunk]) -> _ActiveCase:
        parents = self._append_case_chunks(table_block, chunks)
        return _ActiveCase(
            parents=parents,
            open_table_parent_indices=[parent.parent_index for parent in parents],
        )

    def _append_case_table(
        self, active: _ActiveCase, table_block: _Block, chunks: list[Chunk]
    ) -> None:
        self._attach_open_table_tail(active)
        parents = self._append_case_chunks(table_block, chunks)
        active.parents.extend(parents)
        active.open_table_parent_indices = [parent.parent_index for parent in parents]

    def _append_case_chunks(
        self, table_block: _Block, chunks: list[Chunk]
    ) -> list[_ActiveParent]:
        parents: list[_ActiveParent] = []
        case_ids = list(table_block.case_ids)
        for case_id in case_ids:
            parent_id = len(chunks)
            chunks.append(
                Chunk(
                    chunk_id=parent_id,
                    text="",
                    chunk_type="parent",
                    page=table_block.page,
                    source=table_block.source,
                    diagram_id=case_id,
                    parent_id=None,
                    location=None,
                    party_type=None,
                    image_path=None,
                )
            )
            chunks.append(
                Chunk(
                    chunk_id=len(chunks),
                    text=table_block.text,
                    chunk_type="child",
                    page=table_block.page,
                    source=table_block.source,
                    diagram_id=case_id,
                    parent_id=parent_id,
                    location=None,
                    party_type=None,
                    image_path=None,
                )
            )
            parents.append(
                _ActiveParent(
                    case_id=case_id,
                    parent_index=parent_id,
                    text_parts=[table_block.text],
                )
            )
        return parents

    def _attach_table_image(self, active: _ActiveCase, image_block: _Block, chunks: list[Chunk]) -> None:
        for parent_index in active.open_table_parent_indices:
            table_child_index = parent_index + 1
            child_chunk = chunks[table_child_index]
            chunks[table_child_index] = Chunk(
                chunk_id=child_chunk.chunk_id,
                text=f"{child_chunk.text}\n{image_block.text}".strip(),
                chunk_type=child_chunk.chunk_type,
                page=child_chunk.page,
                source=child_chunk.source,
                diagram_id=child_chunk.diagram_id,
                parent_id=child_chunk.parent_id,
                location=child_chunk.location,
                party_type=child_chunk.party_type,
                image_path=image_block.image_path,
            )
            parent_chunk = chunks[parent_index]
            chunks[parent_index] = Chunk(
                chunk_id=parent_chunk.chunk_id,
                text=parent_chunk.text,
                chunk_type=parent_chunk.chunk_type,
                page=parent_chunk.page,
                source=parent_chunk.source,
                diagram_id=parent_chunk.diagram_id,
                parent_id=parent_chunk.parent_id,
                location=parent_chunk.location,
                party_type=parent_chunk.party_type,
                image_path=image_block.image_path,
            )
            active_parent = self._parent_for_index(active, parent_index)
            if active_parent is not None and active_parent.text_parts:
                active_parent.text_parts[0] = (
                    f"{active_parent.text_parts[0]}\n{image_block.text}".strip()
                )

    def _only_image_for_active_table(self, active: _ActiveCase, image_block: _Block) -> bool:
        # Image right after the table belongs to the table/image child if no section started yet.
        return (
            not self._has_body_section(active)
            and bool(active.open_table_parent_indices)
            and image_block.image_path is not None
        )

    def _finalize_section(self, active: _ActiveCase, chunks: list[Chunk]) -> None:
        if not active.section_blocks:
            return
        self._attach_open_table_tail(active)
        section_text = "\n\n".join(
            block.text for block in active.section_blocks if block.text
        ).strip()
        if not section_text:
            active.section_blocks = []
            return
        first_block = active.section_blocks[0]
        section_image = next(
            (block.image_path for block in active.section_blocks if block.kind == "image" and block.image_path),
            None,
        )
        for parent in active.parents:
            parent_section_text = self._section_text_for_case(section_text, parent.case_id)
            if not parent_section_text:
                continue
            chunks.append(
                Chunk(
                    chunk_id=len(chunks),
                    text=parent_section_text,
                    chunk_type="child",
                    page=first_block.page,
                    source=first_block.source,
                    diagram_id=parent.case_id,
                    parent_id=parent.parent_index,
                    location=None,
                    party_type=None,
                    image_path=section_image,
                )
            )
            parent.text_parts.append(parent_section_text)
        active.section_blocks = []

    def _finalize_case(
        self, active: _ActiveCase | None, chunks: list[Chunk]
    ) -> _ActiveCase | None:
        if active is None:
            return None
        self._finalize_section(active, chunks)
        for parent in active.parents:
            parent_text = "\n\n".join(part for part in parent.text_parts if part).strip()
            parent_chunk = chunks[parent.parent_index]
            chunks[parent.parent_index] = Chunk(
                chunk_id=parent_chunk.chunk_id,
                text=parent_text,
                chunk_type=parent_chunk.chunk_type,
                page=parent_chunk.page,
                source=parent_chunk.source,
                diagram_id=parent_chunk.diagram_id,
                parent_id=parent_chunk.parent_id,
                location=parent_chunk.location,
                party_type=parent_chunk.party_type,
                image_path=parent_chunk.image_path,
            )
        return None

    def _parent_for_index(
        self, active: _ActiveCase, parent_index: int
    ) -> _ActiveParent | None:
        for parent in active.parents:
            if parent.parent_index == parent_index:
                return parent
        return None

    def _has_body_section(self, active: _ActiveCase) -> bool:
        return any(
            block.kind == "heading" and self._is_child_heading(block.text)
            for block in active.section_blocks
        )

    def _attach_open_table_tail(self, active: _ActiveCase) -> None:
        if not active.open_table_parent_indices or self._has_body_section(active):
            return
        tail_text = "\n\n".join(
            block.text for block in active.section_blocks if block.text
        ).strip()
        if not tail_text:
            return
        for parent_index in active.open_table_parent_indices:
            parent = self._parent_for_index(active, parent_index)
            if parent is not None:
                parent.text_parts.append(tail_text)
        active.section_blocks = []

    def _section_text_for_case(self, section_text: str, case_id: str) -> str:
        case_labeled_text = self._case_labeled_section_text(section_text, case_id)
        if case_labeled_text is not None:
            return case_labeled_text

        variant_match = re.search(r"\(([가나다라마바사])\)$", case_id)
        if variant_match is None:
            return section_text
        label = variant_match.group(1)
        lines = section_text.splitlines()
        selected: list[str] = []
        saw_variant_line = False
        for line in lines:
            line_label = self._variant_label_for_line(line)
            if line_label is None:
                selected.append(line)
                continue
            saw_variant_line = True
            if line_label == label:
                selected.append(line)
        if not saw_variant_line:
            return section_text
        return "\n".join(selected).strip()

    def _case_labeled_section_text(self, section_text: str, case_id: str) -> str | None:
        base_case_id = self._base_case_id(case_id)
        lines = section_text.splitlines()
        selected: list[str] = []
        saw_case_labeled_line = False
        for line in lines:
            line_case_id = self._case_id_label_for_line(line)
            if line_case_id is None:
                selected.append(line)
                continue
            saw_case_labeled_line = True
            if line_case_id == base_case_id:
                selected.append(line)
        if not saw_case_labeled_line:
            return None
        return "\n".join(selected).strip()

    def _base_case_id(self, case_id: str) -> str:
        return re.sub(r"\([가나다라마바사]\)$", "", case_id)

    def _case_id_label_for_line(self, line: str) -> str | None:
        stripped = self._strip_list_and_symbol_prefix(line)
        match = CASE_ID_PATTERN.match(stripped)
        if match is None:
            return None
        return match.group(0)

    def _variant_label_for_line(self, line: str) -> str | None:
        stripped = self._strip_list_and_symbol_prefix(line)
        match = VARIANT_LABEL_PATTERN.match(stripped)
        if match is None:
            return None
        return match.group(1)

    def _strip_list_and_symbol_prefix(self, line: str) -> str:
        stripped = line.strip()
        stripped = re.sub(r"^(?:[*\-ㆍ•◉⊙]\s*)+", "", stripped).strip()
        stripped = re.sub(r"^\*\*([^*]+)\*\*", r"\1", stripped).strip()
        return stripped

    def _split_general_detail_examples(self, text: str) -> list[str]:
        if "세부적용 예" not in text:
            return [text]
        lines = text.splitlines()
        item_starts = [
            index
            for index, line in enumerate(lines)
            if re.match(r"^\s*(?:\*\*)?[①②③④⑤⑥⑦⑧⑨⑩]", line)
        ]
        if len(item_starts) < 2:
            return [text]
        prefix = lines[: item_starts[0]]
        chunks: list[str] = []
        for position, start in enumerate(item_starts):
            end = item_starts[position + 1] if position + 1 < len(item_starts) else len(lines)
            item_lines = lines[start:end]
            chunks.append("\n".join(prefix + item_lines).strip())
        return chunks

    def _is_child_heading(self, heading_text: str) -> bool:
        first_line = heading_text.splitlines()[0].strip() if heading_text else ""
        return CHILD_HEADING_PATTERN.match(first_line) is not None

    def _collect_blocks(self, documents: list[Document]) -> list[_Block]:
        blocks: list[_Block] = []
        for document in documents:
            page = self._page_for(document)
            source = str(document.metadata.get("source", ""))
            for block in self._split_blocks(document.page_content):
                block.page = page
                block.source = source
                blocks.append(block)
        return blocks

    def _split_blocks(self, markdown: str) -> list[_Block]:
        lines = markdown.splitlines()
        blocks: list[_Block] = []
        index = 0
        while index < len(lines):
            line = lines[index]
            stripped = line.strip()
            if not stripped:
                index += 1
                continue
            if self._is_horizontal_rule(stripped):
                index += 1
                continue
            if self._is_running_line(stripped):
                index += 1
                continue

            if stripped.startswith("|") and stripped.endswith("|"):
                end = index
                while end < len(lines) and lines[end].strip().startswith("|"):
                    end += 1
                table_text = "\n".join(lines[index:end]).strip()
                if not self._is_running_table(table_text):
                    case_ids = self._case_ids_in_table(table_text)
                    blocks.append(_Block(kind="table", text=table_text, case_ids=tuple(case_ids)))
                index = end
                continue

            image_match = IMAGE_LINK_PATTERN.match(stripped)
            if image_match is not None:
                blocks.append(
                    _Block(kind="image", text=stripped, image_path=image_match.group(1))
                )
                index += 1
                continue

            heading_match = HEADING_PATTERN.match(stripped)
            if heading_match is not None:
                end = index + 1
                while end < len(lines):
                    next_stripped = lines[end].strip()
                    if HEADING_PATTERN.match(next_stripped):
                        break
                    if next_stripped.startswith("|"):
                        break
                    if IMAGE_LINK_PATTERN.match(next_stripped):
                        break
                    end += 1
                heading_text = self._clean_running_text("\n".join(lines[index:end]))
                if heading_text:
                    blocks.append(_Block(kind="heading", text=heading_text))
                index = end
                continue

            decorated_child_heading = self._decorated_child_heading(stripped)
            if decorated_child_heading is not None:
                blocks.append(_Block(kind="heading", text=f"### {decorated_child_heading}"))
                index += 1
                continue

            end = index + 1
            while end < len(lines):
                next_stripped = lines[end].strip()
                if not next_stripped:
                    break
                if self._is_horizontal_rule(next_stripped):
                    break
                if HEADING_PATTERN.match(next_stripped):
                    break
                if self._decorated_child_heading(next_stripped) is not None:
                    break
                if next_stripped.startswith("|"):
                    break
                if IMAGE_LINK_PATTERN.match(next_stripped):
                    break
                end += 1
            paragraph_text = self._clean_running_text("\n".join(lines[index:end]))
            if paragraph_text and not self._is_running_text(paragraph_text):
                blocks.append(_Block(kind="paragraph", text=paragraph_text))
            index = end

        return blocks

    def _decorated_child_heading(self, text: str) -> str | None:
        normalized = re.sub(r"</?u>", "", text).strip()
        normalized = normalized.strip("*").strip()
        match = CHILD_HEADING_TITLE_PATTERN.match(normalized)
        if match is None:
            return None
        return re.sub(r"\s+", " ", match.group(1)).strip()

    def _is_running_text(self, text: str) -> bool:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return bool(lines) and all(self._is_running_line(line) for line in lines)

    def _clean_running_text(self, text: str) -> str:
        lines = [
            line
            for line in text.splitlines()
            if not self._is_running_line(line.strip())
        ]
        return "\n".join(lines).strip()

    def _is_running_line(self, line: str) -> bool:
        if not line:
            return False
        if line.startswith("#"):
            return False
        normalized = self._normalize_running_line(line)
        if normalized == "목차":
            return True
        if normalized.startswith("자동차사고 과실비율 인정기준 |"):
            return True
        return CHAPTER_NAV_PATTERN.match(normalized) is not None

    def _is_horizontal_rule(self, line: str) -> bool:
        return re.fullmatch(r"-{3,}", line) is not None

    def _normalize_running_line(self, line: str) -> str:
        normalized = line.replace("\\*", "*")
        normalized = re.sub(r"</?[^>]+>", "", normalized)
        normalized = normalized.strip()
        normalized = normalized.strip("|").strip()
        normalized = normalized.strip("*_` ").strip()
        normalized = re.sub(r"\s+", " ", normalized)
        return normalized

    def _is_running_table(self, table_text: str) -> bool:
        meaningful_cells: list[str] = []
        for row in table_text.splitlines():
            cells = [cell.strip() for cell in row.strip().strip("|").split("|")]
            for cell in cells:
                if not cell or re.fullmatch(r"[-:\s]+", cell):
                    continue
                meaningful_cells.append(cell)
        return bool(meaningful_cells) and all(
            self._is_running_line(cell) for cell in meaningful_cells
        )

    def _case_ids_in_table(self, table_text: str) -> list[str]:
        base_ids: list[str] = []
        for row in table_text.splitlines():
            leading_cell = self._leading_table_cell(row)
            if leading_cell is None:
                continue
            for match in CASE_ID_PATTERN.finditer(leading_cell):
                case_id = match.group(0)
                if case_id not in base_ids:
                    base_ids.append(case_id)

        if len(base_ids) == 1:
            variants: list[str] = []
            for match in VARIANT_LABEL_PATTERN.finditer(table_text):
                label = match.group(1)
                if label not in variants:
                    variants.append(label)
            if len(variants) >= 2:
                return [f"{base_ids[0]}({label})" for label in variants]
        return base_ids

    def _leading_table_cell(self, row: str) -> str | None:
        stripped = row.strip()
        if not stripped.startswith("|"):
            return None
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if cells and all(re.fullmatch(r"[-:\s]+", cell) for cell in cells):
            return None
        for cell in cells:
            if not cell:
                continue
            if re.fullmatch(r"[-:\s]+", cell):
                continue
            return cell
        return None

    def _documents_for(self, parsed_input: str | Document | Iterable[Document]) -> list[Document]:
        if isinstance(parsed_input, str):
            return [Document(page_content=parsed_input, metadata={})]
        if isinstance(parsed_input, Document):
            return [parsed_input]
        return list(parsed_input)

    def _page_for(self, document: Document) -> int:
        page = document.metadata.get("page", 0)
        if isinstance(page, int):
            return page
        if isinstance(page, str) and page.isdigit():
            return int(page)
        return 0
