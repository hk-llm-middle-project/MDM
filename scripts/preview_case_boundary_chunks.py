"""Preview CaseBoundaryChunker output for selected LlamaParse markdown pages."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from io import StringIO
from pathlib import Path

from langchain_core.documents import Document

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag.chunkers import CaseBoundaryChunker


DEFAULT_MARKDOWN_DIR = PROJECT_ROOT / "data" / "llama_md" / "main_pdf"
DEFAULT_PAGES = [85, 389, 24]
RESULTS_DIR = PROJECT_ROOT / "scripts" / "results"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preview case-boundary chunks from page-level markdown files."
    )
    parser.add_argument(
        "--pages",
        nargs="+",
        type=int,
        default=DEFAULT_PAGES,
        help="Page numbers to chunk. Default: 85 389 24",
    )
    parser.add_argument(
        "--page-range",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        help="Inclusive page range to chunk, e.g. --page-range 85 90.",
    )
    parser.add_argument(
        "--markdown-dir",
        type=Path,
        default=DEFAULT_MARKDOWN_DIR,
        help="Directory containing page markdown files named like 085.md.",
    )
    parser.add_argument(
        "--diagram-id",
        help="Only print chunks for this diagram id, e.g. 보20 or 차43-7(가).",
    )
    parser.add_argument(
        "--chunk-type",
        choices=["preface", "general", "parent", "child"],
        help="Only print chunks with this chunk_type.",
    )
    parser.add_argument(
        "--show-text",
        action="store_true",
        help="Deprecated. Markdown output always includes full chunk text.",
    )
    parser.add_argument(
        "--combined",
        action="store_true",
        help="Chunk selected pages as one continuous document sequence.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print chunks as JSON for easier inspection or piping.",
    )
    return parser.parse_args()


def load_documents(markdown_dir: Path, pages: list[int]) -> list[Document]:
    documents: list[Document] = []
    for page in pages:
        path = markdown_dir / f"{page:03}.md"
        if not path.exists():
            raise FileNotFoundError(f"Markdown page not found: {path}")
        documents.append(
            Document(
                page_content=path.read_text(encoding="utf-8"),
                metadata={"page": page, "source": str(path.relative_to(PROJECT_ROOT))},
            )
        )
    return documents


def compact_text(text: str, limit: int = 180) -> str:
    preview = " ".join(text.split())
    if len(preview) <= limit:
        return preview
    return f"{preview[: limit - 3]}..."


def selected_pages(args: argparse.Namespace) -> list[int]:
    if args.page_range is None:
        return args.pages
    start, end = args.page_range
    if start > end:
        raise ValueError("--page-range START must be less than or equal to END")
    return list(range(start, end + 1))


def chunk_to_dict(chunk) -> dict:
    return {
        "chunk_id": chunk.chunk_id,
        "chunk_type": chunk.chunk_type,
        "diagram_id": chunk.diagram_id,
        "parent_id": chunk.parent_id,
        "page": chunk.page,
        "source": chunk.source,
        "image_path": chunk.image_path,
        "text": chunk.text,
    }


def filtered_chunks(chunks, diagram_id: str | None, chunk_type: str | None):
    for chunk in chunks:
        if diagram_id is not None and chunk.diagram_id != diagram_id:
            continue
        if chunk_type is not None and chunk.chunk_type != chunk_type:
            continue
        yield chunk


def write_summary(buffer: StringIO, chunks) -> None:
    type_counts = Counter(chunk.chunk_type for chunk in chunks)
    diagram_counts = Counter(
        chunk.diagram_id for chunk in chunks if chunk.diagram_id is not None
    )
    buffer.write("# Case Boundary Chunk Preview\n\n")
    buffer.write("## Summary\n\n")
    buffer.write(f"- total chunks: {len(chunks)}\n")
    buffer.write(f"- chunk_type counts: `{dict(type_counts)}`\n")
    buffer.write(f"- diagram_id counts: `{dict(diagram_counts)}`\n")
    buffer.write("\n")


def render_human_readable(chunks) -> str:
    buffer = StringIO()
    write_summary(buffer, chunks)
    for chunk in chunks:
        buffer.write(f"## Chunk {chunk.chunk_id}\n\n")
        buffer.write("| field | value |\n")
        buffer.write("| --- | --- |\n")
        buffer.write(f"| chunk_type | `{chunk.chunk_type}` |\n")
        buffer.write(f"| diagram_id | `{chunk.diagram_id}` |\n")
        buffer.write(f"| parent_id | `{chunk.parent_id}` |\n")
        buffer.write(f"| page | `{chunk.page}` |\n")
        buffer.write(f"| image_path | `{chunk.image_path}` |\n\n")
        buffer.write("```markdown\n")
        buffer.write(f"{chunk.text}\n")
        buffer.write("```\n\n")
    return buffer.getvalue()


def chunk_documents(documents: list[Document], combined: bool):
    chunker = CaseBoundaryChunker(mode="B")
    if combined:
        return chunker.chunk(documents)

    chunks = []
    for document in documents:
        page_chunks = chunker.chunk(document)
        id_offset = len(chunks)
        for chunk in page_chunks:
            original_chunk_id = chunk.chunk_id
            chunk.chunk_id = id_offset + original_chunk_id
            if chunk.parent_id is not None:
                chunk.parent_id += id_offset
            chunks.append(chunk)
    return chunks


def output_path_for(args: argparse.Namespace, pages: list[int]) -> Path:
    if args.page_range is not None:
        page_part = f"{pages[0]:03}-{pages[-1]:03}"
    else:
        page_part = "_".join(f"{page:03}" for page in pages)
    parts = ["case_boundary_chunks", f"pages_{page_part}"]
    if args.combined:
        parts.append("combined")
    if args.diagram_id:
        parts.append(args.diagram_id.replace("/", "_"))
    if args.chunk_type:
        parts.append(args.chunk_type)
    suffix = ".json" if args.json else ".md"
    return RESULTS_DIR / ("__".join(parts) + suffix)


def main() -> None:
    args = parse_args()
    pages = selected_pages(args)
    documents = load_documents(args.markdown_dir, pages)
    chunks = chunk_documents(documents, args.combined)
    chunks = list(filtered_chunks(chunks, args.diagram_id, args.chunk_type))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = output_path_for(args, pages)

    if args.json:
        output_text = json.dumps(
            [chunk_to_dict(chunk) for chunk in chunks],
            ensure_ascii=False,
            indent=2,
        )
    else:
        output_text = render_human_readable(chunks)

    output_path.write_text(output_text, encoding="utf-8")
    print(f"[INFO] saved: {output_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
