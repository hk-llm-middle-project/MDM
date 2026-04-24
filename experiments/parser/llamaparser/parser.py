from __future__ import annotations

import json
from pathlib import Path

from dotenv import load_dotenv
from llama_parse import LlamaParse


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PDF_PATH = PROJECT_ROOT / "data/raw/230630_자동차사고 과실비율 인정기준_최종.pdf"
EXAMPLE_PAGES_PATH = PROJECT_ROOT / "experiments/parser/example_pages.json"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

PAGE_NUMBER_BASE_ZERO_BASED = "zero-based"
PAGE_NUMBER_BASE_ONE_BASED = "one-based"


def load_example_config() -> tuple[dict[str, object], dict[str, list[int]]]:
    with EXAMPLE_PAGES_PATH.open(encoding="utf-8") as file:
        raw_config = json.load(file)

    if not isinstance(raw_config, dict):
        raise ValueError("example_pages.json must contain an object.")

    metadata = raw_config.get("metadata")
    pages = raw_config.get("pages")
    if not isinstance(metadata, dict):
        raise ValueError("example_pages.json must contain a metadata object.")
    if not isinstance(pages, dict):
        raise ValueError("example_pages.json must contain a pages object.")

    pages_by_category: dict[str, list[int]] = {}
    for category, category_pages in pages.items():
        if not isinstance(category, str):
            raise ValueError("example_pages.json category names must be strings.")
        if not isinstance(category_pages, list) or not all(
            isinstance(page, int) for page in category_pages
        ):
            raise ValueError(f"example_pages.json '{category}' must be a list of ints.")
        pages_by_category[category] = category_pages

    return metadata, pages_by_category


def flatten_pages(pages_by_category: dict[str, list[int]]) -> list[int]:
    return sorted({page for pages in pages_by_category.values() for page in pages})


def build_page_categories(pages_by_category: dict[str, list[int]]) -> dict[int, list[str]]:
    page_categories: dict[int, list[str]] = {}
    for category, pages in pages_by_category.items():
        for page in pages:
            page_categories.setdefault(page, []).append(category)
    return page_categories


def to_llamaparse_target_pages(target_pages: list[int], page_number_base: str) -> list[int]:
    if page_number_base == PAGE_NUMBER_BASE_ZERO_BASED:
        return target_pages
    if page_number_base == PAGE_NUMBER_BASE_ONE_BASED:
        return [page - 1 for page in target_pages]
    raise ValueError(
        "metadata.page_number_base must be either "
        f"'{PAGE_NUMBER_BASE_ZERO_BASED}' or '{PAGE_NUMBER_BASE_ONE_BASED}'."
    )


def from_llamaparse_page_number(page_number: int, page_number_base: str) -> int:
    if page_number_base == PAGE_NUMBER_BASE_ZERO_BASED:
        return page_number - 1
    if page_number_base == PAGE_NUMBER_BASE_ONE_BASED:
        return page_number
    raise ValueError(
        "metadata.page_number_base must be either "
        f"'{PAGE_NUMBER_BASE_ZERO_BASED}' or '{PAGE_NUMBER_BASE_ONE_BASED}'."
    )


def make_parser(target_pages: list[int], page_number_base: str) -> LlamaParse:
    llamaparse_target_pages = to_llamaparse_target_pages(target_pages, page_number_base)
    return LlamaParse(
        result_type="markdown",
        target_pages=",".join(str(page) for page in llamaparse_target_pages),
        split_by_page=True,
        language="ko",
    )


def save_page_result(
    page_number: int,
    page_number_base: str,
    categories: list[str],
    markdown: str,
) -> Path:
    category_name = "-".join(categories)
    output_path = RESULTS_DIR / f"{page_number:03d}_{category_name}.md"
    content = "\n".join(
        [
            f"# Page {page_number}",
            "",
            f"- source: {PDF_PATH}",
            f"- categories: {', '.join(categories)}",
            f"- page_number_base: {page_number_base}",
            "",
            markdown.strip(),
            "",
        ]
    )
    output_path.write_text(content, encoding="utf-8")
    return output_path


def clear_previous_results() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    for result_path in RESULTS_DIR.glob("*.md"):
        result_path.unlink()


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")

    metadata, pages_by_category = load_example_config()
    page_number_base = str(metadata.get("page_number_base", PAGE_NUMBER_BASE_ONE_BASED))
    target_pages = flatten_pages(pages_by_category)
    page_categories = build_page_categories(pages_by_category)

    clear_previous_results()

    parser = make_parser(target_pages, page_number_base)
    job_result = parser.parse(str(PDF_PATH))

    saved_paths: list[Path] = []
    for page in job_result.pages:
        page_number = from_llamaparse_page_number(page.page, page_number_base)
        categories = page_categories.get(page_number, ["uncategorized"])
        markdown = page.md or page.text or ""
        saved_paths.append(
            save_page_result(page_number, page_number_base, categories, markdown)
        )

    print(f"Parsed pages: {target_pages}")
    print(f"Saved {len(saved_paths)} files to: {RESULTS_DIR}")
    for path in saved_paths:
        print(f"- {path}")


if __name__ == "__main__":
    main()
