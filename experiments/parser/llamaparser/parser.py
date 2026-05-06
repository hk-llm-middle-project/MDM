from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from llama_parse import LlamaParse


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PDF_PATH = PROJECT_ROOT / "data/raw/230630_자동차사고 과실비율 인정기준_최종.pdf"
EXAMPLE_PAGES_PATH = PROJECT_ROOT / "experiments/parser/example_pages.json"
RESULTS_PARENT_DIR = Path(__file__).resolve().parent
RESULTS_DIR_PREFIX = "results_"

PAGE_NUMBER_BASE_ZERO_BASED = "zero-based"
PAGE_NUMBER_BASE_ONE_BASED = "one-based"

OUTPUT_FORMAT_MARKDOWN = "md"
OUTPUT_FORMAT_TEXT = "txt"

SAVE_MODE_PAGE = "page"
SAVE_MODE_COMBINED = "combined"
SAVE_MODE_BOTH = "both"

IMAGE_CATEGORY = "image"


@dataclass(frozen=True)
class ParserExperimentConfig:
    output_format: str = OUTPUT_FORMAT_MARKDOWN
    save_mode: str = SAVE_MODE_BOTH
    download_images: bool = True
    include_images_in_markdown: bool = True
    include_screenshot_images: bool = False
    include_object_images: bool = True
    image_filename_markers: tuple[str, ...] = ("_picture_", "_table_")
    llamaparse_options: dict[str, Any] = field(
        default_factory=lambda: {
            "split_by_page": True,
            "language": "ko",
            "adaptive_long_table": True,
            "disable_ocr": False,
            "disable_image_extraction": False,
            "do_not_unroll_columns": True,
            "extract_layout": True,
            "auto_mode": True,
            "auto_mode_trigger_on_table_in_page": True,
            "auto_mode_trigger_on_image_in_page": True,
        }
    )


EXPERIMENT = ParserExperimentConfig(
    output_format=OUTPUT_FORMAT_MARKDOWN,
    save_mode=SAVE_MODE_BOTH,
    download_images=True,
    include_images_in_markdown=True,
    include_screenshot_images=False,
    include_object_images=True,
    image_filename_markers=("_picture_", "_table_"),
    llamaparse_options={
        "split_by_page": True,
        "language": "ko",
        "adaptive_long_table": True,
        "disable_ocr": True,
        "disable_image_extraction": False,
        "do_not_unroll_columns": True,
        "extract_layout": True,
        "auto_mode": True,
        "auto_mode_trigger_on_table_in_page": True,
        "auto_mode_trigger_on_image_in_page": True,
    },
)


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


def to_llamaparse_result_type(output_format: str) -> str:
    if output_format == OUTPUT_FORMAT_MARKDOWN:
        return "markdown"
    if output_format == OUTPUT_FORMAT_TEXT:
        return "text"
    raise ValueError(
        f"output_format must be either '{OUTPUT_FORMAT_MARKDOWN}' "
        f"or '{OUTPUT_FORMAT_TEXT}'."
    )


def build_llamaparse_options(
    target_pages: list[int],
    page_number_base: str,
    config: ParserExperimentConfig,
) -> dict[str, Any]:
    llamaparse_target_pages = to_llamaparse_target_pages(target_pages, page_number_base)
    return {
        **config.llamaparse_options,
        "result_type": to_llamaparse_result_type(config.output_format),
        "target_pages": ",".join(str(page) for page in llamaparse_target_pages),
    }


def make_parser(
    target_pages: list[int],
    page_number_base: str,
    config: ParserExperimentConfig,
) -> LlamaParse:
    options = build_llamaparse_options(target_pages, page_number_base, config)
    return LlamaParse(**options)


def create_next_results_dir(
    parent_dir: Path = RESULTS_PARENT_DIR,
    prefix: str = RESULTS_DIR_PREFIX,
) -> Path:
    parent_dir.mkdir(exist_ok=True)
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)$")
    existing_numbers = []
    for path in parent_dir.iterdir():
        if not path.is_dir():
            continue
        match = pattern.match(path.name)
        if match:
            existing_numbers.append(int(match.group(1)))

    next_number = max(existing_numbers, default=0) + 1
    result_dir = parent_dir / f"{prefix}{next_number}"
    result_dir.mkdir()
    return result_dir


def save_hyperparameters(
    result_dir: Path,
    config: ParserExperimentConfig,
    target_pages: list[int],
    page_number_base: str,
) -> Path:
    llamaparse_options = build_llamaparse_options(target_pages, page_number_base, config)
    output_path = result_dir / "hyperparameters.txt"
    content_lines = [
        f"pdf_path={PDF_PATH}",
        f"example_pages_path={EXAMPLE_PAGES_PATH}",
        f"output_format={config.output_format}",
        f"save_mode={config.save_mode}",
        f"download_images={config.download_images}",
        f"include_images_in_markdown={config.include_images_in_markdown}",
        f"include_screenshot_images={config.include_screenshot_images}",
        f"include_object_images={config.include_object_images}",
        f"image_filename_markers={','.join(config.image_filename_markers)}",
        f"page_number_base={page_number_base}",
        f"target_pages={','.join(str(page) for page in target_pages)}",
        "",
        "[llamaparse]",
    ]
    content_lines.extend(
        f"{key}={value}" for key, value in sorted(llamaparse_options.items())
    )
    output_path.write_text("\n".join(content_lines) + "\n", encoding="utf-8")
    return output_path


def select_page_content(page: object, output_format: str) -> str:
    markdown = getattr(page, "md", None) or ""
    text = getattr(page, "text", None) or ""
    if output_format == OUTPUT_FORMAT_MARKDOWN:
        return markdown or text
    if output_format == OUTPUT_FORMAT_TEXT:
        return text or markdown
    raise ValueError(
        f"output_format must be either '{OUTPUT_FORMAT_MARKDOWN}' "
        f"or '{OUTPUT_FORMAT_TEXT}'."
    )


def validate_save_mode(save_mode: str) -> None:
    valid_save_modes = {SAVE_MODE_PAGE, SAVE_MODE_COMBINED, SAVE_MODE_BOTH}
    if save_mode not in valid_save_modes:
        raise ValueError(
            f"save_mode must be one of: {', '.join(sorted(valid_save_modes))}."
        )


def is_selected_image_path(image_path: Path, config: ParserExperimentConfig) -> bool:
    return any(marker in image_path.name for marker in config.image_filename_markers)


def image_path_to_markdown(result_dir: Path, image_path: Path) -> str:
    return image_path.relative_to(result_dir).as_posix()


def build_image_markdown(
    result_dir: Path,
    image_paths: list[Path],
    config: ParserExperimentConfig,
) -> list[str]:
    if not config.include_images_in_markdown or not image_paths:
        return []

    lines = ["", "## Images", ""]
    for index, image_path in enumerate(image_paths, start=1):
        relative_image_path = image_path_to_markdown(result_dir, image_path)
        lines.append(f"![image {index}]({relative_image_path})")
        lines.append("")
    return lines


def save_page_result(
    result_dir: Path,
    config: ParserExperimentConfig,
    output_format: str,
    page_number: int,
    page_number_base: str,
    categories: list[str],
    content_body: str,
    image_paths: list[Path],
) -> Path:
    category_name = "-".join(categories)
    output_path = result_dir / f"{page_number:03d}_{category_name}.{output_format}"
    content_parts = [
        f"# Page {page_number}",
        "",
        f"- source: {PDF_PATH}",
        f"- categories: {', '.join(categories)}",
        f"- page_number_base: {page_number_base}",
        "",
        content_body.strip(),
    ]
    content_parts.extend(build_image_markdown(result_dir, image_paths, config))
    content_parts.append("")
    output_path.write_text("\n".join(content_parts), encoding="utf-8")
    return output_path


def save_combined_result(
    result_dir: Path,
    config: ParserExperimentConfig,
    output_format: str,
    page_results: list[tuple[int, list[str], str, list[Path]]],
) -> Path:
    output_path = result_dir / f"combined.{output_format}"
    content_parts = [
        "# LlamaParse Combined Results",
        "",
        f"- source: {PDF_PATH}",
        "",
    ]

    for page_number, categories, content_body, image_paths in page_results:
        content_parts.extend(
            [
                "---",
                "",
                f"# Page {page_number}",
                "",
                f"- categories: {', '.join(categories)}",
                "",
                content_body.strip(),
            ]
        )
        content_parts.extend(build_image_markdown(result_dir, image_paths, config))
        content_parts.append("")

    output_path.write_text("\n".join(content_parts), encoding="utf-8")
    return output_path


def download_images_by_page(
    job_result: object,
    result_dir: Path,
    page_number_base: str,
    config: ParserExperimentConfig,
    allowed_pages: set[int],
) -> dict[int, list[Path]]:
    if not config.download_images:
        return {}

    image_dir = result_dir / "images"
    image_docs = job_result.get_image_documents(
        include_screenshot_images=config.include_screenshot_images,
        include_object_images=config.include_object_images,
        image_download_dir=str(image_dir),
    )

    images_by_page: dict[int, list[Path]] = {}
    for image_doc in image_docs:
        raw_page_number = image_doc.metadata.get("page_number")
        image_path = getattr(image_doc, "image_path", None)
        if not isinstance(raw_page_number, int) or not image_path:
            continue

        page_number = from_llamaparse_page_number(raw_page_number, page_number_base)
        image_file_path = Path(image_path)
        if page_number not in allowed_pages or not is_selected_image_path(
            image_file_path, config
        ):
            image_file_path.unlink(missing_ok=True)
            continue

        images_by_page.setdefault(page_number, []).append(image_file_path)

    return images_by_page


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    validate_save_mode(EXPERIMENT.save_mode)

    metadata, pages_by_category = load_example_config()
    page_number_base = str(metadata.get("page_number_base", PAGE_NUMBER_BASE_ONE_BASED))
    target_pages = flatten_pages(pages_by_category)
    page_categories = build_page_categories(pages_by_category)

    result_dir = create_next_results_dir()
    save_hyperparameters(result_dir, EXPERIMENT, target_pages, page_number_base)

    parser = make_parser(target_pages, page_number_base, EXPERIMENT)
    job_result = parser.parse(str(PDF_PATH))
    image_pages = set(pages_by_category.get(IMAGE_CATEGORY, []))
    images_by_page = download_images_by_page(
        job_result, result_dir, page_number_base, EXPERIMENT, image_pages
    )

    saved_paths: list[Path] = []
    page_results: list[tuple[int, list[str], str, list[Path]]] = []
    for page in job_result.pages:
        page_number = from_llamaparse_page_number(page.page, page_number_base)
        categories = page_categories.get(page_number, ["uncategorized"])
        content_body = select_page_content(page, EXPERIMENT.output_format)
        image_paths = images_by_page.get(page_number, [])
        page_results.append((page_number, categories, content_body, image_paths))

    if EXPERIMENT.save_mode in {SAVE_MODE_PAGE, SAVE_MODE_BOTH}:
        for page_number, categories, content_body, image_paths in page_results:
            saved_paths.append(
                save_page_result(
                    result_dir,
                    EXPERIMENT,
                    EXPERIMENT.output_format,
                    page_number,
                    page_number_base,
                    categories,
                    content_body,
                    image_paths,
                )
            )

    if EXPERIMENT.save_mode in {SAVE_MODE_COMBINED, SAVE_MODE_BOTH}:
        saved_paths.append(
            save_combined_result(
                result_dir, EXPERIMENT, EXPERIMENT.output_format, page_results
            )
        )

    print(f"Parsed pages: {target_pages}")
    print(f"Saved {len(saved_paths)} files to: {result_dir}")
    for path in saved_paths:
        print(f"- {path}")


if __name__ == "__main__":
    main()
