"""Preview text extracted from selected PDF pages with pdfplumber."""

from pathlib import Path

import pdfplumber


PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_PATH = Path(__file__).resolve().parent / "pdfplumber_results.txt"
PDF_PATH = PROJECT_ROOT / "data/raw/230630_자동차사고 과실비율 인정기준_최종.pdf"
PAGE_NUMBERS = [47, 50, 197, 438, 558]


def main() -> None:
    output_parts: list[str] = []

    with pdfplumber.open(PDF_PATH) as pdf:
        total_pages = len(pdf.pages)
        output_parts.extend(
            [
                f"PDF: {PDF_PATH}",
                f"Total pages: {total_pages}",
                f"Preview pages: {PAGE_NUMBERS}",
            ]
        )

        for page_number in PAGE_NUMBERS:
            if page_number < 1 or page_number > total_pages:
                output_parts.append(f"\n--- Page {page_number} skipped: out of range ---")
                continue

            page = pdf.pages[page_number - 1]
            text = page.extract_text() or ""

            output_parts.extend(
                [
                    "\n" + "=" * 80,
                    f"Page {page_number}",
                    "=" * 80,
                    text if text.strip() else "[No text extracted]",
                ]
            )

    OUTPUT_PATH.write_text("\n".join(output_parts) + "\n", encoding="utf-8")
    print(f"Saved pdfplumber results to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
