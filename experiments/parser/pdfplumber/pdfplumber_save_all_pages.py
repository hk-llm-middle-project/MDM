"""Save pdfplumber text extraction results as one TXT file per PDF page."""

from pathlib import Path

import pdfplumber


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PDF_PATH = PROJECT_ROOT / "data/raw/230630_자동차사고 과실비율 인정기준_최종.pdf"
OUTPUT_DIR = Path(__file__).resolve().parent / "results"


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    with pdfplumber.open(PDF_PATH) as pdf:
        total_pages = len(pdf.pages)
        filename_width = len(str(total_pages - 1))

        for page_index, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            output_path = OUTPUT_DIR / f"{page_index:0{filename_width}d}.txt"
            output_path.write_text(text, encoding="utf-8")

    print(f"Saved {total_pages} page files to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
