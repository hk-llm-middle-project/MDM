"""PDF loading and document extraction utilities."""

import pdfplumber
from langchain_core.documents import Document

from config import PDF_PATH


def load_pdf(path=PDF_PATH) -> list[Document]:
    """Load a PDF with pdfplumber and return page documents."""
    documents: list[Document] = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            if text.strip():
                documents.append(Document(page_content=text, metadata={}))
    return documents
