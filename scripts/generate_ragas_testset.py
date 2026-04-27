"""Generate a synthetic RAG evaluation testset with RAGAS.

This script loads the project PDF, chunks it for dataset generation, and asks
RAGAS to create question/reference/context samples for later evaluation.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import openai
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ragas.embeddings import OpenAIEmbeddings as RagasOpenAIEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.testset import TestsetGenerator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import BASE_DIR, LLM_MODEL, PDF_PATH
from rag.loader import load_pdf


DEFAULT_OUTPUT_PATH = BASE_DIR / "data" / "testsets" / "synthetic_testset_v0_10.jsonl"
DEFAULT_CHUNK_SIZE = 1500
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_TESTSET_SIZE = 10
DEFAULT_MAX_PAGES = 0
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_LANGSMITH_PROJECT = "MDM-RAGAS-DatasetGen"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a RAGAS synthetic testset from the project PDF.",
    )
    parser.add_argument(
        "--pdf-path",
        type=Path,
        default=PDF_PATH,
        help="PDF file to use as the source document.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output path where the generated testset will be saved. Supports .jsonl and .csv.",
    )
    parser.add_argument(
        "--testset-size",
        type=int,
        default=DEFAULT_TESTSET_SIZE,
        help="Number of synthetic questions to generate.",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=DEFAULT_MAX_PAGES,
        help="Use only the first N loaded pages. Set to 0 to use all pages.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Character chunk size used before RAGAS generation.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP,
        help="Character overlap between chunks.",
    )
    parser.add_argument(
        "--llm-model",
        default=LLM_MODEL,
        help="OpenAI chat model used by RAGAS for generation.",
    )
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help="OpenAI embedding model used by RAGAS.",
    )
    parser.add_argument(
        "--langsmith-project",
        default=DEFAULT_LANGSMITH_PROJECT,
        help="LangSmith project name for tracing, if LangSmith is enabled.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load and chunk the PDF without calling OpenAI/RAGAS generation.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.testset_size <= 0:
        raise ValueError("--testset-size must be greater than 0")
    if args.max_pages < 0:
        raise ValueError("--max-pages must be 0 or greater")
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be greater than 0")
    if args.chunk_overlap < 0:
        raise ValueError("--chunk-overlap must be 0 or greater")
    if args.chunk_overlap >= args.chunk_size:
        raise ValueError("--chunk-overlap must be smaller than --chunk-size")
    if not args.pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {args.pdf_path}")


def build_text_chunks(pdf_path: Path, max_pages: int, chunk_size: int, chunk_overlap: int) -> list[str]:
    documents = load_pdf(pdf_path)
    if max_pages:
        documents = documents[:max_pages]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(documents)
    return [chunk.page_content for chunk in chunks if chunk.page_content.strip()]


def build_testset_generator(llm_model: str, embedding_model: str) -> TestsetGenerator:
    generator_llm = LangchainLLMWrapper(ChatOpenAI(model=llm_model))
    generator_embeddings = RagasOpenAIEmbeddings(
        client=openai.OpenAI(),
        model=embedding_model,
    )
    return TestsetGenerator(
        llm=generator_llm,
        embedding_model=generator_embeddings,
    )


def generate_and_save_testset(
    generator: TestsetGenerator,
    text_chunks: list[str],
    testset_size: int,
    output_path: Path,
) -> None:
    testset = generator.generate_with_chunks(
        chunks=text_chunks,
        testset_size=testset_size,
    )
    test_df = testset.to_pandas()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_suffix = output_path.suffix.lower()
    if output_suffix == ".jsonl":
        test_df.to_json(
            output_path,
            orient="records",
            lines=True,
            force_ascii=False,
        )
    elif output_suffix == ".csv":
        test_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    else:
        raise ValueError("Output path must end with .jsonl or .csv")

    print(f"Generated rows: {len(test_df)}")
    print(f"Columns: {list(test_df.columns)}")
    print(f"Saved testset: {output_path}")


def main() -> None:
    load_dotenv()
    args = parse_args()
    validate_args(args)

    os.environ["LANGSMITH_PROJECT"] = args.langsmith_project

    text_chunks = build_text_chunks(
        pdf_path=args.pdf_path,
        max_pages=args.max_pages,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    print(f"PDF: {args.pdf_path}")
    print(f"Text chunks: {len(text_chunks)}")
    if text_chunks:
        print(f"First chunk length: {len(text_chunks[0])}")

    if args.dry_run:
        print("Dry run complete. No testset was generated.")
        return

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required to generate a RAGAS testset.")
    if not text_chunks:
        raise RuntimeError("No text chunks were created from the source PDF.")

    generator = build_testset_generator(
        llm_model=args.llm_model,
        embedding_model=args.embedding_model,
    )
    generate_and_save_testset(
        generator=generator,
        text_chunks=text_chunks,
        testset_size=args.testset_size,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()
