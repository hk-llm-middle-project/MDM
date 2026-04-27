import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from config import LLAMA_MD_DIR, VECTORSTORE_DIR, get_vectorstore_dir
from rag.loader import (
    LOADER_STRATEGIES,
    LlamaParserLoaderConfig,
    PdfPlumberLoaderConfig,
    load_pdf,
)


class LoaderTest(unittest.TestCase):
    def test_load_pdf_routes_to_selected_strategy(self):
        called = {}

        def fake_strategy(path, strategy_config):
            called["args"] = (path, strategy_config)
            return [Document(page_content="loaded")]

        config = PdfPlumberLoaderConfig()

        with patch.dict(LOADER_STRATEGIES, {"fake": fake_strategy}, clear=False):
            documents = load_pdf(Path("source.pdf"), strategy="fake", strategy_config=config)

        self.assertEqual([document.page_content for document in documents], ["loaded"])
        self.assertEqual(called["args"], (Path("source.pdf"), config))

    def test_load_pdf_raises_for_unknown_strategy(self):
        with self.assertRaises(ValueError):
            load_pdf(Path("source.pdf"), strategy="missing")

    def test_get_vectorstore_dir_separates_loader_strategies(self):
        self.assertEqual(get_vectorstore_dir("pdfplumber"), VECTORSTORE_DIR / "pdfplumber")
        self.assertEqual(get_vectorstore_dir("llamaparser"), VECTORSTORE_DIR / "llamaparser")
        self.assertEqual(get_vectorstore_dir("llama-parse"), VECTORSTORE_DIR / "llamaparser")

    def test_get_vectorstore_dir_raises_for_unknown_strategy(self):
        with self.assertRaises(ValueError):
            get_vectorstore_dir("missing")

    def test_pdfplumber_strategy_preserves_page_metadata(self):
        with TemporaryDirectory() as temp_dir:
            pdf_path = Path(temp_dir) / "source.pdf"
            pdf_path.touch()
            first_page = MagicMock()
            first_page.page_number = 1
            first_page.extract_text.return_value = " first page "
            empty_page = MagicMock()
            empty_page.page_number = 2
            empty_page.extract_text.return_value = " "
            fake_pdf = MagicMock()
            fake_pdf.__enter__.return_value.pages = [first_page, empty_page]

            with patch("rag.loader.strategies.pdfplumber_loader.pdfplumber.open", return_value=fake_pdf):
                documents = load_pdf(pdf_path)

        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0].page_content, " first page ")
        self.assertEqual(
            documents[0].metadata,
            {
                "source": str(pdf_path),
                "page": 1,
                "parser": "pdfplumber",
            },
        )

    def test_llamaparser_strategy_builds_documents_and_saves_markdown(self):
        with TemporaryDirectory() as temp_dir:
            pdf_path = Path(temp_dir) / "source.pdf"
            pdf_path.touch()
            output_dir = Path(temp_dir) / "llama_md"
            page = MagicMock()
            page.page = 3
            page.md = "# content"
            page.text = "content"
            job_result = MagicMock()
            job_result.pages = [page]
            parser_instance = MagicMock()
            parser_instance.parse.return_value = job_result
            parser_class = MagicMock(return_value=parser_instance)

            with patch("rag.loader.strategies.llamaparser_loader.LlamaParse", parser_class):
                documents = load_pdf(
                    pdf_path,
                    strategy="llamaparser",
                    strategy_config=LlamaParserLoaderConfig(output_dir=output_dir),
                )
            saved_content = (output_dir / "003.md").read_text(encoding="utf-8")

        self.assertEqual([document.page_content for document in documents], ["# content"])
        self.assertEqual(
            documents[0].metadata,
            {
                "source": str(pdf_path),
                "page": 3,
                "parser": "llamaparser",
            },
        )
        parser_class.assert_called_once_with(
            split_by_page=True,
            language="ko",
            adaptive_long_table=True,
            disable_ocr=True,
            disable_image_extraction=False,
            do_not_unroll_columns=True,
            extract_layout=True,
            auto_mode=True,
            auto_mode_trigger_on_table_in_page=True,
            auto_mode_trigger_on_image_in_page=True,
            result_type="markdown",
        )
        parser_instance.parse.assert_called_once_with(str(pdf_path))
        self.assertEqual(saved_content, "# content\n")

    def test_llamaparser_strategy_reuses_saved_markdown_without_parsing(self):
        with TemporaryDirectory() as temp_dir:
            pdf_path = Path(temp_dir) / "source.pdf"
            pdf_path.touch()
            output_dir = Path(temp_dir) / "llama_md"
            output_dir.mkdir()
            (output_dir / "001.md").write_text("# page one\n", encoding="utf-8")
            (output_dir / "003.md").write_text("# page three\n", encoding="utf-8")
            parser_class = MagicMock()

            with patch("rag.loader.strategies.llamaparser_loader.LlamaParse", parser_class):
                documents = load_pdf(
                    pdf_path,
                    strategy="llamaparser",
                    strategy_config=LlamaParserLoaderConfig(output_dir=output_dir),
                )

        self.assertEqual(
            [document.page_content for document in documents],
            ["# page one\n", "# page three\n"],
        )
        self.assertEqual(
            [document.metadata for document in documents],
            [
                {
                    "source": str(pdf_path),
                    "page": 1,
                    "parser": "llamaparser",
                },
                {
                    "source": str(pdf_path),
                    "page": 3,
                    "parser": "llamaparser",
                },
            ],
        )
        parser_class.assert_not_called()

    def test_llamaparser_default_output_dir_is_data_llama_md(self):
        self.assertEqual(LlamaParserLoaderConfig().output_dir, LLAMA_MD_DIR)


if __name__ == "__main__":
    unittest.main()
