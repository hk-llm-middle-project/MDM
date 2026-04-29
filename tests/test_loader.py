import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from config import LLAMA_MD_DIR, PDFPLUMBER_OUT_DIR, VECTORSTORE_DIR, get_vectorstore_dir
from rag.loader import (
    LOADER_STRATEGIES,
    LlamaParserLoaderConfig,
    PdfPlumberLoaderConfig,
    UpstageLoaderConfig,
    load_pdf,
)
from rag.loader.strategies.llamaparser_loader import get_document_cache_dir
from rag.loader.strategies.pdfplumber_loader import (
    get_document_cache_dir as get_pdfplumber_cache_dir,
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

    def test_get_vectorstore_dir_separates_loader_and_embedding_strategies(self):
        self.assertEqual(get_vectorstore_dir("pdfplumber"), VECTORSTORE_DIR / "pdfplumber" / "bge")
        self.assertEqual(
            get_vectorstore_dir("llamaparser", "google"),
            VECTORSTORE_DIR / "llamaparser" / "google",
        )
        self.assertEqual(
            get_vectorstore_dir("llama-parse", "openai"),
            VECTORSTORE_DIR / "llamaparser" / "openai",
        )

    def test_get_vectorstore_dir_raises_for_unknown_strategy(self):
        with self.assertRaises(ValueError):
            get_vectorstore_dir("missing")

    def test_pdfplumber_strategy_preserves_page_metadata(self):
        with TemporaryDirectory() as temp_dir:
            pdf_path = Path(temp_dir) / "source.pdf"
            pdf_path.touch()
            output_dir = Path(temp_dir) / "pdfplumber_out"
            first_page = MagicMock()
            first_page.page_number = 1
            first_page.extract_text.return_value = " first page "
            empty_page = MagicMock()
            empty_page.page_number = 2
            empty_page.extract_text.return_value = " "
            fake_pdf = MagicMock()
            fake_pdf.__enter__.return_value.pages = [first_page, empty_page]

            with patch("rag.loader.strategies.pdfplumber_loader.pdfplumber.open", return_value=fake_pdf):
                documents = load_pdf(
                    pdf_path,
                    strategy_config=PdfPlumberLoaderConfig(output_dir=output_dir),
                )
            saved_content = (get_pdfplumber_cache_dir(pdf_path, output_dir) / "001.md").read_text(
                encoding="utf-8"
            )

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
        self.assertEqual(saved_content, "first page\n")

    def test_pdfplumber_strategy_inserts_extracted_table_at_page_position(self):
        with TemporaryDirectory() as temp_dir:
            pdf_path = Path(temp_dir) / "source.pdf"
            pdf_path.touch()
            output_dir = Path(temp_dir) / "pdfplumber_out"

            table = MagicMock()
            table.bbox = (0, 20, 100, 40)
            table.extract.return_value = [["구분", "값"], ["기본 과실비율", "A100:B0"]]
            page = MagicMock()
            page.page_number = 1
            page.extract_text.return_value = "위 본문\n구분 값\n기본 과실비율 A100:B0\n아래 본문"
            page.extract_words.return_value = [
                {"text": "위", "x0": 0, "top": 10, "bottom": 12},
                {"text": "본문", "x0": 10, "top": 10, "bottom": 12},
                {"text": "구분", "x0": 0, "top": 25, "bottom": 27},
                {"text": "값", "x0": 20, "top": 25, "bottom": 27},
                {"text": "기본", "x0": 0, "top": 30, "bottom": 32},
                {"text": "과실비율", "x0": 10, "top": 30, "bottom": 32},
                {"text": "A100:B0", "x0": 40, "top": 30, "bottom": 32},
                {"text": "아래", "x0": 0, "top": 50, "bottom": 52},
                {"text": "본문", "x0": 10, "top": 50, "bottom": 52},
            ]
            page.find_tables.return_value = [table]
            fake_pdf = MagicMock()
            fake_pdf.__enter__.return_value.pages = [page]

            with patch("rag.loader.strategies.pdfplumber_loader.pdfplumber.open", return_value=fake_pdf):
                documents = load_pdf(
                    pdf_path,
                    strategy_config=PdfPlumberLoaderConfig(output_dir=output_dir),
                )

        page_content = documents[0].page_content
        self.assertLess(page_content.index("위 본문"), page_content.index("| 구분 | 값 |"))
        self.assertLess(page_content.index("| 구분 | 값 |"), page_content.index("아래 본문"))
        self.assertIn("| 기본 과실비율 | A100:B0 |", page_content)
        self.assertEqual(page_content.count("기본 과실비율"), 1)

    def test_pdfplumber_strategy_applies_crop_word_and_table_settings(self):
        with TemporaryDirectory() as temp_dir:
            pdf_path = Path(temp_dir) / "source.pdf"
            pdf_path.touch()
            output_dir = Path(temp_dir) / "pdfplumber_out"
            original_page = MagicMock()
            original_page.page_number = 1
            original_page.width = 600
            original_page.height = 800
            cropped_page = MagicMock()
            cropped_page.extract_text.return_value = "cropped text"
            cropped_page.find_tables.return_value = []
            cropped_page.extract_words.return_value = []
            original_page.crop.return_value = cropped_page
            fake_pdf = MagicMock()
            fake_pdf.__enter__.return_value.pages = [original_page]
            word_settings = {"x_tolerance": 5, "y_tolerance": 4}
            table_settings = {"vertical_strategy": "lines"}

            with patch("rag.loader.strategies.pdfplumber_loader.pdfplumber.open", return_value=fake_pdf):
                documents = load_pdf(
                    pdf_path,
                    strategy_config=PdfPlumberLoaderConfig(
                        output_dir=output_dir,
                        crop_margins=(10, 20, 30, 40),
                        word_settings=word_settings,
                        table_settings=table_settings,
                    ),
                )

        original_page.crop.assert_called_once_with((10, 20, 570, 760))
        cropped_page.find_tables.assert_called_once_with(table_settings=table_settings)
        cropped_page.extract_words.assert_not_called()
        self.assertEqual([document.page_content for document in documents], ["cropped text"])

    def test_pdfplumber_strategy_reuses_saved_markdown_without_parsing(self):
        with TemporaryDirectory() as temp_dir:
            pdf_path = Path(temp_dir) / "source.pdf"
            pdf_path.touch()
            output_dir = Path(temp_dir) / "pdfplumber_out"
            document_cache_dir = get_pdfplumber_cache_dir(pdf_path, output_dir)
            document_cache_dir.mkdir(parents=True)
            (document_cache_dir / "001.md").write_text("page one\n", encoding="utf-8")
            (document_cache_dir / "003.md").write_text("page three\n", encoding="utf-8")

            with patch("rag.loader.strategies.pdfplumber_loader.pdfplumber.open") as open_mock:
                documents = load_pdf(
                    pdf_path,
                    strategy_config=PdfPlumberLoaderConfig(output_dir=output_dir),
                )

        self.assertEqual(
            [document.page_content for document in documents],
            ["page one\n", "page three\n"],
        )
        self.assertEqual(
            [document.metadata for document in documents],
            [
                {
                    "source": str(pdf_path),
                    "page": 1,
                    "parser": "pdfplumber",
                },
                {
                    "source": str(pdf_path),
                    "page": 3,
                    "parser": "pdfplumber",
                },
            ],
        )
        open_mock.assert_not_called()

    def test_pdfplumber_default_output_dir_is_data_pdfplumber_out_root(self):
        self.assertEqual(PdfPlumberLoaderConfig().output_dir, PDFPLUMBER_OUT_DIR)

    def test_pdfplumber_cache_dir_uses_main_pdf_directory(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "pdfplumber_out"
            first_pdf = Path(temp_dir) / "source.pdf"
            second_pdf = Path(temp_dir) / "source-copy.pdf"
            first_pdf.write_bytes(b"first")
            second_pdf.write_bytes(b"second")

            first_cache_dir = get_pdfplumber_cache_dir(first_pdf, root)
            second_cache_dir = get_pdfplumber_cache_dir(second_pdf, root)

        self.assertEqual(first_cache_dir, root / "main_pdf")
        self.assertEqual(second_cache_dir, root / "main_pdf")

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
            document_cache_dir = get_document_cache_dir(pdf_path, output_dir)
            saved_content = (document_cache_dir / "003.md").read_text(encoding="utf-8")

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
            document_cache_dir = get_document_cache_dir(pdf_path, output_dir)
            document_cache_dir.mkdir(parents=True)
            (document_cache_dir / "001.md").write_text("# page one\n", encoding="utf-8")
            (document_cache_dir / "003.md").write_text("# page three\n", encoding="utf-8")
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

    def test_llamaparser_default_output_dir_is_data_llama_md_root(self):
        self.assertEqual(LlamaParserLoaderConfig().output_dir, LLAMA_MD_DIR)

    def test_llamaparser_cache_dir_uses_main_pdf_directory(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "llama_md"
            first_pdf = Path(temp_dir) / "source.pdf"
            second_pdf = Path(temp_dir) / "source-copy.pdf"
            first_pdf.write_bytes(b"first")
            second_pdf.write_bytes(b"second")

            first_cache_dir = get_document_cache_dir(first_pdf, root)
            second_cache_dir = get_document_cache_dir(second_pdf, root)

        self.assertEqual(first_cache_dir, root / "main_pdf")
        self.assertEqual(second_cache_dir, root / "main_pdf")

    def test_upstage_strategy_reuses_final_json_without_api_parsing(self):
        with TemporaryDirectory() as temp_dir:
            pdf_path = Path(temp_dir) / "source.pdf"
            pdf_path.touch()
            final_path = Path(temp_dir) / "final" / "chunked_documents_final.json"
            final_path.parent.mkdir(parents=True)
            final_path.write_text(
                json.dumps(
                    [
                        {
                            "page_content": "already chunked",
                            "metadata": {
                                "chunk_type": "general",
                                "diagram_id": None,
                                "page": 3,
                            },
                        }
                    ]
                ),
                encoding="utf-8",
            )
            config = UpstageLoaderConfig(final_documents_path=final_path)

            with patch("rag.loader.strategies.upstage.upstage_loader.create_upstage_loader") as loader_mock:
                documents = load_pdf(pdf_path, strategy="upstage", strategy_config=config)

        self.assertEqual([document.page_content for document in documents], ["already chunked"])
        self.assertEqual(
            documents[0].metadata,
            {
                "chunk_type": "general",
                "page": 3,
                "source": str(pdf_path),
                "parser": "upstage",
            },
        )
        loader_mock.assert_not_called()

    def test_upstage_strategy_saves_raw_json_when_final_cache_is_missing(self):
        with TemporaryDirectory() as temp_dir:
            pdf_path = Path(temp_dir) / "source.pdf"
            pdf_path.touch()
            final_path = Path(temp_dir) / "final" / "chunked_documents_final.json"
            raw_path = Path(temp_dir) / "raw" / "parsed_documents_raw.json"
            loader = MagicMock()
            loader.load.return_value = [
                Document(page_content="raw parsed", metadata={"page": 1, "category": "paragraph"})
            ]
            config = UpstageLoaderConfig(
                final_documents_path=final_path,
                raw_documents_path=raw_path,
                save_images=False,
            )

            with (
                patch(
                    "rag.loader.strategies.upstage.upstage_loader.split_pdf_for_upstage",
                    return_value=[{"path": pdf_path, "page_offset": 0}],
                ),
                patch(
                    "rag.loader.strategies.upstage.upstage_loader.create_upstage_loader",
                    return_value=loader,
                ),
            ):
                documents = load_pdf(pdf_path, strategy="upstage", strategy_config=config)

            self.assertEqual([document.page_content for document in documents], ["raw parsed"])
            self.assertTrue(raw_path.exists())
            saved_payload = json.loads(raw_path.read_text(encoding="utf-8"))
            self.assertEqual(saved_payload[0]["page_content"], "raw parsed")
            self.assertEqual(saved_payload[0]["metadata"]["parser"], "upstage")
            self.assertEqual(saved_payload[0]["metadata"]["source"], str(pdf_path))


if __name__ == "__main__":
    unittest.main()
