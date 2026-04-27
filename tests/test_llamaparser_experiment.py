import unittest
import sys
import types
from pathlib import Path
from tempfile import TemporaryDirectory

fake_llama_parse = types.ModuleType("llama_parse")
fake_llama_parse.LlamaParse = object
sys.modules.setdefault("llama_parse", fake_llama_parse)

from experiments.parser.llamaparser import parser


class LlamaParserExperimentTest(unittest.TestCase):
    def test_create_next_results_dir_uses_next_numeric_suffix(self):
        with TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            (base_dir / "results_1").mkdir()
            (base_dir / "results_3").mkdir()
            (base_dir / "results_md").mkdir()

            result_dir = parser.create_next_results_dir(base_dir)

            self.assertEqual(result_dir, base_dir / "results_4")
            self.assertTrue(result_dir.is_dir())

    def test_build_llamaparse_options_maps_output_format_and_target_pages(self):
        config = parser.ParserExperimentConfig(
            output_format="txt",
            llamaparse_options={"extract_layout": False},
        )

        options = parser.build_llamaparse_options(
            target_pages=[1, 4],
            page_number_base=parser.PAGE_NUMBER_BASE_ONE_BASED,
            config=config,
        )

        self.assertEqual(options["result_type"], "text")
        self.assertEqual(options["target_pages"], "0,3")
        self.assertFalse(options["extract_layout"])

    def test_save_hyperparameters_writes_config_snapshot(self):
        with TemporaryDirectory() as temp_dir:
            result_dir = Path(temp_dir)
            config = parser.ParserExperimentConfig(
                output_format="md",
                llamaparse_options={"extract_layout": False, "language": "ko"},
            )

            output_path = parser.save_hyperparameters(
                result_dir=result_dir,
                config=config,
                target_pages=[1, 4],
                page_number_base=parser.PAGE_NUMBER_BASE_ONE_BASED,
            )

            content = output_path.read_text(encoding="utf-8")
            self.assertIn("output_format=md", content)
            self.assertIn("result_type=markdown", content)
            self.assertIn("extract_layout=False", content)
            self.assertIn("target_pages=1,4", content)


if __name__ == "__main__":
    unittest.main()
