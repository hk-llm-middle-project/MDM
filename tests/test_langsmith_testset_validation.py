import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


def load_validator_module():
    module_path = Path(__file__).resolve().parents[1] / "evaluation" / "validate_langsmith_testsets.py"
    spec = importlib.util.spec_from_file_location("validate_langsmith_testsets", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load validate_langsmith_testsets.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LangSmithTestsetValidationTest(unittest.TestCase):
    def setUp(self):
        self.validator = load_validator_module()

    def test_default_chunks_path_points_to_canonical_upstage_chunks(self):
        with patch.object(sys, "argv", ["validate_langsmith_testsets.py"]):
            args = self.validator.parse_args()

        chunks_path = Path(args.chunks)

        self.assertEqual(chunks_path, Path("data/chunks/upstage/custom/chunks.json"))
        self.assertTrue(chunks_path.exists())

    def test_normalize_text_treats_ocr_letter_o_as_zero_in_fault_ratio(self):
        self.assertEqual(
            self.validator.normalize_text("기본 과실비율 A100 BO"),
            self.validator.normalize_text("기본 과실비율 A100 B0"),
        )


if __name__ == "__main__":
    unittest.main()
