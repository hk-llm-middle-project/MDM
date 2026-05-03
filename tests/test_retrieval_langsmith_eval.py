import importlib.util
import json
import os
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import Mock, patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from main import (
    CHUNKER_STRATEGY_OPTIONS_BY_LOADER,
    EMBEDDING_PROVIDER_OPTIONS,
    LOADER_STRATEGY_OPTIONS,
    RERANKER_STRATEGY_OPTIONS,
    RETRIEVER_STRATEGY_OPTIONS,
)


def load_retrieval_eval_module():
    module_path = Path(__file__).resolve().parents[1] / "evaluation" / "evaluate_retrieval_langsmith.py"
    spec = importlib.util.spec_from_file_location("evaluate_retrieval_langsmith", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load evaluate_retrieval_langsmith.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RetrievalLangSmithEvalTest(unittest.TestCase):
    def test_default_args_target_upstage_custom_langsmith_retrieval_eval(self):
        module = load_retrieval_eval_module()

        with patch.object(sys, "argv", ["evaluate_retrieval_langsmith.py"]):
            args = module.parse_args()

        self.assertEqual(
            args.testset_path,
            module.BASE_DIR / "data" / "testsets" / "langsmith" / "retrieval_eval.jsonl",
        )
        self.assertEqual(args.loader_strategy, "upstage")
        self.assertEqual(args.chunker_strategy, "custom")
        self.assertEqual(args.embedding_provider, "bge")
        self.assertEqual(args.retriever_strategy, "similarity")
        self.assertEqual(args.reranker_strategy, "none")
        self.assertFalse(args.langsmith)

    def test_retriever_and_reranker_choices_follow_streamlit_options(self):
        module = load_retrieval_eval_module()

        self.assertEqual(set(module.RETRIEVER_STRATEGY_CHOICES), {"vectorstore", *RETRIEVER_STRATEGY_OPTIONS})
        self.assertEqual(tuple(module.RERANKER_STRATEGY_CHOICES), RERANKER_STRATEGY_OPTIONS)
        self.assertIn("ensemble_parent", module.RETRIEVER_STRATEGY_CHOICES)
        self.assertIn("cross-encoder", module.RERANKER_STRATEGY_CHOICES)

    def test_langsmith_flag_is_required_for_remote_evaluation_mode(self):
        module = load_retrieval_eval_module()

        with patch.object(sys, "argv", ["evaluate_retrieval_langsmith.py", "--langsmith"]):
            args = module.parse_args()

        self.assertTrue(args.langsmith)

    def test_configure_tracing_disables_langsmith_tracing_for_local_mode(self):
        module = load_retrieval_eval_module()

        with patch.dict(
            os.environ,
            {
                "LANGSMITH_TRACING": "true",
                "LANGCHAIN_TRACING_V2": "true",
                "ANONYMIZED_TELEMETRY": "true",
            },
            clear=False,
        ):
            module.configure_tracing(needs_langsmith=False)

            self.assertEqual(os.environ["LANGSMITH_TRACING"], "false")
            self.assertEqual(os.environ["LANGCHAIN_TRACING_V2"], "false")
            self.assertEqual(os.environ["ANONYMIZED_TELEMETRY"], "False")

    def test_evaluate_local_rows_returns_langsmith_compatible_columns(self):
        module = load_retrieval_eval_module()

        def target(inputs):
            return {
                "query": inputs["question"],
                "retrieved": [{"metadata": {"diagram_id": "A"}, "page_content": "context"}],
                "retrieved_metadata": [{"diagram_id": "A"}],
                "contexts": ["context"],
            }

        rows = [
            {
                "question": "테스트 질문",
                "expected_diagram_ids": ["A"],
                "expected_location": None,
                "expected_party_type": None,
                "expected_chunk_types": [],
                "expected_keywords": [],
            }
        ]

        frame = module.evaluate_local_rows(
            rows=rows,
            target=target,
            evaluators=[module.diagram_id_hit, module.critical_error],
        )

        self.assertEqual(len(frame), 1)
        self.assertEqual(frame.loc[0, "inputs.question"], "테스트 질문")
        self.assertEqual(frame.loc[0, "outputs.query"], "테스트 질문")
        self.assertEqual(frame.loc[0, "feedback.diagram_id_hit"], 1)
        self.assertEqual(frame.loc[0, "feedback.critical_error"], 0)

    def test_default_dataset_name_includes_loader_chunker_and_embedding(self):
        module = load_retrieval_eval_module()

        dataset_name = module.make_dataset_name(
            Path("data/testsets/langsmith/retrieval_eval.jsonl"),
            loader_strategy="upstage",
            embedding_provider="bge",
            chunker_strategy="custom",
        )

        self.assertEqual(
            dataset_name,
            "MDM retrieval testset - upstage-custom-bge - retrieval_eval",
        )

    def test_load_matrix_resolves_named_preset_runs(self):
        module = load_retrieval_eval_module()

        matrix = module.load_eval_matrix(module.DEFAULT_MATRIX_PATH)
        runs = module.resolve_matrix_runs(matrix, "upstage")

        self.assertEqual(
            [(run["loader_strategy"], run["chunker_strategy"], run["embedding_provider"]) for run in runs],
            [
                ("upstage", "custom", "bge"),
                ("upstage", "custom", "openai"),
                ("upstage", "custom", "google"),
                ("upstage", "raw", "bge"),
                ("upstage", "raw", "openai"),
                ("upstage", "raw", "google"),
            ],
        )

    def test_matrix_all_matches_streamlit_sidebar_combinations(self):
        module = load_retrieval_eval_module()

        matrix = module.load_eval_matrix(module.DEFAULT_MATRIX_PATH)
        runs = module.resolve_matrix_runs(matrix, "all")
        actual = {
            (run["loader_strategy"], run["chunker_strategy"], run["embedding_provider"])
            for run in runs
        }
        expected = {
            (loader, chunker, embedding)
            for loader in LOADER_STRATEGY_OPTIONS
            for chunker in CHUNKER_STRATEGY_OPTIONS_BY_LOADER[loader]
            for embedding in EMBEDDING_PROVIDER_OPTIONS
        }

        self.assertEqual(actual, expected)
        self.assertEqual(len(actual), 30)

    def test_preset_arg_implies_matrix_mode(self):
        module = load_retrieval_eval_module()

        with patch.object(sys, "argv", ["evaluate_retrieval_langsmith.py", "--preset", "all"]):
            args = module.parse_args()

        self.assertTrue(module.should_run_matrix(args))

    def test_matrix_run_skips_missing_vectorstore_by_default(self):
        module = load_retrieval_eval_module()
        args = Namespace(
            testset_path=Path("data/testsets/langsmith/retrieval_eval.jsonl"),
            dataset_name=None,
            retriever_strategy="vectorstore",
            reranker_strategy="none",
            k=5,
            max_concurrency=1,
            upload_only=False,
            fail_on_missing_vectorstore=False,
        )
        run = {
            "name": "upstage-custom-google",
            "loader_strategy": "upstage",
            "chunker_strategy": "custom",
            "embedding_provider": "google",
        }

        with (
            patch.object(module, "get_vectorstore_dir", return_value=Path("missing/vectorstore")),
            patch.object(module, "vectorstore_exists", return_value=False),
            patch.object(module, "get_or_create_dataset") as get_or_create_dataset,
        ):
            module.run_retrieval_experiment(
                args=args,
                client=Mock(),
                rows=[{"question": "테스트"}],
                run=run,
                matrix_mode=True,
            )

        get_or_create_dataset.assert_not_called()

    def test_save_experiment_results_writes_csv_and_summary_json(self):
        module = load_retrieval_eval_module()
        import pandas as pd

        results = Mock()
        results.experiment_name = "MDM retrieval eval - upstage-custom-bge"
        results.to_pandas.return_value = pd.DataFrame(
            [
                {"feedback.diagram_id_hit": 1, "feedback.critical_error": 0},
                {"feedback.diagram_id_hit": 0, "feedback.critical_error": 1},
            ]
        )
        run = {
            "name": "upstage-custom-bge",
            "loader_strategy": "upstage",
            "chunker_strategy": "custom",
            "embedding_provider": "bge",
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            paths = module.save_experiment_results(
                results=results,
                output_dir=Path(temp_dir),
                run=run,
                dataset_name="dataset",
                testset_path=Path("data/testsets/langsmith/retrieval_eval.jsonl"),
            )

            self.assertTrue(paths["csv"].exists())
            self.assertTrue(paths["summary_json"].exists())
            summary = json.loads(paths["summary_json"].read_text(encoding="utf-8"))

        self.assertEqual(summary["run_name"], "upstage-custom-bge")
        self.assertEqual(summary["row_count"], 2)
        self.assertEqual(summary["retriever_strategy"], "similarity")
        self.assertEqual(summary["reranker_strategy"], "none")
        self.assertEqual(summary["final_k"], 5)
        self.assertEqual(summary["metrics"]["diagram_id_hit"], 0.5)
        self.assertEqual(summary["metrics"]["critical_error"], 0.5)

    def test_save_experiment_dataframe_records_retriever_and_reranker_metadata(self):
        module = load_retrieval_eval_module()
        import pandas as pd

        run = {
            "name": "upstage-custom-bge",
            "loader_strategy": "upstage",
            "chunker_strategy": "custom",
            "embedding_provider": "bge",
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            paths = module.save_experiment_dataframe(
                dataframe=pd.DataFrame([{"feedback.critical_error": 0}]),
                experiment_name="experiment",
                output_dir=Path(temp_dir),
                run=run,
                dataset_name="dataset",
                testset_path=Path("data/testsets/langsmith/retrieval_eval.jsonl"),
                retriever_strategy="ensemble_parent",
                reranker_strategy="cross-encoder",
                final_k=3,
                candidate_k=30,
            )
            summary = json.loads(paths["summary_json"].read_text(encoding="utf-8"))

        self.assertEqual(summary["retriever_strategy"], "ensemble_parent")
        self.assertEqual(summary["reranker_strategy"], "cross-encoder")
        self.assertEqual(summary["final_k"], 3)
        self.assertEqual(summary["candidate_k"], 30)

    def test_summarize_feedback_metrics_ignores_feedback_comment_columns(self):
        module = load_retrieval_eval_module()
        import pandas as pd

        metrics = module.summarize_feedback_metrics(
            pd.DataFrame(
                [
                    {
                        "feedback.critical_error": 1,
                        "feedback.critical_error.comment": "bad",
                    },
                    {
                        "feedback.critical_error": 0,
                        "feedback.critical_error.comment": "ok",
                    },
                ]
            )
        )

        self.assertEqual(metrics, {"critical_error": 0.5})

    def test_build_retrieval_target_uses_chunker_strategy_for_vectorstore_dir(self):
        module = load_retrieval_eval_module()
        vectorstore_dir = Path("data/vectorstore/upstage/custom/bge")

        with (
            patch.object(module, "get_vectorstore_dir", return_value=vectorstore_dir) as get_dir,
            patch.object(module, "vectorstore_exists", return_value=True),
            patch.object(module, "load_vectorstore", return_value=object()),
            patch.object(module, "build_retrieval_components", return_value=object()),
            patch.object(module, "run_retrieval_pipeline", return_value=[]),
        ):
            target = module.build_retrieval_target(
                loader_strategy="upstage",
                embedding_provider="bge",
                chunker_strategy="custom",
                retriever_strategy="vectorstore",
                reranker_strategy="none",
                k=5,
            )
            outputs = target({"question": "테스트 질문"})

        get_dir.assert_called_once_with("upstage", "bge", chunker_strategy="custom")
        self.assertEqual(outputs["chunker_strategy"], "custom")


if __name__ == "__main__":
    unittest.main()
