import importlib.util
import json
import os
import sys
import tempfile
import threading
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import Mock, patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import RETRIEVER_K
from main import (
    CHUNKER_STRATEGY_OPTIONS_BY_LOADER,
    DEFAULT_CHUNKER_STRATEGY,
    DEFAULT_EMBEDDING_PROVIDER,
    DEFAULT_ENSEMBLE_BM25_WEIGHT,
    DEFAULT_ENSEMBLE_CANDIDATE_K,
    DEFAULT_ENSEMBLE_USE_CHUNK_ID,
    DEFAULT_LOADER_STRATEGY,
    DEFAULT_RERANKER_STRATEGY,
    EMBEDDING_PROVIDER_OPTIONS,
    LOADER_STRATEGY_OPTIONS,
    RERANKER_STRATEGY_OPTIONS,
    RETRIEVER_STRATEGY_OPTIONS,
    build_pipeline_config,
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
    def test_default_args_match_streamlit_initial_retrieval_settings(self):
        module = load_retrieval_eval_module()

        with patch.object(sys, "argv", ["evaluate_retrieval_langsmith.py"]):
            args = module.parse_args()

        self.assertEqual(
            args.testset_path,
            module.BASE_DIR / "data" / "testsets" / "langsmith" / "retrieval_eval.jsonl",
        )
        self.assertEqual(args.loader_strategy, DEFAULT_LOADER_STRATEGY)
        self.assertEqual(args.chunker_strategy, DEFAULT_CHUNKER_STRATEGY)
        self.assertEqual(args.embedding_provider, DEFAULT_EMBEDDING_PROVIDER)
        self.assertEqual(args.retriever_strategy, "similarity")
        self.assertEqual(args.reranker_strategy, DEFAULT_RERANKER_STRATEGY)
        self.assertEqual(args.k, RETRIEVER_K)
        self.assertEqual(args.ensemble_bm25_weight, DEFAULT_ENSEMBLE_BM25_WEIGHT)
        self.assertEqual(args.ensemble_candidate_k, DEFAULT_ENSEMBLE_CANDIDATE_K)
        self.assertEqual(args.ensemble_use_chunk_id, DEFAULT_ENSEMBLE_USE_CHUNK_ID)
        self.assertFalse(args.langsmith)

    def test_retriever_and_reranker_choices_follow_streamlit_options(self):
        module = load_retrieval_eval_module()

        self.assertEqual(set(module.RETRIEVER_STRATEGY_CHOICES), {"vectorstore", *RETRIEVER_STRATEGY_OPTIONS})
        self.assertEqual(tuple(module.RERANKER_STRATEGY_CHOICES), RERANKER_STRATEGY_OPTIONS)
        self.assertIn("ensemble_parent", module.RETRIEVER_STRATEGY_CHOICES)
        self.assertIn("cross-encoder", module.RERANKER_STRATEGY_CHOICES)
        self.assertIn("llm-score", module.RERANKER_STRATEGY_CHOICES)

    def test_reranker_strategy_accepts_underscore_aliases(self):
        module = load_retrieval_eval_module()

        with patch.object(
            sys,
            "argv",
            [
                "evaluate_retrieval_langsmith.py",
                "--reranker-strategy",
                "llm_score",
            ],
        ):
            args = module.parse_args()

        self.assertEqual(args.reranker_strategy, "llm-score")

    def test_langsmith_flag_is_required_for_remote_evaluation_mode(self):
        module = load_retrieval_eval_module()

        with patch.object(sys, "argv", ["evaluate_retrieval_langsmith.py", "--langsmith"]):
            args = module.parse_args()

        self.assertTrue(args.langsmith)

    def test_validate_run_args_rejects_out_of_range_ensemble_weight(self):
        module = load_retrieval_eval_module()

        with patch.object(
            sys,
            "argv",
            ["evaluate_retrieval_langsmith.py", "--ensemble-bm25-weight", "1.5"],
        ):
            args = module.parse_args()

        with self.assertRaisesRegex(ValueError, "ensemble-bm25-weight"):
            module.validate_run_args(args)

    def test_validate_run_args_rejects_non_positive_ensemble_candidate_k(self):
        module = load_retrieval_eval_module()

        with patch.object(
            sys,
            "argv",
            ["evaluate_retrieval_langsmith.py", "--ensemble-candidate-k", "0"],
        ):
            args = module.parse_args()

        with self.assertRaisesRegex(ValueError, "ensemble-candidate-k"):
            module.validate_run_args(args)

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

    def test_evaluate_local_rows_can_run_targets_concurrently_and_preserve_order(self):
        module = load_retrieval_eval_module()
        barrier = threading.Barrier(2, timeout=1)

        def target(inputs):
            barrier.wait()
            return {
                "query": inputs["question"],
                "retrieved": [],
                "retrieved_metadata": [],
                "contexts": [],
            }

        rows = [
            {"id": "case-1", "question": "첫 번째"},
            {"id": "case-2", "question": "두 번째"},
        ]

        frame = module.evaluate_local_rows(
            rows=rows,
            target=target,
            evaluators=[],
            max_concurrency=2,
        )

        self.assertEqual(frame["example_id"].tolist(), ["case-1", "case-2"])
        self.assertEqual(frame["error"].tolist(), [None, None])
        self.assertEqual(frame["outputs.query"].tolist(), ["첫 번째", "두 번째"])

    def test_build_examples_maps_extended_testset_contract(self):
        module = load_retrieval_eval_module()

        examples = module.build_examples(
            [
                {
                    "id": "reranker_001",
                    "question": "테스트 질문",
                    "expected_diagram_ids": ["보1"],
                    "acceptable_diagram_ids": ["보1", "보3"],
                    "near_miss_diagram_ids": ["보5"],
                    "expected_evidence_keywords": ["적색신호", "횡단보도"],
                    "candidate_k": 10,
                    "final_k": 3,
                    "case_type_codes": ["RET_NEAR_MISS"],
                    "difficulty": "medium",
                    "case_family": "pedestrian_crosswalk",
                    "inference_type": "implicit_metadata",
                }
            ]
        )

        example = examples[0]
        self.assertEqual(example["inputs"]["candidate_k"], 10)
        self.assertEqual(example["inputs"]["final_k"], 3)
        self.assertEqual(example["outputs"]["expected_keywords"], ["적색신호", "횡단보도"])
        self.assertEqual(example["outputs"]["acceptable_diagram_ids"], ["보1", "보3"])
        self.assertEqual(example["outputs"]["near_miss_diagram_ids"], ["보5"])
        self.assertEqual(example["metadata"]["case_type_codes"], ["RET_NEAR_MISS"])
        self.assertEqual(example["metadata"]["difficulty"], "medium")
        self.assertEqual(example["metadata"]["case_family"], "pedestrian_crosswalk")
        self.assertEqual(example["metadata"]["inference_type"], "implicit_metadata")

    def test_diagram_id_hit_accepts_acceptable_diagram_ids(self):
        module = load_retrieval_eval_module()

        feedback = module.diagram_id_hit(
            outputs={
                "retrieved": [{"metadata": {"diagram_id": "보3"}}],
                "retrieved_metadata": [{"diagram_id": "보3"}],
            },
            reference_outputs={
                "expected_diagram_ids": ["보1"],
                "acceptable_diagram_ids": ["보1", "보3"],
            },
        )

        self.assertEqual(feedback["score"], 1)

    def test_evaluate_local_rows_exports_case_metadata_for_dashboard(self):
        module = load_retrieval_eval_module()

        frame = module.evaluate_local_rows(
            rows=[
                {
                    "id": "case_001",
                    "question": "테스트 질문",
                    "case_type_codes": ["RET_DIAGRAM"],
                    "difficulty": "hard",
                    "case_family": "car_intersection",
                    "inference_type": "explicit_keyword",
                    "expected_evidence_keywords": ["차1-4"],
                }
            ],
            target=lambda inputs: {
                "query": inputs["question"],
                "retrieved": [{"metadata": {"diagram_id": "차1-4"}, "page_content": "차1-4"}],
                "retrieved_metadata": [{"diagram_id": "차1-4"}],
                "contexts": ["차1-4"],
            },
            evaluators=[module.keyword_coverage],
        )

        self.assertEqual(frame.loc[0, "example_id"], "case_001")
        self.assertEqual(frame.loc[0, "case_type_codes"], json.dumps(["RET_DIAGRAM"], ensure_ascii=False))
        self.assertEqual(frame.loc[0, "difficulty"], "hard")
        self.assertEqual(frame.loc[0, "case_family"], "car_intersection")
        self.assertEqual(frame.loc[0, "inference_type"], "explicit_keyword")
        self.assertEqual(frame.loc[0, "feedback.keyword_coverage"], 1.0)

    def test_default_dataset_name_is_fixed_by_testset_file(self):
        module = load_retrieval_eval_module()

        dataset_name = module.make_dataset_name(
            Path("data/testsets/langsmith/retrieval_eval.jsonl"),
        )

        self.assertEqual(
            dataset_name,
            "MDM retrieval testset - retrieval_eval",
        )

    def test_dataset_name_for_run_does_not_append_matrix_run_name(self):
        module = load_retrieval_eval_module()
        args = Namespace(
            testset_path=Path("data/testsets/langsmith/retrieval_eval.jsonl"),
            dataset_name="Shared retrieval dataset",
        )
        run = {
            "name": "upstage-custom-bge",
            "loader_strategy": "upstage",
            "chunker_strategy": "custom",
            "embedding_provider": "bge",
        }

        self.assertEqual(
            module.dataset_name_for_run(args, run, matrix_mode=True),
            "Shared retrieval dataset",
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

    def test_all_strategies_arg_expands_retriever_and_reranker_product(self):
        module = load_retrieval_eval_module()

        with patch.object(sys, "argv", ["evaluate_retrieval_langsmith.py", "--all-strategies"]):
            args = module.parse_args()

        combos = module.resolve_strategy_combinations(args)

        self.assertEqual(
            len(combos),
            len(module.RETRIEVER_STRATEGY_CHOICES) * len(module.RERANKER_STRATEGY_CHOICES),
        )
        self.assertIn(
            {"retriever_strategy": "ensemble_parent", "reranker_strategy": "cross-encoder"},
            combos,
        )
        self.assertIn(
            {"retriever_strategy": args.retriever_strategy, "reranker_strategy": args.reranker_strategy},
            combos,
        )

    def test_strategy_list_args_expand_only_requested_product(self):
        module = load_retrieval_eval_module()

        with patch.object(
            sys,
            "argv",
            [
                "evaluate_retrieval_langsmith.py",
                "--retriever-strategies",
                "vectorstore,ensemble_parent",
                "--reranker-strategies",
                "none,cross_encoder,llm_score",
            ],
        ):
            args = module.parse_args()

        self.assertEqual(
            module.resolve_strategy_combinations(args),
            [
                {"retriever_strategy": "vectorstore", "reranker_strategy": "none"},
                {"retriever_strategy": "vectorstore", "reranker_strategy": "cross-encoder"},
                {"retriever_strategy": "vectorstore", "reranker_strategy": "llm-score"},
                {"retriever_strategy": "ensemble_parent", "reranker_strategy": "none"},
                {"retriever_strategy": "ensemble_parent", "reranker_strategy": "cross-encoder"},
                {"retriever_strategy": "ensemble_parent", "reranker_strategy": "llm-score"},
            ],
        )

    def test_build_execution_plan_multiplies_base_runs_by_strategy_combinations(self):
        module = load_retrieval_eval_module()
        args = Namespace(
            retriever_strategy="similarity",
            reranker_strategy="none",
            retriever_strategies="vectorstore,ensemble_parent",
            reranker_strategies="none,cross-encoder",
            all_strategies=False,
        )
        runs = [
            {
                "name": "upstage-custom-bge",
                "loader_strategy": "upstage",
                "chunker_strategy": "custom",
                "embedding_provider": "bge",
            },
            {
                "name": "upstage-raw-bge",
                "loader_strategy": "upstage",
                "chunker_strategy": "raw",
                "embedding_provider": "bge",
            },
        ]

        plan = module.build_execution_plan(args, runs)

        self.assertEqual(len(plan), 8)
        self.assertEqual(plan[0][0]["name"], "upstage-custom-bge")
        self.assertEqual(plan[0][1], {"retriever_strategy": "vectorstore", "reranker_strategy": "none"})
        self.assertEqual(plan[-1][0]["name"], "upstage-raw-bge")
        self.assertEqual(
            plan[-1][1],
            {"retriever_strategy": "ensemble_parent", "reranker_strategy": "cross-encoder"},
        )

    def test_strategy_values_reject_unknown_names(self):
        module = load_retrieval_eval_module()
        args = Namespace(
            retriever_strategy="similarity",
            reranker_strategy="none",
            retriever_strategies="missing",
            reranker_strategies=None,
            all_strategies=False,
        )

        with self.assertRaisesRegex(ValueError, "Unknown retriever strategy"):
            module.resolve_strategy_combinations(args)

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

    def test_local_run_passes_max_concurrency_to_local_evaluator(self):
        module = load_retrieval_eval_module()
        args = Namespace(
            testset_path=Path("data/testsets/langsmith/retrieval_eval.jsonl"),
            dataset_name=None,
            retriever_strategy="similarity",
            reranker_strategy="none",
            k=5,
            candidate_k=0,
            max_concurrency=4,
            upload_only=False,
            langsmith=False,
            output_dir=Path("evaluation/results/langsmith"),
            fail_on_missing_vectorstore=False,
        )
        run = {
            "name": "upstage-custom-bge",
            "loader_strategy": "upstage",
            "chunker_strategy": "custom",
            "embedding_provider": "bge",
        }

        with (
            patch.object(module, "get_vectorstore_dir", return_value=Path("vectorstore")),
            patch.object(module, "vectorstore_exists", return_value=True),
            patch.object(module, "build_retrieval_target", return_value=Mock()),
            patch.object(module, "evaluate_local_rows") as evaluate_local_rows,
            patch.object(module, "save_experiment_dataframe") as save_experiment_dataframe,
        ):
            evaluate_local_rows.return_value = Mock()
            save_experiment_dataframe.return_value = {
                "csv": Path("result.csv"),
                "summary_json": Path("result.summary.json"),
            }

            module.run_retrieval_experiment(
                args=args,
                client=None,
                rows=[{"question": "테스트"}],
                run=run,
                matrix_mode=False,
            )

        self.assertEqual(evaluate_local_rows.call_args.kwargs["max_concurrency"], 4)

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
        self.assertEqual(summary["final_k"], module.DEFAULT_K)
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
        self.assertIn(
            "upstage-custom-bge-ensemble_parent-cross-encoder",
            paths["csv"].stem,
        )

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

    def test_sanitize_metadata_converts_numpy_scalar_scores_for_flashrank(self):
        module = load_retrieval_eval_module()
        import numpy as np

        metadata = module.sanitize_metadata(
            {
                "diagram_id": "A",
                "rerank_score": np.float32(0.25),
            }
        )

        self.assertIsInstance(metadata["rerank_score"], float)
        json.dumps(metadata)

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

    def test_build_retrieval_target_uses_row_specific_reranker_k_values(self):
        module = load_retrieval_eval_module()
        vectorstore_dir = Path("data/vectorstore/upstage/custom/bge")
        seen_configs = []

        def fake_run_retrieval_pipeline(**kwargs):
            seen_configs.append(kwargs["pipeline_config"])
            return []

        with (
            patch.object(module, "get_vectorstore_dir", return_value=vectorstore_dir),
            patch.object(module, "vectorstore_exists", return_value=True),
            patch.object(module, "load_vectorstore", return_value=object()),
            patch.object(module, "build_retrieval_components", return_value=object()),
            patch.object(module, "run_retrieval_pipeline", side_effect=fake_run_retrieval_pipeline),
        ):
            target = module.build_retrieval_target(
                loader_strategy="upstage",
                embedding_provider="bge",
                chunker_strategy="custom",
                retriever_strategy="vectorstore",
                reranker_strategy="cross-encoder",
                k=5,
                candidate_k=20,
            )
            outputs = target({"question": "테스트 질문", "candidate_k": 10, "final_k": 3})

        self.assertEqual(seen_configs[0].candidate_k, 10)
        self.assertEqual(seen_configs[0].final_k, 3)
        self.assertEqual(outputs["candidate_k"], 10)
        self.assertEqual(outputs["k"], 3)

    def test_build_retrieval_target_matches_streamlit_ensemble_defaults(self):
        module = load_retrieval_eval_module()
        vectorstore_dir = Path("data/vectorstore/upstage/custom/openai")
        seen_configs = []

        def fake_run_retrieval_pipeline(**kwargs):
            seen_configs.append(kwargs["pipeline_config"])
            return []

        with (
            patch.object(module, "get_vectorstore_dir", return_value=vectorstore_dir),
            patch.object(module, "vectorstore_exists", return_value=True),
            patch.object(module, "load_vectorstore", return_value=object()),
            patch.object(module, "build_retrieval_components", return_value=object()),
            patch.object(module, "run_retrieval_pipeline", side_effect=fake_run_retrieval_pipeline),
        ):
            target = module.build_retrieval_target(
                loader_strategy="upstage",
                embedding_provider="openai",
                chunker_strategy="custom",
                retriever_strategy="ensemble_parent",
                reranker_strategy="none",
                k=module.DEFAULT_K,
            )
            outputs = target({"question": "테스트 질문"})

        expected = build_pipeline_config("ensemble_parent")
        actual = seen_configs[0]
        self.assertEqual(actual.retriever_strategy, expected.retriever_strategy)
        self.assertEqual(actual.final_k, expected.final_k)
        self.assertEqual(actual.candidate_k, expected.candidate_k)
        self.assertEqual(actual.retriever_config.weights, expected.retriever_config.weights)
        self.assertEqual(actual.retriever_config.bm25_k, expected.retriever_config.bm25_k)
        self.assertEqual(actual.retriever_config.dense_k, expected.retriever_config.dense_k)
        self.assertEqual(actual.retriever_config.id_key, expected.retriever_config.id_key)
        self.assertEqual(outputs["k"], expected.final_k)
        self.assertEqual(outputs["ensemble_candidate_k"], expected.retriever_config.bm25_k)
        self.assertEqual(outputs["ensemble_use_chunk_id"], True)


if __name__ == "__main__":
    unittest.main()
