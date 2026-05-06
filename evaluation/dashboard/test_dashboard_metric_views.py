"""Tests for dashboard metric views and charts."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.dashboard import transforms
from evaluation.dashboard.loaders import (
    discover_result_bundles,
    discover_result_sets,
    load_results,
)
from evaluation.dashboard.test_dashboard_support import write_json


class DashboardMetricViewTest(unittest.TestCase):
    def test_metric_comparison_defaults_to_one_selected_metric(self) -> None:
        from evaluation.dashboard.views import metric_comparison

        self.assertEqual(
            metric_comparison.default_metric_selection(
                ["retrieval_relevance", "critical_error", "router_overall"]
            ),
            ["critical_error"],
        )
        self.assertEqual(
            metric_comparison.default_metric_selection(
                ["router_overall", "metadata_filter_overall"]
            ),
            ["router_overall"],
        )

    def test_metric_comparison_caption_includes_metric_description(self) -> None:
        from evaluation.dashboard.views import metric_comparison

        critical_caption = metric_comparison.metric_caption("critical_error")
        router_caption = metric_comparison.metric_caption("router_overall")

        self.assertIn("critical_error", critical_caption)
        self.assertIn("낮을수록", critical_caption)
        self.assertIn("router_overall", router_caption)

    def test_metric_comparison_exposes_execution_time_with_time_caption(self) -> None:
        from evaluation.dashboard.views import metric_comparison

        available = metric_comparison._available_metrics(
            pd.DataFrame(columns=["critical_error", "execution_time"])
        )
        caption = metric_comparison.metric_caption("execution_time")

        self.assertIn("execution_time", available)
        self.assertIn("초", caption)
        self.assertIn("낮을수록", caption)

    def test_metric_comparison_group_by_options_include_short_retrieval_axes(self) -> None:
        from evaluation.dashboard.views import metric_comparison

        options = metric_comparison.available_group_by_options(
            pd.DataFrame(
                columns=[
                    "run_label",
                    "retriever_strategy",
                    "reranker_strategy",
                    "retriever_reranker",
                ]
            )
        )

        self.assertIn("retriever_strategy", options)
        self.assertIn("reranker_strategy", options)
        self.assertIn("retriever_reranker", options)
        self.assertEqual(
            metric_comparison.group_by_label("retriever_strategy"),
            "retriever",
        )
        self.assertEqual(
            metric_comparison.group_by_label("reranker_strategy"),
            "reranker",
        )
        self.assertEqual(
            metric_comparison.group_by_label("retriever_reranker"),
            "retriever + reranker",
        )
        self.assertEqual(
            metric_comparison.default_group_by_index(options),
            options.index("retriever_reranker"),
        )


    def test_metric_comparison_uses_intake_folder_defaults(self) -> None:
        from evaluation.dashboard.views import metric_comparison

        available = [
            "critical_error",
            "intake_is_sufficient",
            "party_type_match",
            "location_match",
            "missing_fields_match",
            "follow_up_contains",
            "forbidden_filter_absent",
            "intake_overall",
        ]
        options = ["run_label", "nickname", "retriever_reranker"]

        self.assertEqual(
            metric_comparison.default_metric_selection(available, "intake"),
            [
                "intake_is_sufficient",
                "party_type_match",
                "location_match",
                "missing_fields_match",
                "follow_up_contains",
                "forbidden_filter_absent",
                "intake_overall",
            ],
        )
        self.assertEqual(
            metric_comparison.default_group_by_index(options, "intake"),
            options.index("nickname"),
        )

    def test_metric_comparison_uses_folder_config_defaults(self) -> None:
        from evaluation.dashboard.views import metric_comparison

        available = [
            "diagram_id_hit",
            "retrieval_relevance",
            "critical_error",
            "keyword_coverage",
            "near_miss_not_above_expected",
            "execution_time",
            "intake_overall",
        ]
        options = ["run_label", "nickname", "retriever_reranker"]
        config = {
            "metrics": [
                "diagram_id_hit",
                "retrieval_relevance",
                "critical_error",
                "keyword_coverage",
                "near_miss_not_above_expected",
                "execution_time",
            ],
            "group_by": "nickname",
        }

        self.assertEqual(
            metric_comparison.default_metric_selection(
                available,
                "compare-cache-final",
                config,
            ),
            [
                "diagram_id_hit",
                "retrieval_relevance",
                "critical_error",
                "keyword_coverage",
                "near_miss_not_above_expected",
                "execution_time",
            ],
        )
        self.assertEqual(
            metric_comparison.default_group_by_index(
                options,
                "compare-cache-final",
                config,
            ),
            options.index("nickname"),
        )

    def test_load_results_reads_dashboard_config(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_json(
                root / "dashboard.json",
                {
                    "metric_comparison": {
                        "metrics": ["critical_error", "execution_time"],
                        "group_by": "nickname",
                    }
                },
            )
            write_json(
                root / "run-a.summary.json",
                {
                    "nickname": "Fast",
                    "run_name": "run-a",
                    "loader_strategy": "llamaparser",
                    "chunker_strategy": "case-boundary",
                    "embedding_provider": "bge",
                    "metrics": {"critical_error": 0.0, "execution_time": 1.2},
                },
            )

            results = load_results(root)

        self.assertEqual(
            results.config,
            {
                "metric_comparison": {
                    "metrics": ["critical_error", "execution_time"],
                    "group_by": "nickname",
                }
            },
        )

    def test_metric_comparison_does_not_treat_with_intake_as_intake_suite(self) -> None:
        from evaluation.dashboard.views import metric_comparison

        self.assertEqual(
            metric_comparison.default_metric_selection(
                ["critical_error", "intake_overall"],
                "with_intake",
            ),
            ["critical_error"],
        )

    def test_metric_comparison_filters_runs_by_metric_value(self) -> None:
        from evaluation.dashboard.views import metric_comparison

        summary = pd.DataFrame(
            [
                {
                    "run_name": "run-a",
                    "run_label": "same label",
                    "result_stem": "run-a-stem",
                    "critical_error": 0.0,
                    "diagram_id_hit": 1.0,
                },
                {
                    "run_name": "run-b",
                    "run_label": "same label",
                    "result_stem": "run-b-stem",
                    "critical_error": 0.5,
                    "diagram_id_hit": 0.0,
                },
            ]
        )
        metrics = transforms.build_metric_frame(summary)

        filtered_summary, filtered_metrics = metric_comparison.filter_metric_frames(
            summary,
            metrics,
            "critical_error",
            "==",
            0.0,
        )

        self.assertEqual(filtered_summary["run_name"].tolist(), ["run-a"])
        self.assertEqual(set(filtered_metrics["result_stem"]), {"run-a-stem"})
        self.assertEqual(
            set(filtered_metrics["metric"]),
            {"critical_error", "diagram_id_hit"},
        )

    def test_metric_comparison_filter_supports_range_operators(self) -> None:
        from evaluation.dashboard.views import metric_comparison

        summary = pd.DataFrame(
            [
                {"run_name": "run-a", "result_stem": "a", "critical_error": 0.0},
                {"run_name": "run-b", "result_stem": "b", "critical_error": 0.25},
                {"run_name": "run-c", "result_stem": "c", "critical_error": 0.75},
                {"run_name": "run-d", "result_stem": "d", "critical_error": None},
            ]
        )
        metrics = transforms.build_metric_frame(summary)

        filtered_summary, filtered_metrics = metric_comparison.filter_metric_frames(
            summary,
            metrics,
            "critical_error",
            "<=",
            0.25,
        )

        self.assertEqual(filtered_summary["run_name"].tolist(), ["run-a", "run-b"])
        self.assertEqual(set(filtered_metrics["result_stem"]), {"a", "b"})

        filtered_summary, _ = metric_comparison.filter_metric_frames(
            summary,
            metrics,
            "critical_error",
            "!=",
            0.0,
        )
        self.assertEqual(filtered_summary["run_name"].tolist(), ["run-b", "run-c"])

    def test_metric_comparison_defaults_to_nickname_when_present(self) -> None:
        from evaluation.dashboard.views import metric_comparison

        options = metric_comparison.available_group_by_options(
            pd.DataFrame(
                [
                    {
                        "nickname": "Pro",
                        "run_label": "upstage-custom-bge / ensemble_parent / llm-score",
                        "retriever_reranker": "ensemble_parent / llm-score",
                    }
                ]
            )
        )

        self.assertIn("nickname", options)
        self.assertEqual(metric_comparison.group_by_label("nickname"), "nickname")
        self.assertEqual(
            metric_comparison.default_group_by_index(options),
            options.index("nickname"),
        )

    def test_metric_bar_chart_uses_seconds_axis_for_execution_time(self) -> None:
        from evaluation.dashboard.charts import metric_bar_chart

        chart = metric_bar_chart(
            pd.DataFrame(
                [
                    {
                        "run_label": "run-a / similarity / none",
                        "metric": "execution_time",
                        "score": 2.25,
                    }
                ]
            ),
            metric="execution_time",
            group_by="run_label",
        )

        x_encoding = chart.to_dict()["encoding"]["x"]
        y_encoding = chart.to_dict()["encoding"]["y"]
        self.assertEqual(x_encoding["title"], "seconds")
        self.assertNotIn("domain", x_encoding.get("scale", {}))
        self.assertEqual(y_encoding["sort"], "x")
        self.assertIsNone(y_encoding.get("title"))
        self.assertGreaterEqual(y_encoding["axis"]["labelLimit"], 320)

    def test_metric_bar_chart_can_use_short_group_label(self) -> None:
        from evaluation.dashboard.charts import metric_bar_chart

        chart = metric_bar_chart(
            pd.DataFrame(
                [
                    {
                        "retriever_reranker": "ensemble_parent / cross-encoder",
                        "metric": "critical_error",
                        "score": 0.25,
                    }
                ]
            ),
            metric="critical_error",
            group_by="retriever_reranker",
            group_label="retriever + reranker",
        )

        y_encoding = chart.to_dict()["encoding"]["y"]
        self.assertIsNone(y_encoding.get("title"))

    def test_parser_chunker_heatmap_accepts_execution_time_without_score_domain(self) -> None:
        from evaluation.dashboard.charts import parser_chunker_heatmap

        chart = parser_chunker_heatmap(
            pd.DataFrame(
                [
                    {
                        "loader_strategy": "upstage",
                        "chunker_strategy": "custom",
                        "execution_time": 2.25,
                    }
                ]
            ),
            metric="execution_time",
        )

        scale = chart.to_dict()["encoding"]["color"]["scale"]
        self.assertNotIn("domain", scale)
        self.assertTrue(scale["reverse"])

    def test_metric_descriptions_cover_failure_metrics(self) -> None:
        for metric in transforms.METRIC_COLUMNS:
            with self.subTest(metric=metric):
                description = transforms.describe_metric(metric)
                self.assertIn(metric, description)
                self.assertIn("0", description)
                self.assertIn("1", description)


if __name__ == "__main__":
    unittest.main()
