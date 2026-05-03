import unittest

from main import (
    CHUNKER_STRATEGY_OPTIONS_BY_LOADER,
    RETRIEVER_STRATEGY_OPTIONS,
    build_ensemble_slider_css,
    build_ensemble_weight_caption_html,
    build_ensemble_weight_label,
    build_pipeline_config,
    get_chunker_strategy_options,
    normalize_chunker_strategy,
)
from rag.pipeline.retriever import EnsembleRetrieverConfig


class StreamlitUiTest(unittest.TestCase):
    def test_build_pipeline_config_uses_selected_retriever_strategy(self):
        config = build_pipeline_config("ensemble")

        self.assertEqual(config.retriever_strategy, "ensemble")

    def test_retriever_strategy_options_expose_similarity_instead_of_vectorstore(self):
        self.assertIn("similarity", RETRIEVER_STRATEGY_OPTIONS)
        self.assertIn("ensemble_parent", RETRIEVER_STRATEGY_OPTIONS)
        self.assertNotIn("vectorstore", RETRIEVER_STRATEGY_OPTIONS)

    def test_chunker_strategy_options_are_filtered_by_loader(self):
        self.assertEqual(
            get_chunker_strategy_options("pdfplumber"),
            ("fixed", "recursive", "semantic"),
        )
        self.assertEqual(
            get_chunker_strategy_options("llamaparser"),
            ("fixed", "recursive", "markdown", "case-boundary", "semantic"),
        )
        self.assertEqual(get_chunker_strategy_options("upstage"), ("raw", "custom"))
        self.assertNotIn("case-boundary", CHUNKER_STRATEGY_OPTIONS_BY_LOADER["pdfplumber"])

    def test_normalize_chunker_strategy_falls_back_to_loader_compatible_default(self):
        self.assertEqual(normalize_chunker_strategy("case-boundary", "pdfplumber"), "fixed")
        self.assertEqual(normalize_chunker_strategy("fixed", "upstage"), "raw")
        self.assertEqual(normalize_chunker_strategy("semantic", "llamaparser"), "semantic")

    def test_build_pipeline_config_uses_ensemble_weight_slider_value(self):
        config = build_pipeline_config(
            "ensemble",
            ensemble_bm25_weight=0.7,
            ensemble_candidate_k=30,
            ensemble_use_chunk_id=False,
        )

        self.assertIsInstance(config.retriever_config, EnsembleRetrieverConfig)
        self.assertEqual(config.retriever_config.weights, (0.7, 0.3))
        self.assertEqual(config.retriever_config.bm25_k, 30)
        self.assertEqual(config.retriever_config.dense_k, 30)
        self.assertIsNone(config.retriever_config.id_key)

    def test_build_pipeline_config_uses_ensemble_parent_options(self):
        config = build_pipeline_config(
            "ensemble_parent",
            ensemble_bm25_weight=0.6,
            ensemble_candidate_k=10,
            ensemble_use_chunk_id=True,
        )

        self.assertEqual(config.retriever_strategy, "ensemble_parent")
        self.assertIsInstance(config.retriever_config, EnsembleRetrieverConfig)
        self.assertEqual(config.retriever_config.weights, (0.6, 0.4))
        self.assertEqual(config.retriever_config.bm25_k, 10)
        self.assertEqual(config.retriever_config.dense_k, 10)
        self.assertEqual(config.retriever_config.id_key, "chunk_id")

    def test_build_pipeline_config_skips_ensemble_config_for_other_strategies(self):
        config = build_pipeline_config("similarity", ensemble_bm25_weight=0.7)

        self.assertIsNone(config.retriever_config)

    def test_build_ensemble_weight_label_is_short(self):
        label = build_ensemble_weight_label(0.65)

        self.assertEqual(label, "앙상블 검색 가중치")

    def test_build_ensemble_weight_caption_html_shows_percentages_below_slider(self):
        html = build_ensemble_weight_caption_html(0.65)

        self.assertIn("BM25 65%", html)
        self.assertIn("Dense 35%", html)
        self.assertIn("justify-content: space-between", html)

    def test_build_ensemble_slider_css_colors_slider_track_by_weight(self):
        css = build_ensemble_slider_css(0.65)

        self.assertIn("#2563eb 0%", css)
        self.assertIn("#2563eb 65%", css)
        self.assertIn("#059669 65%", css)
        self.assertIn("#059669 100%", css)
        self.assertIn("[role=\"slider\"]", css)
        self.assertIn("width: 4px", css)
        self.assertIn("[data-testid=\"stTickBar\"]", css)
        self.assertIn("display: none", css)


if __name__ == "__main__":
    unittest.main()
