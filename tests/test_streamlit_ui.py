import unittest

from main import (
    CHUNKER_STRATEGY_OPTIONS_BY_LOADER,
    DEFAULT_RERANKER_CANDIDATE_K,
    DEFAULT_RERANKER_FINAL_K,
    RERANKER_STRATEGY_OPTIONS,
    RETRIEVER_STRATEGY_OPTIONS,
    build_ensemble_slider_css,
    build_ensemble_weight_caption_html,
    build_ensemble_weight_label,
    build_pipeline_config,
    delete_session_and_select_fallback,
    get_chunker_strategy_options,
    normalize_chunker_strategy,
)
from rag.pipeline.reranker import CrossEncoderRerankerConfig, FlashrankRerankerConfig
from rag.pipeline.retriever import EnsembleRetrieverConfig
from rag.service.session.memory_store import MemoryConversationStore


class StreamlitUiTest(unittest.TestCase):
    def test_build_pipeline_config_uses_selected_retriever_strategy(self):
        config = build_pipeline_config("ensemble")

        self.assertEqual(config.retriever_strategy, "ensemble")

    def test_delete_session_and_select_fallback_uses_remaining_session(self):
        store = MemoryConversationStore()
        first = store.create_session("local", title="첫 세션")
        second = store.create_session("local", title="둘째 세션")
        store.set_active_session("local", first.session_id)

        active_session = delete_session_and_select_fallback(store, first.session_id)

        self.assertEqual(active_session, second.session_id)
        self.assertEqual(store.get_active_session("local"), second.session_id)
        self.assertEqual(store.list_sessions("local"), [second])

    def test_delete_session_and_select_fallback_creates_session_when_empty(self):
        store = MemoryConversationStore()
        session = store.create_session("local", title="마지막 세션")
        store.set_active_session("local", session.session_id)

        active_session = delete_session_and_select_fallback(store, session.session_id)

        sessions = store.list_sessions("local")
        self.assertEqual(len(sessions), 1)
        self.assertEqual(active_session, sessions[0].session_id)
        self.assertEqual(store.get_active_session("local"), sessions[0].session_id)

    def test_retriever_strategy_options_expose_similarity_instead_of_vectorstore(self):
        self.assertIn("similarity", RETRIEVER_STRATEGY_OPTIONS)
        self.assertNotIn("vectorstore", RETRIEVER_STRATEGY_OPTIONS)

    def test_reranker_strategy_options_expose_supported_app_choices(self):
        self.assertEqual(RERANKER_STRATEGY_OPTIONS, ("none", "cross-encoder", "flashrank"))

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
        config = build_pipeline_config("ensemble", ensemble_bm25_weight=0.7)

        self.assertIsInstance(config.retriever_config, EnsembleRetrieverConfig)
        self.assertEqual(config.retriever_config.weights, (0.7, 0.3))

    def test_build_pipeline_config_skips_ensemble_config_for_other_strategies(self):
        config = build_pipeline_config("similarity", ensemble_bm25_weight=0.7)

        self.assertIsNone(config.retriever_config)

    def test_build_pipeline_config_uses_cross_encoder_reranker(self):
        config = build_pipeline_config("similarity", reranker_strategy="cross-encoder")

        self.assertEqual(config.reranker_strategy, "cross-encoder")
        self.assertIsInstance(config.reranker_config, CrossEncoderRerankerConfig)
        self.assertEqual(config.candidate_k, DEFAULT_RERANKER_CANDIDATE_K)
        self.assertEqual(config.final_k, DEFAULT_RERANKER_FINAL_K)

    def test_build_pipeline_config_uses_flashrank_reranker_with_ensemble(self):
        config = build_pipeline_config(
            "ensemble",
            ensemble_bm25_weight=0.7,
            reranker_strategy="flashrank",
        )

        self.assertEqual(config.retriever_strategy, "ensemble")
        self.assertIsInstance(config.retriever_config, EnsembleRetrieverConfig)
        self.assertEqual(config.reranker_strategy, "flashrank")
        self.assertIsInstance(config.reranker_config, FlashrankRerankerConfig)
        self.assertEqual(config.candidate_k, DEFAULT_RERANKER_CANDIDATE_K)
        self.assertEqual(config.final_k, DEFAULT_RERANKER_FINAL_K)

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
