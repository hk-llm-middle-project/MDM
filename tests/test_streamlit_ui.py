import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from langchain_core.documents import Document
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
    build_retrieved_context_metadata,
    create_progress_reporter,
    delete_session_and_select_fallback,
    get_chunker_strategy_options,
    normalize_chunker_strategy,
    render_answer_area,
    render_chat,
)
from config import DEFAULT_MODE, MODE_PRESETS, get_debug_progress_enabled
from rag.pipeline.reranker import (
    CrossEncoderRerankerConfig,
    FlashrankRerankerConfig,
    LLMScoreRerankerConfig,
)
from rag.pipeline.reranker.strategies.cross_encoder import rerank_with_cross_encoder
from rag.pipeline.retriever import EnsembleRetrieverConfig
from rag.service.analysis.answer_schema import RetrievedContext
from rag.service.intake.schema import IntakeState
from rag.service.session.schema import ChatMessage
from rag.service.session.memory_store import MemoryConversationStore


class FakeCrossEncoder:
    def __init__(self):
        self.pairs = []

    def score(self, pairs):
        self.pairs = pairs
        return [0.2, 0.9]


class FakeProgressSlot:
    def __init__(self):
        self.markdowns = []
        self.markdown_kwargs = []

    def markdown(self, label, **kwargs):
        self.markdowns.append(label)
        self.markdown_kwargs.append(kwargs)


class FakeStatusBox:
    def __init__(self):
        self.writes = []
        self.updates = []
        self.status_boxes = []
        self.status_calls = []

    def write(self, label):
        self.writes.append(label)

    def update(self, **kwargs):
        self.updates.append(kwargs)

    def status(self, label, expanded=False):
        self.status_calls.append({"label": label, "expanded": expanded})
        status_box = FakeStatusBox()
        self.status_boxes.append(status_box)
        return status_box


class FakeStreamlit:
    def __init__(self):
        self.slot = FakeProgressSlot()
        self.status_boxes = []
        self.status_calls = []
        self.empty_calls = 0

    def empty(self):
        self.empty_calls += 1
        return self.slot

    def status(self, label, expanded=False):
        self.status_calls.append({"label": label, "expanded": expanded})
        status_box = FakeStatusBox()
        self.status_boxes.append(status_box)
        return status_box


class FakeColumn:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


class FakeSessionState(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(name) from error

    def __setattr__(self, name, value):
        self[name] = value


class FakeChatStreamlit:
    def __init__(self):
        self.session_state = FakeSessionState(
            active_session="session-1",
            pending_request={
                "session_id": "session-1",
                "question": "야간 사고였어",
            },
        )
        self.columns_calls = []

    def columns(self, spec, gap=None):
        self.columns_calls.append({"spec": spec, "gap": gap})
        return [FakeColumn(), FakeColumn()]

    def chat_input(self, label):
        return None

    def rerun(self):
        raise RuntimeError("rerun")


class FakeChatStore:
    def __init__(self):
        self.messages = [
            ChatMessage(role="user", content="기존 사고 설명"),
            ChatMessage(role="assistant", content="기존 답변", metadata={}),
            ChatMessage(role="user", content="야간 사고였어"),
        ]
        self.intake_state = IntakeState()
        self.appended_messages = []

    def get_messages(self, user_id, session_id):
        return list(self.messages)

    def append_message(self, user_id, session_id, role, content, metadata=None):
        self.appended_messages.append((role, content, metadata or {}))

    def get_intake_state(self, user_id, session_id):
        return self.intake_state

    def set_intake_state(self, user_id, session_id, state):
        self.intake_state = state


class StreamlitUiTest(unittest.TestCase):
    def test_mode_presets_expose_fast_and_thinking_only(self):
        self.assertEqual(DEFAULT_MODE, "Fast")
        self.assertEqual(tuple(MODE_PRESETS), ("Fast", "Thinking"))

    def test_fast_mode_uses_upstage_google_parent_without_reranker(self):
        self.assertEqual(
            MODE_PRESETS["Fast"],
            {
                "loader_strategy": "upstage",
                "chunker_strategy": "custom",
                "embedding_provider": "google",
                "retriever_strategy": "parent",
                "reranker_strategy": "none",
                "ensemble_bm25_weight": 0.5,
                "ensemble_candidate_k": 20,
                "ensemble_use_chunk_id": True,
            },
        )

    def test_thinking_mode_uses_llamaparser_bge_parent_with_llm_score(self):
        self.assertEqual(
            MODE_PRESETS["Thinking"],
            {
                "loader_strategy": "llamaparser",
                "chunker_strategy": "case-boundary",
                "embedding_provider": "bge",
                "retriever_strategy": "parent",
                "reranker_strategy": "llm-score",
                "ensemble_bm25_weight": 0.5,
                "ensemble_candidate_k": 20,
                "ensemble_use_chunk_id": True,
            },
        )

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
        self.assertIn("ensemble_parent", RETRIEVER_STRATEGY_OPTIONS)
        self.assertNotIn("vectorstore", RETRIEVER_STRATEGY_OPTIONS)

    def test_reranker_strategy_options_expose_supported_app_choices(self):
        self.assertEqual(
            RERANKER_STRATEGY_OPTIONS,
            ("none", "cross-encoder", "flashrank", "llm-score"),
        )

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

    def test_build_pipeline_config_uses_cross_encoder_reranker(self):
        config = build_pipeline_config("similarity", reranker_strategy="cross-encoder")

        self.assertEqual(config.reranker_strategy, "cross-encoder")
        self.assertIsInstance(config.reranker_config, CrossEncoderRerankerConfig)
        self.assertEqual(config.candidate_k, DEFAULT_RERANKER_CANDIDATE_K)
        self.assertEqual(config.final_k, DEFAULT_RERANKER_FINAL_K)

    def test_cross_encoder_reranker_scores_truncated_text_and_returns_original_document(self):
        model = FakeCrossEncoder()
        documents = [
            Document(page_content="a" * 20, metadata={"id": "first"}),
            Document(page_content="b" * 20, metadata={"id": "second"}),
        ]

        reranked = rerank_with_cross_encoder(
            "query",
            documents,
            k=1,
            strategy_config=CrossEncoderRerankerConfig(
                model=model,
                max_chars=5,
                batch_size=2,
            ),
        )

        self.assertEqual(model.pairs, [("query", "a" * 5), ("query", "b" * 5)])
        self.assertEqual(reranked[0].page_content, "b" * 20)
        self.assertEqual(reranked[0].metadata["id"], "second")
        self.assertEqual(reranked[0].metadata["rerank_score"], 0.9)

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

    def test_build_pipeline_config_uses_llm_score_reranker(self):
        config = build_pipeline_config("similarity", reranker_strategy="llm-score")

        self.assertEqual(config.reranker_strategy, "llm-score")
        self.assertIsInstance(config.reranker_config, LLMScoreRerankerConfig)
        self.assertEqual(config.candidate_k, DEFAULT_RERANKER_CANDIDATE_K)
        self.assertEqual(config.final_k, DEFAULT_RERANKER_FINAL_K)
    def test_retrieved_context_metadata_uses_similarity_score_without_reranker(self):
        contexts = [
            RetrievedContext(
                content="본문에는 질문 키워드가 없습니다.",
                metadata={"similarity_score": 0.82},
            )
        ]

        rendered = build_retrieved_context_metadata(
            contexts,
            question="추돌 사고",
            reranker_strategy="none",
        )

        self.assertEqual(rendered[0]["match_percent"], 81)


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

    def test_debug_progress_env_flag_controls_verbose_status_box(self):
        with patch.dict(os.environ, {"DEBUG_PROGRESS": "true"}):
            self.assertTrue(get_debug_progress_enabled())

        with patch.dict(os.environ, {"DEBUG_PROGRESS": "false"}):
            self.assertFalse(get_debug_progress_enabled())

    def test_progress_reporter_uses_plain_text_when_debug_is_off(self):
        fake_st = FakeStreamlit()

        reporter = create_progress_reporter(False, streamlit_module=fake_st)
        reporter.update("대화 의도를 판단하는 중")
        reporter.detail("대화 의도: 사고 분석")
        reporter.complete()

        self.assertEqual(fake_st.status_calls, [])
        self.assertEqual(fake_st.empty_calls, 1)
        self.assertEqual(len(fake_st.slot.markdowns), 3)
        self.assertTrue(all("progress-spinner" in html for html in fake_st.slot.markdowns))
        self.assertTrue(all("progress-line" in html for html in fake_st.slot.markdowns))
        self.assertIn("사고 설명을 확인하는 중", fake_st.slot.markdowns[0])
        self.assertIn("대화 의도를 판단하는 중", fake_st.slot.markdowns[1])
        self.assertIn("참고 근거를 준비하는 중", fake_st.slot.markdowns[2])
        self.assertTrue(
            all(
                kwargs == {"unsafe_allow_html": True}
                for kwargs in fake_st.slot.markdown_kwargs
            )
        )

    def test_progress_reporter_uses_collapsible_status_steps_when_debug_is_on(self):
        fake_st = FakeStreamlit()

        reporter = create_progress_reporter(True, streamlit_module=fake_st)
        reporter.update("대화 의도를 판단하는 중")
        reporter.detail("대화 의도: 사고 분석")
        reporter.update("사고 단서를 추출하는 중")
        reporter.detail("사고 대상: 자동차")
        reporter.complete()

        self.assertEqual(
            fake_st.status_calls,
            [
                {"label": "분석 진행 상황", "expanded": True},
            ],
        )
        self.assertEqual(fake_st.empty_calls, 0)
        self.assertEqual(
            fake_st.status_boxes[0].status_calls,
            [
                {"label": "사고 설명을 확인하는 중", "expanded": True},
                {"label": "대화 의도를 판단하는 중", "expanded": True},
                {"label": "사고 단서를 추출하는 중", "expanded": True},
                {"label": "참고 근거를 준비하는 중", "expanded": True},
            ],
        )
        inner_status_boxes = fake_st.status_boxes[0].status_boxes
        self.assertEqual(
            inner_status_boxes[1].writes,
            ["대화 의도: 사고 분석"],
        )
        self.assertEqual(
            inner_status_boxes[2].writes,
            ["사고 대상: 자동차"],
        )
        self.assertEqual(
            inner_status_boxes[0].updates[-1],
            {"state": "complete", "expanded": False},
        )
        self.assertEqual(
            inner_status_boxes[1].updates[-1],
            {"state": "complete", "expanded": False},
        )
        self.assertEqual(
            inner_status_boxes[-1].updates[-1],
            {"state": "complete", "expanded": False},
        )
        self.assertEqual(
            fake_st.status_boxes[0].updates[-1],
            {"state": "complete", "expanded": False},
        )

    def test_render_answer_area_places_pending_progress_after_question_history(self):
        events = []
        fake_progress = SimpleNamespace(
            update=lambda label: None,
            detail=lambda detail: None,
            complete=lambda: None,
            error=lambda: None,
        )

        def record_progress():
            events.append("progress")
            return fake_progress

        with (
            patch("main.render_fault_ratio", side_effect=lambda *args: events.append("fault")),
            patch("main.render_question_history", side_effect=lambda *args: events.append("questions")),
            patch("main.render_previous_answers", side_effect=lambda *args: events.append("previous")),
            patch("main.st.markdown", side_effect=lambda *args, **kwargs: events.append("markdown")),
        ):
            reporter = render_answer_area(
                {},
                "기존 답변",
                [ChatMessage(role="user", content="Q1")],
                progress_reporter_factory=record_progress,
            )

        self.assertIs(reporter, fake_progress)
        self.assertLess(events.index("questions"), events.index("progress"))
        self.assertLess(events.index("progress"), events.index("previous"))

    def test_render_chat_uses_progress_reporter_created_inside_answer_area(self):
        fake_progress = SimpleNamespace(
            update=lambda label: None,
            detail=lambda detail: None,
            complete=lambda: None,
            error=lambda: None,
        )
        fake_result = SimpleNamespace(
            answer="새 답변",
            fault_ratio_a=None,
            fault_ratio_b=None,
            retrieved_contexts=[],
            intake_state=IntakeState(),
        )

        def return_progress_from_answer_area(*args, **kwargs):
            return kwargs["progress_reporter_factory"]()

        with (
            patch("main.st", FakeChatStreamlit()),
            patch("main.render_answer_area", side_effect=return_progress_from_answer_area),
            patch("main.render_right_panel"),
            patch("main.create_progress_reporter", return_value=fake_progress),
            patch("main.get_debug_progress_enabled", return_value=False),
            patch("main.answer_question_with_intake", return_value=fake_result) as answer_mock,
        ):
            with self.assertRaises(RuntimeError):
                render_chat(FakeChatStore())

        self.assertIs(answer_mock.call_args.kwargs["progress_callback"], fake_progress)


if __name__ == "__main__":
    unittest.main()
