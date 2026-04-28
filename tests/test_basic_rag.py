import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from rag.chunker import chunk_text, split_documents
from rag.embeddings import EMBEDDING_STRATEGIES, create_embeddings
from rag.embeddings.strategies.bge import BGEM3Embeddings
from rag.embeddings.strategies.google import GoogleGeminiEmbeddings
from rag.indexer import build_vectorstore, vectorstore_exists
from rag.pipeline.retriever import build_retrieval_components
from rag.pipeline.retrieval import RetrievalPipelineConfig, run_retrieval_pipeline
from rag.pipeline.reranker import (
    RERANKER_STRATEGIES,
    CohereRerankerConfig,
    FlashrankRerankerConfig,
    LLMScoreRerankerConfig,
    rerank,
)
from rag.pipeline.retriever import EnsembleRetrieverConfig
from rag.pipeline.retriever import RETRIEVAL_STRATEGIES, retrieve
from rag.service.conversation.app_service import answer_question, answer_question_with_intake
from rag.service.conversation.app_service import answer_question_without_intake
from rag.service.intake.filter_service import build_metadata_filters
from rag.service.intake.intake_service import (
    evaluate_input_sufficiency,
    normalize_metadata_response,
)
from rag.service.intake.schema import IntakeDecision, IntakeState, UserSearchMetadata
from rag.service.presentation.result_service import format_context_preview, truncate_context
from rag.service.tracing import TraceContext


class BasicRagTest(unittest.TestCase):
    def test_chunk_text_uses_fixed_size_without_overlap(self):
        chunks = chunk_text("a" * 1201, chunk_size=500, overlap=0)

        self.assertEqual([len(chunk) for chunk in chunks], [500, 500, 201])

    def test_split_documents_preserves_metadata(self):
        documents = [Document(page_content="abcdef", metadata={"page": 1})]

        chunks = split_documents(documents, chunk_size=3, overlap=0)

        self.assertEqual([chunk.page_content for chunk in chunks], ["abc", "def"])
        self.assertEqual([chunk.metadata for chunk in chunks], [{"page": 1}, {"page": 1}])
        self.assertIsNot(chunks[0].metadata, documents[0].metadata)

    def test_retrieve_uses_vectorstore_strategy_by_default(self):
        fake_retriever = MagicMock()
        fake_retriever.invoke.return_value = [Document(page_content="result")]
        vectorstore = MagicMock()
        vectorstore.as_retriever.return_value = fake_retriever
        components = build_retrieval_components(vectorstore)

        results = retrieve(components, "query")

        self.assertEqual(results[0].page_content, "result")
        vectorstore.as_retriever.assert_called_once_with(
            search_type="similarity",
            search_kwargs={"k": 3},
        )
        fake_retriever.invoke.assert_called_once_with("query")

    def test_retrieve_routes_strategy_config_to_selected_strategy(self):
        called = {}
        config = EnsembleRetrieverConfig(weights=(0.7, 0.3))

        def fake_strategy(vectorstore, query, k, filters, strategy_config, trace_context=None):
            called["args"] = (vectorstore, query, k, filters, strategy_config, trace_context)
            return [Document(page_content="routed")]

        with patch.dict(RETRIEVAL_STRATEGIES, {"ensemble": fake_strategy}, clear=False):
            components = build_retrieval_components(MagicMock())
            results = retrieve(
                components,
                "query",
                strategy="ensemble",
                filters={"page": 3},
                strategy_config=config,
            )

        self.assertEqual(results[0].page_content, "routed")
        self.assertEqual(
            called["args"],
            (components, "query", 3, {"page": 3}, config, None),
        )

    def test_retrieve_raises_for_unknown_strategy(self):
        with self.assertRaises(ValueError):
            retrieve(build_retrieval_components(MagicMock()), "query", strategy="unknown")

    def test_rerank_with_none_preserves_order(self):
        documents = [
            Document(page_content="first"),
            Document(page_content="second"),
            Document(page_content="third"),
        ]

        results = rerank("query", documents, k=2)

        self.assertEqual([document.page_content for document in results], ["first", "second"])

    def test_rerank_routes_strategy_config_to_selected_strategy(self):
        called = {}
        config = FlashrankRerankerConfig(model_name="test-model")

        def fake_strategy(query, documents, k, strategy_config, trace_context=None):
            called["args"] = (query, documents, k, strategy_config, trace_context)
            return [Document(page_content="reranked")]

        with patch.dict(RERANKER_STRATEGIES, {"flashrank": fake_strategy}, clear=False):
            documents = [Document(page_content="candidate")]
            results = rerank(
                "query",
                documents,
                k=1,
                strategy="flashrank",
                strategy_config=config,
            )

        self.assertEqual(results[0].page_content, "reranked")
        self.assertEqual(called["args"], ("query", documents, 1, config, None))

    def test_reranker_registry_includes_new_strategies(self):
        self.assertIn("cohere", RERANKER_STRATEGIES)
        self.assertIn("llm-score", RERANKER_STRATEGIES)

    def test_pipeline_accepts_cohere_reranker_config(self):
        pipeline_config = RetrievalPipelineConfig(
            reranker_strategy="cohere",
            reranker_config=CohereRerankerConfig(model="rerank-v3.5"),
        )

        self.assertEqual(pipeline_config.reranker_strategy, "cohere")
        self.assertIsInstance(pipeline_config.reranker_config, CohereRerankerConfig)

    def test_pipeline_accepts_llm_score_reranker_config(self):
        pipeline_config = RetrievalPipelineConfig(
            reranker_strategy="llm-score",
            reranker_config=LLMScoreRerankerConfig(model="gpt-4o-mini"),
        )

        self.assertEqual(pipeline_config.reranker_strategy, "llm-score")
        self.assertIsInstance(pipeline_config.reranker_config, LLMScoreRerankerConfig)

    def test_rerank_raises_for_unknown_strategy(self):
        with self.assertRaises(ValueError):
            rerank("query", [], k=1, strategy="unknown")

    def test_run_retrieval_pipeline_routes_retriever_and_reranker(self):
        retriever_config = EnsembleRetrieverConfig(weights=(0.6, 0.4))
        pipeline_config = RetrievalPipelineConfig(
            retriever_strategy="ensemble",
            retriever_config=retriever_config,
            reranker_strategy="flashrank",
            reranker_config=FlashrankRerankerConfig(model_name="ranker"),
            candidate_k=5,
            final_k=2,
        )
        candidate_documents = [
            Document(page_content="candidate-1"),
            Document(page_content="candidate-2"),
        ]
        final_documents = [Document(page_content="final-1")]
        components = build_retrieval_components(MagicMock())

        with (
            patch("rag.pipeline.retrieval.retrieve", return_value=candidate_documents) as retrieve_mock,
            patch("rag.pipeline.retrieval.rerank", return_value=final_documents) as rerank_mock,
        ):
            results = run_retrieval_pipeline(
                components,
                "query",
                filters={"page": 7},
                pipeline_config=pipeline_config,
            )

        self.assertEqual(results, final_documents)
        retrieve_mock.assert_called_once_with(
            components=components,
            query="query",
            k=5,
            strategy="ensemble",
            filters={"page": 7},
            strategy_config=retriever_config,
        )
        rerank_mock.assert_called_once_with(
            query="query",
            documents=candidate_documents,
            k=2,
            strategy="flashrank",
            strategy_config=pipeline_config.reranker_config,
        )

    def test_run_retrieval_pipeline_falls_back_without_filters_when_filtered_results_empty(self):
        pipeline_config = RetrievalPipelineConfig(candidate_k=5, final_k=2)
        fallback_documents = [Document(page_content="fallback")]
        final_documents = [Document(page_content="final")]
        components = build_retrieval_components(MagicMock())

        with (
            patch("rag.pipeline.retrieval.retrieve", side_effect=[[], fallback_documents]) as retrieve_mock,
            patch("rag.pipeline.retrieval.rerank", return_value=final_documents) as rerank_mock,
        ):
            results = run_retrieval_pipeline(
                components,
                "query",
                filters={"party_type": "자동차"},
                pipeline_config=pipeline_config,
            )

        self.assertEqual(results, final_documents)
        self.assertEqual(retrieve_mock.call_count, 2)
        first_call, second_call = retrieve_mock.call_args_list
        self.assertEqual(first_call.kwargs["filters"], {"party_type": "자동차"})
        self.assertIsNone(second_call.kwargs["filters"])
        rerank_mock.assert_called_once_with(
            query="query",
            documents=fallback_documents,
            k=2,
            strategy="none",
            strategy_config=None,
        )

    def test_selfquery_strategy_placeholder_raises(self):
        with self.assertRaises(NotImplementedError):
            retrieve(build_retrieval_components(MagicMock()), "query", strategy="selfquery")

    def test_build_retrieval_components_reuses_bm25_retriever(self):
        vectorstore = MagicMock()
        components = build_retrieval_components(
            vectorstore,
            source_documents=[Document(page_content="문서")],
        )
        fake_bm25 = MagicMock()

        with patch("rag.pipeline.retriever.components.BM25Retriever.from_documents", return_value=fake_bm25) as bm25_mock:
            from rag.pipeline.retriever.components import get_or_create_bm25_retriever

            first = get_or_create_bm25_retriever(components)
            second = get_or_create_bm25_retriever(components)

        self.assertIs(first, second)
        bm25_mock.assert_called_once()

    def test_vectorstore_exists_ignores_placeholder_files(self):
        self.assertFalse(vectorstore_exists(Path("missing-vectorstore")))

    def test_build_vectorstore_adds_documents_in_batches(self):
        class FakeChroma:
            def __init__(self, persist_directory, embedding_function):
                self.persist_directory = persist_directory
                self.embedding_function = embedding_function
                self.batches = []

            def add_documents(self, documents):
                self.batches.append(list(documents))

        documents = [Document(page_content=str(index)) for index in range(5)]

        with (
            patch("pathlib.Path.mkdir"),
            patch("rag.indexer.Chroma", FakeChroma),
            patch("rag.indexer.create_embeddings", return_value="embeddings") as embeddings_mock,
        ):
            vectorstore = build_vectorstore(
                documents,
                Path("vectorstore"),
                batch_size=2,
                embedding_provider="google",
            )

        self.assertEqual([len(batch) for batch in vectorstore.batches], [2, 2, 1])
        embeddings_mock.assert_called_once_with("google")

    def test_create_embeddings_returns_bge_provider(self):
        with patch.dict(EMBEDDING_STRATEGIES, {"bge": lambda: "bge"}, clear=False):
            self.assertEqual(create_embeddings("bge"), "bge")

    def test_embedding_registry_includes_current_providers(self):
        self.assertIn("openai", EMBEDDING_STRATEGIES)
        self.assertIn("bge", EMBEDDING_STRATEGIES)
        self.assertIn("google", EMBEDDING_STRATEGIES)

    def test_bge_embeddings_calls_embedding_endpoint(self):
        response = MagicMock()
        response.json.return_value = {"data": [{"dense": [0.1, 0.2]}, {"dense": [0.3, 0.4]}]}

        with (
            patch.dict("os.environ", {"BGE_BASE_URL": "https://bge.example", "BGE_API_KEY": "secret"}),
            patch("rag.embeddings.strategies.bge.requests.Session") as session_mock,
        ):
            post_mock = session_mock.return_value.post
            post_mock.return_value = response
            embeddings = BGEM3Embeddings()
            result = embeddings.embed_documents(["one", "two"])

        self.assertEqual(result, [[0.1, 0.2], [0.3, 0.4]])
        post_mock.assert_called_once_with(
            "https://bge.example/v1/embeddings/m3",
            headers={"Authorization": "Bearer secret"},
            json={"input": ["one", "two"], "return_dense": True},
            timeout=120,
        )
        response.raise_for_status.assert_called_once()

    def test_bge_embeddings_splits_document_batches(self):
        first_response = MagicMock()
        first_response.json.return_value = {
            "data": [{"dense": [float(index)]} for index in range(16)]
        }
        second_response = MagicMock()
        second_response.json.return_value = {"data": [{"dense": [16.0]}]}
        texts = [str(index) for index in range(17)]

        with (
            patch.dict("os.environ", {"BGE_BASE_URL": "https://bge.example", "BGE_API_KEY": "secret"}),
            patch("rag.embeddings.strategies.bge.requests.Session") as session_mock,
        ):
            post_mock = session_mock.return_value.post
            post_mock.side_effect = [first_response, second_response]
            embeddings = BGEM3Embeddings()
            result = embeddings.embed_documents(texts)

        self.assertEqual(result, [[float(index)] for index in range(17)])
        self.assertEqual(post_mock.call_count, 2)
        self.assertEqual(post_mock.call_args_list[0].kwargs["json"]["input"], texts[:16])
        self.assertEqual(post_mock.call_args_list[1].kwargs["json"]["input"], texts[16:])
        first_response.raise_for_status.assert_called_once()
        second_response.raise_for_status.assert_called_once()

    def test_google_embeddings_use_default_dimension(self):
        with patch("rag.embeddings.strategies.google.GoogleGenerativeAIEmbeddings") as google_mock:
            embeddings = GoogleGeminiEmbeddings()
            embeddings.embed_documents(["one", "two"])
            embeddings.embed_query("query")

        google_mock.assert_called_once_with(model="models/gemini-embedding-001")
        google_mock.return_value.embed_documents.assert_called_once_with(["one", "two"])
        google_mock.return_value.embed_query.assert_called_once_with("query")

    def test_vectorstore_service_uses_loader_specific_directory(self):
        from config import BASE_DIR
        from rag.service.vectorstore import vectorstore_service as vectorstore_service

        vectorstore_service.get_vectorstore.cache_clear()
        loaded_documents = [Document(page_content="doc", metadata={"page": 1})]
        enriched_documents = [
            Document(
                page_content="doc",
                metadata={"page": 1, "party_type": "pedestrian", "location": "crosswalk"},
            )
        ]

        with (
            patch("rag.service.vectorstore.vectorstore_service.get_vectorstore_dir", return_value=Path("vectorstore/llamaparser/google")) as dir_mock,
            patch("rag.service.vectorstore.vectorstore_service.vectorstore_exists", return_value=False),
            patch("rag.service.vectorstore.vectorstore_service.load_pdf", return_value=loaded_documents) as load_mock,
            patch(
                "rag.service.vectorstore.vectorstore_service.enrich_documents_with_llm_metadata",
                return_value=enriched_documents,
            ) as enrich_mock,
            patch("rag.service.vectorstore.vectorstore_service.split_documents", return_value=[Document(page_content="chunk")]),
            patch("rag.service.vectorstore.vectorstore_service.build_vectorstore", return_value="vectorstore") as build_mock,
        ):
            result = vectorstore_service.get_vectorstore("llamaparser", "google")

        self.assertEqual(result, "vectorstore")
        dir_mock.assert_called_once_with("llamaparser", "google")
        load_mock.assert_called_once()
        self.assertEqual(load_mock.call_args.kwargs["strategy"], "llamaparser")
        enrich_mock.assert_called_once_with(
            loaded_documents,
            cache_path=BASE_DIR / "data" / "metadata" / "main_pdf_page_metadata.json",
        )
        build_mock.assert_called_once()
        self.assertEqual(build_mock.call_args.args[1], Path("vectorstore/llamaparser/google"))
        self.assertEqual(build_mock.call_args.kwargs["embedding_provider"], "google")

        vectorstore_service.get_vectorstore.cache_clear()

    def test_vectorstore_service_loads_embedding_specific_directory(self):
        from rag.service.vectorstore import vectorstore_service

        vectorstore_service.get_vectorstore.cache_clear()

        with (
            patch("rag.service.vectorstore.vectorstore_service.get_vectorstore_dir", return_value=Path("vectorstore/pdfplumber/openai")),
            patch("rag.service.vectorstore.vectorstore_service.vectorstore_exists", return_value=True),
            patch("rag.service.vectorstore.vectorstore_service.load_vectorstore", return_value="vectorstore") as load_mock,
        ):
            result = vectorstore_service.get_vectorstore("pdfplumber", "openai")

        self.assertEqual(result, "vectorstore")
        load_mock.assert_called_once_with(
            Path("vectorstore/pdfplumber/openai"),
            embedding_provider="openai",
        )

        vectorstore_service.get_vectorstore.cache_clear()

    def test_vectorstore_service_does_not_split_upstage_final_chunks(self):
        from rag.service.vectorstore import vectorstore_service

        vectorstore_service.get_vectorstore.cache_clear()
        upstage_documents = [Document(page_content="already chunked")]

        with (
            patch("rag.service.vectorstore.vectorstore_service.get_vectorstore_dir", return_value=Path("vectorstore/upstage/bge")),
            patch("rag.service.vectorstore.vectorstore_service.vectorstore_exists", return_value=False),
            patch("rag.service.vectorstore.vectorstore_service.load_pdf", return_value=upstage_documents),
            patch("rag.service.vectorstore.vectorstore_service.enrich_documents_with_llm_metadata") as enrich_mock,
            patch("rag.service.vectorstore.vectorstore_service.split_documents") as split_mock,
            patch("rag.service.vectorstore.vectorstore_service.build_vectorstore", return_value="vectorstore") as build_mock,
        ):
            result = vectorstore_service.get_vectorstore("upstage", "bge")

        self.assertEqual(result, "vectorstore")
        enrich_mock.assert_not_called()
        split_mock.assert_not_called()
        build_mock.assert_called_once()
        self.assertEqual(build_mock.call_args.args[0], upstage_documents)

        vectorstore_service.get_vectorstore.cache_clear()

    def test_format_context_preview_returns_empty_without_contexts(self):
        self.assertEqual(format_context_preview([]), "")

    def test_format_context_preview_truncates_debug_contexts(self):
        result = format_context_preview(
            ["a" * 50, "b" * 50],
            max_context_chars=20,
        )

        self.assertIn("...(중략)", result)
        self.assertIn("[1]", result)
        self.assertIn("[2]", result)
        self.assertLessEqual(len(truncate_context("a" * 50, 20)), 20)

    def test_normalize_metadata_response_accepts_confident_allowed_values(self):
        decision = normalize_metadata_response(
            {
                "party_type": "자동차",
                "location": "신호등 없는 교차로",
                "confidence": {"party_type": 0.95, "location": 0.9},
                "missing_fields": [],
                "follow_up_questions": [],
            }
        )

        self.assertTrue(decision.is_sufficient)
        self.assertEqual(decision.search_metadata.party_type, "자동차")
        self.assertEqual(decision.search_metadata.location, "신호등 없는 교차로")
        self.assertEqual(decision.missing_fields, [])

    def test_normalize_metadata_response_rejects_invalid_or_low_confidence_values(self):
        decision = normalize_metadata_response(
            {
                "party_type": "오토바이",
                "location": "교차로 사고",
                "confidence": {"party_type": 0.95, "location": 0.3},
                "follow_up_questions": [],
            }
        )

        self.assertFalse(decision.is_sufficient)
        self.assertIsNone(decision.search_metadata.party_type)
        self.assertIsNone(decision.search_metadata.location)
        self.assertEqual([field.name for field in decision.missing_fields], ["party_type", "location"])
        self.assertTrue(decision.follow_up_questions)

    def test_evaluate_input_sufficiency_uses_llm_metadata_response(self):
        fake_llm = MagicMock()
        fake_llm.invoke.return_value = MagicMock(
            content='{"party_type":"보행자","location":"횡단보도 내","confidence":{"party_type":0.9,"location":0.9},"missing_fields":[],"follow_up_questions":[]}'
        )

        decision = evaluate_input_sufficiency("횡단보도에서 보행자와 사고가 났어요.", llm=fake_llm)

        self.assertTrue(decision.is_sufficient)
        self.assertEqual(decision.normalized_description, "횡단보도에서 보행자와 사고가 났어요.")
        self.assertEqual(decision.search_metadata.party_type, "보행자")
        self.assertEqual(decision.search_metadata.location, "횡단보도 내")
        fake_llm.invoke.assert_called_once()

    def test_evaluate_input_sufficiency_rejects_empty_input_without_llm_call(self):
        fake_llm = MagicMock()

        decision = evaluate_input_sufficiency("   ", llm=fake_llm)

        self.assertFalse(decision.is_sufficient)
        self.assertTrue(decision.follow_up_questions)
        fake_llm.invoke.assert_not_called()

    def test_build_metadata_filters_uses_party_type_and_location(self):
        filters = build_metadata_filters(
            UserSearchMetadata(
                party_type="자동차",
                location="교차로 사고",
            )
        )

        self.assertEqual(
            filters,
            {"$and": [{"party_type": "자동차"}, {"location": "교차로 사고"}]},
        )

    def test_build_metadata_filters_uses_single_condition_without_and(self):
        filters = build_metadata_filters(UserSearchMetadata(party_type="자전거"))

        self.assertEqual(filters, {"party_type": "자전거"})

    def test_build_metadata_filters_returns_none_without_values(self):
        self.assertIsNone(build_metadata_filters(UserSearchMetadata()))
        self.assertIsNone(build_metadata_filters(None))

    def test_answer_question_returns_follow_up_without_analysis_when_intake_is_insufficient(self):
        with (
            patch(
                "rag.service.conversation.app_service.evaluate_input_sufficiency",
                return_value=IntakeDecision(
                    is_sufficient=False,
                    normalized_description="사고났어요",
                    follow_up_questions=["사고 상대는 무엇인가요?"],
                ),
            ) as intake_mock,
            patch("rag.service.conversation.app_service.analyze_question") as analyze_mock,
        ):
            answer, contexts = answer_question("사고났어요")

        self.assertIn("사고 상대는 무엇인가요?", answer)
        self.assertEqual(contexts, [])
        intake_mock.assert_called_once_with("사고났어요")
        analyze_mock.assert_not_called()

    def test_answer_question_uses_llm_follow_up_questions_first(self):
        with (
            patch(
                "rag.service.conversation.app_service.evaluate_input_sufficiency",
                return_value=IntakeDecision(
                    is_sufficient=False,
                    normalized_description="사고났어요",
                    follow_up_questions=["LLM이 만든 추가 질문입니다."],
                ),
            ),
            patch("rag.service.conversation.app_service.analyze_question") as analyze_mock,
        ):
            answer, contexts = answer_question("사고났어요")

        self.assertIn("LLM이 만든 추가 질문입니다.", answer)
        self.assertNotIn("사고 상대는 보행자", answer)
        self.assertEqual(contexts, [])
        analyze_mock.assert_not_called()

    def test_answer_question_with_intake_allows_two_follow_up_attempts(self):
        with (
            patch(
                "rag.service.conversation.app_service.evaluate_input_sufficiency",
                return_value=IntakeDecision(
                    is_sufficient=False,
                    normalized_description="애매한 입력",
                    follow_up_questions=["사고 상대는 무엇인가요?"],
                ),
            ),
            patch("rag.service.conversation.app_service.analyze_question") as analyze_mock,
        ):
            result = answer_question_with_intake(
                "애매한 입력",
                intake_state=IntakeState(attempt_count=1),
            )

        self.assertTrue(result.needs_more_input)
        self.assertIn("사고 상대는 무엇인가요?", result.answer)
        self.assertEqual(result.contexts, [])
        self.assertEqual(result.intake_state.attempt_count, 2)
        analyze_mock.assert_not_called()

    def test_answer_question_with_intake_falls_back_after_two_follow_up_attempts(self):
        metadata = UserSearchMetadata()
        with (
            patch(
                "rag.service.conversation.app_service.evaluate_input_sufficiency",
                return_value=IntakeDecision(
                    is_sufficient=False,
                    normalized_description="계속 애매한 입력",
                    search_metadata=metadata,
                    follow_up_questions=["사고 상대는 무엇인가요?"],
                ),
            ),
            patch("rag.service.conversation.app_service.analyze_question", return_value=("fallback answer", ["context"])) as analyze_mock,
        ):
            result = answer_question_with_intake(
                "계속 애매한 입력",
                intake_state=IntakeState(attempt_count=2),
            )

        self.assertFalse(result.needs_more_input)
        self.assertIn("정확도가 낮을 수 있습니다", result.answer)
        self.assertIn("fallback answer", result.answer)
        self.assertEqual(result.contexts, ["context"])
        self.assertEqual(result.intake_state, IntakeState())
        analyze_mock.assert_called_once_with(
            "계속 애매한 입력",
            search_metadata=metadata,
            pipeline_config=None,
            loader_strategy="pdfplumber",
            embedding_provider="bge",
        )

    def test_answer_question_with_intake_accumulates_metadata_across_turns(self):
        previous_state = IntakeState(
            attempt_count=1,
            search_metadata=UserSearchMetadata(party_type="자동차"),
            last_missing_fields=["location"],
            last_follow_up_questions=["사고 상황은 어디에 가까운가요?"],
        )
        with (
            patch(
                "rag.service.conversation.app_service.evaluate_input_sufficiency",
                return_value=IntakeDecision(
                    is_sufficient=False,
                    normalized_description="교차로였어요",
                    search_metadata=UserSearchMetadata(location="교차로 사고"),
                ),
            ),
            patch("rag.service.conversation.app_service.analyze_question", return_value=("answer", ["context"])) as analyze_mock,
        ):
            result = answer_question_with_intake(
                "교차로였어요",
                intake_state=previous_state,
            )

        self.assertFalse(result.needs_more_input)
        self.assertEqual(result.answer, "answer")
        self.assertEqual(result.intake_state, IntakeState())
        analyze_mock.assert_called_once_with(
            "교차로였어요",
            search_metadata=UserSearchMetadata(party_type="자동차", location="교차로 사고"),
            pipeline_config=None,
            loader_strategy="pdfplumber",
            embedding_provider="bge",
        )

    def test_answer_question_passes_intake_metadata_to_analysis(self):
        metadata = UserSearchMetadata(party_type="자동차", location="교차로 사고")
        with (
            patch(
                "rag.service.conversation.app_service.evaluate_input_sufficiency",
                return_value=IntakeDecision(
                    is_sufficient=True,
                    normalized_description="정규화된 설명",
                    search_metadata=metadata,
                ),
            ),
            patch("rag.service.conversation.app_service.analyze_question", return_value=("answer", ["context"])) as analyze_mock,
        ):
            answer, contexts = answer_question("원문 입력")

        self.assertEqual(answer, "answer")
        self.assertEqual(contexts, ["context"])
        analyze_mock.assert_called_once_with(
            "정규화된 설명",
            search_metadata=metadata,
            pipeline_config=None,
            loader_strategy="pdfplumber",
            embedding_provider="bge",
        )

    def test_answer_question_passes_loader_and_embedding_strategy_to_analysis(self):
        metadata = UserSearchMetadata(party_type="자동차", location="교차로 사고")
        with (
            patch(
                "rag.service.conversation.app_service.evaluate_input_sufficiency",
                return_value=IntakeDecision(
                    is_sufficient=True,
                    normalized_description="정규화된 설명",
                    search_metadata=metadata,
                ),
            ),
            patch("rag.service.conversation.app_service.analyze_question", return_value=("answer", ["context"])) as analyze_mock,
        ):
            answer, contexts = answer_question(
                "원문 입력",
                loader_strategy="llamaparser",
                embedding_provider="google",
            )

        self.assertEqual(answer, "answer")
        self.assertEqual(contexts, ["context"])
        analyze_mock.assert_called_once_with(
            "정규화된 설명",
            search_metadata=metadata,
            pipeline_config=None,
            loader_strategy="llamaparser",
            embedding_provider="google",
        )

    def test_answer_question_without_intake_bypasses_intake(self):
        with (
            patch("rag.service.conversation.app_service.evaluate_input_sufficiency") as intake_mock,
            patch("rag.service.conversation.app_service.analyze_question", return_value=("answer", ["context"])) as analyze_mock,
        ):
            answer, contexts = answer_question_without_intake("평가용 질문")

        self.assertEqual(answer, "answer")
        self.assertEqual(contexts, ["context"])
        intake_mock.assert_not_called()
        analyze_mock.assert_called_once_with(
            "평가용 질문",
            search_metadata=None,
            pipeline_config=None,
            chat_history=None,
            trace_context=None,
        )

    def test_analyze_question_passes_metadata_filters_to_retrieval_pipeline(self):
        from rag.service.analysis.analysis_service import analyze_question

        metadata = UserSearchMetadata(party_type="자동차", location="교차로 사고")
        fake_document = Document(page_content="context")
        fake_llm = MagicMock()
        fake_llm.invoke.return_value = MagicMock(
            content='{"fault_ratio_a":70,"fault_ratio_b":30,"response":"answer"}'
        )

        with (
            patch("rag.service.analysis.analysis_service.get_retrieval_components", return_value="components"),
            patch("rag.service.analysis.analysis_service.run_retrieval_pipeline", return_value=[fake_document]) as pipeline_mock,
            patch("rag.service.analysis.analysis_service.ChatOpenAI", return_value=fake_llm),
        ):
            answer, contexts = analyze_question("query", search_metadata=metadata)

        self.assertEqual(answer, "answer")
        self.assertEqual(contexts, ["context"])
        pipeline_mock.assert_called_once_with(
            "components",
            "query",
            filters={"$and": [{"party_type": "자동차"}, {"location": "교차로 사고"}]},
            pipeline_config=None,
        )

    def test_analyze_question_uses_loader_and_embedding_strategy_for_retrieval_components(self):
        from rag.service.analysis.analysis_service import analyze_question

        fake_document = Document(page_content="context")
        fake_llm = MagicMock()
        fake_llm.invoke.return_value = MagicMock(
            content='{"fault_ratio_a":70,"fault_ratio_b":30,"response":"answer"}'
        )

        with (
            patch("rag.service.analysis.analysis_service.get_retrieval_components", return_value="components") as components_mock,
            patch("rag.service.analysis.analysis_service.run_retrieval_pipeline", return_value=[fake_document]),
            patch("rag.service.analysis.analysis_service.ChatOpenAI", return_value=fake_llm),
        ):
            answer, contexts = analyze_question(
                "query",
                loader_strategy="llamaparser",
                embedding_provider="google",
            )

        self.assertEqual(answer, "answer")
        self.assertEqual(contexts, ["context"])
        components_mock.assert_called_once_with("llamaparser", "google")

    def test_analyze_question_passes_trace_context_to_langchain_runs(self):
        from rag.service.analysis.analysis_service import analyze_question

        fake_document = Document(page_content="context")
        fake_llm = MagicMock()
        fake_llm.invoke.return_value = MagicMock(
            content='{"fault_ratio_a":70,"fault_ratio_b":30,"response":"answer"}'
        )
        trace_context = TraceContext(
            thread_id="session-1",
            session_id="session-1",
            user_id="local",
            tags=("mdm", "pdfplumber", "bge"),
            metadata={"loader_strategy": "pdfplumber", "embedding_provider": "bge"},
        )

        with (
            patch("rag.service.analysis.analysis_service.get_retrieval_components", return_value="components"),
            patch("rag.service.analysis.analysis_service.run_retrieval_pipeline", return_value=[fake_document]) as pipeline_mock,
            patch("rag.service.analysis.analysis_service.ChatOpenAI", return_value=fake_llm),
        ):
            answer, contexts = analyze_question("query", trace_context=trace_context)

        self.assertEqual(answer, "answer")
        self.assertEqual(contexts, ["context"])
        self.assertIs(pipeline_mock.call_args.kwargs["trace_context"], trace_context)
        langchain_config = fake_llm.invoke.call_args.kwargs["config"]
        self.assertEqual(langchain_config["run_name"], "mdm.answer")
        self.assertEqual(langchain_config["metadata"]["thread_id"], "session-1")
        self.assertEqual(langchain_config["metadata"]["session_id"], "session-1")
        self.assertEqual(langchain_config["metadata"]["user_id"], "local")
        self.assertEqual(langchain_config["tags"], ["mdm", "pdfplumber", "bge"])


if __name__ == "__main__":
    unittest.main()
