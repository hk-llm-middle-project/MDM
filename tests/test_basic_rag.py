import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from rag.chunker import chunk_text, split_documents
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
from rag.service.intake.intake_service import (
    evaluate_input_sufficiency,
    normalize_metadata_response,
)
from rag.service.result_service import format_context_preview, truncate_context


class BasicRagTest(unittest.TestCase):
    def test_chunk_text_uses_fixed_size_without_overlap(self):
        chunks = chunk_text("a" * 1201, chunk_size=500, overlap=0)

        self.assertEqual([len(chunk) for chunk in chunks], [500, 500, 201])

    def test_split_documents_does_not_add_metadata(self):
        documents = [Document(page_content="abcdef", metadata={"page": 1})]

        chunks = split_documents(documents, chunk_size=3, overlap=0)

        self.assertEqual([chunk.page_content for chunk in chunks], ["abc", "def"])
        self.assertEqual([chunk.metadata for chunk in chunks], [{}, {}])

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

        def fake_strategy(vectorstore, query, k, filters, strategy_config):
            called["args"] = (vectorstore, query, k, filters, strategy_config)
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
            (components, "query", 3, {"page": 3}, config),
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

        def fake_strategy(query, documents, k, strategy_config):
            called["args"] = (query, documents, k, strategy_config)
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
        self.assertEqual(called["args"], ("query", documents, 1, config))

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
            patch("rag.indexer.create_embeddings", return_value="embeddings"),
        ):
            vectorstore = build_vectorstore(documents, Path("vectorstore"), batch_size=2)

        self.assertEqual([len(batch) for batch in vectorstore.batches], [2, 2, 1])

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
                "location": "교차로 사고",
                "confidence": {"party_type": 0.95, "location": 0.9},
                "missing_fields": [],
                "follow_up_questions": [],
            }
        )

        self.assertTrue(decision.is_sufficient)
        self.assertEqual(decision.search_metadata.party_type, "자동차")
        self.assertEqual(decision.search_metadata.location, "교차로 사고")
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


if __name__ == "__main__":
    unittest.main()
