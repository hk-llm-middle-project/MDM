import json
import tempfile
import threading
import time
import unittest
from pathlib import Path

from langchain_core.documents import Document


class PageMetadataClassifierTest(unittest.TestCase):
    def test_normalize_page_metadata_accepts_allowed_values(self):
        from rag.metadata.classifier import normalize_page_metadata_response

        result = normalize_page_metadata_response(
            {
                "party_type": "보행자",
                "location": "횡단보도 내",
                "confidence": {
                    "party_type": 0.91,
                    "location": 0.86,
                },
            }
        )

        self.assertEqual(result.party_type, "보행자")
        self.assertEqual(result.location, "횡단보도 내")
        self.assertEqual(result.confidence["party_type"], 0.91)
        self.assertEqual(result.confidence["location"], 0.86)

    def test_normalize_page_metadata_rejects_invalid_or_low_confidence_values(self):
        from rag.metadata.classifier import normalize_page_metadata_response

        result = normalize_page_metadata_response(
            {
                "party_type": "트럭",
                "location": "교차로 사고",
                "confidence": {
                    "party_type": 0.95,
                    "location": 0.2,
                },
            }
        )

        self.assertIsNone(result.party_type)
        self.assertIsNone(result.location)

    def test_prompt_return_schema_lists_allowed_location_values(self):
        from rag.metadata.classifier import build_page_metadata_prompt
        from rag.service.intake.values import LOCATIONS

        prompt = build_page_metadata_prompt("사고 내용")

        self.assertNotIn("허용된 location 값", prompt)
        for location in LOCATIONS:
            self.assertIn(location, prompt)
        self.assertIn('"location": "횡단보도 내" | "횡단보도 부근"', prompt)

    def test_enrich_documents_merges_classifier_metadata(self):
        from rag.metadata.classifier import enrich_documents_with_llm_metadata

        class FakeLLM:
            def invoke(self, prompt):
                self.prompt = prompt
                return (
                    '{"party_type": "자동차", "location": "교차로 사고", '
                    '"confidence": {"party_type": 0.9, "location": 0.8}}'
                )

        llm = FakeLLM()
        documents = [
            Document(
                page_content="교차로에서 자동차끼리 충돌한 사고 유형 설명",
                metadata={"source": "source.pdf", "page": 3, "parser": "pdfplumber"},
            )
        ]

        enriched = enrich_documents_with_llm_metadata(documents, llm=llm)

        self.assertEqual(len(enriched), 1)
        self.assertEqual(enriched[0].page_content, documents[0].page_content)
        self.assertEqual(enriched[0].metadata["source"], "source.pdf")
        self.assertEqual(enriched[0].metadata["party_type"], "자동차")
        self.assertEqual(enriched[0].metadata["location"], "교차로 사고")
        self.assertEqual(enriched[0].metadata["metadata_source"], "llm")
        self.assertEqual(enriched[0].metadata["metadata_confidence_party_type"], 0.9)
        self.assertEqual(enriched[0].metadata["metadata_confidence_location"], 0.8)
        self.assertNotIn("party_type", documents[0].metadata)
        self.assertIn("교차로에서 자동차끼리", llm.prompt)

    def test_enrich_documents_writes_page_metadata_cache(self):
        from rag.metadata.classifier import enrich_documents_with_llm_metadata

        class FakeLLM:
            def invoke(self, prompt):
                return (
                    '{"party_type": "보행자", "location": "횡단보도 내", '
                    '"confidence": {"party_type": 0.92, "location": 0.81}}'
                )

        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "pdfplumber_page_metadata.json"
            documents = [Document(page_content="횡단보도 보행자 사고", metadata={"page": 39})]

            enrich_documents_with_llm_metadata(documents, llm=FakeLLM(), cache_path=cache_path)

            cache = json.loads(cache_path.read_text(encoding="utf-8"))

        self.assertEqual(
            cache,
            {
                "39": {
                    "party_type": "보행자",
                    "location": "횡단보도 내",
                    "confidence": {
                        "party_type": 0.92,
                        "location": 0.81,
                    },
                }
            },
        )

    def test_enrich_documents_reuses_default_llm_for_cache_misses(self):
        from rag.metadata.classifier import enrich_documents_with_llm_metadata

        class FakeChatOpenAI:
            instances = []

            def __init__(self, model, temperature):
                self.model = model
                self.temperature = temperature
                self.invoke_count = 0
                FakeChatOpenAI.instances.append(self)

            def invoke(self, prompt):
                self.invoke_count += 1
                return (
                    '{"party_type": "자동차", "location": "교차로 사고", '
                    '"confidence": {"party_type": 0.9, "location": 0.9}}'
                )

        documents = [
            Document(page_content="교차로 자동차 사고 1", metadata={"page": 1}),
            Document(page_content="교차로 자동차 사고 2", metadata={"page": 2}),
        ]

        with unittest.mock.patch("rag.metadata.classifier.ChatOpenAI", FakeChatOpenAI):
            enrich_documents_with_llm_metadata(documents)

        self.assertEqual(len(FakeChatOpenAI.instances), 1)
        self.assertEqual(FakeChatOpenAI.instances[0].invoke_count, 2)

    def test_enrich_documents_classifies_cache_misses_concurrently(self):
        from rag.metadata.classifier import enrich_documents_with_llm_metadata

        class SlowFakeLLM:
            def __init__(self):
                self.active_count = 0
                self.max_active_count = 0
                self.lock = threading.Lock()

            def invoke(self, prompt):
                with self.lock:
                    self.active_count += 1
                    self.max_active_count = max(self.max_active_count, self.active_count)
                time.sleep(0.05)
                with self.lock:
                    self.active_count -= 1
                return (
                    '{"party_type": "자동차", "location": "교차로 사고", '
                    '"confidence": {"party_type": 0.9, "location": 0.9}}'
                )

        llm = SlowFakeLLM()
        documents = [
            Document(page_content=f"교차로 자동차 사고 {index}", metadata={"page": index})
            for index in range(1, 5)
        ]

        enriched = enrich_documents_with_llm_metadata(documents, llm=llm)

        self.assertEqual([document.metadata["page"] for document in enriched], [1, 2, 3, 4])
        self.assertGreater(llm.max_active_count, 1)

    def test_enrich_documents_uses_default_metadata_when_page_classification_fails(self):
        from rag.metadata.classifier import enrich_documents_with_llm_metadata

        class PartiallyFailingLLM:
            def invoke(self, prompt):
                if "실패 페이지" in prompt:
                    raise ValueError("invalid json")
                return (
                    '{"party_type": "자동차", "location": "교차로 사고", '
                    '"confidence": {"party_type": 0.9, "location": 0.9}}'
                )

        documents = [
            Document(page_content="실패 페이지", metadata={"page": 1}),
            Document(page_content="교차로 자동차 사고", metadata={"page": 2}),
        ]

        enriched = enrich_documents_with_llm_metadata(documents, llm=PartiallyFailingLLM())

        self.assertEqual(len(enriched), 2)
        self.assertEqual(enriched[0].metadata["metadata_source"], "llm_error")
        self.assertNotIn("party_type", enriched[0].metadata)
        self.assertNotIn("location", enriched[0].metadata)
        self.assertEqual(enriched[0].metadata["metadata_confidence_party_type"], 0.0)
        self.assertEqual(enriched[0].metadata["metadata_confidence_location"], 0.0)
        self.assertEqual(enriched[1].metadata["party_type"], "자동차")
        self.assertEqual(enriched[1].metadata["location"], "교차로 사고")

    def test_enrich_documents_uses_cached_page_metadata_without_llm_call(self):
        from rag.metadata.classifier import enrich_documents_with_llm_metadata

        class FailingLLM:
            def invoke(self, prompt):
                raise AssertionError("LLM should not be called when page metadata is cached.")

        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "llamaparser_page_metadata.json"
            cache_path.write_text(
                json.dumps(
                    {
                        "3": {
                            "party_type": "자동차",
                            "location": "교차로 사고",
                            "confidence": {
                                "party_type": 0.88,
                                "location": 0.79,
                            },
                        }
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            documents = [Document(page_content="cached page", metadata={"page": 3})]

            enriched = enrich_documents_with_llm_metadata(
                documents,
                llm=FailingLLM(),
                cache_path=cache_path,
            )

        self.assertEqual(enriched[0].metadata["party_type"], "자동차")
        self.assertEqual(enriched[0].metadata["location"], "교차로 사고")
        self.assertEqual(enriched[0].metadata["metadata_source"], "llm_cache")
        self.assertEqual(enriched[0].metadata["metadata_confidence_party_type"], 0.88)
        self.assertEqual(enriched[0].metadata["metadata_confidence_location"], 0.79)

    def test_save_page_metadata_cache_sorts_numeric_page_keys(self):
        from rag.metadata.classifier import save_page_metadata_cache

        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "main_pdf_page_metadata.json"
            save_page_metadata_cache(
                cache_path,
                {
                    "10": {"party_type": None, "location": None, "confidence": {}},
                    "2": {"party_type": None, "location": None, "confidence": {}},
                    "1": {"party_type": None, "location": None, "confidence": {}},
                },
            )

            content = cache_path.read_text(encoding="utf-8")

        self.assertLess(content.index('"1"'), content.index('"2"'))
        self.assertLess(content.index('"2"'), content.index('"10"'))


if __name__ == "__main__":
    unittest.main()
