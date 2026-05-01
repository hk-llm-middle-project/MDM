import json
import tempfile
import unittest
from pathlib import Path

from langchain_core.documents import Document


class PageMetadataClassifierTest(unittest.TestCase):
    def test_normalize_page_metadata_accepts_allowed_values(self):
        from rag.metadata.classifier import normalize_page_metadata_cache_entry

        result = normalize_page_metadata_cache_entry(
            {
                "party_type": "보행자",
                "location": "횡단보도 내",
            }
        )

        self.assertEqual(result.party_type, "보행자")
        self.assertEqual(result.location, "횡단보도 내")

    def test_normalize_page_metadata_rejects_invalid_values(self):
        from rag.metadata.classifier import normalize_page_metadata_cache_entry

        result = normalize_page_metadata_cache_entry(
            {
                "party_type": "트럭",
                "location": "신호등 없는 교차로",
            }
        )

        self.assertIsNone(result.party_type)
        self.assertIsNone(result.location)

    def test_build_rule_based_page_metadata_cache_uses_case_ranges(self):
        from rag.metadata.classifier import build_rule_based_page_metadata_cache

        documents = [
            Document(page_content="p39", metadata={"page": 39}),
            Document(page_content="p70", metadata={"page": 70}),
            Document(page_content="p148", metadata={"page": 148}),
            Document(page_content="p481", metadata={"page": 481}),
            Document(page_content="p501", metadata={"page": 501}),
            Document(page_content="p579", metadata={"page": 579}),
            Document(page_content="p30", metadata={"page": 30}),
        ]

        cache = build_rule_based_page_metadata_cache(documents)

        self.assertEqual(cache["39"]["party_type"], "보행자")
        self.assertEqual(cache["39"]["location"], "횡단보도 내")
        self.assertEqual(cache["70"]["location"], "횡단보도 부근")
        self.assertEqual(cache["148"]["party_type"], "자동차")
        self.assertEqual(cache["148"]["location"], "교차로 사고")
        self.assertEqual(cache["481"]["location"], "기타")
        self.assertEqual(cache["501"]["party_type"], "자전거")
        self.assertEqual(cache["501"]["location"], "교차로 사고")
        self.assertEqual(cache["579"]["location"], "기타")
        self.assertEqual(cache["30"]["party_type"], None)
        self.assertEqual(cache["30"]["location"], None)

    def test_enrich_documents_merges_cached_metadata_into_case_chunks(self):
        from rag.metadata.classifier import enrich_documents_with_page_metadata

        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "main_pdf_page_metadata.json"
            cache_path.write_text(
                json.dumps(
                    {
                        "147": {
                            "party_type": "자동차",
                            "location": "교차로 사고",
                        }
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            documents = [
                Document(
                    page_content="차1 사례",
                    metadata={"page": 147, "chunk_type": "parent", "diagram_id": "차1-1"},
                )
            ]

            enriched = enrich_documents_with_page_metadata(documents, cache_path=cache_path)

        self.assertEqual(enriched[0].metadata["party_type"], "자동차")
        self.assertEqual(enriched[0].metadata["location"], "교차로 사고")
        self.assertNotIn("metadata_confidence_party_type", enriched[0].metadata)
        self.assertNotIn("metadata_confidence_location", enriched[0].metadata)

    def test_enrich_documents_skips_general_and_preface_chunks(self):
        from rag.metadata.classifier import enrich_documents_with_page_metadata

        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "main_pdf_page_metadata.json"
            cache_path.write_text(
                json.dumps(
                    {
                        "147": {
                            "party_type": "자동차",
                            "location": "교차로 사고",
                        }
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            documents = [
                Document(page_content="총설", metadata={"page": 147, "chunk_type": "general"}),
                Document(page_content="목차", metadata={"page": 147, "chunk_type": "preface"}),
            ]

            enriched = enrich_documents_with_page_metadata(documents, cache_path=cache_path)

        self.assertNotIn("party_type", enriched[0].metadata)
        self.assertNotIn("location", enriched[0].metadata)
        self.assertNotIn("metadata_confidence_party_type", enriched[0].metadata)
        self.assertNotIn("metadata_confidence_location", enriched[0].metadata)
        self.assertNotIn("party_type", enriched[1].metadata)
        self.assertNotIn("location", enriched[1].metadata)
        self.assertNotIn("metadata_confidence_party_type", enriched[1].metadata)
        self.assertNotIn("metadata_confidence_location", enriched[1].metadata)

    def test_enrich_documents_merges_into_documents_without_chunk_type(self):
        from rag.metadata.classifier import enrich_documents_with_page_metadata

        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "main_pdf_page_metadata.json"
            cache_path.write_text(
                json.dumps(
                    {
                        "500": {
                            "party_type": "자전거",
                            "location": "교차로 사고",
                        }
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            documents = [Document(page_content="fixed chunk", metadata={"page": 500})]

            enriched = enrich_documents_with_page_metadata(documents, cache_path=cache_path)

        self.assertEqual(enriched[0].metadata["party_type"], "자전거")
        self.assertEqual(enriched[0].metadata["location"], "교차로 사고")

    def test_enrich_documents_skips_confidence_fields_for_uncategorized_pages(self):
        from rag.metadata.classifier import enrich_documents_with_page_metadata

        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "main_pdf_page_metadata.json"
            cache_path.write_text(
                json.dumps(
                    {
                        "1": {
                            "party_type": None,
                            "location": None,
                        }
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            documents = [Document(page_content="표지", metadata={"page": 1, "chunk_type": "flat"})]

            enriched = enrich_documents_with_page_metadata(documents, cache_path=cache_path)

        self.assertNotIn("party_type", enriched[0].metadata)
        self.assertNotIn("location", enriched[0].metadata)
        self.assertNotIn("metadata_confidence_party_type", enriched[0].metadata)
        self.assertNotIn("metadata_confidence_location", enriched[0].metadata)

    def test_save_page_metadata_cache_sorts_numeric_page_keys(self):
        from rag.metadata.classifier import save_page_metadata_cache

        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "main_pdf_page_metadata.json"
            save_page_metadata_cache(
                cache_path,
                {
                    "10": {"party_type": None, "location": None},
                    "2": {"party_type": None, "location": None},
                    "1": {"party_type": None, "location": None},
                },
            )

            content = cache_path.read_text(encoding="utf-8")

        self.assertLess(content.index('"1"'), content.index('"2"'))
        self.assertLess(content.index('"2"'), content.index('"10"'))

    def test_ensure_page_metadata_cache_reuses_existing_non_empty_cache(self):
        from rag.metadata.generator import ensure_page_metadata_cache

        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "main_pdf_page_metadata.json"
            existing_cache = {
                "147": {
                    "party_type": "자동차",
                    "location": "교차로 사고",
                }
            }
            cache_path.write_text(
                json.dumps(existing_cache, ensure_ascii=False),
                encoding="utf-8",
            )

            cache = ensure_page_metadata_cache(
                [Document(page_content="doc", metadata={"page": 147})],
                cache_path,
            )

        self.assertEqual(cache, existing_cache)


if __name__ == "__main__":
    unittest.main()
