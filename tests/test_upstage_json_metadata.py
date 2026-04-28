import json
import unittest
from collections import Counter

from config import UPSTAGE_FINAL_DOCUMENTS_PATH
from rag.service.intake.values import LOCATIONS, PARTY_TYPES


class UpstageJsonMetadataTest(unittest.TestCase):
    def test_party_type_values_match_intake_allowed_values(self):
        chunks = self._load_chunks()

        invalid_values = self._invalid_metadata_values(chunks, "party_type", PARTY_TYPES)

        self._assert_no_invalid_values("party_type", invalid_values)

    def test_location_values_match_intake_allowed_values(self):
        chunks = self._load_chunks()

        invalid_values = self._invalid_metadata_values(chunks, "location", LOCATIONS)

        self._assert_no_invalid_values("location", invalid_values)

    def _load_chunks(self):
        with UPSTAGE_FINAL_DOCUMENTS_PATH.open("r", encoding="utf-8") as fp:
            return json.load(fp)

    def _invalid_metadata_values(
        self,
        chunks: list[dict],
        metadata_key: str,
        allowed_values: list[str],
    ) -> dict[str, int]:
        allowed = set(allowed_values)
        counts = Counter(
            chunk.get("metadata", {}).get(metadata_key)
            for chunk in chunks
            if chunk.get("metadata", {}).get(metadata_key) is not None
        )
        return {
            value: count
            for value, count in sorted(counts.items())
            if value not in allowed
        }

    def _assert_no_invalid_values(self, metadata_key: str, invalid_values: dict[str, int]) -> None:
        if invalid_values:
            print(f"\nInvalid {metadata_key} values:")
            for value, count in invalid_values.items():
                print(f"- {value}: {count}")

        self.assertEqual(
            invalid_values,
            {},
            f"Invalid {metadata_key} values: {invalid_values}",
        )


if __name__ == "__main__":
    unittest.main()
