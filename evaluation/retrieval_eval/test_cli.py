"""Tests for retrieval evaluation CLI defaults."""

from __future__ import annotations

import argparse
import unittest

from config import DEFAULT_RERANKER_FINAL_K, RETRIEVER_K
from evaluation.retrieval_eval.cli import effective_final_k


class RetrievalEvalCliTest(unittest.TestCase):
    def test_effective_final_k_matches_streamlit_when_k_is_not_explicit(self) -> None:
        self.assertEqual(effective_final_k("none", None), RETRIEVER_K)
        self.assertEqual(effective_final_k("llm-score", None), DEFAULT_RERANKER_FINAL_K)
        self.assertEqual(effective_final_k("cross-encoder", None), DEFAULT_RERANKER_FINAL_K)
        self.assertEqual(effective_final_k("flashrank", None), DEFAULT_RERANKER_FINAL_K)

    def test_effective_final_k_keeps_explicit_k_for_all_rerankers(self) -> None:
        self.assertEqual(effective_final_k("none", 7), 7)
        self.assertEqual(effective_final_k("llm-score", 7), 7)

    def test_args_with_strategy_preserves_unset_k_for_strategy_specific_default(self) -> None:
        from evaluation.retrieval_eval.cli import args_with_strategy

        args = argparse.Namespace(k=None, retriever_strategy="parent", reranker_strategy="none")
        run_args = args_with_strategy(args, {"reranker_strategy": "llm-score"})

        self.assertIsNone(run_args.k)
        self.assertEqual(effective_final_k(run_args.reranker_strategy, run_args.k), DEFAULT_RERANKER_FINAL_K)


if __name__ == "__main__":
    unittest.main()
