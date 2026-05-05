# Evaluation Dashboard

Local dashboard for LangSmith evaluation result exports.

Run:

```bash
uv run streamlit run evaluation/dashboard/app.py
```

The dashboard reads result-set folders under `evaluation/results/`; by default,
evaluation scripts write uncategorized runs to
`evaluation/results/uncategorized/*.summary.json` with sibling CSV files. LangSmith remains the source for
trace-level details; this dashboard focuses on combination-level comparison,
metric charts, and failed-row lookup.

Result summaries may include parser, chunker, embedder, retriever, and reranker
metadata. The dashboard treats `run_name / retriever_strategy /
reranker_strategy` as the visible run label so repeated parser/chunker/embedder
runs with different retrieval strategies do not collapse into one column.

Each result-set folder can override dashboard defaults with a sibling
`dashboard.json` file:

```json
{
  "metric_comparison": {
    "group_by": "nickname",
    "metrics": [
      "diagram_id_hit",
      "retrieval_relevance",
      "critical_error",
      "keyword_coverage",
      "near_miss_not_above_expected",
      "execution_time"
    ]
  }
}
```

Useful local evaluation commands:

```bash
uv run python evaluation/evaluate_retrieval_langsmith.py --preset all
uv run python evaluation/evaluate_retrieval_langsmith.py --preset all --retriever-strategy ensemble_parent
uv run python evaluation/evaluate_retrieval_langsmith.py --preset parser-baseline --retriever-strategy ensemble_parent --reranker-strategy cross-encoder --candidate-k 30 --k 3
uv run python evaluation/evaluate_retrieval_langsmith.py --preset all --all-strategies
uv run python evaluation/evaluate_retrieval_langsmith.py --preset parser-baseline --retriever-strategies vectorstore,ensemble_parent --reranker-strategies none,cross-encoder
uv run python evaluation/evaluate_decision_suites.py --suite all
```
