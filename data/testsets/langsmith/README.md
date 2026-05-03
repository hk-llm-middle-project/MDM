# LangSmith Testsets

These JSONL files evaluate internal RAG decisions, not final answer prose.

Canonical evidence source: `data/chunks/upstage/custom/chunks.json`. Legacy q1/q2/q3 datasets are wording hints only. Expected diagram IDs, evidence keywords, ratios, modifiers, and cross-references must be traceable to chunk `page_content` and metadata.

Run validation:

```bash
python3 evaluation/validate_langsmith_testsets.py
```

Run the default retrieval evaluation locally (`upstage/custom/bge`). This does
not create LangSmith traces. The default retriever follows the Streamlit app and
uses `similarity` with no reranker:

```bash
uv run python evaluation/evaluate_retrieval_langsmith.py
```

Results are written locally under:

```text
evaluation/results/langsmith/
```

Each completed run writes:

- `<timestamp>-<run-name>-<retriever>-<reranker>.csv`
- `<timestamp>-<run-name>-<retriever>-<reranker>.summary.json`

Only run through LangSmith when you explicitly need remote experiments/traces:

```bash
uv run python evaluation/evaluate_retrieval_langsmith.py --langsmith
```

List configured retrieval evaluation combinations:

```bash
uv run python evaluation/evaluate_retrieval_langsmith.py --list-matrix
```

Run the default matrix preset locally. Missing vectorstores are skipped by
default, and no LangSmith traces are created:

```bash
uv run python evaluation/evaluate_retrieval_langsmith.py --matrix
```

Run every Streamlit-selectable parser/chunker/embedder combination locally:

```bash
uv run python evaluation/evaluate_retrieval_langsmith.py --preset all
```

Evaluate the same parser/chunker/embedder matrix with a different retrieval
strategy or reranker:

```bash
uv run python evaluation/evaluate_retrieval_langsmith.py \
  --preset all \
  --retriever-strategy ensemble_parent

uv run python evaluation/evaluate_retrieval_langsmith.py \
  --preset parser-baseline \
  --retriever-strategy ensemble_parent \
  --reranker-strategy cross-encoder \
  --candidate-k 30 \
  --k 3
```

Run every configured parser/chunker/embedder combination against every exposed
retriever and reranker strategy:

```bash
uv run python evaluation/evaluate_retrieval_langsmith.py \
  --preset all \
  --all-strategies
```

Run a narrower strategy product when the full matrix is too large:

```bash
uv run python evaluation/evaluate_retrieval_langsmith.py \
  --preset parser-baseline \
  --retriever-strategies vectorstore,ensemble_parent \
  --reranker-strategies none,cross-encoder
```

Run the same matrix through LangSmith only when trace quota is acceptable:

```bash
uv run python evaluation/evaluate_retrieval_langsmith.py --preset all --langsmith
```

Fail instead of skipping missing vectorstores:

```bash
uv run python evaluation/evaluate_retrieval_langsmith.py --preset all --fail-on-missing-vectorstore
```
