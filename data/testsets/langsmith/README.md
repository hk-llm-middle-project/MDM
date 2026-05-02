# LangSmith Testsets

These JSONL files evaluate internal RAG decisions, not final answer prose.

Canonical evidence source: `data/chunks/upstage/custom/chunks.json`. Legacy q1/q2/q3 datasets are wording hints only. Expected diagram IDs, evidence keywords, ratios, modifiers, and cross-references must be traceable to chunk `page_content` and metadata.

Run validation:

```bash
python3 evaluation/validate_langsmith_testsets.py
```

Run the default retrieval LangSmith evaluation (`upstage/custom/bge`):

```bash
uv run python evaluation/evaluate_retrieval_langsmith.py
```

Results are stored in LangSmith and summarized locally under:

```text
evaluation/results/langsmith/
```

Each completed run writes:

- `<timestamp>-<run-name>.csv`
- `<timestamp>-<run-name>.summary.json`

List configured retrieval evaluation combinations:

```bash
uv run python evaluation/evaluate_retrieval_langsmith.py --list-matrix
```

Run the default matrix preset (`upstage/custom/bge`, `upstage/raw/bge`, `upstage/raw/openai`):

```bash
uv run python evaluation/evaluate_retrieval_langsmith.py --matrix
```

Run every Streamlit-selectable parser/chunker/embedder combination. Missing vectorstores are skipped by default:

```bash
uv run python evaluation/evaluate_retrieval_langsmith.py --preset all
```

Fail instead of skipping missing vectorstores:

```bash
uv run python evaluation/evaluate_retrieval_langsmith.py --preset all --fail-on-missing-vectorstore
```
