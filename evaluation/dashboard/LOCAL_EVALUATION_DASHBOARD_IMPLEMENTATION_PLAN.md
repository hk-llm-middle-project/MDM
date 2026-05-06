# Local Evaluation Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Separate production conversation tracing from offline evaluation so LangSmith is used only for real Streamlit user sessions, while repeated parser/chunker/embedder evaluation is inspected through a local dashboard with detailed pass/fail views.

**Architecture:** LangSmith remains the observability backend for real conversations and small targeted debugging runs. Offline evaluation writes local CSV/JSON artifacts under `evaluation/results/langsmith/`, and `evaluation/dashboard/` reads those artifacts to show combination summaries, test-case pass/fail matrices, failure explorers, and case-level expected-vs-actual comparisons. Trace-level detail for tests is added later as optional local JSONL, not as LangSmith runs.

**Tech Stack:** Python 3.11, Streamlit, pandas, Altair, existing retrieval evaluator in `evaluation/evaluate_retrieval_langsmith.py`, existing result files in `evaluation/results/langsmith/`, existing test runner style using `uv run python tests/<file>.py`.

---

## Current Baseline

The project already has these pieces:

- `evaluation/evaluate_retrieval_langsmith.py`
  - Runs retrieval evaluation for one parser/chunker/embedder combination or a matrix preset.
  - Default mode should stay local-only and write CSV/summary JSON.
  - LangSmith should only be used when `--langsmith` is explicitly passed.
- `evaluation/results/langsmith/*.csv`
  - Example-level outputs from evaluation runs.
  - Includes input question, retrieved outputs, references, `feedback.*` score columns, and `feedback.*.comment` columns.
- `evaluation/results/langsmith/*.summary.json`
  - Combination-level metadata and averaged feedback metrics.
- `evaluation/dashboard/`
  - Basic Streamlit dashboard exists.
  - Currently useful for summary metrics, simple charts, and failure rows.
- LangSmith
  - Reserved for real user Streamlit conversation tracing and occasional small manual debug runs.
  - Full matrix evaluation must not use LangSmith by default.

## Product Direction

Use two separate evaluation/observability modes:

```text
Real Streamlit user sessions
  -> LangSmith tracing
  -> thread/session view, step input/output/attributes
  -> pay-as-you-go acceptable because volume is real usage

Offline evaluation matrix
  -> local CSV/JSON results
  -> local dashboard
  -> no LangSmith traces, no monthly trace burn
```

The local dashboard should answer:

- Which combination is best overall?
- Which combinations fail on each test case?
- Which metric causes failures?
- For a given failed case, what was expected and what did each combination actually retrieve?
- Which two combinations differ on the same test case?
- Which failures are systematic by parser, chunker, embedder, case type, or difficulty?

## Non-Goals

- Do not rebuild a full LangSmith clone.
- Do not store full unbounded retrieved context in dashboard state.
- Do not send matrix evaluation traces to LangSmith by default.
- Do not migrate to Langfuse in this implementation.
- Do not evaluate final natural-language answer quality in this phase.

## Target File Structure

```text
evaluation/
  dashboard/
    __init__.py
    app.py
    charts.py
    loaders.py
    transforms.py
    views/
      __init__.py
      overview.py
      metric_comparison.py
      test_case_matrix.py
      failure_explorer.py
      case_detail.py
      combo_compare.py
    README.md
  evaluate_retrieval_langsmith.py
  results/
    langsmith/
      <timestamp>-<run-name>.csv
      <timestamp>-<run-name>.summary.json
      traces/
        <timestamp>-<run-name>.jsonl
tests/
  test_evaluation_dashboard.py
  test_retrieval_langsmith_eval.py
```

## Data Contract

### Summary JSON

Each summary file should contain:

```json
{
  "experiment_name": "MDM retrieval eval - upstage-custom-bge-vectorstore-none-local",
  "dataset_name": "MDM retrieval testset - upstage-custom-bge - retrieval_eval",
  "testset_path": "/home/nyong/mdm/data/testsets/langsmith/retrieval_eval.jsonl",
  "run_name": "upstage-custom-bge",
  "loader_strategy": "upstage",
  "chunker_strategy": "custom",
  "embedding_provider": "bge",
  "retriever_strategy": "vectorstore",
  "reranker_strategy": "none",
  "row_count": 30,
  "metrics": {
    "diagram_id_hit": 0.73,
    "location_match": 1.0,
    "party_type_match": 1.0,
    "chunk_type_match": 1.0,
    "keyword_coverage": null,
    "retrieval_relevance": 0.91,
    "critical_error": 0.27
  }
}
```

Required fields for dashboard grouping:

- `run_name`
- `loader_strategy`
- `chunker_strategy`
- `embedding_provider`
- `row_count`
- `metrics`

Recommended additional fields:

- `retriever_strategy`
- `reranker_strategy`
- `dataset_name`
- `testset_path`
- `experiment_name`

### Evaluation CSV

Each CSV row represents one test case under one combination.

Required columns:

```text
inputs.question
outputs.query
outputs.retrieved
outputs.retrieved_metadata
outputs.contexts
reference.expected_diagram_ids
reference.expected_location
reference.expected_party_type
reference.expected_chunk_types
reference.expected_keywords
feedback.diagram_id_hit
feedback.location_match
feedback.party_type_match
feedback.chunk_type_match
feedback.keyword_coverage
feedback.retrieval_relevance
feedback.critical_error
execution_time
example_id
```

Recommended columns:

```text
feedback.diagram_id_hit.comment
feedback.location_match.comment
feedback.party_type_match.comment
feedback.chunk_type_match.comment
feedback.keyword_coverage.comment
feedback.retrieval_relevance.comment
feedback.critical_error.comment
```

If `example_id` is missing, the dashboard must derive a stable key from `inputs.question`.

### Local Trace JSONL

This is optional and should be implemented only after the dashboard pass/fail views are useful.

Each line should represent one evaluated example:

```json
{
  "run_name": "upstage-custom-bge",
  "example_id": "retrieval_001",
  "question": "신호기 없는 교차로에서...",
  "events": [
    {
      "step": "mdm.retrieve.vectorstore",
      "latency_ms": 123.4,
      "inputs": {
        "k": 5,
        "filters": null
      },
      "outputs": {
        "count": 5,
        "top_documents": [
          {
            "rank": 1,
            "diagram_id": "차16-1",
            "page": 123,
            "chunk_type": "case",
            "party_type": "자동차",
            "location": "교차로",
            "preview": "..."
          }
        ]
      }
    },
    {
      "step": "mdm.rerank.none",
      "latency_ms": 1.7,
      "outputs": {
        "final_count": 5,
        "top_documents": []
      }
    }
  ]
}
```

Do not store full page content by default. Store metadata and a short preview.

## Implementation Tasks

### Task 1: Keep LangSmith Off For Offline Evaluation By Default

**Files:**

- Modify: `evaluation/evaluate_retrieval_langsmith.py`
- Modify: `data/testsets/langsmith/README.md`
- Test: `tests/test_retrieval_langsmith_eval.py`

- [ ] **Step 1: Add tests for local default mode**

Add or keep tests that assert:

```python
def test_default_args_do_not_enable_langsmith(self):
    module = load_retrieval_eval_module()
    with patch.object(sys, "argv", ["evaluate_retrieval_langsmith.py"]):
        args = module.parse_args()
    self.assertFalse(args.langsmith)


def test_langsmith_flag_enables_remote_mode(self):
    module = load_retrieval_eval_module()
    with patch.object(sys, "argv", ["evaluate_retrieval_langsmith.py", "--langsmith"]):
        args = module.parse_args()
    self.assertTrue(args.langsmith)
```

- [ ] **Step 2: Add tests that tracing env vars are disabled in local mode**

```python
def test_configure_tracing_disables_langsmith_tracing_for_local_mode(self):
    module = load_retrieval_eval_module()
    with patch.dict(
        os.environ,
        {
            "LANGSMITH_TRACING": "true",
            "LANGCHAIN_TRACING_V2": "true",
        },
        clear=False,
    ):
        module.configure_tracing(needs_langsmith=False)
        self.assertEqual(os.environ["LANGSMITH_TRACING"], "false")
        self.assertEqual(os.environ["LANGCHAIN_TRACING_V2"], "false")
```

- [ ] **Step 3: Verify the tests fail before implementation**

Run:

```bash
uv run python tests/test_retrieval_langsmith_eval.py
```

Expected:

```text
FAILED
AttributeError or argparse error related to args.langsmith/configure_tracing
```

- [ ] **Step 4: Implement local default mode**

In `parse_args()` add:

```python
parser.add_argument(
    "--langsmith",
    action="store_true",
    help=(
        "Run through LangSmith evaluate(). By default this script evaluates "
        "locally and writes CSV/JSON only to avoid consuming trace quota."
    ),
)
```

Add:

```python
def configure_tracing(needs_langsmith: bool) -> None:
    if needs_langsmith:
        return
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
```

In `main()`:

```python
needs_langsmith = bool(args.langsmith or args.upload_only)
configure_tracing(needs_langsmith)
if needs_langsmith and not os.getenv("LANGSMITH_API_KEY"):
    raise RuntimeError("LANGSMITH_API_KEY is required.")
client = Client() if needs_langsmith else None
```

- [ ] **Step 5: Implement local evaluation path**

Add a function that runs target/evaluators locally and returns a DataFrame:

```python
def evaluate_local_rows(rows, target, evaluators) -> pd.DataFrame:
    records = []
    examples = build_examples(rows)
    for example in examples:
        inputs = example["inputs"]
        reference_outputs = example["outputs"]
        outputs = target(inputs)
        record = {
            "inputs.question": inputs["question"],
            "outputs.query": outputs.get("query"),
            "outputs.retrieved": json.dumps(outputs.get("retrieved", []), ensure_ascii=False),
            "outputs.retrieved_metadata": json.dumps(outputs.get("retrieved_metadata", []), ensure_ascii=False),
            "outputs.contexts": json.dumps(outputs.get("contexts", []), ensure_ascii=False),
            "example_id": example.get("metadata", {}).get("id"),
        }
        for key, value in reference_outputs.items():
            record[f"reference.{key}"] = json.dumps(value, ensure_ascii=False) if isinstance(value, list | dict) else value
        for evaluator in evaluators:
            feedback = evaluator(outputs, reference_outputs)
            record[f"feedback.{feedback['key']}"] = feedback.get("score")
            record[f"feedback.{feedback['key']}.comment"] = feedback.get("comment")
        records.append(record)
    return pd.DataFrame.from_records(records)
```

Use the project's existing implementation style if this function already exists.

- [ ] **Step 6: Save local output with the same summary schema**

Create or reuse:

```python
def save_experiment_dataframe(dataframe, experiment_name, output_dir, run, dataset_name, testset_path):
    ...
```

It must write:

```text
<timestamp>-<run-name>.csv
<timestamp>-<run-name>.summary.json
```

- [ ] **Step 7: Update docs**

In `data/testsets/langsmith/README.md`, make these commands explicit:

```bash
# local only, no LangSmith traces
uv run python evaluation/evaluate_retrieval_langsmith.py --preset all

# remote LangSmith only when explicitly needed
uv run python evaluation/evaluate_retrieval_langsmith.py --preset all --langsmith
```

- [ ] **Step 8: Verify**

Run:

```bash
uv run python tests/test_retrieval_langsmith_eval.py
uv run python evaluation/evaluate_retrieval_langsmith.py --max-examples 1 --output-dir /tmp/mdm-local-eval
```

Expected:

```text
LangSmith skipped. Local result files were written.
```

### Task 2: Normalize Dashboard Tables For Test-Case Views

**Files:**

- Modify: `evaluation/dashboard/transforms.py`
- Modify: `evaluation/dashboard/loaders.py`
- Test: `tests/test_evaluation_dashboard.py`

- [ ] **Step 1: Add tests for stable example keys**

```python
def test_build_example_frame_derives_case_key_from_example_id_or_question(self):
    ...
    frame = build_example_frame(discover_result_bundles(root))
    self.assertIn("case_key", frame.columns)
    self.assertEqual(frame.loc[0, "case_key"], "retrieval_001")
```

If `example_id` is missing:

```python
self.assertTrue(frame.loc[0, "case_key"].startswith("question:"))
```

- [ ] **Step 2: Add tests for feedback score/comment aliasing**

```python
def test_build_example_frame_adds_feedback_score_and_comment_aliases(self):
    ...
    self.assertEqual(frame.loc[0, "critical_error"], 1)
    self.assertEqual(frame.loc[0, "critical_error_comment"], "1 means ...")
```

- [ ] **Step 3: Verify failing tests**

Run:

```bash
uv run python tests/test_evaluation_dashboard.py
```

Expected:

```text
FAILED because case_key/comment alias columns do not exist
```

- [ ] **Step 4: Implement `case_key`**

In `build_example_frame()`:

```python
def make_case_key(row: pd.Series) -> str:
    example_id = row.get("example_id")
    if pd.notna(example_id) and str(example_id).strip():
        return str(example_id)
    question = str(row.get("inputs.question", "")).strip()
    return f"question:{hashlib.sha1(question.encode('utf-8')).hexdigest()[:12]}"
```

Apply:

```python
frame["case_key"] = frame.apply(make_case_key, axis=1)
```

- [ ] **Step 5: Implement score/comment aliases**

For every metric in `METRIC_COLUMNS`:

```python
score_column = f"feedback.{metric_name}"
comment_column = f"feedback.{metric_name}.comment"
if score_column in frame.columns:
    frame[metric_name] = pd.to_numeric(frame[score_column], errors="coerce")
if comment_column in frame.columns:
    frame[f"{metric_name}_comment"] = frame[comment_column].fillna("")
```

- [ ] **Step 6: Verify**

Run:

```bash
uv run python tests/test_evaluation_dashboard.py
```

Expected:

```text
OK
```

### Task 3: Add Test Case Matrix View

**Files:**

- Create: `evaluation/dashboard/views/__init__.py`
- Create: `evaluation/dashboard/views/test_case_matrix.py`
- Modify: `evaluation/dashboard/app.py`
- Modify: `evaluation/dashboard/transforms.py`
- Test: `tests/test_evaluation_dashboard.py`

- [ ] **Step 1: Add transform tests for matrix data**

Add:

```python
from evaluation.dashboard.transforms import build_case_metric_matrix

def test_build_case_metric_matrix_pivots_cases_by_run(self):
    examples = pd.DataFrame([
        {"case_key": "retrieval_001", "inputs.question": "q1", "run_name": "upstage-custom-bge", "critical_error": 0},
        {"case_key": "retrieval_001", "inputs.question": "q1", "run_name": "upstage-raw-bge", "critical_error": 1},
    ])
    matrix = build_case_metric_matrix(examples, metric="critical_error")
    self.assertEqual(matrix.loc[0, "upstage-custom-bge"], 0)
    self.assertEqual(matrix.loc[0, "upstage-raw-bge"], 1)
```

- [ ] **Step 2: Verify failing test**

Run:

```bash
uv run python tests/test_evaluation_dashboard.py
```

Expected:

```text
FAILED because build_case_metric_matrix is missing
```

- [ ] **Step 3: Implement `build_case_metric_matrix()`**

In `transforms.py`:

```python
def build_case_metric_matrix(examples: pd.DataFrame, metric: str) -> pd.DataFrame:
    if examples.empty or metric not in examples.columns:
        return pd.DataFrame()
    base = examples[["case_key", "inputs.question", "run_name", metric]].copy()
    pivot = base.pivot_table(
        index=["case_key", "inputs.question"],
        columns="run_name",
        values=metric,
        aggfunc="first",
    ).reset_index()
    pivot.columns = [str(column) for column in pivot.columns]
    return pivot
```

- [ ] **Step 4: Create matrix view**

In `evaluation/dashboard/views/test_case_matrix.py`:

```python
import pandas as pd
import streamlit as st

from evaluation.dashboard.transforms import METRIC_COLUMNS, build_case_metric_matrix


def render(examples: pd.DataFrame) -> None:
    st.subheader("Test Case Matrix")
    if examples.empty:
        st.info("No example-level results found.")
        return

    metrics = [metric for metric in METRIC_COLUMNS if metric in examples.columns]
    metric = st.selectbox(
        "Matrix metric",
        metrics,
        index=metrics.index("critical_error") if "critical_error" in metrics else 0,
    )
    matrix = build_case_metric_matrix(examples, metric)
    st.dataframe(matrix, use_container_width=True, hide_index=True)
```

- [ ] **Step 5: Add dashboard tab**

In `app.py`, import:

```python
from evaluation.dashboard.views import test_case_matrix
```

Add a tab:

```python
tabs = st.tabs(["Overview", "Metrics", "Matrix", "Test Cases", "Failures"])
...
with tabs[3]:
    test_case_matrix.render(examples)
```

- [ ] **Step 6: Verify**

Run:

```bash
uv run python tests/test_evaluation_dashboard.py
uv run python -m py_compile evaluation/dashboard/app.py evaluation/dashboard/views/test_case_matrix.py
```

Expected:

```text
OK
```

Manual smoke:

```bash
uv run streamlit run evaluation/dashboard/app.py
```

Open the dashboard and confirm a `Test Cases` tab appears.

### Task 4: Improve Failure Explorer

**Files:**

- Create: `evaluation/dashboard/views/failure_explorer.py`
- Modify: `evaluation/dashboard/app.py`
- Modify: `evaluation/dashboard/transforms.py`
- Test: `tests/test_evaluation_dashboard.py`

- [ ] **Step 1: Add failure filtering tests**

```python
from evaluation.dashboard.transforms import filter_failed_examples

def test_filter_failed_examples_treats_critical_error_as_bad_when_score_is_one(self):
    examples = pd.DataFrame([
        {"case_key": "a", "critical_error": 1},
        {"case_key": "b", "critical_error": 0},
    ])
    failed = filter_failed_examples(examples, "critical_error")
    self.assertEqual(failed["case_key"].tolist(), ["a"])


def test_filter_failed_examples_treats_hit_metric_as_bad_when_score_below_one(self):
    examples = pd.DataFrame([
        {"case_key": "a", "diagram_id_hit": 0},
        {"case_key": "b", "diagram_id_hit": 1},
    ])
    failed = filter_failed_examples(examples, "diagram_id_hit")
    self.assertEqual(failed["case_key"].tolist(), ["a"])
```

- [ ] **Step 2: Verify failing tests**

Run:

```bash
uv run python tests/test_evaluation_dashboard.py
```

Expected:

```text
FAILED because filter_failed_examples is missing
```

- [ ] **Step 3: Implement failure filtering**

In `transforms.py`:

```python
def filter_failed_examples(examples: pd.DataFrame, metric: str) -> pd.DataFrame:
    if examples.empty or metric not in examples.columns:
        return pd.DataFrame()
    scores = pd.to_numeric(examples[metric], errors="coerce")
    if metric == "critical_error":
        return examples[scores > 0].reset_index(drop=True)
    return examples[scores < 1].reset_index(drop=True)
```

- [ ] **Step 4: Create failure explorer view**

In `evaluation/dashboard/views/failure_explorer.py`:

```python
import pandas as pd
import streamlit as st

from evaluation.dashboard.transforms import METRIC_COLUMNS, filter_failed_examples


def render(examples: pd.DataFrame) -> None:
    st.subheader("Failure Explorer")
    if examples.empty:
        st.info("No example-level results found.")
        return

    metrics = [metric for metric in METRIC_COLUMNS if metric in examples.columns]
    metric = st.selectbox(
        "Failure metric",
        metrics,
        index=metrics.index("critical_error") if "critical_error" in metrics else 0,
    )
    failed = filter_failed_examples(examples, metric)
    st.caption(f"{len(failed)} failed rows")

    group_columns = [
        "loader_strategy",
        "chunker_strategy",
        "embedding_provider",
        "run_name",
    ]
    breakdown = (
        failed.groupby(group_columns, dropna=False)
        .size()
        .reset_index(name="failed_count")
        .sort_values("failed_count", ascending=False)
    )
    st.dataframe(breakdown, use_container_width=True, hide_index=True)

    detail_columns = [
        "case_key",
        "run_name",
        "inputs.question",
        metric,
        f"{metric}_comment",
        "reference.expected_diagram_ids",
        "outputs.retrieved_metadata",
    ]
    existing = [column for column in detail_columns if column in failed.columns]
    st.dataframe(failed[existing], use_container_width=True, hide_index=True)
```

- [ ] **Step 5: Replace old failure tab implementation**

In `app.py`, import:

```python
from evaluation.dashboard.views import failure_explorer
```

Call:

```python
with tabs[-1]:
    failure_explorer.render(examples)
```

- [ ] **Step 6: Verify**

Run:

```bash
uv run python tests/test_evaluation_dashboard.py
uv run python -m py_compile evaluation/dashboard/views/failure_explorer.py
```

Expected:

```text
OK
```

### Task 5: Add Case Detail View

**Files:**

- Create: `evaluation/dashboard/views/case_detail.py`
- Modify: `evaluation/dashboard/app.py`
- Modify: `evaluation/dashboard/transforms.py`
- Test: `tests/test_evaluation_dashboard.py`

- [ ] **Step 1: Add tests for case detail selection**

```python
from evaluation.dashboard.transforms import rows_for_case

def test_rows_for_case_returns_all_runs_for_one_case(self):
    examples = pd.DataFrame([
        {"case_key": "retrieval_001", "run_name": "a"},
        {"case_key": "retrieval_001", "run_name": "b"},
        {"case_key": "retrieval_002", "run_name": "c"},
    ])
    rows = rows_for_case(examples, "retrieval_001")
    self.assertEqual(rows["run_name"].tolist(), ["a", "b"])
```

- [ ] **Step 2: Verify failing test**

Run:

```bash
uv run python tests/test_evaluation_dashboard.py
```

Expected:

```text
FAILED because rows_for_case is missing
```

- [ ] **Step 3: Implement `rows_for_case()`**

In `transforms.py`:

```python
def rows_for_case(examples: pd.DataFrame, case_key: str) -> pd.DataFrame:
    if examples.empty or "case_key" not in examples.columns:
        return pd.DataFrame()
    return examples[examples["case_key"] == case_key].reset_index(drop=True)
```

- [ ] **Step 4: Create case detail view**

In `evaluation/dashboard/views/case_detail.py`:

```python
import pandas as pd
import streamlit as st

from evaluation.dashboard.transforms import METRIC_COLUMNS, rows_for_case


def render(examples: pd.DataFrame) -> None:
    st.subheader("Case Detail")
    if examples.empty or "case_key" not in examples.columns:
        st.info("No cases available.")
        return

    case_options = examples["case_key"].dropna().astype(str).sort_values().unique().tolist()
    selected_case = st.selectbox("Test case", case_options)
    rows = rows_for_case(examples, selected_case)
    first = rows.iloc[0]

    st.markdown("#### Question")
    st.write(first.get("inputs.question", ""))

    st.markdown("#### Expected")
    expected_columns = [
        "reference.expected_diagram_ids",
        "reference.expected_location",
        "reference.expected_party_type",
        "reference.expected_chunk_types",
        "reference.expected_keywords",
    ]
    expected = {column: first.get(column) for column in expected_columns if column in rows.columns}
    st.json(expected)

    st.markdown("#### Results By Combination")
    score_columns = [metric for metric in METRIC_COLUMNS if metric in rows.columns]
    display_columns = [
        "run_name",
        "loader_strategy",
        "chunker_strategy",
        "embedding_provider",
        *score_columns,
        "outputs.retrieved_metadata",
    ]
    existing = [column for column in display_columns if column in rows.columns]
    st.dataframe(rows[existing], use_container_width=True, hide_index=True)
```

- [ ] **Step 5: Add dashboard tab**

In `app.py`:

```python
from evaluation.dashboard.views import case_detail
...
tabs = st.tabs(["Overview", "Metrics", "Matrix", "Test Cases", "Failures", "Case Detail"])
...
with tabs[5]:
    case_detail.render(examples)
```

- [ ] **Step 6: Verify**

Run:

```bash
uv run python tests/test_evaluation_dashboard.py
uv run python -m py_compile evaluation/dashboard/views/case_detail.py
```

Expected:

```text
OK
```

### Task 6: Add Combo Compare View

**Files:**

- Create: `evaluation/dashboard/views/combo_compare.py`
- Modify: `evaluation/dashboard/app.py`
- Modify: `evaluation/dashboard/transforms.py`
- Test: `tests/test_evaluation_dashboard.py`

- [ ] **Step 1: Add tests for pairwise case comparison**

```python
from evaluation.dashboard.transforms import compare_runs_for_case

def test_compare_runs_for_case_returns_two_rows_for_selected_runs(self):
    examples = pd.DataFrame([
        {"case_key": "retrieval_001", "run_name": "run-a", "critical_error": 0},
        {"case_key": "retrieval_001", "run_name": "run-b", "critical_error": 1},
        {"case_key": "retrieval_001", "run_name": "run-c", "critical_error": 0},
    ])
    rows = compare_runs_for_case(examples, "retrieval_001", "run-a", "run-b")
    self.assertEqual(rows["run_name"].tolist(), ["run-a", "run-b"])
```

- [ ] **Step 2: Verify failing test**

Run:

```bash
uv run python tests/test_evaluation_dashboard.py
```

Expected:

```text
FAILED because compare_runs_for_case is missing
```

- [ ] **Step 3: Implement compare helper**

In `transforms.py`:

```python
def compare_runs_for_case(
    examples: pd.DataFrame,
    case_key: str,
    left_run: str,
    right_run: str,
) -> pd.DataFrame:
    rows = rows_for_case(examples, case_key)
    return rows[rows["run_name"].isin([left_run, right_run])].reset_index(drop=True)
```

- [ ] **Step 4: Create combo compare view**

In `evaluation/dashboard/views/combo_compare.py`:

```python
import pandas as pd
import streamlit as st

from evaluation.dashboard.transforms import METRIC_COLUMNS, compare_runs_for_case


def render(examples: pd.DataFrame) -> None:
    st.subheader("Combo Compare")
    if examples.empty:
        st.info("No example-level results found.")
        return

    case_options = examples["case_key"].dropna().astype(str).sort_values().unique().tolist()
    run_options = examples["run_name"].dropna().astype(str).sort_values().unique().tolist()
    selected_case = st.selectbox("Test case", case_options, key="combo_compare_case")
    col_left, col_right = st.columns(2)
    left_run = col_left.selectbox("Left run", run_options, key="combo_compare_left")
    right_run = col_right.selectbox("Right run", run_options, key="combo_compare_right")

    rows = compare_runs_for_case(examples, selected_case, left_run, right_run)
    score_columns = [metric for metric in METRIC_COLUMNS if metric in rows.columns]
    columns = [
        "run_name",
        *score_columns,
        "outputs.retrieved_metadata",
        "outputs.contexts",
    ]
    existing = [column for column in columns if column in rows.columns]
    st.dataframe(rows[existing], use_container_width=True, hide_index=True)
```

- [ ] **Step 5: Add dashboard tab**

In `app.py`:

```python
from evaluation.dashboard.views import combo_compare
...
tabs = st.tabs(["Overview", "Metrics", "Matrix", "Test Cases", "Failures", "Case Detail", "Compare"])
...
with tabs[6]:
    combo_compare.render(examples)
```

- [ ] **Step 6: Verify**

Run:

```bash
uv run python tests/test_evaluation_dashboard.py
uv run python -m py_compile evaluation/dashboard/views/combo_compare.py
```

Expected:

```text
OK
```

### Task 7: Improve Overview And Metric Comparison

**Files:**

- Create: `evaluation/dashboard/views/overview.py`
- Create: `evaluation/dashboard/views/metric_comparison.py`
- Modify: `evaluation/dashboard/app.py`
- Modify: `evaluation/dashboard/charts.py`
- Test: `tests/test_evaluation_dashboard.py`

- [ ] **Step 1: Add transform tests for best/worst combinations**

```python
from evaluation.dashboard.transforms import rank_combinations

def test_rank_combinations_sorts_critical_error_ascending(self):
    summary = pd.DataFrame([
        {"run_name": "bad", "critical_error": 1.0},
        {"run_name": "good", "critical_error": 0.0},
    ])
    ranked = rank_combinations(summary, "critical_error")
    self.assertEqual(ranked["run_name"].tolist(), ["good", "bad"])


def test_rank_combinations_sorts_hit_metrics_descending(self):
    summary = pd.DataFrame([
        {"run_name": "bad", "diagram_id_hit": 0.0},
        {"run_name": "good", "diagram_id_hit": 1.0},
    ])
    ranked = rank_combinations(summary, "diagram_id_hit")
    self.assertEqual(ranked["run_name"].tolist(), ["good", "bad"])
```

- [ ] **Step 2: Implement ranking helper**

```python
def rank_combinations(summary: pd.DataFrame, metric: str) -> pd.DataFrame:
    if summary.empty or metric not in summary.columns:
        return pd.DataFrame()
    ascending = metric == "critical_error"
    return summary.sort_values(metric, ascending=ascending).reset_index(drop=True)
```

- [ ] **Step 3: Move overview rendering to `views/overview.py`**

The view should show:

- Total experiments
- Total combinations
- Average `retrieval_relevance`
- Average `critical_error`
- Best 5 by `critical_error`
- Best 5 by `retrieval_relevance`
- Summary table

- [ ] **Step 4: Move metric comparison rendering to `views/metric_comparison.py`**

The view should show:

- Metric selector
- Group-by selector
- Bar chart
- Explanation that `critical_error` is lower-is-better

- [ ] **Step 5: Keep app.py thin**

`app.py` should only:

- Load results.
- Render sidebar filters.
- Filter frames.
- Dispatch each tab to a view module.

- [ ] **Step 6: Verify**

Run:

```bash
uv run python tests/test_evaluation_dashboard.py
uv run python -m py_compile evaluation/dashboard/app.py evaluation/dashboard/views/*.py
```

Expected:

```text
OK
```

### Task 8: Add Optional Local Retrieval Trace JSONL

**Files:**

- Modify: `rag/service/tracing.py`
- Modify: `rag/pipeline/retrieval.py`
- Modify: `evaluation/evaluate_retrieval_langsmith.py`
- Modify: `evaluation/dashboard/loaders.py`
- Create: `evaluation/dashboard/views/local_trace.py`
- Test: `tests/test_retrieval_langsmith_eval.py`
- Test: `tests/test_evaluation_dashboard.py`

Implement this task only after Tasks 1-7 are useful. This task gives LangSmith-like step evidence for offline tests without external quota.

- [ ] **Step 1: Add local trace data model tests**

In a new or existing test:

```python
from rag.service.tracing import LocalTraceRecorder

def test_local_trace_recorder_records_step_events(self):
    recorder = LocalTraceRecorder(run_name="upstage-custom-bge", example_id="retrieval_001")
    recorder.record_step(
        step="mdm.retrieve.vectorstore",
        inputs={"k": 5},
        outputs={"count": 1},
        latency_ms=12.3,
    )
    payload = recorder.to_dict()
    self.assertEqual(payload["run_name"], "upstage-custom-bge")
    self.assertEqual(payload["example_id"], "retrieval_001")
    self.assertEqual(payload["events"][0]["step"], "mdm.retrieve.vectorstore")
```

- [ ] **Step 2: Extend `TraceContext`**

In `rag/service/tracing.py`:

```python
@dataclass
class LocalTraceRecorder:
    run_name: str
    example_id: str | None = None
    question: str | None = None
    events: list[dict[str, Any]] = field(default_factory=list)

    def record_step(self, step: str, inputs: dict[str, Any], outputs: dict[str, Any], latency_ms: float) -> None:
        self.events.append({
            "step": step,
            "latency_ms": latency_ms,
            "inputs": inputs,
            "outputs": outputs,
        })

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_name": self.run_name,
            "example_id": self.example_id,
            "question": self.question,
            "events": self.events,
        }
```

Add to `TraceContext`:

```python
local_recorder: LocalTraceRecorder | None = None

def record_step(...):
    if self.local_recorder is not None:
        self.local_recorder.record_step(...)
```

- [ ] **Step 3: Record retrieval pipeline events**

In `run_retrieval_pipeline()`:

- Record `mdm.retrieve.<strategy>`.
- Record fallback retrieval when filters produce zero results.
- Record `mdm.rerank.<strategy>`.

Use a document serializer that stores:

```text
rank
page
diagram_id
chunk_id
chunk_type
party_type
location
preview
```

- [ ] **Step 4: Add evaluator CLI options**

In `parse_args()`:

```python
parser.add_argument(
    "--trace-mode",
    choices=["off", "failed", "all"],
    default="off",
    help="Write local trace JSONL for offline dashboard inspection.",
)
```

Add:

```python
parser.add_argument(
    "--trace-dir",
    type=Path,
    default=DEFAULT_OUTPUT_DIR / "traces",
)
```

- [ ] **Step 5: Write JSONL traces**

During `evaluate_local_rows()`:

- Create a recorder per example when `trace_mode` is not `off`.
- Attach it to `TraceContext`.
- After feedback scores are computed:
  - Save all traces if `trace_mode == "all"`.
  - Save only failed traces if `trace_mode == "failed"` and `critical_error > 0` or any selected metric failed.

- [ ] **Step 6: Add dashboard trace loader**

In `loaders.py`, read `evaluation/results/langsmith/traces/*.jsonl` and expose a trace DataFrame keyed by:

```text
run_name
case_key/example_id
step
event_index
```

- [ ] **Step 7: Add local trace view**

In `views/local_trace.py`, show:

- Test case selector.
- Run selector.
- Step list.
- Step input JSON.
- Step output JSON.

- [ ] **Step 8: Verify**

Run:

```bash
uv run python tests/test_retrieval_langsmith_eval.py
uv run python tests/test_evaluation_dashboard.py
uv run python evaluation/evaluate_retrieval_langsmith.py --max-examples 2 --trace-mode all --output-dir /tmp/mdm-local-eval
```

Expected:

```text
/tmp/mdm-local-eval/traces/<timestamp>-upstage-custom-bge.jsonl exists
```

### Task 9: Keep LangSmith For Streamlit User Sessions Only

**Files:**

- Modify: `main.py`
- Modify: `rag/service/tracing.py`
- Modify: `rag/service/conversation/app_service.py`
- Modify: `README.md`
- Test: existing conversation tests if available

- [ ] **Step 1: Define tracing policy**

Production/user-facing Streamlit should support:

```text
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=MDM
```

Offline evaluation should not require these variables.

- [ ] **Step 2: Ensure Streamlit creates stable thread/session IDs**

In Streamlit session state, keep:

```python
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
```

Use this when constructing `TraceContext`:

```python
trace_context = TraceContext(
    thread_id=st.session_state.thread_id,
    session_id=st.session_state.thread_id,
    metadata={"surface": "streamlit"},
)
```

- [ ] **Step 3: Confirm route/intake/retrieve/rerank/answer receive TraceContext**

TraceContext should flow through:

```text
app_service
conversation orchestrator
router
intake
analysis
retrieval
reranker
answer
```

- [ ] **Step 4: Add README policy**

Document:

```text
Use LangSmith for real Streamlit sessions.
Use local dashboard for matrix evaluation.
Do not run --preset all --langsmith unless intentionally spending trace quota.
```

- [ ] **Step 5: Manual smoke**

Run Streamlit:

```bash
uv run streamlit run main.py
```

Ask one accident question and confirm in LangSmith:

- One thread/session is visible.
- route/intake/retrieve/rerank/answer steps are visible.
- Input/output/attributes are visible.

Run local eval:

```bash
uv run python evaluation/evaluate_retrieval_langsmith.py --max-examples 1
```

Confirm no new LangSmith trace is created.

### Task 10: Documentation And Usage Guide

**Files:**

- Modify: `evaluation/dashboard/README.md`
- Modify: `data/testsets/langsmith/README.md`
- Modify: `README.md`

- [ ] **Step 1: Document local evaluation commands**

Add:

```bash
uv run python evaluation/evaluate_retrieval_langsmith.py
uv run python evaluation/evaluate_retrieval_langsmith.py --matrix
uv run python evaluation/evaluate_retrieval_langsmith.py --preset all
```

- [ ] **Step 2: Document dashboard command**

Add:

```bash
uv run streamlit run evaluation/dashboard/app.py
```

- [ ] **Step 3: Document LangSmith remote commands**

Add:

```bash
uv run python evaluation/evaluate_retrieval_langsmith.py --langsmith --max-examples 3
```

Warn:

```text
Avoid --preset all --langsmith unless trace quota/cost is intentional.
```

- [ ] **Step 4: Document interpretation**

Explain:

- `critical_error`: lower is better.
- `diagram_id_hit`: higher is better.
- `retrieval_relevance`: higher is better.
- `chunk_type_match`, `party_type_match`, `location_match`: higher is better.

- [ ] **Step 5: Verify docs**

Run:

```bash
rg -n "preset all --langsmith|critical_error|evaluation/dashboard" README.md data/testsets/langsmith/README.md evaluation/dashboard/README.md
```

Expected:

```text
`preset all --langsmith` appears in `data/testsets/langsmith/README.md`.
`critical_error` appears in `evaluation/dashboard/README.md`.
`evaluation/dashboard` appears in `README.md`.
```

## Verification Checklist

Before claiming the implementation is complete, run:

```bash
uv run python tests/test_retrieval_langsmith_eval.py
uv run python tests/test_evaluation_dashboard.py
uv run python -m py_compile evaluation/evaluate_retrieval_langsmith.py evaluation/dashboard/app.py evaluation/dashboard/*.py evaluation/dashboard/views/*.py
```

Run one local evaluation:

```bash
rm -rf /tmp/mdm-local-eval
env -u LANGSMITH_API_KEY LANGSMITH_TRACING=true LANGCHAIN_TRACING_V2=true \
  uv run python evaluation/evaluate_retrieval_langsmith.py \
  --max-examples 1 \
  --output-dir /tmp/mdm-local-eval
```

Expected:

```text
LangSmith skipped. Local result files were written.
```

Run dashboard smoke:

```bash
uv run streamlit run evaluation/dashboard/app.py --server.port 8502 --server.address 127.0.0.1 --server.headless true
curl -sSf http://127.0.0.1:8502/_stcore/health
```

Expected:

```text
ok
```

## Recommended Commit Split

Use small commits:

```text
test: keep retrieval evaluation local by default
feat: add local evaluation dashboard test matrix
feat: add evaluation failure explorer
feat: add evaluation case detail comparison
feat: add optional local retrieval traces
docs: document local evaluation and LangSmith tracing policy
```

## Rollout Plan

1. Merge Tasks 1-4 first.
   - This immediately prevents trace quota burn and makes current CSV/JSON results easier to inspect.
2. Merge Tasks 5-7 next.
   - This turns the dashboard into a real evaluation analysis tool.
3. Implement Task 8 only if the dashboard still lacks enough root-cause detail.
   - Local trace JSONL is useful, but it should not block pass/fail analysis.
4. Keep Task 9 separate.
   - It touches real Streamlit user tracing and should be tested manually against LangSmith with very small usage.
5. Finish with Task 10.
   - Docs should make the split between LangSmith user tracing and local offline evaluation impossible to miss.
