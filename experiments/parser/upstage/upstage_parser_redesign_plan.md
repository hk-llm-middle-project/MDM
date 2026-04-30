# Upstage Parser Redesign Plan

Status: paused design note. This document records the agreed parser design and should not be treated as a completed migration by itself.

## Goal

Rebuild the final chunk file from the Upstage raw parsed file:

`data/upstage_output/main_pdf/raw/parsed_documents.json`

Do not patch the existing final file directly:

`data/upstage_output/main_pdf/final/chunked_documents_final.json`

Known issues in the existing final file:

- General explanatory pages are incorrectly attached to the first diagram case, such as `보1`.
- Cases that share explanation pages, such as `거2-1` and `거2-2`, are merged into the wrong parent.
- Boilerplate notes such as `※사고발생...` are treated like headings.
- Next section headings can be appended to the previous case text.
- Existing child chunks use `parent_id` as a diagram-id string, but the new schema requires an integer parent `chunk_id`.

## Final JSON Shape

Keep the existing project and LangChain-compatible JSON shape:

```json
{
  "page_content": "...",
  "metadata": {
    "chunk_id": 1,
    "chunk_type": "parent",
    "diagram_id": "거5-3",
    "parent_id": null,
    "page": 528,
    "source": "data/upstage_output/main_pdf/raw/parsed_documents.json",
    "party_type": "자전거",
    "location": "교차로 사고",
    "image_path": "data/upstage_output/main_pdf/final/img/page_528_table_1.png"
  }
}
```

Use `page_content` in the JSON output, not a top-level `text` field.

## Target Chunk Schema

```python
@dataclass
class Chunk:
    chunk_id: int
    text: str
    chunk_type: str  # "parent", "child", "preface", "general"
    page: int
    source: str
    diagram_id: str | None  # example: "차43-7(가)"
    parent_id: int | None   # for child chunks only: parent chunk_id
    location: str | None
    party_type: str | None
    image_path: str | None  # parent only; child must be null
```

## Allowed Metadata Values

```python
PARTY_TYPES = [
    "보행자",
    "자동차",
    "자전거",
]

LOCATIONS = [
    "횡단보도 내",
    "횡단보도 부근",
    "횡단보도 없음",
    "교차로 사고",
    "기타",
    "마주보는 방향 진행차량 상호 간의 사고",
    "같은 방향 진행차량 상호간의 사고",
    "자동차 대 이륜차 특수유형",
]
```

## Chunk Type Policy

- `parent`: one parent chunk per case diagram or sub-diagram.
- `child`: a meaningful child item under a parent.
- `preface`: cover pages, table of contents, pre-document information, appendix-like administrative content.
- `general`: non-case explanatory content that does not belong to a specific diagram.
- Do not create standalone `image` chunks.
- Store table/diagram content as a `child` chunk, not as an `image` chunk.
- The table child should keep the markdown table text in `page_content`.
- Store `image_path` only on the `parent` chunk.
- Every child chunk must have `"image_path": null`.

Example structure:

```text
거5-1 parent
  child: 거5-1 table
  child: 거5-1 accident situation
  child: 거5-1 basic fault-ratio explanation
  child: shared modification-factor explanation
  child: shared related statutes
  child: shared precedent

거5-2 parent
  child: 거5-2 table
  child: 거5-2 accident situation
  child: 거5-2 basic fault-ratio explanation
  child: shared modification-factor explanation
  child: shared related statutes
  child: shared precedent
```

## Diagram ID Policy

Normal diagram ids:

- `보1`
- `차12-2`
- `거2-1`

Sub-diagram ids:

- If a table or explanation is split by `(가)`, `(나)`, `(다)`, create separate diagram ids.
- Examples: `차43-7(가)`, `차43-7(나)`.
- Remove unnecessary spaces inside diagram ids.
  - `차43 -7` -> `차43-7`
  - `거2 - 1` -> `거2-1`

When sub-diagrams share one table image:

- Create separate parents, such as `차43-7(가)` and `차43-7(나)`.
- Both parents may use the same `image_path`.
- Text children should be split by the corresponding variant marker.
- Shared modification factors, statutes, and precedents should be included in each relevant parent.

## Parsing Strategy

Use deterministic parsing rules, not an LLM, for metadata assignment and case ownership.

The Upstage raw file is element-level, not page-level. It contains elements such as `paragraph`, `heading1`, `table`, `list`, `header`, and `footer`. The parser should walk these elements in order while tracking current page, current heading, and active diagram ids.

Recommended flow:

1. Load `raw/parsed_documents.json`.
2. Preserve raw `metadata.page` as the actual page.
3. Separate or skip headers, footers, and table-of-contents/index material.
4. Detect case tables from `category == "table"` and table text.
5. Extract diagram ids from the first markdown table cell.
6. If `(가)`, `(나)`, `(다)` variants exist, create variant diagram ids.
7. Attach the table image path to the corresponding `parent.image_path`.
8. Split explanatory text by markers such as:
   - `⊙ 거2-1`
   - `- 거2-1`
   - `거2-1`
9. If multiple cases share one explanation page, split targeted text by diagram id.
10. Text explicitly labeled with a diagram id belongs only to that parent.
11. Common text without a diagram id should be included in all currently active parents.
12. End the current parent group when a new case group, new section, or new diagram boundary is reached.
13. Split each parent into child chunks by allowed headings and table boundaries.
14. Assign sequential integer `chunk_id` values after all chunks are assembled.

## Heading Policy

Only the following labels should be treated as structural headings:

- `사고 상황`
- `기본 과실비율 해설`
- `수정요소(인과관계를 감안한 과실비율 조정) 해설`
- `활용시 참고 사항`
- `관련 법규`
- `참고 판례`

Do not treat the following as headings:

- `※사고발생, 손해확대와의 인과관계를 감안하여...`
- old-reference notes such as `※舊 414 기준`
- arbitrary first lines

## Page Metadata Policy

- `parent.page`: the page where the case table starts.
  - Example: if the `거5-1` table is on page 524 and the explanation is on page 525, parent page is `524`.
- `child.page`: the actual page where that child content appears.
  - Example: table child is `524`, accident-situation child is `525`.
- When shared statute or precedent children are copied to multiple parents, their `page` is the page where that shared section actually starts.
  - Example: if related statutes span pages 525-526 and start on 525, the child page is `525`.

## Required Validation Cases

The rebuilt output should pass at least these checks:

- `1. 적용 범위` around raw page 31 must not be attached to `보1`.
- `거2-1` parent must include the `거2-1` accident situation and basic fault-ratio explanation.
- `거2-2` parent must include the `거2-2` accident situation and basic fault-ratio explanation.
- `거2-1` text must not be stored under `거2-2`.
- Cases sharing explanation pages, such as `거5-1` and `거5-2`, must become separate parents.
- If `차43-7` is split into `(가)` and `(나)`, create `차43-7(가)` and `차43-7(나)`.
- Tables must be stored as child chunks, but `image_path` must be stored only on parents.
- Every child chunk must have `image_path == null`.
- Every child `parent_id` must point to an existing parent `chunk_id`.
- Every non-null `location` must be in `LOCATIONS`.
- Every non-null `party_type` must be in `PARTY_TYPES`.

## Output Versioning

Do not overwrite the current final file on the first run.

Recommended generated files:

- `data/upstage_output/main_pdf/final/chunked_documents_final.v2.json`
- `data/upstage_output/main_pdf/final/chunked_documents_final.v2.report.json`

After review, the v2 file can be promoted to:

- `data/upstage_output/main_pdf/final/chunked_documents_final.json`

## Implementation Location

Implement and run the new parser under:

`experiments/parser/upstage`

Existing files in that directory may be moved to a timestamped legacy folder before writing the new implementation.

표 + 주석
사고 상황
기본 과실비율 해설
수정요소 해설
활용시 참고 사항
관련 법규
참고 판례