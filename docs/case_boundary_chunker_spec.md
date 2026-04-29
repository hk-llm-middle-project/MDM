# Case-boundary chunker implementation spec

## Purpose

Implement the domain-specific case-boundary chunker for the accident-ratio RAG pipeline.

This chunker should process page-level parsed documents, especially LlamaParse markdown, and return standard `Chunk` objects. It must preserve both:

- front/back matter as `preface` chunks
- non-case explanatory content as `general` chunks
- case content as `parent` chunks with `child` chunks

The primary target mode is Mode B: case parent plus child chunks.

## Output Chunk Types

This chunker produces only these chunk types:

| chunk_type | Meaning |
| --- | --- |
| `preface` | Front matter, back matter, appendices, tables of contents, publication notes, revision history |
| `general` | General principles, definitions, application scope, modification-factor explanations |
| `parent` | Case parent chunk containing the whole case |
| `child` | Case child chunk containing one item under a case |

Do not produce `flat` chunks from this chunker. Do not produce standalone `image` chunks here. Table/image markdown should be included inside the relevant case child text.

Image handling:

- Keep the original chunk type. Do not change a chunk to `chunk_type="image"` just because it contains an image.
- If a chunk contains one representative image, set `image_path` to that image path.
- If a chunk does not contain an image, set `image_path=None`.
- If a parent chunk contains multiple markdown image links because it preserves the whole case, set `image_path` to the first representative image path and keep all image markdown links in `text`.
- Prefer splitting child chunks so that each table/image child has at most one image.

## Preface Chunk Policy

Use `preface` for content that is neither a case nor reusable legal/domain explanation for accident analysis.

Examples:

- 목차
- 발간사
- 제1편 개정경과
- 문서 안내 or 이용 안내
- `※ (별첨) 변경대비표`
- Other front/back matter or appendix-like administrative content

Preface chunks are flat:

```text
chunk_type="preface"
diagram_id=None
parent_id=None
image_path=None
```

The document is not expected to use old criterion numbers during retrieval, so the appendix change-comparison tables should be grouped as `preface`, not `general`.

Split preface content by large document heading or page-sized section. It does not need parent-child structure.

## General Chunk Policy

General content is not parent-child. Each general chunk is independent:

```text
chunk_type="general"
diagram_id=None
parent_id=None
```

General chunks must include breadcrumb headings at the top so that each chunk keeps its meaning after splitting.

Example:

```markdown
# 3. 수정요소(인과관계를 감안한 과실비율 조정)의 해설
### (1) 야간, 기타 시야장애, 차의 등화 및 감속

...
```

### 제2편 총설

Target pages: roughly 13p-28p.

Large headings:

- `# 1. 과실비율 인정기준의 필요성`
- `# 2. 과실과 과실상계`
- `# 3. 과실비율 인정기준의 기본원칙`
- `# 4. 과실비율 인정기준의 적용`
- `# 5. 인적 손해에서의 과실상계 별도적용기준`

Chunking rules:

- Split by meaningful subsection under the large heading.
- Prefer `### (n) ...` as a general chunk boundary.
- Use `### n) ...` as a boundary when it is the meaningful subsection level.
- Keep short `#### ① ...`, `② ...`, `③ ...` items together if they form one small conceptual unit.
- For `# 5` > `### (4) 세부적용 예`, split each `①` through `⑩` item into a separate `general` chunk because the section is long.

Examples:

```markdown
# 1. 과실비율 인정기준의 필요성
### (1) 신속한 보상처리

...
```

```markdown
# 5. 인적 손해에서의 과실상계 별도적용기준
### (4) 세부적용 예
**① 보호자의 자녀감호 태만**

...
```

### 제3편 각 장의 공통 설명

Targets:

- 제1장 자동차와 보행자의 사고: roughly 31p-38p
- 제2장 자동차와 자동차(이륜차 포함)의 사고: chapter intro before cases
- 제3장 자동차와 자전거(농기계 포함)의 사고: chapter intro before cases

Chunking rules:

- `# 1. 적용 범위`: keep as one `general` chunk.
- `# 2. 용어 정의`: split by numbered subsection, such as `### (1) ...`, `### (2) ...`.
- `# 3. 수정요소(인과관계를 감안한 과실비율 조정)의 해설`: split by numbered subsection, such as `### (1) ...`, `### (2) ...`.
- Include the parent heading as breadcrumb in each resulting chunk.

Examples:

```markdown
# 2. 용어 정의
### (2) 동일폭 교차로, 대로/소로 교차로

...
```

```markdown
# 3. 수정요소(인과관계를 감안한 과실비율 조정)의 해설
### (1) 야간, 기타 시야장애, 차의 등화 및 감속

...
```

## Case Chunk Policy

Cases are identified by table boundaries.

Assumption:

- A case starts with a markdown table.
- The table contains a case id such as `보20`, `차43-7`, `거10-3`.
- A table image markdown line immediately following a table belongs to that case.

Case parent chunks:

```text
chunk_type="parent"
diagram_id="<case id>"
parent_id=None
```

Case child chunks:

```text
chunk_type="child"
diagram_id="<same as parent>"
parent_id=<parent chunk_id>
```

Child item categories:

- table/image
- 사고 상황
- 기본 과실비율 해설
- 수정요소(인과관계를 감안한 과실비율 조정) 해설
- 활용시 참고 사항
- 관련 법규
- 참고 판례

The `parent` chunk should contain the whole case content. Each child should contain one meaningful item under that parent.

For table/image children:

```text
chunk_type="child"
diagram_id="<same as parent>"
parent_id=<parent chunk_id>
image_path="<table image path>"
```

The child `text` should include both the markdown table and the markdown image link.

For case parent chunks:

```text
chunk_type="parent"
diagram_id="<case id>"
parent_id=None
image_path="<first representative image path, if any>"
```

The parent chunk `text` field should preserve the full case text, including all markdown image links.

## Multi-case Pages

### Multiple Tables With Shared Explanation

Example: page 85, `보20` and `보21`.

When multiple cases are followed by shared explanation, create separate parents for each case.

```text
보20 parent
  child: 보20 table/image
  child: 보20-specific 사고 상황, if detectable
  child: shared explanation duplicated if common

보21 parent
  child: 보21 table/image
  child: 보21-specific 사고 상황, if detectable
  child: shared explanation duplicated if common
```

If a section contains sub-items explicitly labeled by case id, split the section by those labels. Common text without a case label should be duplicated into every active case parent.

### One Table Containing Multiple Variants

Example: page 389, `차43-7` with `(가)` and `(나)`.

Create a separate parent for each variant:

```text
diagram_id="차43-7(가)"
diagram_id="차43-7(나)"
```

If the parsed table is too noisy to split safely, duplicate the original table/image child into each variant parent to avoid information loss.

Prefer information preservation over aggressive table slicing.

## Page Boundary Policy

Input documents are page-level, but chunking state must continue across pages.

Rules:

- If a case starts on one page and its explanatory sections continue on the next page, attach the next page sections to the existing parent.
- If a general section starts on one page and continues on the next page without a new boundary, continue the same general section.
- Child chunks should use the actual page/source where their content appears.
- Parent chunks should use the page/source where the parent starts.

Example:

```text
chunk_id=0 | type=parent | diagram_id=차2-5 | parent_id=None | page=175 | source=175.md
chunk_id=1 | type=child | diagram_id=차2-5 | parent_id=0    | page=176 | source=176.md
chunk_id=2 | type=child | diagram_id=차2-5 | parent_id=0    | page=176 | source=176.md
```

## Diagram ID Rules

Base case id pattern:

```regex
[차보거]\d+(?:-\d+)?
```

Variant id pattern:

```text
<base id>(가)
<base id>(나)
```

Use variant ids when a single case id clearly contains multiple variants labeled `(가)`, `(나)`, etc.

Examples:

```text
차43-7(가)
차43-7(나)
```

## Recommended Implementation Shape

Implement one class:

```python
CaseBoundaryChunker(mode="B")
```

Mode B is the primary target:

- front/back matter chunks: `preface`
- parent case chunk: `parent`
- child chunks under each case: `child`
- general chunks: `general`
- no standalone `image` chunks; use `image_path` on `parent`/`child` chunks instead

Mode A can later reuse the same boundary detection but skip child creation and return only case parent chunks for cases.

## Safety Principles

- Do not drop text because a boundary is ambiguous.
- If a section is common to multiple active cases, duplicate it.
- If table parsing is broken, duplicate the table into each relevant variant rather than trying to split it unsafely.
- Preserve headings inside chunk text.
- Include breadcrumb headings in `general` chunks.
- Keep `diagram_id=None` for `preface`.
- Keep `diagram_id=None` for `general`.
- Keep `parent_id=None` for `preface`, `general`, and `parent` chunks.
