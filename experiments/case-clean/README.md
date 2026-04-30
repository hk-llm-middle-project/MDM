# Case Boundary + Table Clean Experiment

Pipeline:

1. Load original `data/llama_md/main_pdf/{page:03}.md` files.
2. Run `CaseBoundaryChunker(mode="B")` on the page sequence.
3. Convert chunks to document dicts (`page_content` + metadata).
4. Run `clean_case_boundary_tables()` to clean table child chunks.
5. Rebuild changed parent chunks from cleaned children.

Important interpretation rule: for `차43-7`, `(가)/(나)` changes only the base fault ratio; adjustment factors are common to both variants.

## Runs

### pages_039-042

- pages: `39-42`
- raw chunks: `9`
- cleaned chunks: `9`
- changed chunk_ids after table clean/rebuild: `[1, 2]`
- raw markdown: `pages_039-042__raw.md`
- cleaned markdown: `pages_039-042__cleaned.md`
- raw json: `pages_039-042__raw.json`
- cleaned json: `pages_039-042__cleaned.json`

### pages_078-081

- pages: `78-81`
- raw chunks: `21`
- cleaned chunks: `21`
- changed chunk_ids after table clean/rebuild: `[0, 1, 2, 3, 4, 5]`
- raw markdown: `pages_078-081__raw.md`
- cleaned markdown: `pages_078-081__cleaned.md`
- raw json: `pages_078-081__raw.json`
- cleaned json: `pages_078-081__cleaned.json`

### pages_389-391

- pages: `389-391`
- raw chunks: `14`
- cleaned chunks: `14`
- changed chunk_ids after table clean/rebuild: `[0, 1, 2, 3]`
- raw markdown: `pages_389-391__raw.md`
- cleaned markdown: `pages_389-391__cleaned.md`
- raw json: `pages_389-391__raw.json`
- cleaned json: `pages_389-391__cleaned.json`
