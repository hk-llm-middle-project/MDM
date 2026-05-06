"""Clean LlamaParse accident diagram tables.

Temporary helper for tables where the left accident diagram is parsed as table
text. It keeps the image link, but rewrites the preceding markdown table to the
right-side fault-ratio data only.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


TABLE_LINE_PATTERN = re.compile(r"^\s*\|.*\|\s*$")
SEPARATOR_CELL_PATTERN = re.compile(r"^:?-{3,}:?$")
MARKER_PATTERN = re.compile(r"^[①②③④⑤⑥⑦⑧⑨⑩]$")
VALUE_PATTERN = re.compile(r"^(?:[+-]?\d+|비적용|[AB]\d+(?::?B?\d+)?)$")
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
CASE_A_VALUE_PATTERN = re.compile(r"\(([^)]+)\)\s*A\s*(\d+)")
A_VALUE_PATTERN = re.compile(r"A\s*(\d+)")
B_VALUE_PATTERN = re.compile(r"B\s*(\d+)")
ADJUSTMENT_PATTERN = re.compile(r"^[+-]\d+$")
TARGET_ITEM_PATTERN = re.compile(r"^([AB])\s*(.+)$")
CIRCLED_NUMBER_PATTERN = re.compile(r"[①②③④⑤⑥⑦⑧⑨⑩]")
BR_TAG_PATTERN = re.compile(r"<br\s*/?>", re.IGNORECASE)
KR_CASE_LABEL_PATTERN = re.compile(r"^\((?:가|나|다|라|마)\)\s*(.+)$")
PLAIN_NUMBER_PATTERN = re.compile(r"^\d+$")
ADJ_VALUE_PATTERN = re.compile(r"^[+-]\d+$|^비적용$")

SKIP_ITEM_CELLS = {
    "과실비율",
    "과실비율 조정예시",
}
PEDESTRIAN_HEADER_KEYWORDS = {"보행자 기본 과실비율", "기본 과실비율", "과실비율 조정예시", "과실비율 조정 예시"}


def clean_markdown(markdown: str) -> str:
    """Rewrite diagram-heavy fault-ratio tables in a markdown document."""
    lines = markdown.splitlines()
    cleaned: list[str] = []
    index = 0

    while index < len(lines):
        if not _is_table_line(lines[index]):
            cleaned.append(lines[index])
            index += 1
            continue

        start = index
        while index < len(lines) and _is_table_line(lines[index]):
            index += 1
        table_lines = lines[start:index]
        rewritten = rewrite_fault_ratio_table("\n".join(table_lines))
        cleaned.extend(rewritten.splitlines())

    trailing_newline = "\n" if markdown.endswith("\n") else ""
    return "\n".join(cleaned) + trailing_newline


def rewrite_fault_ratio_table(table_markdown: str) -> str:
    """Return a compact right-side fault-ratio table when one can be inferred."""
    rows = [
        _parse_markdown_row(line)
        for line in table_markdown.splitlines()
        if _is_table_line(line) and not _is_separator_row(line)
    ]
    ab_table = _rewrite_ab_fault_ratio_table(rows)
    if ab_table is not None:
        return ab_table

    ped_table = _rewrite_pedestrian_fault_ratio_table(rows)
    if ped_table is not None:
        return ped_table

    simple_ab = _rewrite_simple_ab_table(rows)
    if simple_ab is not None:
        return simple_ab

    fault_rows = _extract_fault_rows(rows)
    has_basic_row = any(group == "기본 과실비율" for group, _, _ in fault_rows)
    if len(fault_rows) < 2 or not has_basic_row:
        return table_markdown
    return _format_fault_rows(fault_rows)


SIMPLE_A_RATIO_PATTERN = re.compile(r"^A(\d+)$")
SIMPLE_B_RATIO_PATTERN = re.compile(r"^B(\d+)$")
SIMPLE_TARGET_PATTERN = re.compile(r"^([AB])\s+(.+)$")
BARE_KR_CASE_PATTERN = re.compile(r"^\(([가나다라마])\)$")
VALUE_TOKEN_PATTERN = re.compile(r"[AB]\d+|[+-]\d+|비적용")
_AB_RATIO_PREFIX = re.compile(r"^([AB])(\d+)")
_ADJ_TOKEN = re.compile(r"[+-]\d+|비적용")


def _try_ab_separated_cells(rows: list[list[str]]) -> str | None:
    """Handle tables where (가)/(나) bare labels, A values, and B values are
    packed in separate <br/>-cells within the same row.

    Example row:  | (가)<br/>(나) | A30<br/>A80 | B70<br/>B20 |
    Adjustment rows use the same "A item | +value" structure as simple AB tables.
    """
    for row_idx, row in enumerate(rows):
        for cell_idx, cell in enumerate(row):
            if "<br" not in cell.lower():
                continue
            parts = _split_br_cell(cell)
            if len(parts) < 2:
                continue
            case_letters: list[str] = []
            for p in parts:
                m = BARE_KR_CASE_PATTERN.match(p)
                if not m:
                    break
                case_letters.append(m.group(1))
            else:
                pass
            if len(case_letters) < 2 or "나" not in case_letters:
                continue

            a_values: list[str] = []
            b_values: list[str] = []
            for next_idx in range(cell_idx + 1, len(row)):
                next_parts = _split_br_cell(row[next_idx])
                if not next_parts:
                    continue
                if not a_values and all(SIMPLE_A_RATIO_PATTERN.match(p) for p in next_parts):
                    a_values = [SIMPLE_A_RATIO_PATTERN.match(p).group(1) for p in next_parts]
                elif not b_values and all(SIMPLE_B_RATIO_PATTERN.match(p) for p in next_parts):
                    b_values = [SIMPLE_B_RATIO_PATTERN.match(p).group(1) for p in next_parts]

            if len(a_values) < len(case_letters) or len(b_values) < len(case_letters):
                continue

            basic_rows = [
                (case_letters[i], a_values[i], b_values[i])
                for i in range(len(case_letters))
            ]
            adj_rows = _extract_simple_ab_adj_from_rows(rows, skip_row=row_idx)

            lines = [
                "### 기본 과실비율",
                "",
                "| 유형 | A 과실 | B 과실 |",
                "| --- | --- | --- |",
            ]
            for case_label, a_val, b_val in basic_rows:
                lines.append(f"| ({case_label}) | A{a_val} | B{b_val} |")

            if adj_rows:
                lines.extend(
                    [
                        "",
                        "### 과실비율 조정예시",
                        "",
                        "| 대상 | 수정요소 | A 조정 | B 조정 |",
                        "| --- | --- | --- | --- |",
                    ]
                )
                for target, item, a_adj, b_adj in adj_rows:
                    lines.append(f"| {target} | {_escape_cell(item)} | {a_adj} | {b_adj} |")

            return "\n".join(lines)

    return None


def _rewrite_simple_ab_table(rows: list[list[str]]) -> str | None:
    """Handle simple A/B fault ratio tables (no (가)/(나) case variants).

    Supports two layouts:
    - Proper rows: one row has A<n> and B<m> cells as basic ratios, other rows
      have 'A/B item | +value' adjustment rows.
    - Crammed: a single <br/>-cell starts with A<n> followed by 'A/B item' names,
      with adjacent <br/>-cells holding B<m> and adjustment values.
    """
    result = _try_simple_ab_proper_rows(rows)
    if result is not None:
        return result
    return _try_simple_ab_crammed(rows)


def _try_simple_ab_proper_rows(rows: list[list[str]]) -> str | None:
    """Pattern A: multi-row table where one row contains A<n> and B<m> cells."""
    basic_row_idx = None
    a_value = b_value = ""

    for row_idx, row in enumerate(rows):
        norm = [_normalize_cell(c) for c in row]
        a_match = next((SIMPLE_A_RATIO_PATTERN.match(c) for c in norm if SIMPLE_A_RATIO_PATTERN.match(c)), None)
        b_match = next((SIMPLE_B_RATIO_PATTERN.match(c) for c in norm if SIMPLE_B_RATIO_PATTERN.match(c)), None)
        if a_match and b_match:
            a_value = a_match.group(1)
            b_value = b_match.group(1)
            basic_row_idx = row_idx
            break

    if basic_row_idx is None:
        return None

    adj_rows = _extract_simple_ab_adj_from_rows(rows, skip_row=basic_row_idx)
    if not adj_rows:
        return None
    return _format_simple_ab_table(a_value, b_value, adj_rows)


def _extract_value_tokens(p: str) -> list[str]:
    """Safely extract A/B ratio and adjustment value tokens from a string.

    Handles concatenated values like 'B100+10' and '+20비적용'.
    Avoids false matches in case identifiers like '차31-1' where '-1' is a
    sub-number (preceded by another digit), not an adjustment value.
    """
    result: list[str] = []
    # If the string starts with an A/B ratio, consume it first, then extract
    # any immediately following adjustment values from the remainder.
    m = _AB_RATIO_PREFIX.match(p)
    if m:
        result.append(m.group(1) + m.group(2))
        for tok in _ADJ_TOKEN.findall(p[m.end():]):
            result.append(tok)
        return result
    # No leading ratio: extract standalone adjustment values, but skip any
    # token whose sign character is immediately preceded by a digit (e.g. '차31-1').
    for tok_m in _ADJ_TOKEN.finditer(p):
        start = tok_m.start()
        tok = tok_m.group()
        # Skip [+-]\d+ tokens immediately preceded by a digit (e.g. '차31-1' → '-1').
        # Don't apply this guard to '비적용' — it's a Korean word, never a sub-number.
        if tok[0] in "+-" and start > 0 and p[start - 1].isdigit():
            continue
        result.append(tok)
    return result


def _try_simple_ab_crammed(rows: list[list[str]]) -> str | None:
    """Pattern B: crammed rows where A/B ratios and items may be scattered
    across cells with repeated title prefixes, and values may be concatenated
    (e.g. 'B100+10', '+20비적용').

    Strategy:
    - Scan all <br/>-cells in the row to find A ratio, B ratio, items, adj vals.
    - Cells before the B-ratio cell contribute A adj vals.
    - The B-ratio cell itself: last len(B-items) adj vals go to B, the rest to A.
    - Cells after the B-ratio cell: treated as residual A adj vals.
    - If no B adj vals found at all, fall back to sequential pairing.
    """
    for row in rows:
        a_value: str | None = None
        b_value: str | None = None
        a_cell_idx = b_cell_idx = -1
        all_items: list[str] = []
        cell_adj_vals: dict[int, list[str]] = {}

        for cell_idx, cell in enumerate(row):
            if "<br" not in cell.lower():
                continue
            parts = _split_br_cell(cell)
            adj_in_cell: list[str] = []

            for p in parts:
                p_norm = _normalize_adjustment_item(p)
                if SIMPLE_TARGET_PATTERN.match(p_norm):
                    all_items.append(p_norm)
                elif SIMPLE_A_RATIO_PATTERN.match(p) and a_value is None:
                    a_value = SIMPLE_A_RATIO_PATTERN.match(p).group(1)
                    a_cell_idx = cell_idx
                elif SIMPLE_B_RATIO_PATTERN.match(p) and b_value is None:
                    b_value = SIMPLE_B_RATIO_PATTERN.match(p).group(1)
                    b_cell_idx = cell_idx
                elif ADJUSTMENT_PATTERN.match(p) or p == "비적용":
                    adj_in_cell.append(p)
                else:
                    # Handle concatenated tokens like "B100+10" or "+20비적용"
                    for tok in _extract_value_tokens(p):
                        if SIMPLE_A_RATIO_PATTERN.match(tok) and a_value is None:
                            a_value = SIMPLE_A_RATIO_PATTERN.match(tok).group(1)
                            a_cell_idx = cell_idx
                        elif SIMPLE_B_RATIO_PATTERN.match(tok) and b_value is None:
                            b_value = SIMPLE_B_RATIO_PATTERN.match(tok).group(1)
                            b_cell_idx = cell_idx
                        elif ADJUSTMENT_PATTERN.match(tok) or tok == "비적용":
                            adj_in_cell.append(tok)

            if adj_in_cell:
                cell_adj_vals[cell_idx] = adj_in_cell

        if a_value is None or b_value is None or not all_items:
            continue

        b_item_count = sum(
            1 for item in all_items
            if SIMPLE_TARGET_PATTERN.match(item)
            and SIMPLE_TARGET_PATTERN.match(item).group(1) == "B"
        )

        a_adj_vals: list[str] = []
        b_adj_vals: list[str] = []
        for ci in sorted(cell_adj_vals):
            vals = cell_adj_vals[ci]
            if ci < b_cell_idx:
                a_adj_vals.extend(vals)
            elif ci == b_cell_idx:
                # Split: last b_item_count are B adj, any surplus at front are A adj
                if len(vals) > b_item_count:
                    a_adj_vals.extend(vals[:-b_item_count])
                    b_adj_vals.extend(vals[-b_item_count:])
                else:
                    b_adj_vals.extend(vals)
            else:
                # Cells after B ratio: residual A adj vals
                a_adj_vals.extend(vals)

        if not b_adj_vals:
            adj_rows = _pair_simple_ab_items_values(all_items, a_adj_vals)
        else:
            if not a_adj_vals:
                a_adj_vals = b_adj_vals
            adj_rows = _pair_simple_ab_items_targeted(all_items, a_adj_vals, b_adj_vals)

        if not adj_rows:
            continue
        return _format_simple_ab_table(a_value, b_value, adj_rows)

    return None


def _extract_simple_ab_adj_from_rows(
    rows: list[list[str]], skip_row: int
) -> list[tuple[str, str, str, str]]:
    """Return (target, item, a_adj, b_adj) tuples from proper adjustment rows."""
    result: list[tuple[str, str, str, str]] = []
    for row_idx, row in enumerate(rows):
        if row_idx == skip_row:
            continue
        norm = [_normalize_cell(c) for c in row]
        nonempty = [c for c in norm if c]
        if not nonempty:
            continue

        # Find target+item cell (starts with "A " or "B ")
        target_cell = next((c for c in nonempty if SIMPLE_TARGET_PATTERN.match(c)), None)
        if target_cell is None:
            continue
        m = SIMPLE_TARGET_PATTERN.match(target_cell)
        target, item = m.group(1), _normalize_adjustment_item(m.group(2))

        # Find adjustment value
        value = next(
            (c for c in reversed(nonempty) if ADJUSTMENT_PATTERN.match(c) or c == "비적용"),
            None,
        )
        if value is None:
            continue

        a_adj = value if target == "A" else ""
        b_adj = value if target == "B" else ""
        result.append((target, item, a_adj, b_adj))
    return result


def _pair_simple_ab_items_values(
    items: list[str], values: list[str]
) -> list[tuple[str, str, str, str]]:
    """Pair crammed A/B items with their adjustment values."""
    result: list[tuple[str, str, str, str]] = []
    val_iter = iter(values)
    for item_cell in items:
        m = SIMPLE_TARGET_PATTERN.match(item_cell)
        if m is None:
            continue
        target = m.group(1)
        item = _normalize_adjustment_item(m.group(2))
        value = next(val_iter, None)
        if value is None:
            break
        a_adj = value if target == "A" else ""
        b_adj = value if target == "B" else ""
        result.append((target, item, a_adj, b_adj))
    return result


def _pair_simple_ab_items_targeted(
    items: list[str], a_adj_vals: list[str], b_adj_vals: list[str]
) -> list[tuple[str, str, str, str]]:
    """Pair A/B items with target-specific adjustment values.

    A-prefixed items consume from a_adj_vals; B-prefixed items consume from
    b_adj_vals. Items with no matching value are dropped.
    """
    result: list[tuple[str, str, str, str]] = []
    a_iter = iter(a_adj_vals)
    b_iter = iter(b_adj_vals)
    for item_cell in items:
        m = SIMPLE_TARGET_PATTERN.match(item_cell)
        if m is None:
            continue
        target = m.group(1)
        item = _normalize_adjustment_item(m.group(2))
        if target == "A":
            value = next(a_iter, None)
            if value is None:
                continue
            result.append((target, item, value, ""))
        else:
            value = next(b_iter, None)
            if value is None:
                continue
            result.append((target, item, "", value))
    return result


def _format_simple_ab_table(
    a_value: str, b_value: str, adj_rows: list[tuple[str, str, str, str]]
) -> str:
    lines = [
        "### 기본 과실비율",
        "",
        "| A 과실 | B 과실 |",
        "| --- | --- |",
        f"| A{a_value} | B{b_value} |",
        "",
        "### 과실비율 조정예시",
        "",
        "| 대상 | 수정요소 | A 조정 | B 조정 |",
        "| --- | --- | --- | --- |",
    ]
    for target, item, a_adj, b_adj in adj_rows:
        lines.append(f"| {target} | {_escape_cell(item)} | {a_adj} | {b_adj} |")
    return "\n".join(lines)


def _split_br_cell(cell: str) -> list[str]:
    """Split a raw cell by <br/> tags and normalize each part, filtering empties."""
    parts = BR_TAG_PATTERN.split(cell)
    return [_normalize_cell(p) for p in parts if _normalize_cell(p)]


def _is_kr_case_label(s: str) -> bool:
    return bool(KR_CASE_LABEL_PATTERN.match(s))


def _rewrite_pedestrian_fault_ratio_table(rows: list[list[str]]) -> str | None:
    """Handle pedestrian tables where (가)/(나) cases are packed into <br/> cells."""
    result = _try_pedestrian_type_row_format(rows)
    if result is not None:
        return result
    result = _try_pedestrian_crammed_format(rows)
    if result is not None:
        return result
    return _try_pedestrian_single_ratio(rows)


def _try_pedestrian_single_ratio(rows: list[list[str]]) -> str | None:
    """Single-ratio pedestrian table with no (가)/(나) sub-types.

    Pattern: a cell starts with a plain integer (basic ratio) then lists
    adjustment items; the next cell starts with the same number then lists
    adjustment values. The row must contain a pedestrian header keyword to
    avoid false positives.
    """
    for row in rows:
        has_ped_header = any(
            any(kw in _normalize_cell(c) for kw in PEDESTRIAN_HEADER_KEYWORDS)
            for c in row
        )
        if not has_ped_header:
            continue

        for cell_idx, cell in enumerate(row):
            if "<br" not in cell.lower():
                continue
            parts = _split_br_cell(cell)
            if len(parts) < 2 or not PLAIN_NUMBER_PATTERN.match(parts[0]):
                continue
            basic_ratio = parts[0]
            items = [p for p in parts[1:] if _is_item_cell(p)]
            if not items:
                continue

            for val_idx in range(cell_idx + 1, len(row)):
                val_parts = _split_br_cell(row[val_idx])
                if not val_parts or val_parts[0] != basic_ratio:
                    continue
                adj_values = [v for v in val_parts[1:] if ADJ_VALUE_PATTERN.match(v)]
                if not adj_values:
                    continue
                adj_rows = list(zip(items, adj_values[: len(items)]))
                lines = [
                    "### 기본 과실비율",
                    "",
                    "| 보행자 과실 |",
                    "| --- |",
                    f"| {basic_ratio} |",
                    "",
                    "### 과실비율 조정예시",
                    "",
                    "| 수정요소 | 조정 |",
                    "| --- | --- |",
                ]
                for item, value in adj_rows:
                    lines.append(f"| {_escape_cell(item)} | {value} |")
                return "\n".join(lines)

    return None


def _try_pedestrian_type_row_format(rows: list[list[str]]) -> str | None:
    """Format A: one row has a <br/>-cell containing (가)/(나) type labels
    (possibly preceded by repeated-title prefix parts) and an adjacent cell
    with the corresponding plain integers. Adjustment items may follow the
    basic ratios in the value cell, with values in the next adjacent cell."""
    for row_idx, row in enumerate(rows):
        for cell_idx, cell in enumerate(row):
            if "<br" not in cell.lower():
                continue
            parts = _split_br_cell(cell)
            if len(parts) < 2:
                continue

            # Find the first consecutive block of (가)/(나)/... case labels.
            # Pre-label parts (title prefix) are allowed; post-label parts may
            # only be circled-number markers (e.g. "①②") — adjustment items
            # appearing after the labels mean this is a crammed table, not a
            # type-row table.
            first_case_idx = next(
                (i for i, p in enumerate(parts) if _is_kr_case_label(p)), -1
            )
            if first_case_idx == -1:
                continue
            case_parts: list[str] = []
            last_case_idx = first_case_idx
            for i in range(first_case_idx, len(parts)):
                if _is_kr_case_label(parts[i]):
                    case_parts.append(parts[i])
                    last_case_idx = i
                else:
                    break
            if len(case_parts) < 2 or not any(p.startswith("(나)") for p in case_parts):
                continue
            post_label = parts[last_case_idx + 1 :]
            if post_label and not all(
                MARKER_PATTERN.match(p) or bool(CIRCLED_NUMBER_PATTERN.search(p))
                for p in post_label
            ):
                continue

            # Find the adjacent cell that has at least len(case_parts) plain integers
            for val_idx in range(cell_idx + 1, len(row)):
                val_parts = _split_br_cell(row[val_idx])
                if not val_parts:
                    norm = _normalize_cell(row[val_idx])
                    if norm and PLAIN_NUMBER_PATTERN.match(norm):
                        val_parts = [norm]
                    else:
                        continue

                numeric_indices = [i for i, p in enumerate(val_parts) if PLAIN_NUMBER_PATTERN.match(p)]
                if len(numeric_indices) < len(case_parts):
                    continue

                basic_ratios = list(zip(case_parts, [val_parts[i] for i in numeric_indices[: len(case_parts)]]))
                last_ratio_idx = numeric_indices[len(case_parts) - 1]

                # Items packed after the basic ratios in the same cell
                val_items = [p for p in val_parts[last_ratio_idx + 1 :] if _is_item_cell(p)]
                if val_items:
                    # Adjustment values live in the next cell(s)
                    for adj_val_idx in range(val_idx + 1, len(row)):
                        adj_val_parts = _split_br_cell(row[adj_val_idx])
                        adj_values = [p for p in adj_val_parts if ADJ_VALUE_PATTERN.match(p)]
                        if adj_values:
                            adj_rows = list(zip(val_items, adj_values[: len(val_items)]))
                            return _format_pedestrian_table(basic_ratios, adj_rows)

                # Fall back: adjustments are in separate table rows
                adj_rows = _extract_ped_adj_from_rows(rows, skip_row=row_idx)
                return _format_pedestrian_table(basic_ratios, adj_rows)

    return None


def _try_pedestrian_crammed_format(rows: list[list[str]]) -> str | None:
    """Format B (pages 74/82): a single row has ALL items (types + adjustments)
    packed in one <br/>-cell and ALL values in another <br/>-cell."""
    for row in rows:
        for cell_idx, cell in enumerate(row):
            if "<br" not in cell.lower():
                continue
            parts = _split_br_cell(cell)
            if len(parts) < 3:
                continue
            if not parts[0].startswith("(가)"):
                continue

            # Split into type labels and adjustment items
            type_labels: list[str] = []
            adj_items: list[str] = []
            past_types = False
            for p in parts:
                if not past_types and _is_kr_case_label(p):
                    type_labels.append(p)
                else:
                    past_types = True
                    if _is_item_cell(p):
                        adj_items.append(p)

            if len(type_labels) < 2:
                continue

            # Find value cell: starts with N plain integers for basic ratios
            for val_idx in range(cell_idx + 1, len(row)):
                val_cell = row[val_idx]
                if "<br" not in val_cell.lower():
                    continue
                vparts = _split_br_cell(val_cell)
                if len(vparts) < len(type_labels):
                    continue
                if not all(PLAIN_NUMBER_PATTERN.match(vparts[i]) for i in range(len(type_labels))):
                    continue

                basic_ratios = list(zip(type_labels, vparts[: len(type_labels)]))
                adj_values = [v for v in vparts[len(type_labels) :] if ADJ_VALUE_PATTERN.match(v)]
                adj_rows = list(zip(adj_items, adj_values[: len(adj_items)]))
                return _format_pedestrian_table(basic_ratios, adj_rows)

    return None


def _extract_ped_adj_from_rows(rows: list[list[str]], skip_row: int) -> list[tuple[str, str]]:
    """Extract (item, value) adjustment pairs from table rows, skipping the basic ratio row."""
    adj: list[tuple[str, str]] = []
    for row_idx, row in enumerate(rows):
        if row_idx == skip_row:
            continue
        cells = [_normalize_cell(c) for c in row]
        nonempty = [c for c in cells if c]
        if not nonempty:
            continue
        if len(nonempty) == 1 and nonempty[0] in PEDESTRIAN_HEADER_KEYWORDS:
            continue
        value = next(
            (c for c in reversed(nonempty) if ADJUSTMENT_PATTERN.match(c) or c == "비적용"),
            None,
        )
        if value is None:
            continue
        value_pos = len(nonempty) - 1 - list(reversed(nonempty)).index(value)
        item = _nearest_item_before_value(nonempty, value_pos)
        if item:
            adj.append((item, value))
    return adj


def _format_pedestrian_table(
    basic_ratios: list[tuple[str, str]], adj_rows: list[tuple[str, str]]
) -> str:
    lines = [
        "### 기본 과실비율",
        "",
        "| 유형 | 보행자 과실 |",
        "| --- | --- |",
    ]
    for type_label, value in basic_ratios:
        lines.append(f"| {_escape_cell(type_label)} | {value} |")

    if adj_rows:
        lines.extend(
            [
                "",
                "### 과실비율 조정예시",
                "",
                "| 수정요소 | 조정 |",
                "| --- | --- |",
            ]
        )
        for item, value in adj_rows:
            lines.append(f"| {_escape_cell(item)} | {value} |")

    return "\n".join(lines)


def _rewrite_ab_fault_ratio_table(rows: list[list[str]]) -> str | None:
    normalized_rows = [[_normalize_cell(cell) for cell in row] for row in rows]
    basic_rows = _extract_ab_basic_rows(normalized_rows)
    adjustment_rows = _extract_ab_adjustment_rows(normalized_rows)
    if not basic_rows or not adjustment_rows:
        return _try_ab_separated_cells(rows)

    lines = [
        "### 기본 과실비율",
        "",
        "| 유형 | A 과실 | B 과실 |",
        "| --- | --- | --- |",
    ]
    for case_label, a_value, b_value in basic_rows:
        lines.append(f"| ({case_label}) | A{a_value} | B{b_value} |")

    lines.extend(
        [
            "",
            "### 과실비율 조정예시",
            "",
            "| 대상 | 수정요소 | A 조정 | B 조정 |",
            "| --- | --- | --- | --- |",
        ]
    )
    for _, target, item, adjustment in adjustment_rows:
        a_adjustment = adjustment if target == "A" else ""
        b_adjustment = adjustment if target == "B" else ""
        lines.append(
            f"| {target} | {_escape_cell(item)} | {a_adjustment} | {b_adjustment} |"
        )
    return "\n".join(lines)


def _extract_ab_basic_rows(rows: list[list[str]]) -> list[tuple[str, str, str]]:
    for row in rows:
        cells = [_normalize_cell(cell) for cell in row]
        joined = " ".join(cell for cell in cells if cell)
        case_a_values = CASE_A_VALUE_PATTERN.findall(joined)
        b_values = B_VALUE_PATTERN.findall(joined)
        if case_a_values and len(b_values) >= len(case_a_values):
            return [
                (case_label, a_value, b_values[index])
                for index, (case_label, a_value) in enumerate(case_a_values)
            ]
    return []


def _extract_ab_adjustment_rows(rows: list[list[str]]) -> list[tuple[str, str, str, str]]:
    adjustment_rows: list[tuple[str, str, str, str]] = []
    current_case = ""

    for row in rows:
        nonempty_cells = [_normalize_cell(cell) for cell in row if _normalize_cell(cell)]
        if not nonempty_cells:
            continue

        explicit_case = next(
            (cell.strip("()") for cell in nonempty_cells if cell in {"(가)", "(나)", "(다)"}),
            "",
        )
        target_item = _target_item(nonempty_cells)
        if explicit_case and target_item is None:
            current_case = explicit_case
            continue
        if explicit_case:
            current_case = explicit_case
        if not current_case or target_item is None:
            continue

        adjustment = next((cell for cell in reversed(nonempty_cells) if ADJUSTMENT_PATTERN.match(cell)), "")
        if not adjustment:
            continue

        target, item = target_item
        adjustment_rows.append((current_case, target, item, adjustment))

    return adjustment_rows


def _target_item(cells: list[str]) -> tuple[str, str] | None:
    for cell in cells:
        match = TARGET_ITEM_PATTERN.match(cell)
        if match is None:
            continue
        target = match.group(1)
        item = _normalize_adjustment_item(match.group(2))
        if item:
            return target, item
    return None


def _normalize_adjustment_item(item: str) -> str:
    item = CIRCLED_NUMBER_PATTERN.sub("", item)
    item = item.replace("진로변경", "진로변경 ")
    return " ".join(item.split())


def _extract_fault_rows(rows: list[list[str]]) -> list[tuple[str, str, str]]:
    fault_rows: list[tuple[str, str, str]] = []
    current_marker = ""
    previous_value = ""

    for row in rows:
        cells = [_normalize_cell(cell) for cell in row]
        nonempty_cells = [cell for cell in cells if cell]
        if not nonempty_cells:
            continue

        marker = next((cell for cell in nonempty_cells if MARKER_PATTERN.match(cell)), "")
        if marker:
            current_marker = marker

        basic_row = _basic_fault_row(nonempty_cells)
        if basic_row is not None:
            fault_rows.append(basic_row)
            continue

        value_cells = _value_cells(nonempty_cells)
        if not value_cells and previous_value == "비적용":
            item = _nearest_item_before_value(nonempty_cells, len(nonempty_cells))
            if item is not None:
                group = current_marker or "조정"
                fault_rows.append((group, item, previous_value))
            continue

        for value_index, value in value_cells:
            item = _nearest_item_before_value(nonempty_cells, value_index)
            if item is None:
                continue
            group = current_marker or "조정"
            fault_rows.append((group, item, value))
            previous_value = value

    return fault_rows


def _basic_fault_row(cells: list[str]) -> tuple[str, str, str] | None:
    item = next((cell for cell in cells if "기본 과실비율" in cell), None)
    if item is None:
        return None
    value = next((cell for cell in reversed(cells) if cell.isdigit()), None)
    if value is None:
        return None
    return ("기본 과실비율", item, value)


def _value_cells(cells: list[str]) -> list[tuple[int, str]]:
    values: list[tuple[int, str]] = []
    for index, cell in enumerate(cells):
        if VALUE_PATTERN.match(cell) and not MARKER_PATTERN.match(cell):
            values.append((index, cell))
    return values


def _nearest_item_before_value(cells: list[str], value_index: int) -> str | None:
    for candidate in reversed(cells[:value_index]):
        if _is_item_cell(candidate):
            return candidate
    return None


def _is_item_cell(cell: str) -> bool:
    if not cell or cell in SKIP_ITEM_CELLS:
        return False
    if MARKER_PATTERN.match(cell) or VALUE_PATTERN.match(cell):
        return False
    if "상황도" in cell:
        return False
    return True


def _format_fault_rows(rows: list[tuple[str, str, str]]) -> str:
    lines = [
        "| 구분 | 항목 | 과실 |",
        "| --- | --- | --- |",
    ]
    for group, item, value in rows:
        lines.append(f"| {_escape_cell(group)} | {_escape_cell(item)} | {_escape_cell(value)} |")
    return "\n".join(lines)


def _parse_markdown_row(line: str) -> list[str]:
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return stripped.split("|")


def _normalize_cell(cell: str) -> str:
    no_tags = HTML_TAG_PATTERN.sub(" ", cell)
    return " ".join(no_tags.replace("\\", "").split())


def _escape_cell(cell: str) -> str:
    return cell.replace("|", "\\|")


def _is_table_line(line: str) -> bool:
    return TABLE_LINE_PATTERN.match(line) is not None


def _is_separator_row(line: str) -> bool:
    cells = [_normalize_cell(cell) for cell in _parse_markdown_row(line)]
    nonempty_cells = [cell for cell in cells if cell]
    return bool(nonempty_cells) and all(SEPARATOR_CELL_PATTERN.match(cell) for cell in nonempty_cells)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rewrite LlamaParse accident diagram tables to fault-ratio-only markdown."
    )
    parser.add_argument("input", type=Path, help="Input markdown file.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output markdown file. Defaults to stdout.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cleaned = clean_markdown(args.input.read_text(encoding="utf-8"))
    if args.output is None:
        print(cleaned, end="")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(cleaned, encoding="utf-8")


if __name__ == "__main__":
    main()
