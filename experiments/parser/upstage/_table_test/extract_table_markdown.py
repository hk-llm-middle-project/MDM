"""
extract_table_markdown.py
─────────────────────────
Regex-only cleaner for Upstage page_content table strings.
LLM 없이 정규식만으로 두 가지 버전의 마크다운 표를 생성합니다.

  버전 1: ![image]() 및 <figcaption>…</table> 이미지 플레이스홀더 제거
  버전 2: 플레이스홀더를 한 줄로 압축하여 유지

Usage:
    # page_content 문자열을 직접 전달
    python extract_table_markdown.py '| 보19 | ...'

    # stdin에서 읽기
    echo '| 보19 | ...' | python extract_table_markdown.py --stdin

    # JSON 전체 표 처리 결과를 파일로 저장
    python extract_table_markdown.py --json data/upstage_output/main_pdf/raw/parsed_documents.json

    # 무작위 N개 표 샘플만 출력 (기본 5개)
    python extract_table_markdown.py --json <path> --sample 3

    # 전체 표 통계 및 처리 실패 케이스 확인
    python extract_table_markdown.py --json <path> --check
"""

from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path


# ─── 이미지/차트 플레이스홀더 패턴 ──────────────────────────────────────────

# Upstage 마크다운 이미지 플레이스홀더: ![alt](url)
MD_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")

# Upstage 차트 플레이스홀더: <figcaption>…</figcaption><table>…</table>
# 표 셀 안에 인라인으로 들어오는 HTML 차트 블록
FIGCAP_RE = re.compile(
    r"<figcaption\b[^>]*>.*?</figcaption>\s*<table\b[^>]*>.*?</table>",
    re.DOTALL | re.IGNORECASE,
)

# 나머지 잔여 HTML 태그
HTML_TAG_RE = re.compile(r"<[^>]+>")

# 마크다운 표 구분선 행: | --- | --- |
SEPARATOR_RE = re.compile(r"^\|[\s\-|]+\|$")


# ─── 행/셀 파싱 ──────────────────────────────────────────────────────────────

def _collect_rows(table_text: str) -> list[list[str]]:
    """
    마크다운 표를 행 단위로 분리한 뒤 셀 목록으로 반환합니다.

    Upstage는 셀 내부에 개행을 포함할 수 있습니다.
    '|'로 시작하지 않는 줄은 이전 행의 연속으로 처리합니다.

    예:
        | ![image](...)         ← 새 행 시작
        (가) 녹색               ← 셀 내부 개행 (같은 행)
        (나) 적색 | 다음셀 | .. ← 셀 구분자 포함, 여전히 같은 행
    """
    raw_rows: list[list[str]] = []
    current: list[str] = []

    for line in table_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("|"):
            if current:
                raw_rows.append(current)
            current = [stripped]
        elif current:
            current.append(stripped)

    if current:
        raw_rows.append(current)

    rows: list[list[str]] = []
    for line_group in raw_rows:
        # 같은 행의 여러 줄을 공백으로 합쳐서 셀 단위로 분리
        joined = " ".join(part for part in line_group if part)
        if SEPARATOR_RE.match(joined):
            continue  # 구분선은 제거 후 재생성
        cells = [c.strip() for c in joined.strip("|").split("|")]
        rows.append(cells)

    return rows


def _render(rows: list[list[str]]) -> str:
    """행 목록을 마크다운 표 문자열로 렌더링합니다."""
    if not rows:
        return ""
    n = max(len(r) for r in rows)
    lines: list[str] = []
    for i, row in enumerate(rows):
        padded = row + [""] * (n - len(row))
        lines.append("| " + " | ".join(padded) + " |")
        if i == 0:
            lines.append("| " + " | ".join(["---"] * n) + " |")
    return "\n".join(lines)


# ─── 셀 정제 함수 ────────────────────────────────────────────────────────────

def _clean_no_images(cell: str) -> str:
    """셀에서 이미지/차트 플레이스홀더를 모두 제거합니다."""
    cell = MD_IMAGE_RE.sub("", cell)
    cell = FIGCAP_RE.sub("", cell)
    cell = HTML_TAG_RE.sub("", cell)
    return " ".join(cell.split())


def _clean_keep_images(cell: str) -> str:
    """셀의 멀티라인 HTML 블록을 한 줄로 압축하되 내용은 유지합니다."""
    cell = FIGCAP_RE.sub(lambda m: " ".join(m.group().split()), cell)
    return " ".join(cell.split())


# ─── 표 외 텍스트 정제 ───────────────────────────────────────────────────────

def _strip_placeholders(text: str) -> str:
    """표 외 텍스트에서 이미지/HTML 플레이스홀더를 제거합니다."""
    text = MD_IMAGE_RE.sub("", text)
    text = FIGCAP_RE.sub("", text)
    text = HTML_TAG_RE.sub("", text)
    return text


def _condense_placeholders(text: str) -> str:
    """표 외 텍스트의 멀티라인 HTML 블록을 한 줄로 압축합니다."""
    return FIGCAP_RE.sub(lambda m: " ".join(m.group().split()), text)


# ─── 콘텐츠 분리 ─────────────────────────────────────────────────────────────

def _segments(content: str) -> list[tuple[str, bool]]:
    """
    콘텐츠를 (텍스트, is_table) 쌍의 목록으로 분리합니다.
    '|'로 시작하는 연속된 줄 블록을 표로 판별합니다.
    """
    segs: list[tuple[str, bool]] = []
    cur: list[str] = []
    in_table = False

    for line in content.splitlines():
        is_tbl = line.strip().startswith("|")
        if is_tbl != in_table:
            if cur:
                segs.append(("\n".join(cur), in_table))
                cur = []
            in_table = is_tbl
        cur.append(line)

    if cur:
        segs.append(("\n".join(cur), in_table))
    return segs


# ─── 핵심 처리 함수 ──────────────────────────────────────────────────────────

def render_both(raw: str) -> tuple[str, str]:
    """
    page_content 문자열을 받아 두 가지 버전의 마크다운을 반환합니다.

    Returns:
        (버전1_플레이스홀더제거, 버전2_플레이스홀더유지)
    """
    # JSON raw string의 이스케이프된 \\n을 실제 개행으로 변환
    content = raw.replace("\\n", "\n")

    parts_no: list[str] = []
    parts_yes: list[str] = []

    for text, is_table in _segments(content):
        if is_table:
            rows = _collect_rows(text)
            parts_no.append(_render([[_clean_no_images(c) for c in r] for r in rows]))
            parts_yes.append(_render([[_clean_keep_images(c) for c in r] for r in rows]))
        else:
            parts_no.append(_strip_placeholders(text).strip())
            parts_yes.append(_condense_placeholders(text).strip())

    def join(parts: list[str]) -> str:
        return "\n\n".join(p for p in parts if p.strip())

    return join(parts_no), join(parts_yes)


# ─── 출력 ────────────────────────────────────────────────────────────────────

def print_result(raw: str, label: str = "") -> None:
    without, with_ = render_both(raw)
    if label:
        print(f"### {label}\n")
    print("# 버전 1: 이미지 플레이스홀더 제거\n")
    print(without)
    print("\n---\n")
    print("# 버전 2: 이미지 플레이스홀더 유지\n")
    print(with_)


# ─── 검증 / 통계 ─────────────────────────────────────────────────────────────

def _placeholder_type(raw: str) -> str:
    if "![image]" in raw:
        return "md_image"
    if "<figcaption" in raw:
        return "figcaption"
    if "<table" in raw:
        return "nested_html"
    return "none"


def check_all(items: list[dict]) -> None:
    """
    전체 표 청크를 처리하고 플레이스홀더 유형별 통계와
    빈 셀이 과도한 이상 케이스를 출력합니다.
    """
    table_items = [
        (i, d) for i, d in enumerate(items)
        if str(d.get("metadata", {}).get("category", "")).lower() == "table"
    ]

    stats: dict[str, int] = {"md_image": 0, "figcaption": 0, "nested_html": 0, "none": 0}
    problems: list[tuple[int, str, str]] = []  # (index, reason, preview)

    for idx, doc in table_items:
        raw = doc.get("page_content", "")
        ptype = _placeholder_type(raw)
        stats[ptype] += 1

        without, _ = render_both(raw)
        rows = _collect_rows(without.replace("\\n", "\n"))
        if not rows:
            problems.append((idx, "행 추출 결과 없음", raw[:80]))
            continue

        # 데이터 행(헤더 제외)에서 모든 셀이 비어있는 행 비율 체크
        empty_rows = sum(
            1 for r in rows[1:] if all(c == "" or c == "---" for c in r)
        )
        if rows[1:] and empty_rows / len(rows[1:]) > 0.5:
            problems.append((idx, f"빈 행 비율 높음 ({empty_rows}/{len(rows[1:])})", raw[:80]))

    print("=" * 50)
    print("전체 표 청크 통계")
    print("=" * 50)
    print(f"  총 표 수          : {len(table_items)}")
    print(f"  ![image]() 포함   : {stats['md_image']}")
    print(f"  <figcaption> 포함 : {stats['figcaption']}")
    print(f"  <table> 중첩      : {stats['nested_html']}")
    print(f"  플레이스홀더 없음 : {stats['none']}")
    print()
    if problems:
        print(f"⚠️  이상 케이스 {len(problems)}건")
        print("-" * 50)
        for idx, reason, preview in problems[:10]:
            print(f"  chunk index {idx}: {reason}")
            print(f"    {preview!r}")
    else:
        print("✅ 이상 케이스 없음 — 모든 표 정상 처리")


# ─── 진입점 ──────────────────────────────────────────────────────────────────

def main() -> None:
    args = sys.argv[1:]

    if not args:
        print(__doc__)
        sys.exit(0)

    if args[0] == "--stdin":
        print_result(sys.stdin.read())
        return

    if args[0] == "--json" and len(args) >= 2:
        path = Path(args[1])
        data = json.loads(path.read_text(encoding="utf-8"))
        items = data if isinstance(data, list) else [data]

        # --check: 전체 통계 및 이상 케이스 확인
        if "--check" in args:
            check_all(items)
            return

        # 표 청크만 추림
        table_items = [
            (i, d) for i, d in enumerate(items)
            if str(d.get("metadata", {}).get("category", "")).lower() == "table"
        ]

        # --sample N: 무작위 N개만 출력
        if "--sample" in args:
            n = int(args[args.index("--sample") + 1]) if args.index("--sample") + 1 < len(args) else 5
            table_items = random.sample(table_items, min(n, len(table_items)))

        for i, doc in table_items:
            raw = doc.get("page_content", "")
            page = doc.get("metadata", {}).get("page", "?")
            print_result(raw, label=f"chunk {i}  |  page {page}  |  placeholder: {_placeholder_type(raw)}")
            print("\n" + "=" * 60 + "\n")
        return

    # 나머지 인자를 raw page_content 문자열로 처리
    for raw in args:
        print_result(raw)


if __name__ == "__main__":
    main()
