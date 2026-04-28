# 테스트셋

## 파일명 규칙

평가 데이터셋은 계속 버전이 늘어날 수 있으므로 아래 형식을 사용합니다.

```text
{source}_{task}_v{major}.{minor}.{patch}_{yyyymmdd}.jsonl
```

예시:

```text
upstage_retrieval_v0.1.0_20260428.jsonl
pdfplumber_retrieval_v0.1.0_20260428.jsonl
upstage_answer_eval_v0.1.0_20260428.jsonl
```

## 현재 파일

- `upstage_retrieval_v0.1.0_20260428.jsonl`
  - 기준 데이터: Upstage 최종 청크
  - 목적: 검색 성능 평가
  - 규모: 20개 케이스
  - 주요 필드: `question`, `reference`, `expected_diagram_ids`, `expected_party_type`, `expected_location`, `expected_chunk_types`, `expected_keywords`
