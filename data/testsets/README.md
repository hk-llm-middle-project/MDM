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

- `common_retrieval_v0.1.0_20260428.jsonl`
  - 기준 데이터: Upstage 테스트셋을 공통 파서 비교용으로 변환
  - 목적: pdfplumber, llamaparser, upstage 검색 성능 비교
  - 규모: 20개 케이스
  - 주요 필드: `question`, `reference`, `expected_party_type`, `expected_location`, `expected_keywords`
  - 참고 필드: `expected_diagram_ids_for_reference_only`
  - 주의: 공통 비교에서는 `diagram_id`를 채점하지 않고, 공통 metadata인 `party_type`, `location`과 키워드 포함 여부를 중심으로 봅니다.
