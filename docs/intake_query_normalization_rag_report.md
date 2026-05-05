# Intake 질문 정규화 전후 RAG 검색 성능 비교 보고서

## 요약

Intake 단계에서 사용자 원문 질문을 문서 taxonomy에 가까운 `retrieval_query`로 정규화하면, `llamaparser/fixed/bge + vectorstore + top-30` 기준으로 RAG 검색 후보군 품질이 개선된다. `top-30`은 reranker가 후단에서 재정렬할 후보군으로 볼 수 있으므로, 최종 답변용 `top-5`보다 정규화의 후보 회수 효과를 더 잘 보여준다.

최신 수정 후 clean query 기준으로, 동일한 intake metadata filter를 적용했을 때 `keyword_coverage`는 원문 검색 `0.5650`에서 정규화 검색 `0.5911`로 올랐다. 변화량은 `+0.0261p`, 상대 개선율은 약 `+4.6%`다. 정규화 질의와 원문 질의를 함께 검색한 뒤 RRF로 합친 hybrid 후보군은 `0.6039`까지 올라, 정규화 단독 대비 `+0.0128p`, 원문+filter 대비 `+0.0389p` 개선됐다.

다만 `diagram_id_hit`는 모든 조건에서 `0.0333`으로 변하지 않았다. 즉, 정규화와 hybrid 검색은 후보군 안의 관련 키워드 회수율을 높였지만, 기대 도표 ID를 직접 top-30 후보군 안으로 끌어올리는 구조적 개선까지는 아직 확인되지 않았다.

## 평가 방법

평가 데이터는 `data/testsets/langsmith/retrieval_eval.jsonl`의 retrieval 테스트셋 30건을 사용했다. 검색 조합은 로컬에 존재하는 `llamaparser/fixed/bge` 벡터스토어, retriever는 `vectorstore`, reranker는 `none`, 검색 후보 수는 `top-30`으로 고정했다.

비교 조건은 다음과 같다.

| 조건 | 설명 |
| --- | --- |
| `raw_no_filter` | 사용자 원문 질문만 사용 |
| `raw_with_intake_filter` | 사용자 원문 질문 + intake metadata filter |
| `normalized_with_intake_filter` | intake 정규화 질의 + intake metadata filter |
| `hybrid_rrf_with_intake_filter` | 정규화 질의 top-30과 원문 질의 top-30을 검색한 뒤 RRF로 병합 |

정규화 효과만 보기 위한 주 비교는 `raw_with_intake_filter` 대비 `normalized_with_intake_filter`이다. Hybrid 실험은 정규화가 틀릴 때 원문 검색이 안전망이 되는지 확인하기 위한 보조 비교다.

## 전체 지표

| 지표 | Raw + filter | Normalized + filter | Hybrid RRF + filter |
| --- | ---: | ---: | ---: |
| `diagram_id_hit` | 0.0333 | 0.0333 | 0.0333 |
| `keyword_coverage` | 0.5650 | 0.5911 | 0.6039 |
| `retrieval_relevance` | 0.7197 | 0.7249 | 0.7274 |
| `critical_error` | 0.9667 | 0.9667 | 0.9667 |

정규화 단독 효과:

| 비교 | `keyword_coverage` | `retrieval_relevance` |
| --- | ---: | ---: |
| Normalized - Raw filter | +0.0261p | +0.0052p |

Hybrid 효과:

| 비교 | `keyword_coverage` | `retrieval_relevance` |
| --- | ---: | ---: |
| Hybrid - Normalized | +0.0128p | +0.0026p |
| Hybrid - Raw filter | +0.0389p | +0.0078p |

원문 질문만 사용한 baseline과 최신 hybrid 결과를 비교하면 `keyword_coverage`는 `0.5028 -> 0.6039`로 `+0.1011p`, 상대 개선율 약 `+20.1%`다. 이는 intake filter, query normalization, raw-query safety net이 함께 적용된 후보군 기준 총 효과로 볼 수 있다.

## 수정한 개선사항

이번 개선에서 실제 코드에 반영한 항목은 세 가지다.

1. LLM이 `"null"`, `"None"`, `"unknown"` 같은 문자열로 보낸 결측값을 실제 `None`으로 정리했다.
2. 정규화 질의 조립 단계에서 `null추돌`, `nullnull` 같은 오염 토큰이 생성되거나 유지되지 않게 했다.
3. 자전거 비보호 좌회전 케이스에서 원문에 자전거가 명시되어 있으면, intake가 `party_type="자동차"`로 잘못 뽑아도 party-aware 보정 규칙이 동작하도록 했다.

수정 후 평가 산출물의 정규화 질의에는 `null`/`none` 토큰이 남지 않는다.

## 케이스 관찰

정규화로 크게 좋아진 케이스는 추돌, 진로변경, 신호 없는 교차로처럼 문서 표제어가 분명한 사고 유형이었다.

| 케이스 | Raw keyword | Normalized keyword | 변화 | 정규화 질의 |
| --- | ---: | ---: | ---: | --- |
| `retrieval_028` | 0.2500 | 0.7500 | +0.5000 | 추돌사고 |
| `retrieval_015` | 0.0000 | 0.4000 | +0.4000 | 직진 대 좌회전 사고, 상대차량이 측면에서 진입, 오른쪽 도로, 왼쪽 도로 |
| `retrieval_019` | 0.0000 | 0.4000 | +0.4000 | 진로변경 대 직진 사고, 진로변경 사고, 진로변경, 같은 방향 |
| `retrieval_018` | 0.2000 | 0.6000 | +0.4000 | 추돌사고, 후행 차량이 선행 차량을 들이받음, 같은 방향 |

정규화 단독으로 하락한 케이스도 있었다. 특히 보행자 횡단보도 케이스와 일부 자전거 교차로 케이스에서 원문 표현이 오히려 더 직접적인 검색 단서가 됐다.

| 케이스 | Raw keyword | Normalized keyword | Hybrid keyword | 관찰 |
| --- | ---: | ---: | ---: | --- |
| `retrieval_003` | 1.0000 | 0.6667 | 0.6667 | 보행자 정상 횡단 원문이 더 강한 단서 |
| `retrieval_004` | 1.0000 | 0.6667 | 0.6667 | 짧은 보행자 사고 원문이 더 직접적 |
| `retrieval_026` | 0.5000 | 0.2500 | 0.5000 | hybrid가 원문 검색을 안전망으로 회복 |
| `retrieval_027` | 0.2500 | 0.0000 | 0.0000 | 자전거 동일폭 교차로는 추가 규칙 필요 |

`retrieval_026`은 이번 실험의 핵심 사례다. 정규화 단독에서는 자전거/자동차 관계가 충분히 보존되지 않아 `0.25`로 떨어졌지만, hybrid RRF에서는 원문 검색 결과가 합쳐지면서 `0.50`으로 회복됐다.

## 해석

정규화는 자연어 질문을 문서 표제어에 맞추는 효과가 있다. 그래서 추돌, 진로변경, 직진 대 좌회전, 신호 없는 교차로처럼 문서 내 반복 표현이 뚜렷한 유형에서는 검색 문맥의 키워드 적중률이 올라간다.

반면 정규화는 원문의 모든 정보를 보존하지 못할 수 있다. 특히 보행자나 자전거처럼 party type 자체가 중요한 검색 축인 케이스에서는 원문 질의가 더 좋은 단서가 되는 경우가 있다. Hybrid RRF는 이 손실을 일부 완화했다. 다만 `retrieval_027`처럼 자전거 동일폭 교차로의 우선관계까지 필요한 케이스는 아직 회복하지 못했다.

따라서 현재 결론은 다음과 같다.

정규화는 후보군의 평균 품질을 개선한다. Hybrid 검색은 정규화 단독보다 더 안정적이다. 그러나 도표 ID 단위 정답 적중은 별도의 metadata/diagram-aware retrieval 개선이 필요하다.

## 다음 개선 제안

1. 자전거 동일폭 교차로, 오른쪽/왼쪽 도로 우선관계를 party-aware하게 보존한다.
2. Hybrid RRF를 평가 스크립트에 정식 옵션으로 추가하고, cross-encoder reranker까지 붙여 `top-30 -> top-5` 성능을 비교한다.
3. `diagram_id_hit` 개선을 위해 도표 ID, 사고 유형 번호, 표 제목 metadata를 검색 후보에 더 직접적으로 반영한다.

## 산출물

최신 비교 결과는 다음 파일에 저장했다.

- `evaluation/results/intake_query_normalization/20260504-150711-llamaparser-fixed-bge-vectorstore-top30-query-hybrid-party-aware.summary.json`
- `evaluation/results/intake_query_normalization/20260504-150711-llamaparser-fixed-bge-vectorstore-top30-query-hybrid-party-aware.csv`
- `evaluation/results/intake_query_normalization/intake_decisions_retrieval_eval.json`

