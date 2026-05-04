# Intake 질문 정규화 전후 RAG 검색 성능 비교 보고서

## 요약

Intake 단계에서 사용자 원문 질문을 문서 taxonomy에 가까운 `retrieval_query`로 정규화한 뒤 RAG 검색에 사용했을 때, 현재 검증 가능한 `llamaparser/fixed/bge + vectorstore + top-30` 조건에서는 검색 후보군 품질이 개선되었다. `top-30`은 reranker가 후단에서 재정렬할 후보군으로 볼 수 있으므로, 최종 답변용 `top-5`보다 정규화의 검색 회수 효과를 더 잘 보여준다.

핵심 개선은 정답 키워드 포함률에서 나타났다. 동일한 intake metadata filter를 적용한 상태에서 원문 질문을 검색한 경우와 정규화 질의를 검색한 경우를 비교하면, 수정 후 clean query 기준 `keyword_coverage`는 `0.5650 -> 0.5911`로 `+0.0261p`, 상대 개선율 약 `+4.6%`였다. 통합 검색 관련성 점수인 `retrieval_relevance`는 `0.7197 -> 0.7249`로 `+0.0052p`, 상대 개선율 약 `+0.7%`였다.

다만 `diagram_id_hit`는 `0.0333`으로 변하지 않았다. 즉, 질문 정규화는 검색된 문서 안에 기대 키워드를 더 포함시키는 데는 도움이 되었지만, 기대 도표 ID를 top-30 후보군 안으로 끌어올리는 수준의 구조적 개선까지는 확인되지 않았다.

## 비교 목적

이번 비교의 목적은 intake에서 수행하는 질문 정규화가 RAG 검색 성능을 실제로 얼마나 개선했는지 확인하는 것이다. 여기서 질문 정규화는 사용자 원문 질문을 그대로 임베딩 검색에 넣는 대신, 사고 유형을 문서 표제어와 가까운 검색어 조합으로 바꾸는 과정을 뜻한다.

예를 들면 다음과 같다.

| 구분 | 질의 |
| --- | --- |
| 정규화 전 | 양쪽 다 빨간불에 직진하다가 교차로에서 박았습니다. |
| 정규화 후 | 적색직진 대 적색직진, 직진 대 직진 사고, 상대차량이 맞은편에서 진입 |

## 평가 방법

평가 데이터는 `data/testsets/langsmith/retrieval_eval.jsonl`의 retrieval 테스트셋 30건을 사용했다. 검색 조합은 로컬에 실제 문서가 존재하는 `llamaparser/fixed/bge` 벡터스토어를 사용했고, retriever는 `vectorstore`, reranker는 `none`, 최종 검색 문서 수는 `top-30`으로 고정했다.

비교 조건은 다음 네 가지다.

| 조건 | 설명 |
| --- | --- |
| `raw_no_filter` | 사용자 원문 질문만 사용 |
| `raw_with_intake_filter` | 사용자 원문 질문 + intake metadata filter |
| `normalized_no_filter` | intake 정규화 질의만 사용 |
| `normalized_with_intake_filter` | intake 정규화 질의 + intake metadata filter |

정규화 효과만 보기 위한 주 비교는 `raw_with_intake_filter` 대비 `normalized_with_intake_filter`이다. 두 조건 모두 같은 intake filter를 사용하므로, 차이는 검색 질의가 원문인지 정규화 결과인지에서만 발생한다.

## 전체 지표: Top-30

| 지표 | Raw + filter | Normalized + filter | 변화량 | 상대 변화 |
| --- | ---: | ---: | ---: | ---: |
| `diagram_id_hit` | 0.0333 | 0.0333 | +0.0000p | 0.0% |
| `keyword_coverage` | 0.5650 | 0.5911 | +0.0261p | +4.6% |
| `retrieval_relevance` | 0.7197 | 0.7249 | +0.0052p | +0.7% |
| `critical_error` | 0.9667 | 0.9667 | +0.0000p | 0.0% |

원문 질문만 사용한 baseline과 intake 전체 적용 결과를 비교하면 `keyword_coverage`는 `0.5028 -> 0.5911`로 `+0.0883p`, 상대 개선율 약 `+17.6%`였다. 이 값은 질문 정규화와 metadata filter가 함께 적용된 총 효과로 볼 수 있다.

참고로 같은 조건을 `top-5`로 제한했을 때는 정규화만의 효과가 `keyword_coverage +0.0139p`, intake 전체 효과가 `+0.0361p`였다. `top-30`에서 개선 폭이 더 크게 보인 것은 정규화가 최종 5개 문서를 즉시 바꾸기보다, reranker가 재정렬할 후보군 안에 관련 키워드 문서를 더 많이 포함시키는 방향으로 작동한다는 뜻이다. 초기 top-30 측정에서는 `keyword_coverage`가 `0.6078`까지 올랐지만, `null추돌`, `nullnull` 같은 결측 토큰이 질의에 섞인 상태였다. 수정 후 수치는 약간 낮아졌지만, 질의 품질은 더 안정적이다.

## 케이스별 관찰

정규화로 키워드 포함률이 변한 케이스는 30건 중 12건이었다. 이 중 7건은 개선, 5건은 하락했다.

개선 폭이 큰 사례는 교차로 신호/진행방향 조합이나 자전거 추돌/진로변경처럼 문서 표제어가 비교적 분명한 질의였다.

| 케이스 | Raw keyword | Normalized keyword | 변화 | 정규화 질의 |
| --- | ---: | ---: | ---: | --- |
| `retrieval_028` | 0.2500 | 0.7500 | +0.5000 | 추돌사고 |
| `retrieval_015` | 0.0000 | 0.4000 | +0.4000 | 직진 대 좌회전 사고, 상대차량이 측면에서 진입, 오른쪽 도로, 왼쪽 도로 |
| `retrieval_019` | 0.0000 | 0.4000 | +0.4000 | 진로변경 대 직진 사고, 진로변경 사고, 진로변경, 같은 방향 |
| `retrieval_018` | 0.2000 | 0.6000 | +0.4000 | 추돌사고, 후행 차량이 선행 차량을 들이받음, 같은 방향 |

반대로 하락한 사례도 있었다. 특히 보행자 횡단보도 케이스나 자전거 비보호 좌회전 케이스에서 정규화 질의가 원문보다 좁거나 잘못된 방향으로 바뀐 경우가 있었다.

| 케이스 | Raw keyword | Normalized keyword | 변화 | 정규화 질의 |
| --- | ---: | ---: | ---: | --- |
| `retrieval_003` | 1.0000 | 0.6667 | -0.3333 | 녹색 대 적색, 보행자 정상 횡단, 횡단보도 내 사고 |
| `retrieval_004` | 1.0000 | 0.6667 | -0.3333 | 신호등 없음, 횡단보도 내 사고 |
| `retrieval_026` | 0.5000 | 0.2500 | -0.2500 | 녹색 비보호 좌회전 대 맞은편 녹색 직진, 녹색직진 대 적색직진, 비보호 좌회전 대 직진, 녹색직진 대 적색비보호좌회전 |
| `retrieval_027` | 0.2500 | 0.0000 | -0.2500 | 직진 대 직진 사고, 동일 폭 교차로, 상대차량이 측면에서 진입, 오른쪽 도로 |

## 해석

현재 정규화는 “자연어 질문을 문서 표제어에 맞추는 효과”가 있다. 그래서 교차로 신호 조합, 직진 대 직진, 추돌, 진로변경처럼 문서 내 반복 표현이 뚜렷한 유형에서는 검색 문맥의 키워드 적중률이 올라갔다.

하지만 개선 폭은 제한적이다. 이유는 세 가지다.

첫째, `diagram_id_hit`가 거의 움직이지 않았다. 이는 정규화 질의가 관련 키워드는 더 잘 찾더라도, 정답 도표 단위까지 정확히 올리는 데는 청킹/metadata/도표 ID 매핑의 영향이 더 크다는 뜻이다.

둘째, `null추돌`, `nullnull`처럼 LLM이 문자열로 보낸 결측값이 질의에 섞이던 문제는 수정했다. 현재 수정 후 평가 산출물에서는 정규화 질의에 `null`/`none` 토큰이 남지 않는다. 남은 문제는 일부 자전거/비보호 좌회전 케이스에서 `녹색직진 대 적색직진`처럼 실제 사고 관계와 다른 신호 조합이 붙는 경우다. 이런 과잉 정규화는 케이스에 따라 원문보다 검색 품질을 낮춘다.

셋째, 보행자/자전거 케이스에서는 자동차 교차로 중심의 질의 패턴이 과하게 붙는 경우가 있었다. 즉, 정규화가 항상 질의를 “더 정확하게” 만드는 것이 아니라, 잘못된 taxonomy로 좁힐 위험도 있다.

## 결론

질문 정규화는 현재 상태에서 RAG 검색 후보군 품질을 개선했다. 수정 후 `top-30` 기준 가장 직접적인 개선 지표는 `keyword_coverage +2.61%p`, 상대 개선율 `+4.6%`이다. intake filter까지 포함한 전체 intake 적용 효과는 `keyword_coverage +8.83%p`, 상대 개선율 `+17.6%`로 더 크다.

다만 핵심 정답 도표 적중률은 개선되지 않았으므로, “리랭커가 사용할 후보군의 문맥 품질을 끌어올렸다” 정도로 표현하는 것이 정확하다. 발표나 문서에는 “정규화로 RAG 검색 성능이 대폭 개선됐다”보다는 “문서 표제어 기반 키워드 회수율과 rerank 후보군 품질을 개선했지만, 도표 단위 정답 적중은 추가 개선이 필요하다”라고 쓰는 편이 안전하다.

## 다음 개선 제안

1. 자전거/비보호 좌회전 케이스에서 실제 당사자 순서와 신호 조합이 뒤집히지 않도록 party-aware signal pair 규칙을 보강한다.
2. 정규화 질의와 원문 질의를 둘 다 검색한 뒤 union/rerank하는 hybrid query 방식을 실험한다.
3. `diagram_id_hit` 개선을 위해 도표 ID/사고 유형 metadata를 더 직접적으로 검색 후보에 반영한다.
4. 해결 완료 항목: `query_slots`에서 문자열 `"null"`을 실제 `None`처럼 처리하고, 정규화 질의에서 `null추돌`, `nullnull`이 생성되지 않게 했다.

## 산출물

이번 비교 결과는 다음 파일에 저장했다.

- `evaluation/results/intake_query_normalization/20260504-141638-llamaparser-fixed-bge-vectorstore-query-normalization.summary.json`
- `evaluation/results/intake_query_normalization/20260504-141638-llamaparser-fixed-bge-vectorstore-query-normalization.csv`
- `evaluation/results/intake_query_normalization/20260504-144108-llamaparser-fixed-bge-vectorstore-top30-query-normalization.summary.json`
- `evaluation/results/intake_query_normalization/20260504-144108-llamaparser-fixed-bge-vectorstore-top30-query-normalization.csv`
- `evaluation/results/intake_query_normalization/20260504-144906-llamaparser-fixed-bge-vectorstore-top30-query-normalization-after-fix.summary.json`
- `evaluation/results/intake_query_normalization/20260504-144906-llamaparser-fixed-bge-vectorstore-top30-query-normalization-after-fix.csv`
- `evaluation/results/intake_query_normalization/intake_decisions_retrieval_eval.json`
