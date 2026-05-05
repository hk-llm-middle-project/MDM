# Hybrid Query RAG 검색 실험 보고서

## 요약

정규화 질의와 원문 질의를 각각 `top-30`으로 검색한 뒤 RRF로 병합하는 hybrid query 방식을 실험했다. 목적은 intake 정규화가 잘 맞는 케이스의 이점은 유지하면서, 정규화가 원문 단서를 잃는 케이스에서는 원문 검색 결과가 안전망 역할을 하게 하는 것이다.

결과적으로 hybrid query는 정규화 단독보다 후보군 품질을 추가로 개선했다. `keyword_coverage`는 `normalized + filter`의 `0.5911`에서 `hybrid RRF + filter`의 `0.6039`로 상승했다. 변화량은 `+0.0128p`다. 원문 검색+filter 기준 `0.5650`과 비교하면 `+0.0389p` 개선이다.

다만 `diagram_id_hit`는 모든 조건에서 `0.0333`으로 변하지 않았다. Hybrid query는 관련 키워드 문맥을 후보군에 더 많이 포함시키는 데는 효과가 있었지만, 정답 도표 ID를 직접 끌어올리지는 못했다.

## 실험 목적

질문 정규화는 문서 표제어에 가까운 검색어를 만들어 주지만, 항상 원문보다 좋은 것은 아니다. 특히 보행자/자전거처럼 원문에 중요한 party 단서가 들어 있는 경우 정규화 과정에서 일부 단서가 약해질 수 있다.

Hybrid query 실험은 이 문제를 줄이기 위한 것이다.

1. 원문 질문으로 검색한다.
2. intake 정규화 질의로 검색한다.
3. 두 후보군을 중복 제거한 뒤 RRF 방식으로 병합한다.
4. 병합된 후보군을 reranker 투입 전 `top-30` 후보군으로 본다.

## 평가 설정

| 항목 | 값 |
| --- | --- |
| 테스트셋 | `data/testsets/langsmith/retrieval_eval.jsonl` |
| 케이스 수 | 30 |
| 벡터스토어 | `llamaparser/fixed/bge` |
| retriever | `vectorstore` |
| reranker | `none` |
| 후보 수 | `top-30` |
| 병합 방식 | Reciprocal Rank Fusion |

재실행 스크립트는 `evaluation/evaluate_hybrid_query.py`에 추가했다. 같은 조건은 아래 명령으로 다시 돌릴 수 있다.

```bash
uv run python evaluation/evaluate_hybrid_query.py --k 30 --conditions raw_with_intake_filter,normalized_with_intake_filter,hybrid_rrf_with_intake_filter
```

스크립트는 intake decision cache인 `evaluation/results/intake_query_normalization/intake_decisions_retrieval_eval.json`을 기본으로 재사용한다. LLM intake를 다시 호출하고 싶으면 `--refresh-intake`를 붙이면 된다.

비교 조건은 다음 세 가지다.

| 조건 | 설명 |
| --- | --- |
| `raw_with_intake_filter` | 원문 질문 + intake metadata filter |
| `normalized_with_intake_filter` | 정규화 질의 + intake metadata filter |
| `hybrid_rrf_with_intake_filter` | 정규화 질의 top-30과 원문 질의 top-30을 RRF로 병합 |

## 전체 결과

| 지표 | Raw + filter | Normalized + filter | Hybrid RRF + filter |
| --- | ---: | ---: | ---: |
| `diagram_id_hit` | 0.0333 | 0.0333 | 0.0333 |
| `keyword_coverage` | 0.5650 | 0.5911 | 0.6039 |
| `retrieval_relevance` | 0.7197 | 0.7249 | 0.7274 |
| `critical_error` | 0.9667 | 0.9667 | 0.9667 |

Hybrid 개선량은 다음과 같다.

| 비교 | `keyword_coverage` | `retrieval_relevance` |
| --- | ---: | ---: |
| Hybrid - Normalized | +0.0128p | +0.0026p |
| Hybrid - Raw filter | +0.0389p | +0.0078p |

## 케이스 관찰

Hybrid의 장점은 정규화가 손실을 만든 케이스에서 원문 검색을 통해 일부 회복이 가능하다는 점이다.

대표 사례는 `retrieval_026`이다.

| 조건 | 질의 | `keyword_coverage` |
| --- | --- | ---: |
| Raw + filter | 녹색 직진 자전거와 맞은편 비보호 좌회전 자동차 사고 | 0.5000 |
| Normalized + filter | 녹색 비보호 좌회전 대 맞은편 녹색 직진, 비보호 좌회전 대 직진, 직진 대 비보호좌회전 사고, 직진 대 좌회전 사고 | 0.2500 |
| Hybrid RRF + filter | 정규화 질의 + 원문 질의 | 0.5000 |

이 케이스에서는 정규화 질의가 문서 표제어와 가까운 표현을 만들었지만, 원문의 “녹색 직진 자전거” 단서가 약해졌다. Hybrid는 원문 검색 결과를 함께 유지해 정규화 단독의 하락을 회복했다.

반면 `retrieval_027`은 hybrid에서도 회복하지 못했다.

| 조건 | `keyword_coverage` |
| --- | ---: |
| Raw + filter | 0.2500 |
| Normalized + filter | 0.0000 |
| Hybrid RRF + filter | 0.0000 |

이 케이스는 “신호 없는 동일폭 교차로에서 오른쪽 도로 자전거와 왼쪽 도로 자동차가 직진”한 사고다. 단순히 원문과 정규화 질의를 합치는 것만으로는 자전거 동일폭 교차로의 우선관계가 충분히 보존되지 않았다. 별도의 party-aware road priority 규칙이 필요하다.

## 해석

Hybrid query는 정규화 단독보다 안정적이다. 정규화가 잘 맞는 케이스에서는 정규화 검색 결과가 후보군 품질을 끌어올리고, 정규화가 일부 단서를 잃는 케이스에서는 원문 검색이 안전망 역할을 한다.

이번 실험에서는 RRF 병합만 적용했고 reranker는 적용하지 않았다. 따라서 실제 운영 효과는 cross-encoder 같은 reranker를 붙여 `top-30 -> top-5`로 재정렬했을 때 더 정확히 판단할 수 있다.

주의할 점도 있다. Hybrid는 검색 호출이 두 번 발생하므로 latency와 비용이 증가한다. 또한 병합 방식이 부정확하면 중복 제거 과정에서 좋은 후보가 사라질 수 있다. 실제로 초기 실험에서는 dedupe key를 너무 거칠게 잡아 결과가 크게 나빠졌고, `chunk_id` 중심 dedupe로 바꾼 뒤 정상적인 개선을 확인했다.

## 결론

Hybrid RRF는 적용 가치가 있다. 최신 top-30 후보군 기준으로 `keyword_coverage`가 정규화 단독 대비 `+0.0128p`, 원문+filter 대비 `+0.0389p` 개선됐다. 특히 정규화가 특정 party 단서를 약화시키는 케이스에서 원문 검색이 안전망 역할을 할 수 있음을 확인했다.

다만 아직 production 기본값으로 바로 넣기보다는 평가 옵션 또는 실험 플래그로 먼저 운영하는 것이 좋다. 다음 단계는 hybrid 후보군에 cross-encoder reranker를 붙여 최종 `top-5` 품질이 실제로 개선되는지 확인하는 것이다.

## 다음 작업

1. `evaluation` 스크립트에 hybrid RRF 검색 조건을 정식 옵션으로 추가한다.
2. Hybrid RRF 후보군에 cross-encoder reranker를 적용해 `top-5` 성능을 비교한다.
3. `retrieval_027` 회복을 위해 자전거 동일폭 교차로의 오른쪽/왼쪽 도로 우선관계 정규화 규칙을 추가한다.
4. 운영 반영 시 검색 2회 호출에 따른 latency 증가를 측정한다.

## 산출물

- `evaluation/results/intake_query_normalization/20260504-150711-llamaparser-fixed-bge-vectorstore-top30-query-hybrid-party-aware.summary.json`
- `evaluation/results/intake_query_normalization/20260504-150711-llamaparser-fixed-bge-vectorstore-top30-query-hybrid-party-aware.csv`
- `evaluation/results/intake_query_normalization/20260504-151335-llamaparser-fixed-bge-vectorstore-none.summary.json`
- `evaluation/results/intake_query_normalization/20260504-151335-llamaparser-fixed-bge-vectorstore-none.csv`
