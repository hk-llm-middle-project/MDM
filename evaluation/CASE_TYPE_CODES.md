# Case Type Codes

`case_type_codes`는 LangSmith 평가 결과를 케이스 유형별로 나누어 분석하기 위한 평가용 라벨입니다. 모든 테스트셋 row에는 반드시 `case_type_codes`가 들어가야 합니다.

## 사용 규칙

- 최소 1개 이상의 `CLARITY_*` 코드를 포함합니다.
- 최소 1개 이상의 suite/behavior 코드를 포함합니다. 예: `INTAKE_*`, `ROUTE_*`, `FILTER_*`, `RET_*`, `RERANK_*`, `MT_*`, `STRUCT_*`.
- `META_SAME_DIRECTION` 같은 메타데이터/위치 코드는 해당 row가 그 추론을 실제로 평가할 때만 넣습니다.
- 데이터 row 안에서 임의로 새 코드를 만들지 않습니다. 새 코드가 필요하면 이 문서와 validator를 먼저 수정합니다.

## 입력 명확도 코드 (`CLARITY_*`)

| 코드 | 의미 |
| --- | --- |
| `CLARITY_EXPLICIT` | 사용자가 “보행자”, “횡단보도”, “교차로”처럼 공식 용어 또는 공식 용어에 가까운 표현을 직접 말한 경우입니다. |
| `CLARITY_IMPLICIT` | 공식 용어가 직접 나오지 않고, 사고 상황을 보고 시스템이 추론해야 하는 경우입니다. |
| `CLARITY_SYNONYM` | 사용자가 공식 용어 대신 흔한 동의어, 유사 표현, 풀어쓴 표현을 사용한 경우입니다. |
| `CLARITY_INCOMPLETE` | 필수 정보가 부족해서 알 수 없는 상태로 남겨야 하거나 후속 질문이 필요한 경우입니다. |
| `CLARITY_DISTRACTOR` | 핵심 분류를 바꾸면 안 되는 부가 정보나 헷갈리는 정보가 포함된 경우입니다. |

## 메타데이터 추론 코드 (`META_*`)

| 코드 | 의미 |
| --- | --- |
| `META_PARTY` | `party_type` 추출을 평가합니다. 예: 보행자, 자동차, 자전거. |
| `META_LOCATION` | `location` 추출을 평가합니다. |
| `META_CROSSWALK_IN` | `location: "횡단보도 내"` 추론을 평가합니다. |
| `META_CROSSWALK_NEAR` | `location: "횡단보도 부근"` 추론을 평가합니다. |
| `META_NO_CROSSWALK` | `location: "횡단보도 없음"` 추론을 평가합니다. |
| `META_INTERSECTION` | `location: "교차로 사고"` 추론을 평가합니다. |
| `META_SAME_DIRECTION` | `location: "같은 방향 진행차량 상호간의 사고"` 추론을 평가합니다. |
| `META_OPPOSITE_DIRECTION` | `location: "마주보는 방향 진행차량 상호 간의 사고"` 추론을 평가합니다. |
| `META_MOTORCYCLE_SPECIAL` | `location: "자동차 대 이륜차 특수유형"` 추론을 평가합니다. |
| `META_OTHER` | `location: "기타"`처럼 넓은 기타 유형 분류를 평가합니다. |

## Suite 동작 코드

### Intake 코드

| 코드 | 의미 |
| --- | --- |
| `INTAKE_FULL` | Intake가 `party_type`과 `location`을 모두 추출해야 하는 경우입니다. |
| `INTAKE_PARTIAL` | Intake가 필요한 메타데이터 중 일부만 추출해야 하는 경우입니다. |
| `INTAKE_NONE` | Intake가 핵심 메타데이터를 추출하면 안 되는 경우입니다. |
| `INTAKE_FOLLOWUP` | Intake가 부족한 정보를 묻는 후속 질문을 해야 하는 경우입니다. |

### Router 코드

| 코드 | 의미 |
| --- | --- |
| `ROUTE_NEW_ACCIDENT` | 새 사고 분석 요청으로 라우팅해야 하는 경우입니다. |
| `ROUTE_FOLLOWUP` | 짧은 답변을 이전 사고 분석의 후속 응답으로 분류해야 하는 경우입니다. |
| `ROUTE_GENERAL` | 일반 대화 또는 사고 분석 외 요청으로 분류해야 하는 경우입니다. |
| `ROUTE_CORRECTION` | 이전 사고 정보의 정정 요청으로 처리해야 하는 경우입니다. |

### Metadata Filter 코드

| 코드 | 의미 |
| --- | --- |
| `FILTER_STRICT` | party와 location을 모두 사용해 엄격한 metadata filter를 만들어야 하는 경우입니다. |
| `FILTER_PARTIAL` | 사용 가능한 한 가지 필드만 filter에 사용해야 하는 경우입니다. |
| `FILTER_NONE` | metadata filter를 만들면 안 되는 경우입니다. |
| `FILTER_FALLBACK` | filter 검색만으로는 부족해서 unfiltered fallback이 필요할 것으로 예상되는 경우입니다. |

### Retrieval 코드

| 코드 | 의미 |
| --- | --- |
| `RET_DIAGRAM` | 도표/이미지 기반 기준을 찾아야 하는 retrieval 케이스입니다. |
| `RET_TABLE` | 표에 있는 기본 과실비율 또는 수정요소 근거를 찾아야 하는 케이스입니다. |
| `RET_NEAR_MISS` | 그럴듯하지만 틀린 기준이 함께 있어, 이를 구분해야 하는 케이스입니다. |
| `RET_MULTI_ACCEPT` | 정답으로 인정 가능한 기준이 2개 이상인 케이스입니다. |

### Reranker 코드

| 코드 | 의미 |
| --- | --- |
| `RERANK_PROMOTE` | reranker가 정답 후보를 상위로 올려야 하는 케이스입니다. |
| `RERANK_PROTECT` | 이미 좋은 순위에 있는 정답 후보를 reranker가 떨어뜨리지 않아야 하는 케이스입니다. |

### Multi-turn 코드

| 코드 | 의미 |
| --- | --- |
| `MT_ACCUMULATE` | 여러 턴에 걸쳐 부족한 메타데이터를 누적해야 하는 케이스입니다. |
| `MT_CORRECTION` | 사용자의 정정 발화를 반영해 기존 필드를 덮어써야 하는 케이스입니다. |
| `MT_MAX_FOLLOWUP` | 후속 질문 한도에 도달하는 흐름을 평가하는 케이스입니다. |

### Structured Output 코드

| 코드 | 의미 |
| --- | --- |
| `STRUCT_BASE_RATIO` | 기본 과실비율을 구조화해서 추출해야 하는 케이스입니다. |
| `STRUCT_ROLE_REVERSAL` | A/B 역할을 뒤집지 않고 올바르게 매핑하는지 평가하는 케이스입니다. |
| `STRUCT_MODIFIER_SAME` | 수정요소 근거가 같은 기준 안에 있는 케이스입니다. |
| `STRUCT_MODIFIER_CROSS_REF` | 수정요소 또는 적용 기준 판단에 다른 기준/섹션 참조가 필요한 케이스입니다. |
| `STRUCT_NON_APPLICABLE_MODIFIER` | 적용하면 그럴듯해 보이지만 실제로는 적용하면 안 되는 수정요소를 배제해야 하는 케이스입니다. |
| `STRUCT_CANNOT_DETERMINE` | 정보 부족으로 최종 과실비율을 결정하면 안 되는 케이스입니다. |
