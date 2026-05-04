# Intake 꼬리질문 평가용 애매한 질문 후보

## 목적

Intake가 정보가 부족한 사고 설명에서 바로 RAG 검색으로 넘어가지 않고, 필요한 꼬리질문을 잘 하는지 평가하기 위한 후보 질문 모음이다.

중점 평가 포인트는 다음과 같다.

- 사고 대상이 불명확할 때 `party_type`을 확정하지 않는가
- 사고 장소/유형이 불명확할 때 `location`을 확정하지 않는가
- 자동차 사고에서 신호, 진행방향, 상대 차량 위치, 도로 우선관계 등 검색 품질에 필요한 slot을 추가 질문하는가
- 애매한 표현을 임의로 특정 location filter로 고정하지 않는가

JSONL 후보셋은 `data/testsets/langsmith/intake_ambiguous_candidates.jsonl`에 저장했다.

## 후보 요약

| ID | 질문 | 기대 누락 | 기대 꼬리질문 방향 |
| --- | --- | --- | --- |
| `intake_ambiguous_001` | 사고났는데 과실비율 좀 봐주세요. | `party_type`, `location` | 사고 대상, 사고 상황 |
| `intake_ambiguous_002` | 차랑 부딪혔어요. | `location` | 교차로/횡단보도/같은 방향 등 상황 |
| `intake_ambiguous_003` | 상대가 갑자기 튀어나왔어요. | `party_type`, `location` | 상대 대상, 진입 위치 |
| `intake_ambiguous_004` | 길 건너던 사람이랑 사고났는데 정확한 위치가 애매해요. | `location` | 횡단보도 안/부근/없음 |
| `intake_ambiguous_005` | 자전거랑 사고났는데 어디 유형인지 모르겠어요. | `location` | 교차로/같은 방향/기타 |
| `intake_ambiguous_006` | 교차로에서 충돌했는데 누가 더 잘못인지 궁금합니다. | `party_type` | 상대가 자동차/자전거/보행자인지 |
| `intake_ambiguous_007` | 신호 있는 곳에서 부딪혔어요. | `party_type`, `location` | 사고 대상, 교차로/횡단보도 |
| `intake_ambiguous_008` | 오른쪽에서 오던 상대랑 부딪혔어요. | `party_type`, `location` | 상대 대상, 교차로 여부 |
| `intake_ambiguous_009` | 앞에서 가던 대상이랑 부딪혔어요. | `party_type`, `location` | 상대 대상, 추돌/진로변경 |
| `intake_ambiguous_010` | 횡단보도 쪽에서 사고났어요. | `party_type`, `location` | 보행자 여부, 횡단보도 안/부근 |
| `intake_ambiguous_011` | 둘 다 직진하다가 박았습니다. | `party_type`, `location` | 자동차끼리인지, 교차로/같은 방향 |
| `intake_ambiguous_012` | 비보호 좌회전 중에 사고가 났어요. | `party_type`, `location` | 상대 대상, 맞은편 직진 여부 |
| `intake_ambiguous_013` | 주차장인지 도로인지 애매한 곳에서 접촉했어요. | `party_type`, `location` | 상대 대상, 도로/주차장/도로 외 |
| `intake_ambiguous_014` | 문 열다가 사고났다는 얘기를 들었는데 제 상황도 비슷해요. | `party_type`, `location` | 대상, 문 열림/주차장 여부 |
| `intake_ambiguous_015` | 오토바이인지 자전거인지 잘 못 봤고 부딪혔습니다. | `party_type`, `location` | 상대 종류, 사고 위치 |
| `intake_ambiguous_016` | 중앙선 근처에서 상대랑 스쳤어요. | `party_type`, `location` | 대상, 중앙선 침범/마주보는 방향 |
| `intake_ambiguous_017` | 뒤에서 받았는데 제가 뒤인지 앞인지 설명이 헷갈립니다. | `location` | 같은 방향, 추돌 관계 |
| `intake_ambiguous_018` | 차선 바꾸다가 사고났는데 상대도 움직였어요. | `party_type`, `location` | 자동차끼리인지, 진로변경 세부 |
| `intake_ambiguous_019` | 노란불쯤에 들어가다 사고났어요. | `party_type`, `location` | 상대 대상, 신호/진행방향 |
| `intake_ambiguous_020` | 골목에서 나온 상대와 부딪혔어요. | `party_type`, `location` | 상대 대상, 교차로/도로 외 진입 |
| `intake_ambiguous_021` | 신호 없는 교차로에서 사고났어요. | `party_type` | 상대 대상 |
| `intake_ambiguous_022` | 같은 방향으로 가던 중 사고났어요. | `party_type` | 자동차/자전거/이륜차 여부 |
| `intake_ambiguous_023` | 상대가 좌회전했고 저는 가고 있었어요. | `party_type`, `location` | 상대 대상, 내 진행방향/신호 |
| `intake_ambiguous_024` | 보행자랑 닿았는데 횡단보도랑 거리가 애매합니다. | `location` | 횡단보도 안/부근/없음 |
| `intake_ambiguous_025` | 자전거가 제 옆으로 오다가 부딪혔어요. | `location` | 교차로/같은 방향/도로 외 |

