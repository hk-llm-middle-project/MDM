# Intake 단계형 꼬리질문 평가 후보

이 후보셋은 첫 질문이 애매한 상태에서 시작해, intake의 꼬리질문과 사용자 답변을 거치며 최종적으로 RAG 검색 가능한 질의와 메타데이터로 수렴하는지 평가하기 위한 멀티턴 케이스다.

데이터 파일: `data/testsets/langsmith/intake_multistep_clarification_candidates.jsonl`

## 평가 관점

- 첫 턴에서 불확실한 필드를 성급하게 확정하지 않는가
- 이미 확인된 `party_type`, `location`을 다음 턴에서 보존하는가
- 새 답변으로 더 구체적인 사고 유형이 드러나면 이전 추정을 갱신하는가
- 최종 턴에서 검색에 유효한 신호, 진행방향, 위치 표현을 포함한 질의로 수렴하는가
- 자전거, 이륜차, 보행자 케이스를 자동차 사고로 잘못 일반화하지 않는가

## 후보 요약

| id | 초기 질문 | 단계적 수렴 목표 | 최종 메타데이터 | 핵심 검증 포인트 |
|---|---|---|---|---|
| intake_multistep_001 | 사람이랑 사고났는데 과실 좀 봐주세요. | 횡단보도 내 보행자 사고 | `party_type=보행자`, `location=횡단보도 내` | 횡단보도 근처라는 답을 바로 `횡단보도 내`로 확정하지 않기 |
| intake_multistep_002 | 자전거랑 부딪혔어요. | 교차로 우회전 차량 대 직진 자전거 | `party_type=자전거`, `location=교차로 사고` | 자전거 당사자 보존, 진행방향 추가 확인 |
| intake_multistep_003 | 뒤에서 받았는데 상대가 갑자기 끼어들었어요. | 진로변경 사고 | `party_type=자동차`, `location=진로변경 사고` | 단순 추돌로 조기 확정하지 않고 차로변경으로 갱신 |
| intake_multistep_004 | 주차하다가 사고났어요. | 주차장 후진 출차 대 통로 진행 차량 | `party_type=자동차`, `location=주차장 사고` | 장소와 후진 여부를 분리해서 확인 |
| intake_multistep_005 | 비보호 좌회전하다 사고났는데 제가 다 잘못인가요? | 자동차 비보호 좌회전 대 맞은편 직진 | `party_type=자동차`, `location=교차로 사고` | 신호와 진행방향 쌍을 최종 질의에 반영 |
| intake_multistep_006 | 길 건너던 사람이랑 사고났어요. | 횡단보도 밖 보행자 횡단 사고 | `party_type=보행자`, `location=횡단보도 외` | 근처 횡단보도 표현을 `횡단보도 내`로 오인하지 않기 |
| intake_multistep_007 | 자전거가 좌회전하다가 사고났어요. | 자전거 비보호 좌회전 대 자동차 직진 | `party_type=자전거`, `location=교차로 사고` | 자전거 당사자 순서와 신호 조합 유지 |
| intake_multistep_008 | 배달 오토바이랑 사고났어요. | 교차로 좌회전 자동차 대 직진 이륜차 | `party_type=이륜차`, `location=교차로 사고` | 이륜차를 자동차/자전거로 오인하지 않기 |
| intake_multistep_009 | 차 문 때문에 사고났어요. | 정차 차량 개문 대 자전거 | `party_type=자전거`, `location=개문 사고` | 주차장 사고로 성급히 보내지 않기 |
| intake_multistep_010 | 교차로에서 옆 차랑 박았어요. | 녹색 직진 대 적색 직진 신호위반 | `party_type=자동차`, `location=교차로 사고` | 상대 신호와 진행방향이 채워질 때까지 추가 질문 |

## 사용 방식

각 행은 `turns` 배열 안에 사용자 입력과 기대되는 assistant 꼬리질문 방향을 포함한다. 실제 평가는 assistant 문구의 완전일치보다 `expected_questions_after_each_turn`에 있는 핵심 단어가 포함되는지, 그리고 `expected_state_after_each_turn`의 메타데이터 상태가 맞는지를 우선 확인하는 방식이 적합하다.

최종 턴에서는 `expected_final_search_query`를 기준으로 검색 질의가 원문보다 구체적인 사고 맥락을 담는지 확인한다.
