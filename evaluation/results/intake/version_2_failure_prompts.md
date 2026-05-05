# Intake evaluation version 2 실패 프롬프트 정리

- 기준 결과: `evaluation/results/intake/20260505-143950-intake-local.csv`
- 실패 기준: `feedback.*` 중 하나라도 1이 아닌 row
- 실패 수: 10 / 20

| ID | 프롬프트 | 틀린 항목 | 기대값 | 실제값 |
|---|---|---|---|---|
| intake_005 | 앞차가 갑자기 멈춰서 제가 뒤에서 박았습니다. | 충분성, location, missing_fields | sufficient=예<br>party_type=자동차<br>location=같은 방향 진행차량 상호간의 사고<br>missing_fields=없음<br>follow_up_contains=없음<br>forbidden_filters=없음 | sufficient=아니오<br>party_type=자동차<br>location=없음<br>missing_fields=location<br>follow_up=사고가 발생한 장소는 어디인가요?, 신호등이 있었나요? |
| intake_006 | 자전거랑 부딪혔는데 과실이 궁금해요. | follow-up 문구 | sufficient=아니오<br>party_type=자전거<br>location=없음<br>missing_fields=location<br>follow_up_contains=사고 상황, 교차로<br>forbidden_filters={"location": "교차로 사고"} | sufficient=아니오<br>party_type=자전거<br>location=없음<br>missing_fields=location<br>follow_up=사고가 발생한 장소는 어디인가요?, 횡단보도 내, 부근, 또는 없는 경우 어떤 상황이었나요? |
| intake_007 | 보행자랑 사고났어요. | follow-up 문구 | sufficient=아니오<br>party_type=보행자<br>location=없음<br>missing_fields=location<br>follow_up_contains=횡단보도, 사고 상황<br>forbidden_filters={"location": "횡단보도 내"} | sufficient=아니오<br>party_type=보행자<br>location=없음<br>missing_fields=location<br>follow_up=사고가 발생한 위치는 어디인가요? (예: 횡단보도 내, 횡단보도 부근 등) |
| intake_008 | 자동차끼리 사고가 났는데 기준 찾아주세요. | follow-up 문구 | sufficient=아니오<br>party_type=자동차<br>location=없음<br>missing_fields=location<br>follow_up_contains=사고 상황, 같은 방향<br>forbidden_filters={"location": "교차로 사고"} | sufficient=아니오<br>party_type=자동차<br>location=없음<br>missing_fields=location<br>follow_up=사고가 발생한 장소는 어디인가요?, 사고의 구체적인 상황을 설명해 주실 수 있나요? |
| intake_010 | 신호 없는 교차로에서 충돌했습니다. | follow-up 문구 | sufficient=아니오<br>party_type=없음<br>location=교차로 사고<br>missing_fields=party_type<br>follow_up_contains=상대, 보행자<br>forbidden_filters={"party_type": "자동차"} | sufficient=아니오<br>party_type=없음<br>location=교차로 사고<br>missing_fields=party_type<br>follow_up=사고 상대는 어떤 종류의 차량이었나요? |
| intake_012 | 같은 방향으로 가던 옆 자동차가 제 차로로 들어오다가 긁었습니다. | location | sufficient=예<br>party_type=자동차<br>location=같은 방향 진행차량 상호간의 사고<br>missing_fields=없음<br>follow_up_contains=없음<br>forbidden_filters=없음 | sufficient=예<br>party_type=자동차<br>location=기타<br>missing_fields=없음<br>follow_up=사고가 발생한 장소는 횡단보도 내, 부근, 또는 없음 중 어디인가요? |
| intake_014 | 차랑 사고났어요. | follow-up 문구 | sufficient=아니오<br>party_type=자동차<br>location=없음<br>missing_fields=location<br>follow_up_contains=사고 상황<br>forbidden_filters={"location": "교차로 사고"} | sufficient=아니오<br>party_type=자동차<br>location=없음<br>missing_fields=location<br>follow_up=사고가 발생한 장소는 어디인가요?, 사고 상대는 어떤 차량이었나요? |
| intake_016 | 과실비율이 뭐예요? | follow-up 문구 | sufficient=아니오<br>party_type=없음<br>location=없음<br>missing_fields=party_type, location<br>follow_up_contains=사고 상황<br>forbidden_filters={"party_type": "자동차"}, {"location": "기타"} | sufficient=아니오<br>party_type=없음<br>location=없음<br>missing_fields=party_type, location<br>follow_up=사고에 관련된 상대방의 유형은 무엇인가요? (보행자, 자동차, 자전거 중에서 선택), 사고가 발생한 장소는 어디인가요? (횡단보도 내, 횡단보도 부근, 횡단보도 없음, 교차로 사고 등) |
| intake_017 | 횡단보도 근처인지 안인지 헷갈리는데 사람이 건너다가 사고났어요. | 충분성, location, missing_fields, follow-up 문구, 금지 필터 부재 | sufficient=아니오<br>party_type=보행자<br>location=없음<br>missing_fields=location<br>follow_up_contains=횡단보도, 안<br>forbidden_filters={"location": "횡단보도 내"}, {"location": "횡단보도 부근"} | sufficient=예<br>party_type=보행자<br>location=횡단보도 부근<br>missing_fields=없음<br>follow_up=사고가 발생한 장소에 대한 더 구체적인 정보가 있나요?, 사고 상대는 어떤 차량이었나요? |
| intake_018 | 상대가 자전거인지 오토바이인지 정확히 못 봤고 교차로였어요. | 충분성, party_type, missing_fields, 금지 필터 부재 | sufficient=아니오<br>party_type=없음<br>location=교차로 사고<br>missing_fields=party_type<br>follow_up_contains=상대<br>forbidden_filters={"party_type": "자전거"}, {"party_type": "자동차"} | sufficient=예<br>party_type=자전거<br>location=교차로 사고<br>missing_fields=없음<br>follow_up=상대 차량의 신호등 상태는 어땠나요?, 사고가 발생한 교차로의 신호는 어떤 상태였나요? |

## 요약

- 실제 의미 판단 오류가 큰 케이스: `intake_005`, `intake_012`, `intake_017`, `intake_018`
- 나머지는 주로 follow-up 문구가 기대 키워드를 포함하지 못한 케이스입니다.
