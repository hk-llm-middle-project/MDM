# 개허접 버전 version 0

## 채팅
- 사용자가 한 번 입력
- 정보가 부족해도 그냥 RAG에 검색
- 대화 내용은 history 저장

## RAG

### Chunking
    - PDF Parser : pdfplumber (https://github.com/jsvine/pdfplumber)
    - Fixed Size Chunking (text)
    - chunking size : 500
    - overlap : 0
    - metadata : none

### Retriever
    - 유사도 기반
    - k: 3

### LLM
    - prompt : 그냥 때려넣기
    - output format
        1. 의심 사고유형
        2. 관련 공식 문서 근거
        3. 수정요소 후보
        4. 예상 과실비율
        5. 설명

## UI/UX
- streamlit
- 채팅 형태
    - 사이드바
        - 새션 목록
    - 메인화면
        - 기본화면
            - 채팅창
            - 레퍼런스 (./data/raw/reference_v0_1.png)
        - 채탱화면
            - 주고받는 대화형태
            - 레퍼런스 (./data/raw/reference_v0_2.png)
