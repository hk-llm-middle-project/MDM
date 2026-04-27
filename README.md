# mdm

자동차 사고 과실 비율 인정 기준 PDF를 기반으로 질의응답하는 RAG 시스템 실험 프로젝트다.  
현재 목표는 아래 흐름을 단순하게 구현하는 것이다.

```text
문서 로드 -> 청킹 -> 임베딩 -> 저장 -> 검색 -> 평가
```

## 프로젝트 개요

- 도메인: 자동차 사고 과실 비율 인정 기준
- 기준 문서: `data/raw/230630_자동차사고 과실비율 인정기준_최종.pdf`
- 목적: 문서 기반 질문에 답하고, 답변 근거를 함께 보여주는 RAG MVP 만들기

## 폴더 구조

```text
mdm/
├─ main.py
├─ config.py
├─ rag/
│  ├─ __init__.py
│  ├─ loader.py
│  ├─ chunker.py
│  ├─ indexer.py
│  ├─ service/
│  │  ├─ app_service.py
│  │  ├─ analysis_service.py
│  │  ├─ result_service.py
│  │  ├─ vectorstore_service.py
│  │  └─ intake/
│  │     ├─ intake_service.py
│  │     ├─ prompts.py
│  │     └─ schema.py
│  ├─ pipeline/
│  │  ├─ retrieval.py
│  │  ├─ retriever/
│  │  └─ reranker/
│  └─ evaluator.py
├─ data/
│  ├─ raw/
│  │  └─ 230630_자동차사고 과실비율 인정기준_최종.pdf
│  └─ vectorstore/
└─ docs/
   └─ rag-quick-guide.md
```

## 파일 역할

### [main.py](/home/nyong/mdm/main.py)

Streamlit UI 실행 진입점이다.  
앱 설정, 세션 상태, 사이드바, 채팅 화면 렌더링만 맡는다.

### [config.py](/home/nyong/mdm/config.py)

환경 변수와 공통 설정을 모아두는 파일이다.

- PDF 경로
- 벡터스토어 저장 경로
- chunk size / overlap
- top-k
- 임베딩 모델명
- LLM 모델명

### [rag/loader.py](/home/nyong/mdm/rag/loader.py)

PDF 문서를 읽고 텍스트와 메타데이터를 정리한다.

- 문서 로드
- 페이지 단위 텍스트 추출
- `source`, `page` 같은 메타데이터 부여

### [rag/chunker.py](/home/nyong/mdm/rag/chunker.py)

로드한 문서를 검색 가능한 단위로 나눈다.

- 문단 또는 고정 길이 청킹
- overlap 적용
- 의미 단위가 최대한 유지되도록 분할

### [rag/indexer.py](/home/nyong/mdm/rag/indexer.py)

청크를 임베딩하고 벡터스토어에 저장한다.

- 임베딩 생성
- Chroma 또는 FAISS 저장
- 재색인 처리

### [rag/service/app_service.py](/home/nyong/mdm/rag/service/app_service.py)

사용자 흐름을 조율하는 애플리케이션 서비스다.

- 입력 처리 흐름의 대표 진입점
- 분석 서비스 호출
- 기존 UI와 평가 스크립트가 사용하는 답변 API 유지

### [rag/service/analysis_service.py](/home/nyong/mdm/rag/service/analysis_service.py)

사고 질의를 분석하고 RAG 답변을 생성한다.

- 검색 파이프라인 실행
- 검색 컨텍스트 조립
- 답변 프롬프트 생성
- LLM 호출

### [rag/service/result_service.py](/home/nyong/mdm/rag/service/result_service.py)

분석 결과를 화면이나 평가에서 쓰기 좋은 형태로 정리한다.

- 답변 표시 형식 정리
- 검색 문서 조각 첨부
- 이후 예상 사고유형, 과실비율, 주의사항 화면 모델 확장

### [rag/service/vectorstore_service.py](/home/nyong/mdm/rag/service/vectorstore_service.py)

앱 프로세스에서 재사용할 벡터스토어와 검색 컴포넌트를 준비한다.

- 기존 벡터스토어 로드
- 필요 시 PDF 로드, 청킹, 색인 생성
- 실행 중 캐시 정책

### [rag/service/intake/](/home/nyong/mdm/rag/service/intake/)

사고 입력 수집과 충분성 판단을 담당한다.

- 입력 충분성 판단 결과 구조
- 추가 질문 후보
- 분석용 사고 설명 정규화

### [rag/pipeline/retrieval.py](/home/nyong/mdm/rag/pipeline/retrieval.py)

검색과 reranker를 묶어 최종 컨텍스트 문서를 만든다.

- retriever 전략 실행
- reranker 전략 실행
- candidate/final top-k 조정

### [rag/pipeline/retriever/](/home/nyong/mdm/rag/pipeline/retriever/)

질문을 받아 관련 청크를 검색한다.

- 쿼리 임베딩
- similarity search
- top-k 결과 반환

### [rag/pipeline/reranker/](/home/nyong/mdm/rag/pipeline/reranker/)

검색 후보 문서의 순서를 다시 매긴다.

- no-op reranker
- Flashrank/Cohere/LLM score 기반 reranker
- 최종 문서 수 제한

### [rag/evaluator.py](/home/nyong/mdm/rag/evaluator.py)

검색과 답변 품질을 점검하는 모듈이다.

- 샘플 질문셋 기반 테스트
- 검색 적중 여부 확인
- 답변 정확도와 출처 확인
- 문서 밖 질문에 대한 거절 품질 확인

### [docs/rag-quick-guide.md](/home/nyong/mdm/docs/rag-quick-guide.md)

허접 버전 RAG를 만들 때 무엇을 우선 봐야 하는지 정리한 메모다.  
설계 기준이나 구현 체크리스트로 참고하면 된다.

## 데이터 경로

- 원본 문서: [data/raw/230630_자동차사고 과실비율 인정기준_최종.pdf](/home/nyong/mdm/data/raw/230630_%EC%9E%90%EB%8F%99%EC%B0%A8%EC%82%AC%EA%B3%A0%20%EA%B3%BC%EC%8B%A4%EB%B9%84%EC%9C%A8%20%EC%9D%B8%EC%A0%95%EA%B8%B0%EC%A4%80_%EC%B5%9C%EC%A2%85.pdf)
- 벡터스토어 저장 위치: `data/vectorstore/`

`data/vectorstore/`는 로컬 색인 결과물이 저장되는 위치이며 `.gitignore`에 포함되어 있다.

## 구현 기준

이 프로젝트는 아래 원칙으로 MVP를 만든다.

- 처음엔 문서 1개로 시작한다.
- 검색은 복잡하게 가지 않고 벡터 검색 + `top-k`부터 시작한다.
- 청크 메타데이터에 `source`, `page`를 남긴다.
- 답변은 문맥 기반으로만 하게 하고, 근거가 없으면 모른다고 하게 만든다.
- 결과에는 출처를 같이 보여준다.
- "잘 되는 느낌"이 아니라 샘플 질문셋으로 평가한다.

## 추천 구현 순서

1. `loader.py`에서 PDF 텍스트 추출 구현
2. `chunker.py`에서 청킹 구현
3. `indexer.py`에서 임베딩 및 저장 구현
4. `retriever.py`에서 검색 구현
5. `main.py`에서 전체 흐름 연결
6. `evaluator.py`에서 샘플 질문 평가 추가

## 참고

- RAG 체크리스트 문서: [docs/rag-quick-guide.md](/home/nyong/mdm/docs/rag-quick-guide.md)

## Git 컨벤션

이 프로젝트는 초반 MVP 단계이므로 복잡한 규칙보다 일관성을 우선한다.

### 브랜치 규칙

- 기본 브랜치는 `master`를 사용한다.
- 기능 작업은 가능하면 작업 브랜치를 따서 진행한다.
- 브랜치 이름은 아래 형식을 권장한다.

```text
feature/<short-name>
fix/<short-name>
docs/<short-name>
refactor/<short-name>
test/<short-name>
```

예:

- `feature/pdf-loader`
- `fix/chunk-overlap-bug`
- `docs/update-readme`

### 커밋 메시지 규칙

커밋 메시지는 한 줄만 봐도 무엇을 했는지 알 수 있게 쓴다.

권장 형식:

```text
type: summary
```

사용할 `type` 예시:

- `feat`: 기능 추가
- `fix`: 버그 수정
- `docs`: 문서 수정
- `refactor`: 동작 변경 없는 구조 개선
- `test`: 테스트 추가/수정
- `chore`: 설정, 의존성, 기타 작업

예:

```text
feat: add pdf loader for accident ratio guide
fix: preserve page metadata during chunking
docs: document rag project structure in readme
chore: ignore local vectorstore artifacts
```

### 커밋 단위

- 하나의 커밋에는 하나의 의도를 담는다.
- 문서 수정과 기능 수정은 가능하면 분리한다.
- 동작 변경이 있으면 관련 설정이나 문서도 같이 반영한다.
- 의미 없는 중간 커밋보다는 나중에 봐도 이해되는 커밋을 남긴다.

### PR 규칙

원격 저장소를 쓰게 되면 PR은 아래 기준으로 작성한다.

- 제목만 보고 변경 목적이 보여야 한다.
- 본문에는 `무엇을`, `왜`, `어떻게 확인했는지`를 짧게 적는다.
- 큰 PR 하나보다 작은 PR 여러 개가 낫다.
- 리뷰어가 바로 확인할 수 있게 테스트 방법이나 실행 예시를 남긴다.

### .gitignore 원칙

아래와 같은 로컬 파일은 커밋하지 않는다.

- `.env`
- `.venv/`
- `data/vectorstore/`
- 개인 실험 산출물, 캐시 파일, 로컬 로그

민감 정보나 로컬 실행 결과는 저장소에 올리지 않는 것을 기본 원칙으로 한다.
