# 공무원 여비 RAG 챗봇

`DATA` 폴더의 공무원 여비 규정 PDF를 검색해, 문서 근거와 함께 답변하는 Streamlit 챗봇입니다.

## 주요 기능

- OpenAI `text-embedding-3-small` 임베딩과 `InMemoryVectorStore` 의미 검색
- BM25 원문 키워드 검색을 함께 사용하는 하이브리드 검색
- 질문과 직접 관계있는 규정·사례를 재정렬해 답변에 사용
- 문서 근거가 부족하면 답변을 거절
- 답변 아래에 출처 파일명, 페이지, 근거 문장 표시
- `주말`, `공용차량`, `비행기` 같은 일상 표현을 규정 용어로 확장

## 준비 사항

- Python 3.11
- [uv](https://docs.astral.sh/uv/)
- OpenAI API 키와 사용 가능한 API 크레딧

## 로컬 실행

1. 프로젝트 최상단의 `.env` 파일에 API 키를 설정합니다.

   ```env
   OPENAI_API_KEY=sk-...
   ```

2. `run.bat`을 더블클릭하거나 터미널에서 실행합니다.

   ```powershell
   uv run streamlit run app.py
   ```

`.env` 파일은 Git으로 관리하지 않으므로 API 키가 GitHub에 올라가지 않습니다.

## 검색 방식

PDF는 페이지 단위 부모 문서로 읽습니다. 이후 약 600자 크기의 자식 청크를 만들어 검색하고, 답변 모델에는 해당 부모 페이지 전체를 전달해 규정의 조건과 예외가 잘리지 않도록 합니다.

검색 결과는 임베딩 의미 검색과 BM25 정확 단어 검색을 Reciprocal Rank Fusion 방식으로 결합합니다. 최종 후보는 `gpt-4o-mini`가 질문 관련성에 따라 재정렬합니다.

## 평가 실행

외부 API 호출 없이 규정 페이지 검색 품질을 점검하려면 다음을 실행합니다.

```powershell
uv run python evaluate_retrieval.py
```

평가 질문은 [eval_cases.json](eval_cases.json)에 있습니다. 새 기능을 추가하거나 검색 규칙을 수정할 때마다 이 평가를 다시 실행하세요.

## Streamlit Community Cloud 배포

1. Streamlit Community Cloud에서 **Create app**을 선택합니다.
2. 이 저장소의 `main` 브랜치와 `app.py`를 진입 파일로 선택합니다.
3. **Advanced settings**에서 Python 3.11을 선택합니다.
4. **Secrets**에 다음 TOML을 입력합니다.

   ```toml
   OPENAI_API_KEY = "sk-..."
   ```

Cloud의 Secrets는 GitHub 저장소와 분리되어 관리됩니다. `.streamlit/secrets.toml`을 로컬에서 만들더라도 Git에 커밋하지 마세요.

## 파일 구성

```text
.
├── DATA/                     # 검색할 공무원 여비 PDF
├── app.py                    # Streamlit 앱
├── eval_cases.json           # 검색 평가 질문과 기대 출처
├── evaluate_retrieval.py     # 오프라인 검색 평가 스크립트
├── run.bat                   # Windows 더블클릭 실행 파일
├── pyproject.toml            # Python 의존성 정의
└── uv.lock                   # 잠긴 의존성 버전
```
