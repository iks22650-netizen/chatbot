"""DATA 폴더의 규정 PDF만 근거로 답하는 Streamlit RAG 챗봇입니다."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
from pypdf import PdfReader
from rank_bm25 import BM25Okapi


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "DATA"
SUPPORTED_TEXT_EXTENSIONS = {".txt", ".md", ".csv"}
SEMANTIC_CANDIDATE_COUNT = 12
FINAL_SOURCE_COUNT = 4
RRF_K = 60

# 사용자의 일상 표현을 규정에서 쓰는 말로 연결합니다.
QUERY_EXPANSIONS = {
    "주말": ("여행일수", "출장기간", "실제로 필요한 일수", "토요일", "일요일", "개인적인 사정"),
    "휴일": ("여행일수", "출장기간", "실제로 필요한 일수", "공무의 형편", "부득이한 사유"),
    "공휴일": ("여행일수", "출장기간", "실제로 필요한 일수", "공무의 형편", "부득이한 사유"),
    "비행기": ("항공기", "항공기 일정상", "항공기 탑승", "국외 출장을 위해", "부득이하게 숙박"),
    "공항": ("항공기", "항공기 일정상", "국외 출장을 위해", "부득이하게 숙박"),
    "공용차량": ("공용선박", "일비의 2분의 1", "여행일수", "차량을 임차"),
}
COMMON_TERMS = {"공무원", "여비", "출장", "경우", "대한", "포함", "무엇", "어떻게", "알려줘"}
QUESTION_STOP_TERMS = {"사용", "사용할", "있", "있나요", "되", "되나요", "받", "받을", "수", "며칠"}
PARTICLE_SUFFIXES = ("으로", "에게", "에서", "까지", "부터", "하는", "하면", "하고", "인가요", "나요", "어요", "아요", "을", "를", "은", "는", "이", "가", "에", "의", "도", "로", "와", "과")


@dataclass
class RetrievalResult:
    """검색 결과와 문서 근거의 충분성 정보를 함께 보관합니다."""

    candidates: list[Document]
    has_lexical_support: bool


class RerankSelection(BaseModel):
    """재정렬 모델이 선택한 후보 번호입니다."""

    selected_ids: list[int] = Field(description="질문에 직접 답하는 후보 번호. 최대 4개.")


def list_source_files() -> list[Path]:
    """DATA 아래 파일을 이름순으로 찾습니다."""
    return sorted(path for path in DATA_DIR.rglob("*") if path.is_file()) if DATA_DIR.exists() else []


def read_pdf(pdf_path: Path) -> list[Document]:
    """PDF를 페이지 단위 부모 문서로 읽어 파일명·쪽수를 보존합니다."""
    documents: list[Document] = []
    for page_number, page in enumerate(PdfReader(str(pdf_path)).pages, start=1):
        text = (page.extract_text() or "").strip()
        # 목차는 질문 문구를 나열할 뿐 답변 근거가 아니므로 검색 대상에서 제외합니다.
        is_navigation_page = "목 차" in text[:600] or "contents" in text[:1_000].lower()
        if text and not is_navigation_page:
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": pdf_path.name,
                        "page": page_number,
                        "parent_id": f"{pdf_path.name}:{page_number}",
                    },
                )
            )
    return documents


def read_text_file(text_path: Path) -> list[Document]:
    """향후 추가될 텍스트 자료도 부모 문서로 읽습니다."""
    text = text_path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        return []
    return [Document(page_content=text, metadata={"source": text_path.name, "page": None, "parent_id": text_path.name})]


def load_documents() -> tuple[list[Document], list[str]]:
    """지원 파일 전체를 읽고 실패한 파일은 별도로 기록합니다."""
    documents: list[Document] = []
    skipped_files: list[str] = []
    for file_path in list_source_files():
        try:
            if file_path.suffix.lower() == ".pdf":
                documents.extend(read_pdf(file_path))
            elif file_path.suffix.lower() in SUPPORTED_TEXT_EXTENSIONS:
                documents.extend(read_text_file(file_path))
            else:
                skipped_files.append(f"{file_path.name} (지원하지 않는 형식)")
        except Exception as error:
            skipped_files.append(f"{file_path.name} ({type(error).__name__})")
    return documents, skipped_files


def source_signature() -> tuple[tuple[str, int, int], ...]:
    """파일이 바뀌면 Streamlit 캐시를 다시 만들기 위한 값입니다."""
    return tuple(
        (str(path.relative_to(DATA_DIR)), path.stat().st_mtime_ns, path.stat().st_size)
        for path in list_source_files()
    )


def tokenize(text: str) -> list[str]:
    """BM25용 간단한 한국어·영문 토큰을 만듭니다."""
    return re.findall(r"[가-힣A-Za-z0-9]+", text.lower())


def content_terms(text: str) -> list[str]:
    """질문의 조사·일반 동사를 줄여 실제 규정 주제를 추립니다."""
    terms: list[str] = []
    for token in tokenize(text):
        normalized = token
        for suffix in PARTICLE_SUFFIXES:
            if normalized.endswith(suffix) and len(normalized) > len(suffix) + 1:
                normalized = normalized[: -len(suffix)]
                break
        if len(normalized) >= 2 and normalized not in COMMON_TERMS | QUESTION_STOP_TERMS:
            terms.append(normalized)
    return list(dict.fromkeys(terms))


def build_child_chunks(parent_documents: list[Document]) -> list[Document]:
    """작은 자식 청크로 검색하되 부모 페이지 원문은 metadata에 유지합니다."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=80,
        separators=["\n\n", "\n", "Q&A", "•", "○", " ", ""],
    )
    children: list[Document] = []
    for parent in parent_documents:
        for chunk_index, text in enumerate(splitter.split_text(parent.page_content)):
            metadata = {
                **parent.metadata,
                "chunk_index": chunk_index,
                # 검색은 작은 청크로 하되 답변에는 잘린 문맥 대신 부모 페이지를 제공합니다.
                "parent_text": parent.page_content,
            }
            children.append(Document(page_content=text, metadata=metadata))
    return children


@st.cache_resource(show_spinner="문서 구조를 읽고 임베딩을 만들고 있습니다...")
def build_retrieval_indexes(
    signature: tuple[tuple[str, int, int], ...],
) -> tuple[InMemoryVectorStore, list[Document], BM25Okapi, list[str]]:
    """의미 검색 인덱스와 정확 단어 검색(BM25) 인덱스를 함께 만듭니다."""
    parents, skipped_files = load_documents()
    if not parents:
        raise ValueError("읽을 수 있는 문서 내용이 DATA 폴더에 없습니다.")

    children = build_child_chunks(parents)
    bm25 = BM25Okapi([tokenize(child.page_content) or ["_"] for child in children])
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = InMemoryVectorStore(embedding=embeddings)
    vector_store.add_documents(children)
    return vector_store, children, bm25, skipped_files


def expanded_query(question: str) -> tuple[str, tuple[str, ...]]:
    """질문과 함께 검색할 규정 용어를 만듭니다."""
    compact_question = re.sub(r"\s+", "", question)
    hints = [hint for trigger, values in QUERY_EXPANSIONS.items() if trigger in compact_question for hint in values]
    unique_hints = tuple(dict.fromkeys(hints))
    return " ".join((question, *unique_hints)), unique_hints


def keyword_score(document: Document, question: str, hints: tuple[str, ...]) -> int:
    """질문 및 확장 용어가 원문에 실제로 있는지 확인하는 점수입니다."""
    compact_text = re.sub(r"\s+", "", document.page_content)
    question_terms = content_terms(question)
    ordinary_score = sum(1 for term in question_terms if term in compact_text)
    direct_case_hints = {"토요일", "일요일", "개인적인 사정"}
    hint_score = sum(
        12 if hint in direct_case_hints else 4
        for hint in hints
        if re.sub(r"\s+", "", hint) in compact_text
    )
    return ordinary_score + hint_score


def document_key(document: Document) -> tuple[str, int | None, int]:
    """RRF로 합칠 때 같은 자식 청크를 하나로 처리합니다."""
    return (
        str(document.metadata.get("source", "")),
        document.metadata.get("page"),
        int(document.metadata.get("chunk_index", 0)),
    )


def retrieve_candidates(
    vector_store: InMemoryVectorStore,
    children: list[Document],
    bm25: BM25Okapi,
    question: str,
) -> RetrievalResult:
    """임베딩·BM25 결과를 RRF로 합치고 문서 근거가 약한 질문을 감지합니다."""
    search_query, hints = expanded_query(question)
    semantic_documents = vector_store.similarity_search(search_query, k=SEMANTIC_CANDIDATE_COUNT)
    bm25_scores = bm25.get_scores(tokenize(search_query))
    bm25_indices = sorted(range(len(children)), key=lambda index: bm25_scores[index], reverse=True)[:SEMANTIC_CANDIDATE_COUNT]

    scores: dict[tuple[str, int | None, int], float] = {}
    candidates: dict[tuple[str, int | None, int], Document] = {}

    def add_ranked(documents: list[Document]) -> None:
        for rank, document in enumerate(documents, start=1):
            key = document_key(document)
            candidates[key] = document
            scores[key] = scores.get(key, 0.0) + 1 / (RRF_K + rank)

    add_ranked(semantic_documents)
    add_ranked([children[index] for index in bm25_indices if bm25_scores[index] > 0])

    # 확장된 규정 용어는 보너스로만 사용하며, 특정 답을 강제로 고르지는 않습니다.
    lexical_scores = [keyword_score(child, question, hints) for child in children]
    support_terms = content_terms(question)
    support_terms.extend(re.sub(r"\s+", "", hint) for hint in hints)
    has_discriminative_match = any(
        term in re.sub(r"\s+", "", child.page_content)
        for term in support_terms
        for child in children
    )
    for index, lexical_score in enumerate(lexical_scores):
        if lexical_score <= 0:
            continue
        child = children[index]
        key = document_key(child)
        candidates[key] = child
        scores[key] = scores.get(key, 0.0) + lexical_score / 100

    ordered_keys = sorted(candidates, key=lambda key: scores[key], reverse=True)
    return RetrievalResult(
        candidates=[candidates[key] for key in ordered_keys[:SEMANTIC_CANDIDATE_COUNT]],
        has_lexical_support=has_discriminative_match,
    )


def rerank_candidates(question: str, candidates: list[Document]) -> list[Document]:
    """상위 후보 중 질문에 직접 답하는 근거만 LLM으로 최대 4개 고릅니다."""
    if len(candidates) <= FINAL_SOURCE_COUNT:
        return candidates

    candidate_text = "\n\n".join(
        f"[{index}] {document.metadata['source']} {document.metadata.get('page')}쪽\n{document.page_content}"
        for index, document in enumerate(candidates, start=1)
    )
    prompt = """당신은 공무원 여비 규정 검색 결과를 재정렬하는 역할입니다.
질문에 직접 답하거나 예외 조건을 설명하는 후보 번호만 최대 4개 선택하세요.
문서 밖의 추측은 하지 말고, 관련성이 낮은 후보는 선택하지 마세요."""
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(RerankSelection)
    selection = model.invoke([SystemMessage(content=prompt), HumanMessage(content=f"질문: {question}\n\n후보:\n{candidate_text}")])

    selected: list[Document] = []
    valid_ids = set(range(1, len(candidates) + 1))
    for candidate_id in selection.selected_ids:
        if candidate_id in valid_ids and candidates[candidate_id - 1] not in selected:
            selected.append(candidates[candidate_id - 1])
        if len(selected) == FINAL_SOURCE_COUNT:
            break
    return selected or candidates[:FINAL_SOURCE_COUNT]


def parent_context(documents: list[Document]) -> list[Document]:
    """같은 페이지의 여러 자식 청크는 하나의 부모 페이지로 합쳐 문맥을 복원합니다."""
    parents: dict[str, Document] = {}
    for document in documents:
        parent_id = str(document.metadata["parent_id"])
        if parent_id not in parents:
            parents[parent_id] = Document(
                page_content=str(document.metadata["parent_text"]),
                metadata={key: value for key, value in document.metadata.items() if key not in {"parent_text", "chunk_index"}},
            )
    return list(parents.values())


def build_context(documents: list[Document]) -> str:
    """답변 모델이 문장마다 출처 번호를 붙일 수 있는 문맥을 만듭니다."""
    sections: list[str] = []
    for index, document in enumerate(documents, start=1):
        page = document.metadata.get("page")
        location = f"{document.metadata['source']} / {page}쪽" if page else document.metadata["source"]
        sections.append(f"[S{index}] 출처: {location}\n{document.page_content}")
    return "\n\n".join(sections)


def evidence_excerpt(text: str, question: str, max_length: int = 500) -> str:
    """질문과 가까운 원문 구간을 출처 아래에 그대로 표시합니다."""
    _query, hints = expanded_query(question)
    terms = [*hints, *content_terms(question)]
    position = next((text.find(term) for term in terms if text.find(term) >= 0), -1)
    excerpt = text[max(0, position - 180) : position + max_length] if position >= 0 else text[:max_length]
    compact = re.sub(r"\s+", " ", excerpt).strip()
    return compact[:max_length] + ("..." if len(compact) > max_length else "")


def answer_question(
    vector_store: InMemoryVectorStore,
    children: list[Document],
    bm25: BM25Okapi,
    question: str,
) -> tuple[str, list[Document]]:
    """근거 부족 여부를 먼저 확인하고 검색·재정렬된 부모 문맥만으로 답변합니다."""
    retrieval = retrieve_candidates(vector_store, children, bm25, question)
    if not retrieval.has_lexical_support:
        return "제공된 문서에서 확인할 수 없습니다.", []

    sources = parent_context(rerank_candidates(question, retrieval.candidates))
    system_prompt = """당신은 제공된 공무원 여비 문서만 근거로 답하는 도우미입니다.
1. 문서 근거 밖의 일반 지식, 추측, 보완 설명을 추가하지 마세요.
2. 근거가 충분하지 않으면 정확히 '제공된 문서에서 확인할 수 없습니다.'라고 답하세요.
3. 모든 사실 문장 끝에 [S1]처럼 해당 출처 번호를 붙이세요.
4. 조건과 예외가 있으면 반드시 함께 설명하세요.
5. 문서에 없는 금액·날짜·절차는 만들지 마세요."""
    response = ChatOpenAI(model="gpt-4o-mini", temperature=0).invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"문서 근거:\n{build_context(sources)}\n\n질문: {question}"),
        ]
    )
    return str(response.content), sources


def show_openai_error(error: Exception, action: str) -> None:
    """API 잔액 오류를 원본 예외 대신 사용자가 해결할 수 있는 안내로 표시합니다."""
    error_text = str(error)
    if "credit_balance_exhausted" in error_text:
        st.error("OpenAI API 크레딧이 부족하여 문서 색인 또는 답변을 만들 수 없습니다.")
        st.markdown("[OpenAI API Billing에서 크레딧 추가하기](https://platform.openai.com/settings/organization/billing/)")
        st.info("크레딧을 추가한 뒤 이 페이지를 새로고침하거나 '문서 다시 읽기'를 눌러 주세요.")
    else:
        st.error(f"{action} 중 오류가 발생했습니다: {error}")


def configure_openai_api_key() -> bool:
    """Cloud의 Secrets를 우선 사용하고, 로컬 개발 시에는 .env를 사용합니다."""
    load_dotenv(dotenv_path=PROJECT_DIR / ".env")

    # Streamlit Community Cloud의 Secrets 값은 GitHub 저장소에 포함되지 않습니다.
    try:
        secret_key = st.secrets.get("OPENAI_API_KEY")
    except FileNotFoundError:
        secret_key = None

    api_key = secret_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        return False

    # LangChain의 OpenAIEmbeddings와 ChatOpenAI가 표준 환경 변수에서 키를 읽도록 설정합니다.
    os.environ["OPENAI_API_KEY"] = str(api_key)
    return True


def main() -> None:
    """Streamlit 화면의 질문-검색-답변 흐름입니다."""
    st.set_page_config(page_title="공무원 여비 RAG 챗봇", page_icon="📚", layout="wide")
    st.title("📚 공무원 여비 RAG 챗봇")
    st.caption("규정 PDF의 구조·원문 키워드·의미 검색을 함께 사용해 답변합니다.")
    if not configure_openai_api_key():
        st.error("OpenAI API 키가 설정되지 않았습니다.")
        st.code('OPENAI_API_KEY = "sk-..."', language="toml")
        st.caption("로컬에서는 .env를, Streamlit Cloud에서는 App settings의 Secrets를 사용하세요.")
        st.stop()

    files = list_source_files()
    if not files:
        st.error("DATA 폴더에 읽을 파일이 없습니다.")
        st.stop()

    with st.sidebar:
        st.header("문서 상태")
        for file_path in files:
            st.caption(f"- {file_path.name}")
        if st.button("문서 다시 읽기"):
            st.cache_resource.clear()
            st.rerun()

    try:
        vector_store, children, bm25, skipped_files = build_retrieval_indexes(source_signature())
    except Exception as error:
        show_openai_error(error, "문서 색인을 만드는")
        st.stop()

    st.caption(f"검색 준비 완료: {len(children)}개 구조 기반 청크")
    if skipped_files:
        st.warning("읽지 못한 파일: " + ", ".join(skipped_files))

    question = st.chat_input("문서에 대해 질문해 보세요")
    if not question:
        return

    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        with st.spinner("규정과 사례를 다시 확인하고 있습니다..."):
            try:
                answer, sources = answer_question(vector_store, children, bm25, question)
            except Exception as error:
                show_openai_error(error, "답변을 생성하는")
                return
        st.write(answer)
        st.markdown("#### 답변 근거")
        if not sources:
            st.caption("표시할 근거가 없습니다.")
        for index, document in enumerate(sources, start=1):
            page = document.metadata.get("page")
            location = f"{document.metadata['source']} / {page}쪽" if page else document.metadata["source"]
            st.markdown(f"**[S{index}] {location}**")
            st.caption(f"근거 문장: {evidence_excerpt(document.page_content, question)}")


if __name__ == "__main__":
    main()
