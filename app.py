import os
from pathlib import Path

import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

st.set_page_config(page_title="AIVLE School AI 학습 도우미", page_icon="🎓", layout="wide")
DB_DIR = "chroma_db"
COLLECTION = "aivle_docs"

st.markdown("""<style>
.stApp {background:#f5fbf9}
.block-container {max-width:1050px;padding-top:2rem}
[data-testid="stSidebar"] {background:#e5f5f1}
.hero {padding:25px 30px;background:linear-gradient(110deg,#087d77,#28b5a5);border-radius:18px;color:white;margin-bottom:20px}
.hero h1 {color:white;font-size:2rem;margin:0}
.hero p {margin:8px 0 0;color:#eafffb}
.stButton button {border-radius:12px}
[data-testid="stChatMessage"] {background:white;border:1px solid #dceeea;border-radius:14px}
</style>""", unsafe_allow_html=True)

if not os.getenv("OPENAI_API_KEY"):
    try:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    except (KeyError, FileNotFoundError):
        pass

@st.cache_resource
def embeddings():
    return OpenAIEmbeddings(model="text-embedding-3-small")

@st.cache_resource
def base_store():
    if not Path(DB_DIR).is_dir():
        return None
    return Chroma(collection_name=COLLECTION, embedding_function=embeddings(), persist_directory=DB_DIR)

@st.cache_resource
def llm():
    return ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=True)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "extra_docs" not in st.session_state:
    st.session_state.extra_docs = []
if "uploaded_hashes" not in st.session_state:
    st.session_state.uploaded_hashes = set()
if "pending" not in st.session_state:
    st.session_state.pending = None


def upload_pdfs(files):
    import hashlib
    import pymupdf
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    count = 0
    for file in files:
        raw = file.getvalue()
        digest = hashlib.sha256(raw).hexdigest()
        if digest in st.session_state.uploaded_hashes:
            continue
        pdf = pymupdf.open(stream=raw, filetype="pdf")
        pages = [Document(page_content=p.get_text(), metadata={"source":file.name,"page":i})
                 for i,p in enumerate(pdf) if p.get_text().strip()]
        pdf.close()
        if not pages:
            st.warning(f"{file.name}: 텍스트 추출 불가 (스캔 PDF는 OCR 필요)")
            continue
        st.session_state.extra_docs.extend(splitter.split_documents(pages))
        st.session_state.uploaded_hashes.add(digest)
        count += 1
    return count


def source_name(doc):
    name = Path(str(doc.metadata.get("source", "통합 백서"))).name
    page = doc.metadata.get("page")
    return f"{name} / {int(page)+1}페이지" if isinstance(page, int) else name


def search_docs(query, k):
    found = []
    store = base_store()
    if store is not None:
        found.extend(store.similarity_search(query, k=k))
    if st.session_state.extra_docs:
        # 업로드 문서는 세션에만 보관하며 기존 공유 DB를 변경하지 않음
        tokens = [word.lower() for word in query.split() if len(word) >= 2]
        ranked = sorted(st.session_state.extra_docs,
                        key=lambda d: sum(d.page_content.lower().count(t) for t in tokens),
                        reverse=True)
        found.extend(ranked[:k])
    unique, seen = [], set()
    for doc in found:
        key = doc.page_content
        if key not in seen:
            unique.append(doc)
            seen.add(key)
    return unique[:k]


def answer(question, k):
    recent = st.session_state.messages[-6:]
    history = "\n".join(f"{m['role']}: {m['content'][:300]}" for m in recent)
    query = question
    if recent:
        result = llm().invoke([
            ("system", "대화 맥락을 참고해 마지막 질문을 독립적인 검색 질문으로 재작성하세요. 검색 질문만 출력하세요."),
            ("human", f"이전 대화:\n{history}\n\n마지막 질문: {question}")
        ])
        query = result.content.strip() or question
    docs = search_docs(query, k)
    if not docs:
        return "검색 가능한 문서가 없습니다. 백서 DB 또는 업로드한 PDF를 확인해 주세요.", []
    context = "\n\n".join(f"[자료 {i}] {source_name(d)}\n{d.page_content}" for i,d in enumerate(docs,1))
    prompt = [
        ("system", "당신은 AIVLE School 학습 어시스턴트입니다. 제공된 자료만 근거로 정확하고 친절하게 답하세요. 자료에 없는 비용·일정·규정·후기를 지어내지 마세요. 확인할 수 없으면 백서에서 확인되지 않는다고 설명하세요. 근거가 있는 문장에는 [자료 1] 형식으로 표시하세요. 지원과 학습에 도움이 되는 실용적인 조언을 하되 합격이나 취업을 보장하지 마세요."),
        ("human", f"이전 대화(맥락 파악용):\n{history}\n\n검색 자료:\n{context}\n\n질문: {question}")
    ]
    return llm().stream(prompt), docs

with st.sidebar:
    st.title("🎓 AIVLE Assistant")
    st.caption("선배의 경험을 후배의 성장으로")
    if st.button("＋ 새 대화", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.divider()
    k = st.slider("참고 문서 개수", 2, 8, 4)
    files = st.file_uploader("📄 PDF 문서 추가", type="pdf", accept_multiple_files=True)
    if st.button("문서 적용", use_container_width=True, disabled=not files):
        try:
            n = upload_pdfs(files)
            st.success(f"새 PDF {n}개 적용 완료")
        except Exception as exc:
            st.error(f"업로드 오류: {exc}")
    if st.button("추가 문서 초기화", use_container_width=True):
        st.session_state.extra_docs = []
        st.session_state.uploaded_hashes = set()
        st.rerun()
    st.caption(f"추가한 PDF: {len(st.session_state.uploaded_hashes)}개")
    if st.session_state.messages:
        transcript = "\n\n".join(f"{m['role']}: {m['content']}" for m in st.session_state.messages)
        st.download_button("대화 내용 다운로드", transcript, "aivle_chat.txt", use_container_width=True)
    st.caption("※ 공식 일정·규정은 최신 운영 공지를 확인하세요.")

st.markdown('<div class="hero"><h1>🎓 AIVLE School AI 학습 도우미</h1><p>교육과정, 프로젝트, 선배들의 학습 전략을 백서 기반으로 안내합니다.</p></div>', unsafe_allow_html=True)
if not os.getenv("OPENAI_API_KEY"):
    st.error("OPENAI_API_KEY를 Colab 환경변수 또는 Streamlit Secrets에 설정해 주세요.")
if not Path(DB_DIR).is_dir():
    st.info("기본 chroma_db 폴더가 없습니다. PDF를 추가하면 해당 문서로 질문할 수 있습니다.")

st.subheader("💡 추천 질문")
examples = ["비전공자는 입교 전에 무엇을 준비하면 좋을까요?", "DX 트랙에서는 무엇을 배우나요?", "미니프로젝트를 잘 진행하는 방법은?", "선배들이 추천하는 학습 루틴은?"]
cols = st.columns(2)
for i, example in enumerate(examples):
    with cols[i % 2]:
        if st.button(example, key=f"example_{i}", use_container_width=True):
            st.session_state.pending = example

st.divider()
st.subheader("💬 AI 학습 상담")
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📚 참고한 문서"):
                for label, excerpt in msg["sources"]:
                    st.markdown(f"**{label}**")
                    st.caption(excerpt)

question = st.chat_input("궁금한 점을 입력하세요...") or st.session_state.pending
st.session_state.pending = None
if question:
    if not os.getenv("OPENAI_API_KEY"):
        st.error("OPENAI_API_KEY가 필요합니다.")
    else:
        with st.chat_message("user"):
            st.markdown(question)
        try:
            with st.chat_message("assistant"):
                with st.spinner("백서에서 관련 정보를 찾는 중..."):
                    result, docs = answer(question, k)
                response = st.markdown(result) if isinstance(result, str) else st.write_stream(result)
                if isinstance(result, str):
                    response = result
                sources = [(source_name(d), d.page_content[:900]) for d in docs]
                if sources:
                    with st.expander("📚 참고한 문서"):
                        for label, excerpt in sources:
                            st.markdown(f"**{label}**")
                            st.caption(excerpt)
            st.session_state.messages.extend([
                {"role":"user", "content":question},
                {"role":"assistant", "content":response, "sources":sources}
            ])
        except Exception as exc:
            st.error(f"답변 생성 오류: {exc}")
