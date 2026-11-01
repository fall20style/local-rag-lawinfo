"""Streamlit 챗봇 — llama-cpp-python 백엔드 버전.

기존 app-local-qwen-law-rag.py(Ollama 백엔드)와 동일한 UI/UX.
차이점:
  - Ollama 서버 프로세스가 필요 없음: 모델 GGUF를 프로세스 내 직접 로드
  - Llama(n_threads)로 하이브리드 CPU 스케줄링 제어
실행: streamlit run app-local-qwen-law-rag-llamacpp.py
"""

import streamlit as st

from law_rag_core import build_retriever, make_prompt

GGUF_PATH = "models/qwen2.5-7b-instruct-q4_k_m.gguf"

# --- [Streamlit UI 설정] ---
st.set_page_config(page_title="로컬 법률 RAG 에이전트 (llama.cpp)", page_icon="⚖️", layout="centered")
st.title("⚖️ 온프레미스 법률 지식 관리 에이전트")
st.caption("Qwen 2.5 7B GGUF · llama-cpp-python 인프로세스 추론 · Ollama 서버 불필요")


# --- [캐싱: 모델·인덱스가 세션마다 새로 로드되는 것을 방지] ---
@st.cache_resource
def init_rag_system():
    from llama_cpp import Llama

    # 1. llama.cpp 인프로세스 LLM (Ollama 서버 불필요)
    llama = Llama(
        model_path=GGUF_PATH,
        n_ctx=2048,
        n_threads=8,        # 하이브리드 CPU: 4 P-core + 4 E-core 균형점
        n_batch=512,
        verbose=False,
    )

    # 2. 공용 임베딩/인덱스/프롬프트 로드
    retriever = build_retriever(k=1)
    prompt = make_prompt()

    return llama, retriever, prompt


llama, retriever, prompt = init_rag_system()

# --- [챗봇 스타일 메시지 로그 관리] ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- [사용자 입력 및 RAG 추론 프로세스] ---
if user_question := st.chat_input("법률 조항이나 사내 규정에 대해 질문하세요..."):
    with st.chat_message("user"):
        st.markdown(user_question)
    st.session_state.messages.append({"role": "user", "content": user_question})

    with st.chat_message("assistant"):
        # 1. 로컬 검색 엔진 가동
        docs = retriever.invoke(user_question)
        context = "\n\n".join(d.page_content for d in docs)
        msgs = prompt.format_messages(context=context, question=user_question)
        chat = [{"role": m.type, "content": m.content} for m in msgs]

        # 2. UI 레이어에 실시간 텍스트 스트리밍 출력
        response_placeholder = st.empty()
        full_response = ""
        for chunk in llama.create_chat_completion(
                messages=chat, max_tokens=512, temperature=0, stream=True):
            token = chunk["choices"][0].get("delta", {}).get("content") or ""
            full_response += token
            response_placeholder.markdown(full_response + "▌")

        # 3. 출처(Source) 컴포넌트 시각화 레이어
        full_response += f"\n\n\n📌 **[근거 조문 출처]** {docs[0].page_content[:40]}..."
        response_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
