import os
import streamlit as st

from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document




# --- [Streamlit UI 설정] ---
st.set_page_config(page_title="로컬 법률 RAG 에이전트", page_icon="⚖️", layout="centered")
st.title("⚖️ 온프레미스 법률 지식 관리 에이전트")
st.caption("Qwen 2.5 7B 및 로컬 Vector DB 기반 가혹 환경 최적화 프로토타입")

# --- [캐싱 처리: 모델 및 DB가 세션마다 새로 로드되는 메모리 낭비 방지] ---
@st.cache_resource
def init_rag_system():
    # 1. 로컬 LLM 설정
    llm = OllamaLLM(model="qwen2.5:7b", temperature=0)

    
    # 2. 로컬 임베딩 설정
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sbert-nli",
        model_kwargs={'device': 'cpu'}
    )
    
    # 3. 데이터 선언 및 인덱싱
    law_documents_array = [
        "제1조(목적) 이 법은 개인정보의 처리 및 보호에 관한 사항을 정함으로써 개인의 자유와 권리를 보호하고, 나아가 개인의 존엄과 가치를 실현함을 목적으로 한다.",
        "제15조(개인정보의 수집·이용) ① 개인정보처리자는 다음 각 호의 어느 하나에 해당하는 경우에는 개인정보를 수집할 수 있으며 그 수집 목적 범위에서 이용할 수 있다. 1. 정보주체의 동의를 받은 경우 2. 법률에 특별한 규정이 있거나 법령상 의무를 준수하기 위하여 불가피한 경우\n② 개인정보처리자는 제1항제1호에 따른 동의를 받을 때에는 다음 각 호의 사항을 정보주체에게 알려야 한다.\n1. 개인정보의 수집·이용 목적 2. 수집하려는 개인정보의 항목 3. 개인정보의 보유 및 이용 기간",
        "제50조(근로시간) ① 1주간의 근로시간은 휴게시간을 제외하고 40시간을 초과할 수 없다. ② 1일의 근로시간은 휴게시간을 제외하고 8시간을 초과할 수 없다.",
        "제56조(연장·야간 및 휴일 근로) ① 사용자는 연장근로에 대하여는 통상임금의 100분의 50 이상을 가산하여 근로자에게 지급하여야 한다. ② 사용자는 야간근로(오후 10시부터 다음 날 오전 6시 사이의 근로)에 대하여는 통상임금의 100분의 50 이상을 가산하여 지급하여야 한다."
    ]
    docs = [Document(page_content=law_text, metadata={"source": "대한민국 법령집"}) for law_text in law_documents_array]
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=150)
    split_docs = text_splitter.split_documents(docs)
    
    vectorstore = Chroma.from_documents(split_docs, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
    
    # 4. 프롬프트 정의
    prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 대한민국의 법령 문서를 기반으로 답변하는 전문 법률 어시스턴트입니다. 반드시 제공된 [참고 법조문]의 내용만을 엄격한 근거로 삼아 사용자의 질문에 정확하게 답변하세요. 문서에 나와있지 않은 내용은 절대로 상상하거나 지어내어 답변해서는 안 됩니다. 답변은 한국어로 해야 합니다. 답변 전체를 영어나 중국어로 하는 것은 금지됩니다."),
        ("human", "[참고 법조문]\n{context}\n\n사용자 질문: {question}\n\n답변:")
    ])
    
    return llm, retriever, prompt

# 시스템 초기화 실행
llm, retriever, prompt = init_rag_system()

# --- [챗봇 스타일 메시지 로그 관리] ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# 기존 대화 기록 화면에 출력
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- [사용자 입력 및 RAG 추론 프로세스] ---
if user_question := st.chat_input("법률 조항이나 사내 규정에 대해 질문하세요..."):
    # 1. 사용자 질문 UI 출력 및 상태 저장
    with st.chat_message("user"):
        st.markdown(user_question)
    st.session_state.messages.append({"role": "user", "content": user_question})
    
    # 2. 에이전트 RAG 연산 가동
    with st.chat_message("assistant"):
        # 로컬 검색 엔진 가동
        retrieved_docs = retriever.invoke(user_question)
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        formatted_messages = prompt.format_messages(context=context, question=user_question)
        
        # UI 레이어에 실시간 텍스트 스트리밍 출력
        response_placeholder = st.empty()
        full_response = ""
        
        # Ollama의 스트리밍 출력을 받아 UI 업데이트
        for chunk in llm.stream(formatted_messages):
            full_response += chunk
            response_placeholder.markdown(full_response + "▌")
        
        # 출처(Source) 컴포넌트 시각화 레이어 추가
        full_response += f"\n\n\n📌 **[근거 조문 출처]** {retrieved_docs[0].page_content[:40]}..."
        response_placeholder.markdown(full_response)
        
    st.session_state.messages.append({"role": "assistant", "content": full_response})
