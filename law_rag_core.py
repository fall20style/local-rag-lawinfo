"""법률 RAG 파이프라인 공용 코어 (Ollama / llama-cpp-python 양쪽에서 재사용).

- 법조문 데이터, 청킹, 임베딩, Chroma 인덱스, 프롬프트를 산출한다.
- LLM 인스턴스는 백엔드별로 별도 구성하므로 여기서 만들지 않는다.
"""

import os
import time

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 대한민국 법령 표준 체계를 반영한 샘플 법조문 (기존 PoC와 동일한 데이터셋)
LAW_DOCUMENTS = [
    "제1조(목적) 이 법은 개인정보의 처리 및 보호에 관한 사항을 정함으로써 개인의 자유와 권리를 보호하고, 나아가 개인의 존엄과 가치를 실현함을 목적으로 한다.",
    "제15조(개인정보의 수집·이용) ① 개인정보처리자는 다음 각 호의 어느 하나에 해당하는 경우에는 개인정보를 수집할 수 있으며 그 수집 목적 범위에서 이용할 수 있다. 1. 정보주체의 동의를 받은 경우 2. 법률에 특별한 규정이 있거나 법령상 의무를 준수하기 위하여 불가피한 경우\n② 개인정보처리자는 제1항제1호에 따른 동의를 받을 때에는 다음 각 호의 사항을 정보주체에게 알려야 한다.\n1. 개인정보의 수집·이용 목적 2. 수집하려는 개인정보의 항목 3. 개인정보의 보유 및 이용 기간",
    "제50조(근로시간) ① 1주간의 근로시간은 휴게시간을 제외하고 40시간을 초과할 수 없다. ② 1일의 근로시간은 휴게시간을 제외하고 8시간을 초과할 수 없다.",
    "제56조(연장·야간 및 휴일 근로) ① 사용자는 연장근로에 대하여는 통상임금의 100분의 50 이상을 가산하여 근로자에게 지급하여야 한다. ② 사용자는 야간근로(오후 10시부터 다음 날 오전 6시 사이의 근로)에 대하여는 통상임금의 100분의 50 이상을 가산하여 지급하여야 한다.",
]

BENCH_QUESTIONS = [
    "개인정보를 수집할 때 상대방에게 고지해야 하는 항목들을 알려주세요.",
    "밤 11시에 일을 더 하게 되면 수당을 얼마나 추가로 보상받을 수 있나요?",
    "위 참고 법조문을 바탕으로, 사내 근로 관리 안내문을 200자 내외로 작성해 주세요.",
]


def build_retriever(k: int = 1):
    """임베딩 + Chroma 인덱스를 만들고 retriever를 반환한다 (LLM과 무관, 공통 자원)."""
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sbert-nli",
        model_kwargs={"device": "cpu"},
    )
    docs = [Document(page_content=t, metadata={"source": "대한민국 법령집"}) for t in LAW_DOCUMENTS]
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=150)
    vectorstore = Chroma.from_documents(splitter.split_documents(docs), embedding=embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": k})


def make_prompt():
    """기존 Streamlit 앱과 동일한 그라운딩/한국어 고정 프롬프트."""
    return ChatPromptTemplate.from_messages([
        ("system",
         "당신은 대한민국의 법령 문서를 기반으로 답변하는 전문 법률 어시스턴트입니다. "
         "반드시 제공된 [참고 법조문]의 내용만을 엄격한 근거로 삼아 사용자의 질문에 정확하게 답변하세요. "
         "문서에 나와있지 않은 내용은 절대로 상상하거나 지어내어 답변해서는 안 됩니다. "
         "답변은 한국어로 해야 합니다. 답변 전체를 영어나 중국어로 하는 것은 금지됩니다."),
        ("human", "[참고 법조문]\n{context}\n\n사용자 질문: {question}\n\n답변:"),
    ])


def retrieve_context(retriever, question: str, k: int = 1):
    t0 = time.perf_counter()
    docs = retriever.invoke(question)
    ms = (time.perf_counter() - t0) * 1000
    return "\n\n".join(d.page_content for d in docs), ms
