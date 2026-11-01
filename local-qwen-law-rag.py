#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install langchain langchain-community langchain-chroma langchain-huggingface sentence-transformers')


# In[2]:


import os
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# 1. Ollama 로컬 DeepSeek-R1 연결 (노트북 환경에 맞춤)
#llm = Ollama(model="deepseek-r1:1.5b", temperature=0.1)
llm = Ollama(model="gemma2:9b", temperature=0)


# 2. 한국어 가성비 최강 경량 임베딩 모델 로드 (최초 실행 시 1회 다운로드 진행)
print("임베딩 모델 로딩 중...")
embeddings = HuggingFaceEmbeddings(
    model_name="jhgan/ko-sbert-nli",
    model_kwargs={'device': 'cpu'}
)


print("환경 설정 완료!")


# In[3]:


# 1. 대한민국 법령 표준 체계를 반영한 법조문 텍스트 배열
law_documents_array = [
    """
    제1조(목적) 이 법은 개인정보의 처리 및 보호에 관한 사항을 정함으로써 개인의 자유와 권리를 보호하고, 나아가 개인의 존엄과 가치를 실현함을 목적으로 한다.
    """,
    """
    제15조(개인정보의 수집·이용) 
    ① 개인정보처리자는 다음 각 호의 어느 하나에 해당하는 경우에는 개인정보를 수집할 수 있으며 그 수집 목적 범위에서 이용할 수 한다.
      1. 정보주체의 동의를 받은 경우
      2. 법률에 특별한 규정이 있거나 법령상 의무를 준수하기 위하여 불가피한 경우
    ② 개인정보처리자는 제1항제1호에 따른 동의를 받을 때에는 다음 각 호의 사항을 정보주체에게 알려야 한다. 다음 각 호의 사항을 변경하는 경우에도 또한 같다.
      1. 개인정보의 수집·이용 목적
      2. 수집하려는 개인정보의 항목
      3. 개인정보의 보유 및 이용 기간
    """,
    """
    제50조(근로시간) 
    ① 1주간의 근로시간은 휴게시간을 제외하고 40시간을 초과할 수 없다.
    ② 1일의 근로시간은 휴게시간을 제외하고 8시간을 초과할 수 없다.
    ③ 제1항 및 제2항에 따른 근로시간을 산정할 때 근로자가 작업시간 도중에 실제로 작업에 종사하지 않은 대기시간 등은 근로시간으로 본다.
    """,
    """
    제56조(연장·야간 및 휴일 근로) 
    ① 사용자는 연장근로에 대하여는 통상임금의 100분의 50 이상을 가산하여 근로자에게 지급하여야 한다.
    ② 제1항에도 불구하고 사용자는 야간근로(오후 10시부터 다음 날 오전 6시 사이의 근로)에 대하여는 통상임금의 100분의 50 이상을 가산하여 지급하여야 한다.
    """
]

# 2. LangChain 형식의 Document 객체로 매핑 (출처 메타데이터 부여)
docs = [Document(page_content=law_text, metadata={"source": "대한민국 법령집"}) for law_text in law_documents_array]

# 3. 조항이 잘리지 않도록 크기를 조정한 청킹(Split) 규칙 정의
text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=150)
split_docs = text_splitter.split_documents(docs)

# 4. 임시 Chroma Vector DB 생성 및 저장
print("법령 데이터 인덱싱 진행 중...")
vectorstore = Chroma.from_documents(split_docs, embedding=embeddings)


retriever = vectorstore.as_retriever(search_kwargs={"k": 1}) # 연관성 높은 1개 조항만 추출
print("인덱싱 완료! RAG 파이프라인이 준비되었습니다.")


# In[4]:


from langchain_core.prompts import ChatPromptTemplate

# gemma2 모델이 인지하기 가장 좋은 깔끔한 한국어 지시문 구조
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "당신은 대한민국의 법령 문서를 기반으로 답변하는 전문 법률 어시스턴트입니다.\n"
        "반드시 제공된 [참고 법조문]의 내용만을 엄격한 근거로 삼아 사용자의 질문에 답변하세요.\n"
        "문서에 나와있지 않은 내용은 절대로 상상하거나 지어내어 답변해서는 안 됩니다."
    ),
    (
        "human",
        "[참고 법조문]\n{context}\n\n"
        "사용자 질문: {question}\n\n"
        "답변:"
    )
])

def ask_law_rag(question: str):
    print(f"🙋 [질문] {question}\n")

    # 1. 벡터 DB에서 연관된 조문 검색
    retrieved_docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    # 2. 메시지 구조 매핑
    messages = prompt.format_messages(context=context, question=question)

    # 3. 주피터 노트북 실시간 스트리밍 출력
    print("🤖 [로컬 LLM 답변 결과]")
    print("-" * 40)
    for chunk in llm.stream(messages):
        print(chunk, end="", flush=True)
    print("\n" + "="*60)

print("Gemma2 기반 RAG 함수 재등록 완료! 이제 [Cell 5]를 실행하세요.")


# In[5]:


ask_law_rag("개인정보를 수집할 때 상대방에게 고지해야 하는 항목들을 알려주세요.")


# In[7]:


ask_law_rag("밤 11시에 일을 더 하게 되면 수당을 얼마나 추가로 보상받을 수 있나요?")


# 주피터 노트북(.ipynb)의 가장 마지막 Markdown 셀에 기존의 DeepSeek-R1 요약 대신 통째로 복사해서 붙여넣을 수 있는 Gemma 2 기반 법률 RAG 파이프라인 아키텍처 요약본입니다. 과제 및 프로젝트 보고서용 문서화에 최적화하여 작성했습니다.
# 
# 📑 [부록] 로컬 Gemma 2 기반 법률 RAG 파이프라인 아키텍처 요약
# 
# 본 노트북은 16GB RAM 및 CPU 전용 환경에서 한국어 처리 능력과 지시 이행 능력을 극대화하도록 설계된 오픈소스 소형 전문화 모델(SLM) 기반 지식 관리 RAG 프로토타입입니다. 각 실행 셀(Cell)의 역할과 유기적 연동 매커니즘은 다음과 같습니다.
# 
# 🧩 셀별 역할 및 아키텍처 흐름
# 
# • [Cell 1] 의존성 라이브러리 설치 (Infrastructure Setup)
# 	• CPU 환경에 경량화된 최적의 텍스트 임베딩 연산 및 벡터 검색을 위해 langchain, chroma, sentence-transformers 시스템 패키지를 선제적으로 빌드합니다.
# • [Cell 2] 로컬 인프라 및 추론 엔진 로드 (LLM & Embedding Initialization)
# 	• Ollama 백그라운드 엔진: Google의 gemma2:9b 모델을 OpenAI 호환 API 형태로 상주시켜 외부 클라우드 통신 없는 100% 온프레미스 보안 환경을 구성합니다. 90억 개의 파라미터 크기로 한국어 지시 이행 능력이 매우 탁월합니다.
# 	• Ko-SBERT 경량 임베딩: CPU 사양을 고려하여 한국어 문맥 이해도가 높으면서도 메모리 점유가 극도로 적은 오픈소스 다국어 모델(JH-some/ko-sbert-nli)을 탑재합니다.
# • [Cell 3] 정형 법령 데이터 파싱 및 인덱싱 (Data Ingestion Pipeline)
# 	• 법령 표준 구조 보존: 편-장-절-조-항-호 체계를 가진 가상/실제 법조문 데이터를 구조적 훼손 없이 메모리에 적재합니다.
# 	• 컨텍스트 가둠 청킹 (Recursive Character Chunking): 법조문 조항 번호(①, 1.)나 수식어가 잘려 할루시네이션이 일어나는 것을 방지하기 위해 chunk_size=600, chunk_overlap=150 기술을 적용하여 한 조항이 완전한 컨텍스트 덩어리를 유지하도록 자릅니다.
# 	• Chroma Vector DB 스토리지: 쪼개진 텍스트를 고차원 밀집 벡터로 변환 후 다차원 공간에 매핑하여 사용자의 시맨틱(의미론적) 질문을 즉시 유클리드 거리를 기반으로 Top-K 추출(여기서는 가장 부합하는 조항 1개)하는 검색 레이어를 완성합니다.
# • [Cell 4] 추론 최적화 프롬프트 및 RAG 연동 함수 (RAG Orchestration)
# 	• 엄격한 그라운딩(Grounding) 프롬프트: 조문에 없는 허위 답변 생성을 강력하게 억제하기 위해 *"반드시 제공된 참고 법조문의 내용만을 엄격한 근거로 삼으라"*는 페르소나와 제약조건을 컨텍스트 가이드라인으로 바인딩합니다.
# 	• 실시간 스트리밍(Streaming): CPU 연산으로 인한 대기 시간(Latency) 체감을 줄이고 사용자 경험을 개선하기 위해 Gemma 2의 답변 추론 버퍼를 주피터 런타임에 실시간으로 흘려보냅니다.
# • [Cell 5] 실전 검증 테스트 (Evaluation Scenario)
# 	• 개인정보 보호법과 근로기준법에 정의된 실무형 질문을 투입하여, 소형 언어 모델의 한계를 극복하고 영어나 타 언어 섞임(Code-switching) 현상 없이 정확한 한국어 결론에 도달하는지 실증 테스트를 수행합니다.
# 
# 💡 시스템 고도화 및 확장 방안 (Next Steps)
# 
# 1. 엔터프라이즈 마크다운 파싱: 대형 법령 텍스트나 PDF에서 #, ## 헤더를 인식하는 MarkdownHeaderTextSplitter를 적용하여 수천 줄의 법조문을 자동으로 연동하는 데이터 파이프라인 구축.
# 2. 보안 가드레일 접목: NVIDIA NeMo Guardrails 등을 입력 단에 전치 배치하여 악의적인 프롬프트 인젝션 공격 및 법률 외 질문을 사전에 원천 차단하는 레이어 추가 가능.
# Gemma 2 기반의 깔끔한 요약본 문서가 준비되었습니다.
# 혹시 코드를 돌려보시면서 Gemma 2의 답변 속도나 리소스 점유율에 만족하시나요? 아니면 과제 발표용으로 질문과 답변을 웹 브라우저 창에서 깔끔하게 주고받을 수 있는 간이 UI(Streamlit) 코드 작성법이 추가로 필요하신지 말씀해 주세요!
# 

# In[ ]:




