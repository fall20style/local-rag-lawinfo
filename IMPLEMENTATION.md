# Implementation — 구현 상세

본 문서는 각 파일의 구현 내역과 핵심 코드, 파라미터 선정 근거를 설명합니다. 아키텍처 개요는 [ARCHITECTURE.md](ARCHITECTURE.md)를 참고하세요.

## 1. 파일 구성 요약

| 파일 | 형태 | LLM | 용도 |
|---|---|---|---|
| `local-qwen-law-rag.ipynb` | Jupyter 노트북 | `gemma2:9b` (temp=0) | 실험/실습, Cell 단위 학습 |
| `local-gemma2-law-rag.ipynb` | Jupyter 노트북 | `gemma2:9b` (temp=0) | 위 노트북과 동일 구성의 Gemma2 버전 |
| `local-qwen-law-rag.py` | Python 스크립트 | `gemma2:9b` (temp=0) | 노트북 → .py 변환본 (`get_ipython()` 흔적 포함) |
| `app-local-qwen-law-rag.py` | Streamlit 앱 | `qwen2.5:7b` (temp=0) | **최종 사용자 UI** — 채팅 인터페이스 |

> 노트북(.ipynb) 진행 중 모델 비교 실험: `deepseek-r1:1.5b` → 제외(주석), `gemma2:9b` → 채택. Streamlit 앱은 `qwen2.5:7b`로 최종 확정.

## 2. 노트북 / 스크립트 셀(단계)별 구현

### Cell 1 — 의존성 설치

```python
get_ipython().system('pip install langchain langchain-community langchain-chroma langchain-huggingface sentence-transformers')
```

- 노트북 전용 설치 셀. Streamlit 앱에서는 사전 pip 설치를 전제로 함.

### Cell 2 — 로컬 LLM & 임베딩 초기화

```python
llm = Ollama(model="gemma2:9b", temperature=0)     # deepseek-r1:1.5b는 주석 처리로 보류
embeddings = HuggingFaceEmbeddings(
    model_name="jhgan/ko-sbert-nli",
    model_kwargs={'device': 'cpu'}                 # CPU 명시
)
```

- Streamlit 앱(`app-local-qwen-law-rag.py`)은 임포트가 다르다: `from langchain_ollama import OllamaLLM` — LangChain 0.3+에서 권장되는 전용 패키지 사용.

### Cell 3 — 법령 인덱싱 파이프라인

```python
law_documents_array = [ ... ]  # 개인정보보호법 제1조·제15조, 근로기준법 제50조·제56조
docs = [Document(page_content=t, metadata={"source": "대한민국 법령집"}) for t in law_documents_array]

text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=150)
split_docs = text_splitter.split_documents(docs)

vectorstore = Chroma.from_documents(split_docs, embedding=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
```

핵심 결정 3가지:

1. **`Document` + 메타데이터 매핑** — 청크에 출처를 부여해 근거 조문 표시 가능하게 함.
2. **`chunk_size=600 / overlap=150`** — 조문 1개가 온전히 담기는 크기. 경계 잘림으로 인한 "② …" 조각+환각 문제를 overlap으로 방어.
3. **`k=1`** — 가장 유사한 조항만 주입. 소형 LLM 컨텍스트 오염을 회피하기 위한 보수적 회수 전략.

### Cell 4 — 그라운딩 프롬프트 & RAG 함수

```python
prompt = ChatPromptTemplate.from_messages([
    ("system", "...반드시 제공된 [참고 법조문]의 내용만을 엄격한 근거로 삼아 답변하세요..."
              "문서에 나와있지 않은 내용은 절대로 상상하거나 지어내어 답변해서는 안 됩니다."),
    ("human", "[참고 법조문]\n{context}\n\n사용자 질문: {question}\n\n답변:")
])

def ask_law_rag(question: str):
    retrieved_docs = retriever.invoke(question)                       # ① 검색
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    messages = prompt.format_messages(context=context, question=question)  # ② 조립
    for chunk in llm.stream(messages):                               # ③ 스트리밍
        print(chunk, end="", flush=True)
```

- LCEL(`|`) 체인 대신 **명시적 3단계(검색 → 조립 → 스트리밍)** — 각 단계 출력을 노트북에서 눈으로 확인하기 위함.
- 노트북에서는 `llm.stream()`을 `print(..., flush=True)`로 실시간 출력.

### Cell 5 — 평가 질문

```python
ask_law_rag("개인정보를 수집할 때 상대방에게 고지해야 하는 항목들을 알려주세요.")
ask_law_rag("밤 11시에 일을 더 하게 되면 수당을 얼마나 추가로 보상받을 수 있나요?")
```

- 각각 `제15조 ②(고지 항목 3종)` / `제56조 ②(야간근로 50% 가산)` 조문에 매핑되는 검색·정답 도달 검증용 질문.

## 3. Streamlit 앱 구현 (`app-local-qwen-law-rag.py`)

### 3.1 리소스 캐싱 — 세션마다 재로딩 방지

```python
@st.cache_resource
def init_rag_system():
    llm = OllamaLLM(model="qwen2.5:7b", temperature=0)
    embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sbert-nli", model_kwargs={'device': 'cpu'})
    ...
    return llm, retriever, prompt

llm, retriever, prompt = init_rag_system()
```

- 모델 로딩 + 임베딩 + 인덱싱 전체를 `@st.cache_resource`로 감싸 **세션 간/리렌더링마다 재연산을 차단**. 이것이 응답 UX의 핵심 병목 관리 지점.

### 3.2 언어 고정 프롬프트 (앱 버전 차이점)

노트북의 프롬프트에서 노트북에는 없는 제약이 앱에 추가되어 있음:

```python
"답변은 한국어로 해야 합니다. 답변 전체를 영어나 중국어로 하는 것은 금지됩니다."
```

- GUI 서비스 노출 시 관측된 다국어 code-switching 억제용.

### 3.3 채팅 상태 관리

```python
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:           # 기존 로그 렌더
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_question := st.chat_input("법률 조항이나 사내 규정에 대해 질문하세요..."):
    ...                                          # user 메시지 즉시 push
```

- `st.session_state.messages`를 유일한 진실 공급원으로 하는 단순 채팅 로그 방식.
- `:=` (walrus)로 입력 존재 여부를 한 번에 판정.

### 3.4 스트리밍 렌더링

```python
response_placeholder = st.empty()
full_response = ""
for chunk in llm.stream(formatted_messages):
    full_response += chunk
    response_placeholder.markdown(full_response + "▌")   # 타이핑 커서
response_placeholder.markdown(full_response)             # 완료 시 커서 제거
```

- `st.empty()` 홀더를 재사용하며 토큰마다 마크다운을 갱신해 실시간 타이핑 효과.
- CPU 추론의 체감 지연(간헐적 정지 느낌)을 스트리밍으로 완화.

### 3.5 근거 조문 출처 표기

```python
full_response += f"\n\n\n📌 **[근거 조문 출처]** {retrieved_docs[0].page_content[:40]}..."
```

- 검색된 Top-1 청크의 선두 40자를 답변 하단에 출처로 첨부 → 답변의 검증 가능성(Attribution) 제공.

## 4. 파라미터 레퍼런스

| 파라미터 | 값 | 선정 근거 |
|---|---|---|
| LLM `temperature` | `0` | 법률 QA: 창의 배제·결정론적 근거 사용 |
| 청킹 | `chunk_size=600 / overlap=150` | 조문 단위 보존, 경계 잘림 방지 |
| 검색 Top-K | `k=1` | 소형 LLM 컨텍스트 오염 방지 |
| 임베딩 device | `cpu` | 저사양 환경 고정 |
| UI 스트리밍 커서 | `"▌"` | 진행 중임을 시각적으로 표시 |

## 5. 확장 구현 가이드 (Next Steps)

1. **실제 법령 파일 인덱싱**
   ```python
   from langchain_community.document_loaders import PyPDFLoader, TextLoader
   from langchain_text_splitters import MarkdownHeaderTextSplitter  # 헤더(편/장/조) 기반 분할
   ```
   하드코딩 배열 → 로더 기반으로 교체. 헤더 구조를 유지하는 분할이 조문 완전성을 지켜줌.
2. **Top-K 확장 + 근거 카드**
   - `search_kwargs={"k": 3}` 확장 후, 출처를 조문별 카드(`st.expander`)로 여럿 표시.
3. **영속적 Chroma**
   ```python
   vectorstore = Chroma.from_documents(..., persist_directory="./law_chroma")
   ```
   매 시작마다 재임베딩하는 비용 제거.
4. **가드레일**
   - NeMo Guardrails 등에 질문 전처리를 붙여 법률 외 주제·인젝션 차단.

> ⚠️ `Chroma`를 인메모리(`from_documents` 재호출)로 계속 쓰는 경우, 노트북을 반복 실행하면 **같은 문서가 중복 적재**됩니다. 개발 중 인덱스가 이상해지면 커널/앱을 완전 재시작하세요. 상세는 [TROUBLES.md](TROUBLES.md) 참고.
