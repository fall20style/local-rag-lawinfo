# Troubleshooting — 트러블슈팅 기록

개발·실험 과정에서 겪은 문제와 해결 내역을 정리한 문서입니다. 코드에 남은 흔적(주석·파라미터·프롬프트 문구)을 근거로 작성했으며, 새 이슈 발생 시 아래 형식으로 계속 추가합니다.

> 형식: **증상 → 원인 → 해결**

---

## 1. DeepSeek-R1 1.5B의 한국어 답변 품질 부족

- **증상**: 초경량 `deepseek-r1:1.5b`가 질문 맥락을 벗어난 답변을 하거나, 추론 태그(`<think>`) 형식의 출력이 섞여 답변이 지저분해짐.
- **원인**: 1.5B급 모델은 한국어 법률 지시 이행 능력이 크게 부족.
- **해결**: `gemma2:9b`(노트북 실험용)와 `qwen2.5:7b`(앱 확정 모델)로 교체.
  - 흔적: `local-qwen-law-rag.py` Cell 2에 `#llm = Ollama(model="deepseek-r1:1.5b", ...)` 주석으로 남아 있음.

## 2. 답변 언어 섞임 (Code-switching)

- **증상**: 질문이 영어 단어를 포함하거나 프롬프트 문맥 영향으로, 답변 전체 혹은 일부가 영어/중국어로 생성됨.
- **원인**: 소형 LLM의 다국어 훈련 분포 특성 — 지시 언어와 무관하게 고빈도 언어로 붕괴.
- **해결**: 시스템 프롬프트에 언어 제약을 명시.
  ```python
  "답변은 한국어로 해야 합니다. 답변 전체를 영어나 중국어로 하는 것은 금지됩니다."
  ```
  - 흔적: `app-local-qwen-law-rag.py`의 system 프롬프트에만 이 문구가 존재 (노트북 기본 프롬프트에는 없음 → 앱 노출 단계에서 관측된 문제).

## 3. 조문 경계 잘림 → 환각/부정확 답변

- **증상**: 청킹 후 "...② 개인정보처리자는 제1항제1호에 따른 동의를 받을 때에는..."처럼 **항·호 번호 유지 관계가 끊긴 조각**이 검색되고, LLM이 해당 조각을 독립적 규범처럼 해석해 답변이 부정확해짐.
- **원인**: 기본 분할 설정으로는 조문 경계를 인식하지 못함.
- **해결**: 조문 1개가 온전히 담기는 크기로 조정.
  ```python
  RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=150)
  ```
  - 흔적: Cell 3의 주석 — "조항이 잘리지 않도록 크기를 조정한 청킹(Split) 규칙 정의".

## 4. Streamlit 세션마다 모델·인덱스 재로딩 (매우 느림)

- **증상**: 위젯 조작(텍스트 입력 등)마다 Rerun이 발생하며 Ollama 프록시 임베딩·인덱싱이 반복 실행되어 UI가 몇 초씩 멎음.
- **원인**: Streamlit은 스크립트를 위아래로 계속 재실행하므로, 모듈 최상위 선언부가 매번 평가됨.
- **해결**: 초기화 전체 + `@st.cache_resource` 감싸기.
  ```python
  @st.cache_resource
  def init_rag_system(): ...
  ```
  주석에도 명시: "모델 및 DB가 세션마다 새로 로드되는 메모리 낭비 방지".

## 5. 임베딩·LLM 장비 의존성 (GPU 없음)

- **증상**: sentence-transformers가 기본적으로 CUDA를 시도하며 시작 단계에서 느려지거나 실패하는 환경.
- **원인**: GPU 미탑재/메모리 부족 머신.
- **해결**: 디바이스 명시.
  ```python
  model_kwargs={'device': 'cpu'}
  ```
- **컨텍스트**: 16GB RAM CPU 전용 환경을 전제로 설계 — 9B 이하 SLM만 채택.

## 6. 노트북 반복 실행 시 Chroma 중복 적재

- **증상**: `Chroma.from_documents()`를 커널 재시작 없이 여러 번 호출하면 **같은 조문이 중복으로 누적**되어, 검색 결과가 중복 청크로 채워짐.
- **원인**: 인메모리 Chroma(`from_documents`)는 컬렉션을 매번 새로 만들지 않고 기존 상태를 이어 쓸 수 있음.
- **해결**: 노트북 반복 실험 후에는 반드시 **커널 재시작** 또는 컬렉션 재생성. 영속화가 필요하면 `persist_directory` 사용.

## 7. LangChain 임포트 경로 변화

- **증상**: 버전 업그레이드 후 `from langchain_community.llms import Ollama`가 deprecated 경고 또는 오류.
- **원인**: LangChain 패키지 구조 분리 (community → 전용 패키지).
- **해결**: 최종 앱은 전용 패키지 사용.
  ```python
  from langchain_ollama import OllamaLLM
  ```
  - 흔적: 노트북은 `langchain_community.llms.Ollama`, 앱은 `langchain_ollama.OllamaLLM`.

## 8. Ollama 서버 미실행 → 연결 실패

- **증상**: `ConnectionError: Failed to establish a new connection ... ('127.0.0.1', 11434)` 류의 오류.
- **원인**: Ollama 백그라운드 서버가 켜져 있지 않거나, 대상 모델이 pull되지 않음.
- **해결**:
  ```bash
  ollama serve            # 서버 기동 (보통 자동 실행됨)
  ollama list             # 모델 존재 확인
  ollama pull qwen2.5:7b  # 없다면 다운로드
  ```

## 9. 첫 실행 시 임베딩 모델 다운로드로 정지처럼 보임

- **증상**: 첫 실행에서 "임베딩 모델 로딩 중..." 이후 오래 걸림 (프리징 아님).
- **원인**: `jhgan/ko-sbert-nli`를 HuggingFace Hub에서 1회 다운로드.
- **해결**: 기다리면 캐시되어 이후 실행은 즉시. 오프라인 환경은 미리 동일 버전으로 다운로드 후 캐시 복사.

---

## llama.cpp 백엔드 관련 (llama-cpp-python 전환 작업)

## 10. llama-cpp-python 소스 빌드 실패

- **증상**: `pip install llama-cpp-python`이 "Failed building wheel"로 실패.
- **원인**: PyPI에 Windows 바이너리 휠이 없고, 소스 빌드에는 MSVC 빌드 도구가 필요한데 이 머신에 없음.
- **해결**: GitHub 릴리스의 사전빌드 휠 설치.
  ```bash
  pip install https://github.com/abetlen/llama-cpp-python/releases/download/v0.3.36/llama_cpp_python-0.3.36-py3-none-win_amd64.whl
  ```
  (최신 태그는 변형 빌드(vulkan/cuda 등)는 있어도 plain CPU win wheel이 없는 경우가 있어 버전 확인 필요)

## 11. Ollama 실행 관리 프로세스의 이름이 `llama-server`라 메모리 추적 누락

- **증상**: 벤치마크에서 Ollama 쪽 RSS 피크가 38MB 등으로 비정상적으로 작게 기록됨.
- **원인**: 최신 Ollama는 추론을 별도 프로세스(`llama-server`)로 띄우므로, 이름 매칭 `ollama`만 하면 모델이 올라간 프로세스를 놓침. Get-Process로 확인해보니 `llama-server` WorkingSet이 약 4.9GB였음.
- **해결**: 샘플러 매칭을 `"ollama" or "llama-server"`로 수정. 또한 Ollama 공식 `/api/ps`의 모델 상주 크기를 병행 기록.

## 12. Ollama에서 요청 옵션 변경 시 모델이 재로딩됨 (TTFT 25s 폭발)

- **증상**: 벤치마크 첫 질문의 TTFT가 매번 약 25초.
- **원인**: Ollama는 이전 요청과 옵션이 다르면(`num_ctx` 등) 러너를 재기동. 기본 컨텍스트로 워밍업 후 `num_ctx=2048` 지정 요청이 와서 재로딩 발생.
- **해결**: 앱 레벨에서는 동일 옵션 일관성 유지로 방지. 이는 llama.cpp 인프로세스 방식에서는 원천적으로 발생하지 않는 문제 (BENCHMARKS.md §3.2 참고).

## 13. llama-cpp-python의 type_k/type_v 잘못된 값 → 컨텍스트 생성 실패

- **증상**: `Llama(type_k=16)` 전달 시 `ValueError: Failed to create llama_context`.
- **원인**: llama-cpp-python 문서 기본값 `16`은 "미지정" 의미의 센티넬이지 GGML 타입이 아님. 실제 값은 `GGML_TYPE_F16=1`, `GGML_TYPE_Q8_0=8`.
- **해결**: 명시 시 1(f16)/8(q8_0)을 사용하고, 미지정은 None으로 두고 인자 생략. (`python -c "import llama_cpp; print(int(llama_cpp.GGML_TYPE_F16))"`로 확인 가능)

## 14. V 캐시 양자화(type_v=q8_0) 시 컨텍스트 생성 실패

- **증상**: `type_v=8` 단독 지정 시 `Failed to create llama_context` (K만 양자화하면 정상).
- **원인**: llama.cpp는 V 캐시 양자화에 Flash Attention 활성화가 전제.
- **해결**: `Llama(..., flash_attn=True)`를 함께 지정. K+V 양자화 비교 실험 시 f16 기준선도 FA=True로 통일해 공정하게 재측정.

## 15. no-mmap 풀 벤치 실행 중 머신 스래싱으로 타임아웃

- **증상**: `use_mmap=False`로 4.3GB 가중치를 통째로 RAM 적재하는 벤치가 타임아웃, 잔존 프로세스가 5.9GB 점유.
- **원인**: 여유 RAM ~7GB 환경에서 가중치 전체 상주 + 컨퓨트 버퍼 조합이 커밋 한계를 넘어 페이지아웃 스래싱.
- **해결**: RSS 기반 메모리 비교는 폐기하고, 컨텍스트 생성 로그의 공식 KV/compute buffer 값(`benchmarks/kv_size_probe.py`)으로 대체. 저사양 머신에서 no-mmap 실행은 배경 앱 없이 단독으로만 할 것.

---

## 메모 / 미해결

- [ ] Top-1(`k=1`) 회수 방식은 "질문이 두 조문에 걸친 경우" 정밀도가 떨어짐 → Top-K + 출처 카드 확대 필요 (IMPLEMENTATION.md §5).
- [ ] 프롬프트 인젝션 방어가 아직 없음 → 법률 외 질문도 조문을 억지로 근거 삼아 답변할 수 있음.
- [ ] `local-qwen-law-rag.py`는 노트북 변환본이므로 `get_ipython()` 의존성이 남아 있어 **IPython 환경에서만 직접 실행 가능** → 순수 .py화 또는Streamlit 앱으로 통합 필요.
- [ ] llama.cpp가 Ollama 대비 tok/s가 20% 낮음 → P코어 affinity pinning 실험 필요 (BENCHMARKS.md §3.1).
- [ ] llama.cpp 인프로세스 방식은 앱 RSS가 약 8GB로 Ollama(러너 4.9GB + 앱 별도)보다 큼 → KV 캐시 양자화(`--cache-type-k q8_0`)나 더 작은 모델로 완화 검토.
