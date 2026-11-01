# USAGE.md — 실행 가이드

이 저장소의 프로그램을 앱별로 실행하는 방법을 번호로 정리한 문서입니다. 모두 GPU 없는 CPU 전용 환경(15~16GB RAM) 기준입니다.

## 0. 공통 사전 준비 (어느 앱이든 1회)

```bash
# 1) 의존성 설치
pip install streamlit langchain langchain-community langchain-chroma \
    langchain-huggingface langchain-ollama llama-cpp-python sentence-transformers

# 2) Ollama 설치 후 모델 다운로드 (앱 1 사용 시)
ollama pull qwen2.5:7b

# 3) llama.cpp용 GGUF (앱 2 사용 시, Ollama 캐시에서 추출 or 허깅페이스 다운로드)
#    models/qwen2.5-7b-instruct-q4_k_m.gguf (약 4.4GB)
```

> 참고: Windows에서 `pip install llama-cpp-python`이 소스 빌드 실패하면(prebuilt wheel 필요) 사전빌드 휠을 설치하세요.
> `pip install https://github.com/abetlen/llama-cpp-python/releases/download/v0.3.36/llama_cpp_python-0.3.36-py3-none-win_amd64.whl`
> (TROUBLES.md §10 참고)

---

## 1. Streamlit 챗봇 — Ollama 백엔드 (원본 버전)

파일: `app-local-qwen-law-rag.py`

```bash
streamlit run app-local-qwen-law-rag.py
```

- 사전 요구: Ollama 서버 실행(`ollama serve` 또는 앱이 자동 실행) + `ollama pull qwen2.5:7b`
- 특징: 채팅 UI, 스트리밍 응답, 근거 조문 출처 표기
- 성능 기준: warm TTFT ~9.4s, 총 응답 ~24s (BENCHMARKS.md §2)

## 2. Streamlit 챗봇 — llama-cpp-python 백엔드 (개선 버전)

파일: `app-local-qwen-law-rag-llamacpp.py`

```bash
streamlit run app-local-qwen-law-rag-llamacpp.py
```

- 사전 요구: `models/qwen2.5-7b-instruct-q4_k_m.gguf` 파일만 있으면 됨 (Ollama 불필요)
- 특징: 인프로세스 추론, `flash_attn=True` + KV 캐시 q8_0 양자화 내장
- 성능 기준: warm TTFT ~2.2s, 총 응답 ~16s, KV 버퍼 −47% (BENCHMARKS.md §2, §6)
- 장기 컨텍스트(n_ctx≥8192) 사용 시 `type_v`를 f16(값 1)으로 되돌릴 것 (§6.3)

## 3. Jupyter 노트북 — 실험용 파이프라인

파일: `local-qwen-law-rag.ipynb`, `local-gemma2-law-rag.ipynb`

```bash
jupyter lab local-qwen-law-rag.ipynb
# 또는
jupyter lab local-gemma2-law-rag.ipynb
```

- 셀 순서대로 실행: [Cell 1] 의존성 → [Cell 2] 모델 로드 → [Cell 3] 인덱싱 → [Cell 4] RAG 함수 → [Cell 5] 평가 질문
- 기본 LLM은 `gemma2:9b` (`ollama pull gemma2:9b` 필요). DeepSeek-R1 주석 토글 가능
- 노트북 재실행 전 커널 재시작 권장 (Chroma 중복 적재 방지, TROUBLES.md §6)

## 4. 노트북 변환 스크립트 (참고용)

파일: `local-qwen-law-rag.py`

- `.ipynb` → `.py` 변환본. `get_ipython()` 호출이 남아 있어 **IPython 커널 안에서만 정상 실행됨**:

```bash
ipython local-qwen-law-rag.py
```

- 일반 `python local-qwen-law-rag.py`로는 실행되지 않으며, 검증된 서비스 경로는 앱 1/2를 사용할 것

## 5. 성능 벤치마크 — Ollama vs llama.cpp

파일: `bench-rag-perf.py`

```bash
# 두 백엔드 비교 (Ollama → 언로드 → llama.cpp 순서로 자동 진행)
python bench-rag-perf.py --backend both \
    --gguf models/qwen2.5-7b-instruct-q4_k_m.gguf \
    --ollama-model qwen2.5:7b --threads 8

# llama.cpp 단독
python bench-rag-perf.py --backend llamacpp --gguf models/*.gguf --threads 8
```

주요 옵션:

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--backend` | `both` | `ollama` / `llamacpp` / `both` |
| `--gguf` | `models/qwen2.5-7b-instruct-q4_k_m.gguf` | llama.cpp용 가중치 |
| `--ollama-model` | `qwen2.5:7b` | Ollama 모델명 |
| `--threads` | `8` | llama.cpp 스레드 수 |
| `--n-ctx` | `2048` | 컨텍스트 크기 |
| `-k`, `-v` | `None`(f16) | KV 캐시 양자화 (`1`=f16, `8`=q8_0) |
| `--no-mmap` | off | 가중치 RAM 완전 적재 (저사양 머신 주의, TROUBLES.md §15) |
| `--tag` | `qwen25-7b-q4k8` | 결과 파일 접미사 |

- 결과: `benchmarks/results_<tag>.json` + `benchmarks/RESULT_<tag>.md`

## 6. KV 캐시 버퍼 크기 확인

파일: `benchmarks/kv_size_probe.py`

```bash
python benchmarks/kv_size_probe.py
```

- f16 / K=q8_0 / K+V=q8_0 세 구성의 공식 KV 버퍼 크기를 컨텍스트 생성 로그에서 캡처해 출력
- 모델 파일은 `models/qwen2.5-7b-instruct-q4_k_m.gguf` 고정 (수정하려면 스크립트 내 경로 편집)

## 7. 벤치마크 결과 종합

파일: `benchmarks/analyze_kv.py`

```bash
python benchmarks/analyze_kv.py
```

- `results_kv-*.json`, `results_nctx8k-*.json` 등 반복 런을 모아 tok/s·응답 시간·RAM 평균 출력
- 새 태그를 비교에 넣으려면 스크립트의 `TAGS` 목록에 추가

---

## 빠른 추천 경로

- **그냥 서비스 데모**: 앱 2 (`streamlit run app-local-qwen-law-rag-llamacpp.py`)
- **외부 데이터 없이 Ollama 생태계 유지**: 앱 1
- **실험/학습**: 앱 3 노트북
- **성능 재검증**: 앱 5 → 6 → 7 순서

문제 발생 시 TROUBLES.md, 설계 배경은 ARCHITECTURE.md / IMPLEMENTATION.md 참고.
