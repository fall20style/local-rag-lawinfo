# BENCHMARKS.md — Ollama vs llama-cpp-python 성능/메모리 비교

## 1. 측정 환경

| 항목 | 값 |
|---|---|
| CPU | Intel i7-1360P (6 P-core + 8 E-core, 16 thread), GPU 없음 |
| RAM | 15.7 GB (CPU-only) |
| OS | Windows 11 |
| 모델 | Qwen2.5 7B Instruct Q4_K_M - 양쪽 백엔드 동일 가중치 |
| llama.cpp | llama-cpp-python 0.3.36 (Windows 사전빌드 CPU wheel), n_threads=8, n_ctx=2048, n_batch=512 |
| Ollama | qwen2.5:7b (동일 Q4_K_M GGUF), num_ctx=2048 |
| 공통 | 동일 임베딩(ko-sbert-nli) + Chroma 인덱스, 동일 프롬프트, temperature=0, max_tokens=350 |

- 측정 방법: `bench-rag-perf.py`가 질문 3개를 두 백엔드에서 순차 실행. TTFT/토큰수/응답 시간은 스트림 타임스탬프로 계산하고, RAM은 30ms 주기 샘플링 스레드로 기록.
- 재현: `python bench-rag-perf.py --backend both --gguf models/qwen2.5-7b-instruct-q4_k_m.gguf --ollama-model qwen2.5:7b --threads 8`
- 원본 데이터: `benchmarks/results_*.json`, 자동 생성 표: `benchmarks/RESULT_*.md`

## 2. 결과 요약 (3회 측정에서 재현성 확인, 아래는 최종 회차)

| 항목 | Ollama (서버 분리) | llama-cpp-python (인프로세스) | 개선 |
|---|---|---|---|
| 콜드 로드 (모델 첫 접속) | 21.0 s | 9.3 s | 2.3배 빠름 |
| TTFT 평균 (warm) | 9,369 ms | 2,226 ms | 4.2배 빠름 |
| 총 응답 시간 평균 | 23,853 ms | 15,707 ms | -34% |
| 생성 속도 평균 | 6.99 tok/s | 5.44 tok/s | -22% (Ollama 우세) |
| 프로세스 RSS 피크 | 4,872 MB (+앱 프로세스 별도) | 7,997 MB (앱 포함 단일 프로세스) | 메모리는 Ollama 유리 |

### 질문별 상세

| 질문 | Ollama TTFT / tok/s / total | llama.cpp TTFT / tok/s / total |
|---|---|---|
| 개인정보 고지 항목 | 24,788 ms / 7.63 / 31.7 s | 6,273 ms / 5.49 / 18.7 s |
| 야간 수당 | 1,973 ms / 6.74 / 19.0 s | 180 ms / 5.34 / 14.4 s |
| 안내문 작성 (long-form) | 1,346 ms / 6.59 / 20.8 s | 226 ms / 5.50 / 14.0 s |

## 3. 해석

### 3.1 속도 - llama.cpp의 이점은 "콜드 스타트 + 통신 오버헤드"

- TTFT 9.4s 에서 2.2s로 단축. Ollama는 HTTP 라운드트립 + 서버 큐잉이 걸리는 반면 llama.cpp는 프로세스 안에서 바로 계산. warm 구간 TTFT는 Ollama 1.3~2.0s 대비 llama.cpp 0.18~0.23s로, 요청당 고정 오버헤드 약 1.5s 차이가 나고 응답이 길어질수록 누적된다.
- 순수 생성 속도는 Ollama가 오히려 20%가량 빠르다. Ollama 러너(llama-server 기반)가 하이브리드 CPU에 대한 스레드 배치 최적화가 잘 되어 있다. 우리 쪽 n_threads=8 세팅으로는 따라가지 못하며, 12 스레드 실험(`results_qwen25-7b-t12.json`)에서도 5.2~5.5 tok/s로 개선이 없었다. P/E 혼합 CPU에서 tok/s의 한도는 8 스레드 근처.

### 3.2 Ollama의 함정 - 옵션 변경은 곧 모델 재로딩

Ollama는 요청 옵션(num_ctx 등)이 이전 요청과 다르면 모델을 다시 로드한다. 실측: 옵션 없는 첫 요청 후 num_ctx=2048 지정 요청의 TTFT가 24.8s (모델 재로딩 21s 포함). 앱이 항상 동일 옵션을 쓰면 실서비스에서 발생하지 않지만, 옵션 일관성이 깨지면 순간적으로 멈춘 것처럼 보인다. llama.cpp는 인프로세스로 컨텍스트가 고정되어 이 문제가 없다.

### 3.3 메모리 - 절대 메모리는 Ollama가 유리

- llama.cpp 인프로세스: 앱 프로세스 RSS 8.0 GB (GGUF mmap 4.3GB + KV/컴퓨트 버퍼 + ko-sbert 임베딩 + Python 런타임 전부 한 프로세스에 포함).
- Ollama: 러너(llama-server) RSS 약 4.9GB + 앱(임베딩/인덱스)은 별도. 임베딩이 LLM보다 가벼워 총 부하가 낮다.
- llama.cpp의 RSS가 큰 이유는 GGUF mmap 페이지가 앱 RSS에 집계되기 때문. RAM이 극도로 부족하면 양쪽 모두 페이지아웃되므로, 여유 메모리 5GB 이하 환경에서는 체감 차이는 크지 않다.

## 4. 결론

| 목표 | 권장 |
|---|---|
| UX 지연 최소화, 서버 프로세스 제거, 임베딩/LLM 단일 프로세스 통합 | llama-cpp-python (콜드 2.3배, TTFT 4.2배, 총 응답 -34%) |
| 절대 메모리 최소화, 여러 클라이언트 서빙, 도구 호환성 | Ollama (RSS 4.9GB 유지, tok/s 20% 우세) |

이 프로젝트는 Streamlit 앱을 llama.cpp 인프로세스 방식으로 교체하기로 확정했다. 콜드 스타트/TTFT 개선이 메모리 증가 트레이드오프보다 크게 판단됨. 생성 속도는 스레드 affinity 튜닝으로 재검토 여지가 있다.

## 5. 측정 유의사항

- "시스템 used 증가량"은 Windows에서 mmap이 standby 페이지로 집계되는 특성 때문에 실제 체감보다 작게 기록됨. 프로세스 RSS 피크 기준으로 해석할 것.
- Ollama 응답 토큰 수(115/128 tok)와 llama.cpp 응답 토큰 수(76 tok)의 차이는 백엔드 chat 템플릿 처리 방식 차이로 생성 길이가 다른 것. 속도 비교는 tok/s 기준으로 해석.
- 하이브리드 CPU 특성상 tok/s 상한이 낮으며, affinity pinning으로 개선 여지 있음.
- GPU 없는 환경 기준이며, 배경 부하(브라우저 등)에 따라 측정값이 10~20% 변동할 수 있음.

## 6. KV 캐시 양자화 실험 (type_k/type_v q8_0)

### 6.1 공식 메모리 절감 (verbose 로그의 컨텍스트 생성 보고값, n_ctx=8192)

| 구성 | KV 버퍼 | 절감 | compute buffer |
|---|---|---|---|
| f16 / f16 (기본) | 448 MiB | 기준 | 311 MiB |
| K=q8_0 | 343 MiB | −105 MiB (−23%) | 311 MiB |
| **K+V=q8_0** | **238 MiB** | **−210 MiB (−47%)** | 318 MiB |

측정 스크립트: `benchmarks/kv_size_probe.py` (모델 로드 시 출력되는 `KV buffer size` 캡처). n_ctx=2048에서는 K+V 기준 약 −53 MiB로 절감량이 작으며, 컨텍스트가 클수록 절감이 커진다 (선형 비례).

### 6.2 속도 (wall tok/s, flash_attn=True, 2회 반복 평균)

| 구성 | n_ctx=2048 | n_ctx=8192 |
|---|---|---|
| f16 기본 | 5.60 tok/s (기준) | 6.26 tok/s (기준) |
| K+V q8_0 | **6.50 tok/s (+16%)** | 5.72 tok/s (−9%) |

- n_ctx=2048(서비스 사용 범위)에서는 KV 양자화로 디코드 속도가 오히려 개선됨 (bandwidth 절감).
- n_ctx=8192에서는 역전: 디코드 단계 어텐션이 전체 컨텍스트를 훑기 때문에 CPU에서 q8 dequant 비용이 컨텍스트 길이에 비례해 커져서 f16보다 느려짐. → **CPU 장기 컨텍스트에서는 K만 양자화 또는 f16 유지 권장**.
- 반복 4회 표본, 첫 질문(템플릿 예열 편차)은 평균에서 제외.

### 6.3 결론

| 목표 | 권장 설정 |
|---|---|
| 이 앱 같은 단문 법률 QA (n_ctx=2048급) | `flash_attn=True, type_k=q8_0, type_v=q8_0` — RAM −47%, 속도 +16% 동시 개선 |
| 장기 대화/긴 문서 주입 (n_ctx≥8192) | `flash_attn=True, type_k=q8_0, type_v=f16` — K만 양자화 (−23%) |
| 최대 품질 우선, 여유 RAM | 기본 f16 유지 |

- KV 수치 양자화는greedy 결과에 미세 영향: 동일 프롬프트라도 응답 길이가 달라짐(측정 중 55~106 tok 변동) → 법률 정확성 우선이라면 개정 시 품질 점검 권장. V 캐시 양자화는 K보다 품질 영향이 크므로 장기 컨텍스트에서는 V를 f16으로 두는 것이 안전.
- `app-local-qwen-law-rag-llamacpp.py`는 §6.3 첫 줄 설정으로 업데이트함.

### 6.4 측정 유의사항 (추가)

- KV 양자화 실험엔 `flash_attn=True`를 모든 구성에 적용했다. llama.cpp는 **V 캐시 양자화 시 FA 활성화가 필수**이며, 비교 공정성을 위해 f16 기준선도 FA로 재측정했다.
- `type_k`/`type_v`에 널리 알려진 "기본값 16"은 llama-cpp-python의 미지정 센티넬이며 실제 값은 GGML 열거형(F16=1, Q8_0=8)이라야 한다. 잘못된 값(예: 16)을 넘기면 `Failed to create llama_context`가 발생한다.
- RSS 피크 측정은 mmap 페이지 유입량 배경 노이즈(±400MB)가 커서 메모리 비교는 위 공식 보고값(KV buffer size) 기준으로 해석할 것.
