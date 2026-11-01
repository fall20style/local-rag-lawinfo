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
