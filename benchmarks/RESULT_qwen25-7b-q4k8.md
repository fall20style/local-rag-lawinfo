# Ollama vs llama-cpp-python 벤치마크 결과

- 시각: 2026-11-01T14:26:39
- CPU: 12C/16T
- RAM: 15.7 GB / 여유
- 모델: Qwen2.5 7B Instruct Q4_K_M (qwen2.5:7b / qwen2.5-7b-instruct-q4_k_m.gguf)
- 스레드: 8, num_ctx=2048, max_tokens=350, temperature=0

| 항목 | Ollama | llama-cpp-python |
|---|---|---|
| 콜드 로드/모델 로드 | 21.0 s | 9.3 s |
| TTFT 평균 (warm) | 9369.0 ms | 2226.0 ms |
| 생성 속도 평균 | 6.99 tok/s | 5.44 tok/s |
| 총 응답 시간 평균 | 23853.0 ms | 15707.0 ms |
| 시스템 메모리 증가량 | 132.6 MB | 1034.7 MB |
| 프로세스 RSS 피크 | 4872.2 MB (ollama+llama-server 합산) | 7996.7 MB (python 인프로세스) |
| 모델 상주 크기 | 3148.0 MB (ollama /api/ps 보고) | ~모델 파일+버퍼 (python RSS에 포함) |
