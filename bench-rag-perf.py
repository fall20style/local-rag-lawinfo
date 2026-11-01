#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Ollama vs llama-cpp-python 속도/메모리 벤치마크 (동일 모델·동일 프롬프트·동일 면).

측정 항목:
  - 모델 로드(접속) 시간
  - TTFT (첫 토큰까지의 시간, warm 기준)
  - 생성 토큰/초, 전체 응답 시간
  - 전체 사용 메모리 증가량 (system used delta, 샘플링 스레드)
  - 백엔드 프로세스 RSS 피크 (ollama 합산 / python 프로세스)

사용법:
  python bench-rag-perf.py --backend both \
      --gguf models/qwen2.5-7b-instruct-q4_k_m.gguf \
      --ollama-model qwen2.5:7b --threads 8
"""

import argparse
import gc
import json
import threading
import time
from datetime import datetime
from pathlib import Path

import psutil
import requests

from law_rag_core import BENCH_QUESTIONS, build_retriever, make_prompt, retrieve_context

OLLAMA_HOST = "http://127.0.0.1:11434"
OUT_DIR = Path("benchmarks")


# ---------------------------------------------------------------- RAM 샘플러
class RamSampler(threading.Thread):
    """30ms 주기로 '전체 메모리 사용량'과 대상 프로세스 RSS를 기록."""

    def __init__(self, include_ollama: bool = False):
        super().__init__(daemon=True)
        self.include_ollama = include_ollama
        self.stop_flag = False
        self.base_used = psutil.virtual_memory().used
        self.max_used_delta_mb = 0.0
        self.max_proc_rss_mb = 0.0       # python(+ llama.cpp) RSS
        self.max_ollama_rss_mb = 0.0     # ollama 관련 프로세스 RSS 합

    def _backend_rss(self) -> float:
        """ollama/llama-server 관련 프로세스 RSS 합계 (bytes)."""
        total = 0.0
        for p in psutil.process_iter(["name", "memory_info"]):
            name = (p.info["name"] or "").lower()
            if "ollama" in name or "llama-server" in name:
                try:
                    total += p.info["memory_info"].rss
                except Exception:
                    pass
        return total

    def run(self):
        me = psutil.Process()
        while not self.stop_flag:
            used = psutil.virtual_memory().used
            self.max_used_delta_mb = max(self.max_used_delta_mb, (used - self.base_used) / 1024**2)
            self.max_proc_rss_mb = max(self.max_proc_rss_mb, me.memory_info().rss / 1024**2)
            if self.include_ollama:
                self.max_ollama_rss_mb = max(self.max_ollama_rss_mb, self._backend_rss() / 1024**2)
            time.sleep(0.03)

    def stop(self):
        self.stop_flag = True
        self.join(timeout=2)


# ---------------------------------------------------------------- Ollama 백엔드
def bench_ollama(model: str, questions: list[str], retriever, prompt, max_tokens: int):
    print(f"\n=== [1/2] Ollama ({model}) ===")
    results = {"backend": "ollama", "model": model, "runs": [], "ram": {}}

    # ① 콜드 로드: 모델이 메모리에 없는 상태에서 첫 요청 (generate로 즉시 언로드 후 측정)
    requests.post(f"{OLLAMA_HOST}/api/generate",
                  json={"model": model, "keep_alive": 0}, timeout=600)
    payload = {"model": model, "prompt": "빠른 워밍업 질문: 1+1은?", "stream": False,
               "options": {"num_predict": 4, "temperature": 0}}
    t0 = time.perf_counter()
    requests.post(f"{OLLAMA_HOST}/api/generate", json=payload, timeout=600)
    cold_load_ms = (time.perf_counter() - t0) * 1000
    results["cold_load_ms"] = cold_load_ms
    print(f"  콜드 로드(모델 접속) {cold_load_ms/1000:.1f}s")

    total_used_delta = 0.0
    total_ollama_peak = 0.0
    for q in questions:
        _, ctx_ms = retrieve_context(retriever, q)
        msgs = prompt.format_messages(context=_, question=q)
        text_out = "".join(m.type + ": " + m.content + "\n\n" for m in msgs)

        body = {"model": model, "prompt": text_out, "stream": True,
                "keep_alive": "5m",
                "options": {"temperature": 0, "num_predict": max_tokens, "num_ctx": 2048}}

        sampler = RamSampler(include_ollama=True)
        sampler.start()
        t_first = None
        n_tok = 0
        t0 = time.perf_counter()
        with requests.post(f"{OLLAMA_HOST}/api/generate", json=body, stream=True, timeout=600) as r:
            for line in r.iter_lines():
                if not line:
                    continue
                obj = json.loads(line)
                if "response" in obj and obj["response"]:
                    if t_first is None:
                        t_first = time.perf_counter()
                    n_tok += 1
                if obj.get("done"):
                    metrics = obj
        total_ms = (time.perf_counter() - t0) * 1000
        ttft_ms = (t_first - t0) * 1000 if t_first else None
        gen_s = (time.perf_counter() - t_first) if t_first else 0
        run = {
            "question": q[:30] + "...",
            "retrieval_ms": round(ctx_ms, 1),
            "ttft_ms": None if ttft_ms is None else round(ttft_ms, 1),
            "gen_tokens": metrics.get("eval_count") or n_tok,
            "total_ms": round(total_ms, 1),
            "wall_tok_s": round(metrics.get("eval_count", n_tok) / gen_s, 2) if gen_s > 0.05 else None,
            "engine_eval_rate": metrics.get("eval_rate"),          # Ollama 자체 측정 tok/s
            "prompt_eval_count": metrics.get("prompt_eval_count"),
        }
        results["runs"].append(run)
        print(f"  {q[:24]:<26} TTFT {ttft_ms:6.0f}ms  {run['gen_tokens']:>4} tok  "
              f"{run['wall_tok_s'] or 0:>6.2f} tok/s  total {total_ms/1000:5.1f}s")
        sampler.stop()
        total_used_delta = max(total_used_delta, sampler.max_used_delta_mb)
        total_ollama_peak = max(total_ollama_peak, sampler.max_ollama_rss_mb)
        resident_mb = None
        try:
            psj = requests.get(f"{OLLAMA_HOST}/api/ps", timeout=5).json()
            if psj.get("models"):
                resident_mb = psj["models"][0]["size"] / 1024**2
        except Exception:
            pass
        results["ram"] = {
            "system_used_delta_mb": round(total_used_delta, 1),
            "ollama_rss_peak_mb": round(total_ollama_peak, 1),
            "ollama_reported_resident_mb": None if resident_mb is None else round(resident_mb, 0),
        }

    # ⑦ 메모리 반환 (keep_alive=0 으로 즉시 언로드)
    requests.post(f"{OLLAMA_HOST}/api/generate",
                  json={"model": model, "keep_alive": 0}, timeout=600)
    return results


# ---------------------------------------------------------------- llama.cpp 백엔드
def bench_llamacpp(gguf: str, questions: list[str], retriever, prompt,
                   n_threads: int, n_ctx: int, max_tokens: int):
    print(f"\n=== [2/2] llama-cpp-python ({gguf}) ===")
    from llama_cpp import Llama
    results = {"backend": "llamacpp", "model": Path(gguf).name,
               "threads": n_threads, "n_ctx": n_ctx, "runs": [], "ram": {}}

    t0 = time.perf_counter()
    llama = Llama(model_path=gguf, n_ctx=n_ctx, n_threads=n_threads,
                  n_batch=512, verbose=False)
    results["load_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    print(f"  모델 로드 {results['load_ms']/1000:.1f}s")

    sampler = RamSampler()
    sampler.start()

    # 워밍업 (컨텍스트/컴퓨트 버프 예열) — 측정 조합 동일하게 chat template 사용
    llama.create_chat_completion(
        messages=[{"role": "user", "content": "1+1은?"}],
        max_tokens=4, temperature=0)

    for q in questions:
        ctx, ctx_ms = retrieve_context(retriever, q)
        msgs = prompt.format_messages(context=ctx, question=q)
        # Ollama(앱) 경로와 동일: messages -> plain text -> chat template 적용
        chat = [{"role": m.type, "content": m.content} for m in msgs]

        t_first = None
        n_tok = 0
        t0 = time.perf_counter()
        last = None
        for chunk in llama.create_chat_completion(messages=chat, max_tokens=max_tokens,
                                                  temperature=0, stream=True):
            delta = chunk["choices"][0].get("delta", {}).get("content")
            if delta:
                if t_first is None:
                    t_first = time.perf_counter()
                n_tok += 1
            last = chunk
        total_ms = (time.perf_counter() - t0) * 1000
        ttft_ms = (t_first - t0) * 1000 if t_first else None
        gen_s = (time.perf_counter() - t_first) if t_first else 0
        run = {
            "question": q[:30] + "...",
            "retrieval_ms": round(ctx_ms, 1),
            "ttft_ms": None if ttft_ms is None else round(ttft_ms, 1),
            "gen_tokens": n_tok,
            "total_ms": round(total_ms, 1),
            "wall_tok_s": round(n_tok / gen_s, 2) if gen_s > 0.05 else None,
        }
        results["runs"].append(run)
        print(f"  {q[:24]:<26} TTFT {ttft_ms:6.0f}ms  {n_tok:>4} tok  "
              f"{run['wall_tok_s'] or 0:>6.2f} tok/s  total {total_ms/1000:5.1f}s")

    sampler.stop()
    results["ram"] = {
        "system_used_delta_mb": round(sampler.max_used_delta_mb, 1),
        "python_rss_peak_mb": round(sampler.max_proc_rss_mb, 1),
    }

    del llama
    gc.collect()
    return results


# ---------------------------------------------------------------- 보고
def write_report(out: dict, tag: str = "result"):
    def tok_avg(runs):
        vals = [r["wall_tok_s"] for r in runs if r["wall_tok_s"]]
        return round(sum(vals) / len(vals), 2) if vals else None

    def ttft_avg(runs):
        vals = [r["ttft_ms"] for r in runs if r["ttft_ms"]]
        return round(sum(vals) / len(vals), 0) if vals else None

    def total_avg(runs):
        return round(sum(r["total_ms"] for r in runs) / len(runs), 0)

    lines = []
    lines.append("# Ollama vs llama-cpp-python 벤치마크 결과")
    lines.append("")
    lines.append(f"- 시각: {out['meta']['timestamp']}")
    lines.append(f"- CPU: {out['meta']['cpu']}")
    lines.append(f"- RAM: {out['meta']['ram_total_gb']} GB / 여유")
    lines.append(f"- 모델: {out['meta']['model_note']}")
    lines.append(f"- 스레드: {out['meta']['threads']}, num_ctx=2048, max_tokens=350, temperature=0")
    lines.append("")
    lines.append("| 항목 | Ollama | llama-cpp-python |")
    lines.append("|---|---|---|")
    lines.append(f"| 콜드 로드/모델 로드 | {out['ollama']['cold_load_ms']/1000:.1f} s | {out['llamacpp']['load_ms']/1000:.1f} s |")
    lines.append(f"| TTFT 평균 (warm) | {ttft_avg(out['ollama']['runs'])} ms | {ttft_avg(out['llamacpp']['runs'])} ms |")
    lines.append(f"| 생성 속도 평균 | {tok_avg(out['ollama']['runs'])} tok/s | {tok_avg(out['llamacpp']['runs'])} tok/s |")
    lines.append(f"| 총 응답 시간 평균 | {total_avg(out['ollama']['runs'])} ms | {total_avg(out['llamacpp']['runs'])} ms |")
    lines.append(f"| 시스템 메모리 증가량 | {out['ollama']['ram']['system_used_delta_mb']} MB | {out['llamacpp']['ram']['system_used_delta_mb']} MB |")
    lines.append(f"| 프로세스 RSS 피크 | {out['ollama']['ram'].get('ollama_rss_peak_mb', '?')} MB (ollama+llama-server 합산) | {out['llamacpp']['ram']['python_rss_peak_mb']} MB (python 인프로세스) |")
    lines.append("| 모델 상주 크기 | {0} MB (ollama /api/ps 보고) | ~모델 파일+버퍼 (python RSS에 포함) |".format(out['ollama']['ram'].get('ollama_reported_resident_mb', '?')))

    md = "\n".join(lines) + "\n"
    path = OUT_DIR / f"RESULT_{tag}.md"
    path.write_text(md, encoding="utf-8")
    print("\n" + md)
    return md


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["ollama", "llamacpp", "both"], default="both")
    ap.add_argument("--gguf", default="models/qwen2.5-7b-instruct-q4_k_m.gguf")
    ap.add_argument("--ollama-model", default="qwen2.5:7b")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=350)
    ap.add_argument("--tag", default="qwen25-7b-q4k8")
    args = ap.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    print("공용 RAG 인덱스/임베딩 준비중...")
    retriever = build_retriever(k=1)
    prompt = make_prompt()
    out = {"meta": {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "cpu": f"{psutil.cpu_count(logical=False)}C/{psutil.cpu_count()}T",
        "ram_total_gb": round(psutil.virtual_memory().total / 1024**3, 1),
        "model_note": f"Qwen2.5 7B Instruct Q4_K_M ({args.ollama_model} / {Path(args.gguf).name})",
        "threads": args.threads,
    }, "ollama": None, "llamacpp": None}

    if args.backend in ("ollama", "both"):
        out["ollama"] = bench_ollama(args.ollama_model, BENCH_QUESTIONS, retriever, prompt, args.max_tokens)
        # 메모리 해제 대기 (시스템이 회수할 시간)
        time.sleep(8)
        print("Ollama 모델 언로드 완료, 8초 대기")

    if args.backend in ("llamacpp", "both"):
        out["llamacpp"] = bench_llamacpp(args.gguf, BENCH_QUESTIONS, retriever, prompt,
                                         args.threads, 2048, args.max_tokens)

    json.dump(out, open(OUT_DIR / f"results_{args.tag}.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    try:
        write_report(out, tag=args.tag)
    except Exception as e:  # 일부 백엔드 누락 시에도 JSON 우선 유지
        print(f"(report build skip: {e})")
    print(f"\n결과 저장: {OUT_DIR / f'results_{args.tag}.json'}")


if __name__ == "__main__":
    main()
