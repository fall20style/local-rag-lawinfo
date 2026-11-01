"""벤치마크 결과 종합 (일회용 분석 스크립트)."""
import json
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

TAGS = [
    ("f16 @2048", ["kv-f16", "kv-f16-r2"]),
    ("K+V q8_0 @2048", ["kv-kvq8", "kv-kvq8-r2"]),
    ("f16 @8192", ["nctx8k-f16", "nctx8k-f16-r2"]),
    ("K+V q8_0 @8192", ["nctx8k-kvq8", "nctx8k-kvq8-r2"]),
]

for label, tags in TAGS:
    tok_list, ram_list, load_list, tot_list = [], [], [], []
    for t in tags:
        d = json.load(open(f"benchmarks/results_{t}.json", encoding="utf-8"))["llamacpp"]
        vals = [r["wall_tok_s"] for r in d["runs"] if r["wall_tok_s"]][1:]  # q1(템플릿 예열) 제외
        tot = [r["total_ms"] for r in d["runs"]][1:]
        tok_list += vals
        tot_list += tot
        ram_list.append(d["ram"]["python_rss_peak_mb"])
        load_list.append(d["load_ms"] / 1000)
    n = len(tok_list)
    print(f"{label:20s} | tok_s {min(tok_list):.2f}~{max(tok_list):.2f} (avg {sum(tok_list)/n:.2f}, n={n})"
          f" | total {sum(tot_list)/len(tot_list):>7.0f} ms"
          f" | RAM_peak {[round(x) for x in ram_list]} MB"
          f" | load {sum(load_list)/len(load_list):.1f}s")
