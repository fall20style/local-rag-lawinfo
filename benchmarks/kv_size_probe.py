"""KV 캐시 버퍼 크기 측정: verbose 로그에서 공식 'KV self size' 값을 캡처."""
import io
import subprocess
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

CODE = r"""
from llama_cpp import Llama
llama = Llama(model_path="models/qwen2.5-7b-instruct-q4_k_m.gguf",
              n_ctx=8192, n_threads=8, n_batch=512,
              flash_attn=True, type_k={K}, type_v={V}, verbose=True)
print("OK", llama.n_ctx())
"""

configs = [("f16/f16", 1, 1), ("K q8_0", 8, 1), ("K+V q8_0", 8, 8)]
for name, k, v in configs:
    src = CODE.format(K=k, V=v)
    p = subprocess.run([sys.executable, "-c", src], capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=600)
    kv = [l for l in p.stderr.splitlines() if "KV self size" in l or "KV buffer size" in l]
    cb = [l for l in p.stderr.splitlines() if "compute buffer" in l]
    print(f"[{name}] (type_k={k}, type_v={v})")
    for l in kv + cb:
        print("   ", l.strip()[20:90])
    if p.returncode != 0:
        print("    FAILED:", p.stderr.splitlines()[-1][:100])
