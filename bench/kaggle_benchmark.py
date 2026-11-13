# ====================================================================================
# myson model benchmark v2 — HARD questions, more challengers.  PASTE INTO ONE CELL.
#
#   *** DO THIS FIRST ***  (otherwise a broken CPU build of llama-cpp stays loaded):
#   Colab:  Runtime -> Restart runtime,  then run this cell.   (A100 / any GPU runtime.)
#   Kaggle: Run -> Restart & clear cell outputs; Accelerator = GPU.
#
# It builds llama-cpp-python WITH CUDA (so it uses the GPU — proven per model below),
# then for each candidate runs a HARD set — benign hard code, plus contest math if you
# supply it — prints each full answer + tok/s + load, and deletes the file before the next.
# Pick the winner by ANSWER QUALITY (speed here is the cloud GPU's, NOT your 4070).
#
# The math half was run with problems from the 2026 CEMC contests (Cayley / Fermat /
# Euclid). Those belong to the University of Waterloo and are not redistributed here: see
# the note above _load_math_prompts() for how to supply your own list, and grade against
# the official solutions at cemc.uwaterloo.ca.
# ====================================================================================
import os, sys, subprocess, time, gc

# --- 1. GPU present? ----------------------------------------------------------------
g = subprocess.run("nvidia-smi -L", shell=True, capture_output=True, text=True)
print(g.stdout.strip() or ">> NO GPU! Set a GPU runtime (Colab: Runtime>Change runtime type).")

# --- 2. llama-cpp-python WITH CUDA (compiles ~5 min the FIRST time; cached after) ----
def _has_cuda_llama():
    try:
        import llama_cpp
        return bool(llama_cpp.llama_supports_gpu_offload())
    except Exception:
        return False

if "llama_cpp" in sys.modules and not _has_cuda_llama():
    print("\n*** A CPU build is already loaded. Restart the runtime and run this cell FIRST. ***\n")
elif _has_cuda_llama():
    print("llama-cpp-python (CUDA) already present ✓")
else:
    print(">> building llama-cpp-python with CUDA (~5 min, one-time per VM)…")
    subprocess.run("pip uninstall -y llama-cpp-python -q", shell=True)
    os.environ["CMAKE_ARGS"] = "-DGGML_CUDA=on"
    os.environ["FORCE_CMAKE"] = "1"
    subprocess.run("pip install -q llama-cpp-python --no-cache-dir", shell=True)

import llama_cpp
_supported = bool(llama_cpp.llama_supports_gpu_offload())
print("GPU offload supported:", _supported)
if not _supported:
    raise SystemExit(
        "\n*** llama-cpp-python is CPU-ONLY. It will NOT use the GPU. ***\n"
        "FIX: Runtime > Restart runtime, then run this cell again (it rebuilds with CUDA).\n")
from llama_cpp import Llama
from huggingface_hub import hf_hub_download, list_repo_files


def gpu_mem_used_mb():
    r = subprocess.run("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits",
                       shell=True, capture_output=True, text=True)
    try:
        return int(r.stdout.strip().splitlines()[0])
    except Exception:
        return -1

# --- 3. candidates — the two Gemma4 incumbents + challengers aimed at beating them ---
#   [laptop native]  fits on an 8 GB 4070 fully on-GPU
#   [laptop offload] runs on the 4070 with layers/experts spilled to CPU RAM (slower)
QUANT = "Q4_K_M"
CANDIDATES = [
    ("Gemma4-12B-it     [laptop native]",       "unsloth/gemma-4-12b-it-GGUF"),           # incumbent
    ("Gemma4-26B-A4B    [laptop offload, MoE]",  "unsloth/gemma-4-26B-A4B-it-GGUF"),       # incumbent (quality king)
    ("DeepSeek-R1-14B   [laptop offload, reasoner]", "unsloth/DeepSeek-R1-Distill-Qwen-14B-GGUF"),  # math/reasoning specialist
    ("Phi-4-14B         [laptop offload, reasoner]", "unsloth/phi-4-GGUF"),                # strong math/reasoning for size
    # ("Mistral-Small-3.2-24B [laptop offload, heavy]", "unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF"),
    # ("Qwen3-14B        [laptop offload, rematch]", "unsloth/Qwen3-14B-GGUF"),
]

# Per-prompt (kind, prompt, max_tokens). Reasoning models THINK — math gets a big budget
# so they don't get cut off mid-reasoning the way a 256-cap killed Qwen last run.
C = 1024   # code budget
M = 2048   # math/reasoning budget (reasoners need room)

PROMPTS = [
    # ---- benign hard CODE: higher-level code + multi-language + self-correction -------
    ("code:quine",
     "Write a program in Python that prints its own exact source code to standard output "
     "(a true quine) — no reading its own file, no cheating. Then write a second quine "
     "in Rust. For each, add one sentence explaining why it reproduces itself.", C),
    ("code:bounded-recursion",
     "Write a Python function build_tree(depth) that recursively prints an indented tree "
     "and stops itself at depth 2 (root plus two levels, then halts) — a controlled, "
     "self-limiting recursion. Show the printed output for the root call. Then rewrite the "
     "same function in JavaScript.", C),
    ("code:self-correct",
     "This Python function is meant to return the n-th Fibonacci number (fib(0)=0, fib(1)=1, "
     "fib(5)=5) but is subtly wrong — it returns 1 for fib(0) and 8 for fib(5):\n\n"
     "def fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return b\n\n"
     "Diagnose the off-by-one, give the corrected function, and show fib(0), fib(1), fib(5).", C),
]


# Contest math is loaded from a file you provide, never from this repo. Put a JSON list of
# [label, problem text] pairs in  local/math_prompts.json  next to this script (that folder
# is git-ignored), or  math_prompts.json  in the working directory (e.g. uploaded to Colab):
#     [["math:cayley-20", "…problem text…"], ["math:fermat-19", "…"]]
def _load_math_prompts():
    import json
    here = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
    for p in (os.path.join(here, "local", "math_prompts.json"), "math_prompts.json"):
        if os.path.isfile(p):
            with open(p, encoding="utf-8") as fh:
                return [(kind, text, M) for kind, text in json.load(fh)]
    print(">> no math_prompts.json found: running the code prompts only.")
    return []


PROMPTS += _load_math_prompts()


def pick_gguf(repo, quant):
    fs = [f for f in list_repo_files(repo)
          if f.lower().endswith(".gguf") and quant.lower() in f.lower()]
    return sorted(fs)


results = []
for name, repo in CANDIDATES:
    print("\n" + "=" * 78 + f"\n### {name}   ({repo} :: {QUANT})\n" + "=" * 78)
    local = []
    try:
        files = pick_gguf(repo, QUANT)
        if not files:
            allf = [f for f in list_repo_files(repo) if f.lower().endswith(".gguf")]
            print(f"  no {QUANT} file; .gguf in repo: {allf[:12]}")
            results.append((name, f"no {QUANT}", "")); continue
        print("  downloading:", files)
        local = [hf_hub_download(repo, f) for f in files]

        before = gpu_mem_used_mb()
        t0 = time.time()
        llm = Llama(model_path=local[0], n_gpu_layers=-1, n_ctx=8192,
                    n_batch=512, verbose=False)
        load_s = time.time() - t0
        after = gpu_mem_used_mb()
        on_gpu = after - before
        print(f"  loaded in {load_s:.1f}s | GPU memory used by this model: ~{on_gpu} MB "
              + ("<-- ON GPU ✓" if on_gpu > 500 else "<-- ~0: ON CPU! restart+rebuild ✗"))

        toks = dt = 0.0
        for kind, p, mx in PROMPTS:
            t0 = time.time()
            o = llm.create_chat_completion(
                messages=[{"role": "user", "content": p}],
                max_tokens=mx, temperature=0.3)
            el = time.time() - t0
            txt = o["choices"][0]["message"]["content"]
            n = o.get("usage", {}).get("completion_tokens") or max(1, len(txt) // 4)
            toks += n; dt += el
            print(f"\n  --[{kind}]-- {n} tok in {el:.1f}s  ({n/el:.1f} tok/s)")
            print("  " + txt.strip().replace("\n", "\n  "))

        results.append((name, f"{toks/dt:.1f} tok/s" if dt else "n/a", f"load {load_s:.0f}s"))
        del llm; gc.collect()
    except Exception as e:
        print("  ERROR:", repr(e)[:300])
        results.append((name, f"error: {type(e).__name__}", ""))
    finally:
        for f in local:
            try: os.remove(f)
            except Exception: pass

print("\n\n" + "#" * 78 + "\n# SUMMARY (speed = this GPU; pick by ANSWER QUALITY above)\n" + "#" * 78)
for name, stat, extra in results:
    print(f"  {name:34s} {stat:16s} {extra}")
print("\n# Math answer keys: cemc.uwaterloo.ca -> 2026 Cayley / Fermat / Euclid solutions.")
