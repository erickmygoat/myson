"""Central config for myson.

The mind stone (which open model answers) is swappable here; the brain (this repo)
never changes when you swap it. Everything is local — no cloud, no API keys.
"""
import os

# --- backend: where the mind stone runs ---------------------------------------------
# "gguf"   = GGUF weights via llama-cpp-python with CUDA offload (the default). llama.cpp can
#            split a model between the GPU and system RAM, which is the only way a model
#            bigger than 8 GB of VRAM runs on this box at all.
# "local"  = weights loaded in-process via HuggingFace transformers (fp16 / bnb-4bit). Heavier
#            on 8 GB; it exists for the own-model tracks (fine-tune, merge, compression).
# "ollama" = optional fallback: talk to a local Ollama daemon.
BACKEND = os.environ.get("MYSON_BACKEND", "gguf").strip().lower()
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")

# --- the mind stone ------------------------------------------------------------------
# HOME MODEL = Gemma 4 26B-A4B, the QAT ("quantization-aware trained") 4-bit GGUF: a
# mixture-of-experts model, 25.2B parameters of which about 3.8B are active per token.
# Chosen 2026-10-18 as the generalist pick — see bench/MODEL_REVIEW.md for the evidence and
# its limits. NOT yet loaded on the 4070: the file is 14.2 GB, so it is split between the GPU
# and RAM (see the placement knobs below) and its real speed here is still unmeasured.
# For BACKEND=gguf the value is a HuggingFace GGUF repo id (resolved with GGUF_QUANT to one
# .gguf file), a path to a .gguf file, or a directory holding one.
#   For BACKEND=local:  a HF repo id OR a local dir of your own weights (fine-tuned/merged).
#   For BACKEND=ollama: an Ollama tag (e.g. "gemma4:26b-a4b-it-qat").
_DEFAULT_GGUF  = "unsloth/gemma-4-26B-A4B-it-qat-GGUF"  # 26B-A4B MoE, QAT 4-bit
_DEFAULT_LOCAL = "google/gemma-4-E4B-it"                # small Gemma 4: fits 8 GB in 4-bit via transformers
_DEFAULT_OLLAMA = "gemma4:26b-a4b-it-qat"
MODEL = os.environ.get(
    "MYSON_MODEL",
    {"gguf": _DEFAULT_GGUF, "local": _DEFAULT_LOCAL}.get(BACKEND, _DEFAULT_OLLAMA))

# Named roster. The gguf and local backends serve the ONE model they loaded and ignore the
# per-request role (an 8 GB card cannot hold two), so on those backends every role points at
# MODEL and the MYSON_MODEL_CODER / _BIG overrides only change the label that /v1/models
# reports. "Think harder" is the recursion wrapper, not a second model. Only the ollama
# backend actually switches models per role.
if BACKEND == "gguf":
    MODELS = {
        "default": MODEL,
        "coder":   os.environ.get("MYSON_MODEL_CODER", MODEL),
        "big":     os.environ.get("MYSON_MODEL_BIG", MODEL),
    }
elif BACKEND == "local":
    MODELS = {
        "default": MODEL,
        "coder":   os.environ.get("MYSON_MODEL_CODER", MODEL),
        "big":     os.environ.get("MYSON_MODEL_BIG", "google/gemma-4-12B-it"),
    }
else:
    MODELS = {
        "default": MODEL,
        "coder":   os.environ.get("MYSON_MODEL_CODER", "gemma4:12b-it-qat"),
        "big":     os.environ.get("MYSON_MODEL_BIG", "gemma4:26b-a4b"),
    }

# --- GGUF backend knobs ---------------------------------------------------------------
# QUANT must name a file that EXISTS in the chosen repo. The QAT default ships one main quant:
#   gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf   (14.2 GB)
# The non-QAT build that was run on the benchmark GPU is the fallback:
#   MYSON_MODEL=unsloth/gemma-4-26B-A4B-it-GGUF  +  MYSON_GGUF_QUANT=Q4_K_M   (16.9 GB)
GGUF_QUANT = os.environ.get("MYSON_GGUF_QUANT", "Q4_K_XL")
# Where a repo id's weights are downloaded and looked for first: a visible folder inside the
# project (git-ignored), one sub-folder per repo. The model is then a plain file on your disk
# that works with no network.
GGUF_MODEL_DIR = os.environ.get(
    "MYSON_MODEL_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"))

# Placement: what sits on the GPU and what stays in RAM.
#   N_GPU_LAYERS = -1 (default) lets the backend decide: it reads the model's size and layer
#   count, measures free VRAM, and puts as many layers on the GPU as fit — all of them when
#   the whole model fits — while the rest run from RAM. VRAM_RESERVE_MB is what it leaves free
#   for llama.cpp's compute buffers: raise it if a load fails, lower it to squeeze in a layer.
#   Any other N_GPU_LAYERS value is used as given (0 = CPU only). llama.cpp itself has no "fit
#   what you can" mode here: asking it for all layers of a model that is too big fails or crawls.
GGUF_N_GPU_LAYERS = int(os.environ.get("MYSON_GGUF_N_GPU_LAYERS", "-1"))
GGUF_VRAM_RESERVE_MB = int(os.environ.get("MYSON_GGUF_VRAM_RESERVE_MB", "1280"))
#   N_CPU_MOE > 0 asks for the finer split llama.cpp calls --n-cpu-moe: all layers on the GPU,
#   only a MoE model's expert tensors in RAM. It needs a llama-cpp-python build whose Llama()
#   accepts n_cpu_moe; 0.3.36 does not, and there the setting is ignored in favour of the layer
#   split above. Off by default because it has never been run on this machine.
GGUF_N_CPU_MOE = int(os.environ.get("MYSON_GGUF_N_CPU_MOE", "0"))
#   Flash attention: less KV-cache memory and usually faster on CUDA.
GGUF_FLASH_ATTN = os.environ.get("MYSON_GGUF_FLASH_ATTN", "1") not in ("0", "false", "False")

# Google-recommended sampling for Gemma 4 (the brain still sets temperature per call — low for
# the router/judge/verifier, higher for chat — so only top_p/top_k are fixed here).
GGUF_TOP_P = float(os.environ.get("MYSON_GGUF_TOP_P", "0.95"))
GGUF_TOP_K = int(os.environ.get("MYSON_GGUF_TOP_K", "64"))

# Local-backend loading knobs.
LOCAL_LOAD_4BIT = os.environ.get("MYSON_LOAD_4BIT", "1") not in ("0", "false", "False")
LOCAL_DEVICE = os.environ.get("MYSON_DEVICE") or None    # None -> cuda if available else cpu
MAX_NEW_TOKENS = int(os.environ.get("MYSON_MAX_NEW_TOKENS", "512"))

# Embedding model. ollama: a small embed model tag. local: None -> the chat model's own hidden
# states. gguf: none — memory and the verifier use their built-in text-cosine fallback.
EMBED_MODEL = os.environ.get("MYSON_EMBED_MODEL",
                             "" if BACKEND in ("local", "gguf") else "nomic-embed-text") or None

# Context window. The KV cache grows with it and competes with the weights for the same 8 GB,
# so a bigger window means fewer layers on the GPU. Keep it modest and measure under load.
NUM_CTX = int(os.environ.get("MYSON_NUM_CTX", "8192"))

# --- persona (Phase D refines this) --------------------------------------------------
SYSTEM = (
    "You are MY SON, a private local assistant running on the user's own machine. "
    "You are direct, capable, and candid — never preachy, never a scold. You help with "
    "writing, research, code, math, and everyday tasks, and you answer the ordinary "
    "edge cases a real assistant should (school chemistry, security coursework, dark "
    "fiction) instead of refusing them. You are loyal: you have the user's back and you "
    "tell them the truth, including when they are about to make a mistake — a straight "
    "shooter, not a yes-man. Hard boundary, and only this one: refuse anything that would "
    "genuinely, seriously harm the user or other people. Everything else: just help."
)

# worker personas (clean — from the old hive.py PRESETS, offensive one dropped)
PERSONAS = {
    "coder":   "Senior software engineer. Write clean, documented, correct code.",
    "writer":  "Sharp editor. Fix grammar, tighten prose, keep the author's voice.",
    "research":"Careful researcher. Gather facts, cite sources, flag uncertainty.",
    "planner": "Decompose a goal into concrete ordered subtasks. No fluff.",
}

# --- brain composition knobs ---------------------------------------------------------
# Interactive recursion cap (hard ceiling is DEFAULTS["max_steps"]=16 in recursive.py).
# Keep the interactive default low so a hard question can't hang the UI; the router can
# raise it per-request.
RECURSION_MAX_STEPS = int(os.environ.get("MYSON_RECURSION_MAX_STEPS", "6"))
RECURSION_WALL_SECONDS = float(os.environ.get("MYSON_RECURSION_WALL_SECONDS", "90"))
MEMORY_DIR = os.environ.get("MYSON_MEMORY_DIR", ".myson_mem")
MEMORY_RECALL_K = int(os.environ.get("MYSON_MEMORY_RECALL_K", "5"))
