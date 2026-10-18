"""Central config for myson.

The mind stone (which open model answers) is swappable here; the brain (this repo)
never changes when you swap it. Everything is local — no cloud, no API keys.
"""
import os

# --- backend: where the mind stone runs ---------------------------------------------
# "local"  = weights loaded DIRECTLY in-process via HuggingFace transformers. This is the
#            default and the point: the model is yours to edit, fine-tune, merge, Heretic,
#            or compress. Needs torch + transformers (+ accelerate/bitsandbytes for 4-bit).
# "ollama" = optional fallback: talk to a local Ollama daemon (a black box you cannot edit).
BACKEND = os.environ.get("MYSON_BACKEND", "local")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")

# --- the mind stone ------------------------------------------------------------------
# For BACKEND=local: a HuggingFace repo id OR a local directory of YOUR OWN weights
#   (fine-tuned / merged / abliterated). Point MYSON_MODEL at your model when you have one.
# For BACKEND=ollama: an Ollama tag (e.g. "qwen3.5:9b-q4_K_M").
_DEFAULT_LOCAL = "Qwen/Qwen2.5-7B-Instruct"     # real, widely-available, fits 8GB in 4-bit
_DEFAULT_OLLAMA = "qwen3.5:9b"
MODEL = os.environ.get("MYSON_MODEL",
                       _DEFAULT_LOCAL if BACKEND == "local" else _DEFAULT_OLLAMA)

# Named roster of models that FIT this box (8 GB VRAM + 32 GB RAM) — dense <=~12B in 4-bit.
# Datacenter flagships (GLM 5.3, full DeepSeek/Qwen-100B+) are absent; they do not start
# on this hardware. The router may pick a role-fit model; MODEL is the default.
if BACKEND == "local":
    MODELS = {
        "default": MODEL,
        "coder":   os.environ.get("MYSON_MODEL_CODER", "Qwen/Qwen2.5-Coder-7B-Instruct"),
        "big":     os.environ.get("MYSON_MODEL_BIG", "google/gemma-2-9b-it"),
    }
else:
    MODELS = {
        "default": MODEL,
        "coder":   os.environ.get("MYSON_MODEL_CODER", "qwen3.5:9b"),
        "big":     os.environ.get("MYSON_MODEL_BIG", "gemma4:12b"),
    }

# Local-backend loading knobs.
LOCAL_LOAD_4BIT = os.environ.get("MYSON_LOAD_4BIT", "1") not in ("0", "false", "False")
LOCAL_DEVICE = os.environ.get("MYSON_DEVICE") or None    # None -> cuda if available else cpu
MAX_NEW_TOKENS = int(os.environ.get("MYSON_MAX_NEW_TOKENS", "512"))

# Embedding model. local: None -> embed with the chat model's own hidden states (no extra
# download). ollama: a small embed model tag. Degrades to text-cosine if unavailable.
EMBED_MODEL = os.environ.get("MYSON_EMBED_MODEL",
                             "" if BACKEND == "local" else "nomic-embed-text") or None

# Context window. Models advertise huge contexts (256K), but at 8 GB VRAM the KV cache —
# not the weights — is what OOMs you. Keep this modest and measure under load.
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
