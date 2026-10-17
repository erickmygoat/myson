"""Central config for myson.

The mind stone (which open model answers) is swappable here; the brain (this repo)
never changes when you swap it. Everything is local — no cloud, no API keys.
"""
import os

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")

# --- the mind stone ------------------------------------------------------------------
# The active chat model. Swap to your own fine-tune/merge later (e.g. "myson").
MODEL = os.environ.get("MYSON_MODEL", "qwen3.5:9b")

# Named roster of models that FIT this box (8 GB VRAM + 32 GB RAM). Only models that
# actually run locally belong here — dense <=~12B at Q4, or a ~20B MoE with CPU offload.
# Datacenter-class flagships (GLM 5.3, full DeepSeek/Qwen-100B+) are intentionally absent;
# they do not start on this hardware. The router (Phase C) may pick a role-fit model from
# this roster; `MODEL` is the default when no role maps.
MODELS = {
    "default": os.environ.get("MYSON_MODEL", "qwen3.5:9b"),  # all-rounder
    "coder":   os.environ.get("MYSON_MODEL_CODER", "qwen3.5:9b"),
    "big":     os.environ.get("MYSON_MODEL_BIG", "gemma4:12b"),   # more params, quantized
}

# Embedding model — powers the recursion verifier's similarity signal and memory's
# hybrid retrieval. Small and cheap; degrade gracefully to text-cosine if absent.
EMBED_MODEL = os.environ.get("MYSON_EMBED_MODEL", "nomic-embed-text")

# Context window. These models advertise huge contexts (256K), but at 8 GB VRAM the KV
# cache — not the weights — is what OOMs you. Keep this modest and measure under load.
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
