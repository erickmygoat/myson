"""Backend factory — build the mind-stone client the brain talks to.

The brain, server, CLI and UI never construct a model client directly; they call
`make_llm()`, which returns whichever backend config.BACKEND selects. Every backend exposes
the same interface — chat / ask / chat_stream / embed / available, plus num_ctx — so nothing
downstream changes when you swap the mind stone (or your own weights) in.

  * "gguf"   (default) -> GGUFLLM: GGUF weights via llama-cpp-python with CUDA offload. The
                          weights are a file on your disk (a .gguf path, or a HF GGUF repo
                          downloaded once into models/). A model that fits in VRAM runs fully
                          on the GPU; a bigger one is split between the GPU and system RAM.
  * "local"  -> LocalLLM: weights in-process via transformers (fp16/bnb-4bit). Heavier on
                          8 GB; use it for fine-tune / merge / compression work.
  * "ollama" -> LLMClient: a local Ollama daemon.
"""
from __future__ import annotations

from .. import config

BACKENDS = ("gguf", "local", "ollama")


def make_llm(backend: str | None = None, model: str | None = None):
    backend = (backend or config.BACKEND).strip().lower()
    model = model or config.MODEL

    if backend == "ollama":
        from .llm import LLMClient
        return LLMClient(model=model)

    if backend == "local":
        from .local_llm import LocalLLM
        return LocalLLM(
            model=model,
            device=config.LOCAL_DEVICE,
            load_4bit=config.LOCAL_LOAD_4BIT,
            max_new_tokens=config.MAX_NEW_TOKENS,
            num_ctx=config.NUM_CTX,
        )

    if backend == "gguf":
        from .gguf_llm import GGUFLLM
        return GGUFLLM(model=model, n_ctx=config.NUM_CTX, max_new_tokens=config.MAX_NEW_TOKENS)

    # A typo used to fall through to the default backend and fail later with a confusing
    # "could not resolve model" error; say what is wrong instead.
    raise ValueError(f"unknown backend {backend!r} (MYSON_BACKEND): expected one of {BACKENDS}")
