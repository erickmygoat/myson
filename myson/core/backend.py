"""Backend factory — build the mind-stone client the brain talks to.

The brain, server, CLI and UI never construct a model client directly; they call
`make_llm()`, which returns whichever backend config.BACKEND selects. Both backends expose
the same interface: chat / ask / chat_stream / embed / available, so nothing downstream
changes when you swap the mind stone (or your own weights) in.

  * "local"  (default) -> LocalLLM: weights loaded in-process (transformers). Editable,
                          fine-tunable, Heretic-able, compressible. This is the point.
  * "ollama" (fallback) -> LLMClient: a local Ollama daemon (black box; cannot be edited).
"""
from __future__ import annotations

from .. import config


def make_llm(backend: str | None = None, model: str | None = None):
    backend = (backend or config.BACKEND).lower()
    model = model or config.MODEL

    if backend == "ollama":
        from .llm import LLMClient
        return LLMClient(model=model)

    # default: local in-process weights
    from .local_llm import LocalLLM
    return LocalLLM(
        model=model,
        device=config.LOCAL_DEVICE,
        load_4bit=config.LOCAL_LOAD_4BIT,
        max_new_tokens=config.MAX_NEW_TOKENS,
        embed_model=config.EMBED_MODEL,
    )
