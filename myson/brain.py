"""The brain — one seam where myson's layers compose into a single response.

Flow per request:
    recall memory  ->  persona + recalled context  ->  router  ->
        (plain chat | tool loop | recursive reasoning)  ->  save memory

Each layer is a module with its own tested API; this file only wires them. Ollama is
reached through the injected LLMClient, so the whole thing is unit-testable with a mock
and needs no running model.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .core.llm import LLMClient, MindStoneMissing
from .memory.memory import Memory
from . import config
from . import router as _router
from .reasoning.recursive import reason_trace
from .reasoning.longctx import maybe_reduce
from .tools.agent_tools import run_with_tools, build_default_tools


@dataclass
class BrainResult:
    answer: str
    mode: str = "chat"              # which path ran
    model: str = ""                 # which mind stone answered
    route_reason: str = ""
    recalled: list = field(default_factory=list)
    trace: Optional[object] = None  # a recursive.RunTrace when mode == "recursion"


def _make_embed_fn(llm: LLMClient):
    """Pick ONE embedder for the whole session so the memory store never mixes real
    embeddings with the stdlib fallback (which would corrupt cosine). Returns an embed
    callable if Ollama's embed model answers a probe, else None (Memory/Similarity then
    use their own bag-of-words fallback)."""
    try:
        v = llm.embed("probe")
        if v:
            return llm.embed
    except Exception:
        pass
    return None


class Brain:
    def __init__(self, llm: Optional[LLMClient] = None, memory: Optional[Memory] = None):
        self.llm = llm or LLMClient()
        self.embed_fn = _make_embed_fn(self.llm)
        self.mem = memory or Memory(directory=config.MEMORY_DIR, embed_fn=self.embed_fn)
        self.tools = build_default_tools()

    # -- helpers --------------------------------------------------------------------

    def _split(self, messages):
        """(system_override, convo_without_system, last_user_text)."""
        msgs = [{"role": m["role"], "content": m["content"]} if isinstance(m, dict)
                else {"role": m.role, "content": m.content} for m in messages]
        sys_override = next((m["content"] for m in msgs if m["role"] == "system"), None)
        convo = [m for m in msgs if m["role"] != "system"]
        last_user = next((m["content"] for m in reversed(convo) if m["role"] == "user"), "")
        return sys_override, convo, last_user

    def _system_with_memory(self, sys_override: str | None, query: str):
        base = sys_override or config.SYSTEM
        recalled = []
        try:
            recalled = self.mem.recall(query, k=config.MEMORY_RECALL_K) if query else []
        except Exception:
            recalled = []
        if recalled:
            base = base + "\n\nRELEVANT MEMORY (things the user told you before):\n" + \
                "\n".join(f"- {r}" for r in recalled)
        return base, recalled

    # -- the composed response ------------------------------------------------------

    def respond(self, messages, temperature: float = 0.7, force: str | None = None,
                remember: bool = True) -> BrainResult:
        sys_override, convo, last_user = self._split(messages)
        system, recalled = self._system_with_memory(sys_override, last_user)

        # oversized input -> map-reduce it down to fit the context window first
        convo = maybe_reduce(convo, self.llm, num_ctx=self.llm.num_ctx)

        r = _router.route(convo, llm=self.llm, force=force)
        model = config.MODELS.get(r.model_role, config.MODEL)

        if r.mode == "recursion":
            trace = reason_trace(
                convo, llm=self.llm, system=system, embed=self.embed_fn,
                max_steps=config.RECURSION_MAX_STEPS,
            )
            answer, extra_trace = trace.answer, trace
        elif r.mode == "tools":
            answer = run_with_tools(convo, self.llm, tools=self.tools, system=system,
                                    temperature=min(temperature, 0.3))
            extra_trace = None
        else:  # chat
            answer = self.llm.chat(convo, system=system, temperature=temperature,
                                   model=model).get("content", "")
            extra_trace = None

        if remember and last_user:
            try:
                self.mem.save("user", last_user)
                if answer:
                    self.mem.save("assistant", answer)
            except Exception:
                pass

        return BrainResult(answer=answer, mode=r.mode, model=model,
                           route_reason=r.reason, recalled=recalled, trace=extra_trace)

    def respond_stream(self, messages, temperature: float = 0.7, force: str | None = None,
                       remember: bool = True):
        """Yield answer text in pieces. Only the plain-chat path streams natively; the
        tool and recursion paths compute fully (they need the whole answer to verify/act)
        then yield it once. Either way the caller gets a stream of str deltas."""
        sys_override, convo, last_user = self._split(messages)
        system, _ = self._system_with_memory(sys_override, last_user)
        convo = maybe_reduce(convo, self.llm, num_ctx=self.llm.num_ctx)
        r = _router.route(convo, llm=self.llm, force=force)
        model = config.MODELS.get(r.model_role, config.MODEL)

        collected = []
        if r.mode == "chat":
            for piece in self.llm.chat_stream(convo, system=system, temperature=temperature,
                                              model=model):
                collected.append(piece)
                yield piece
        else:
            res = self.respond(messages, temperature=temperature, force=r.mode, remember=False)
            collected.append(res.answer)
            yield res.answer

        answer = "".join(collected)
        if remember and last_user:
            try:
                self.mem.save("user", last_user)
                if answer:
                    self.mem.save("assistant", answer)
            except Exception:
                pass
