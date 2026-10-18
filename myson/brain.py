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

from .core.llm import MindStoneMissing
from .core.backend import make_llm
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


def _make_embed_fn(llm):
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
    def __init__(self, llm=None, memory: Optional[Memory] = None):
        self.llm = llm or make_llm()
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

    def _clean_convo(self, convo, route):
        """Strip a steering prefix (/think, /fast, /tools) from the last user message so it
        never reaches the model. Returns (convo_for_model, clean_last_user_text)."""
        clean = route.cleaned_text
        if not clean:                       # forced route or empty -> leave convo untouched
            last = next((m["content"] for m in reversed(convo) if m["role"] == "user"), "")
            return convo, last
        out = [dict(m) for m in convo]
        for i in range(len(out) - 1, -1, -1):
            if out[i]["role"] == "user":
                out[i]["content"] = clean
                break
        return out, clean

    def _dispatch(self, convo, system, mode, model, temperature):
        """Run one path and return (answer, trace). Shared by respond + respond_stream."""
        if mode == "recursion":
            trace = reason_trace(convo, llm=self.llm, system=system, embed=self.embed_fn,
                                 max_steps=config.RECURSION_MAX_STEPS)
            return trace.answer, trace
        if mode == "tools":
            return run_with_tools(convo, self.llm, tools=self.tools, system=system,
                                  temperature=min(temperature, 0.3)), None
        return self.llm.chat(convo, system=system, temperature=temperature,
                             model=model).get("content", ""), None

    def _served_model(self, requested: str) -> str:
        """The model that ACTUALLY answered. The Ollama backend loads the requested roster
        model on demand, so `requested` is truthful there. The local backend serves one
        loaded model (an 8 GB box can't hold two), so per-request roster switching is a
        no-op and we report the loaded model id instead of the requested one."""
        return getattr(self.llm, "model_id", None) or requested

    def _remember(self, remember, user_text, answer):
        if remember and user_text:
            try:
                self.mem.save("user", user_text)
                if answer:
                    self.mem.save("assistant", answer)
            except Exception:
                pass

    def respond(self, messages, temperature: float = 0.7, force: str | None = None,
                remember: bool = True) -> BrainResult:
        sys_override, convo, last_user = self._split(messages)
        system, recalled = self._system_with_memory(sys_override, last_user)
        convo = maybe_reduce(convo, self.llm, num_ctx=self.llm.num_ctx)
        r = _router.route(convo, llm=self.llm, force=force)
        convo, user_text = self._clean_convo(convo, r)
        model = config.MODELS.get(r.model_role, config.MODEL)

        answer, trace = self._dispatch(convo, system, r.mode, model, temperature)
        self._remember(remember, user_text, answer)
        return BrainResult(answer=answer, mode=r.mode, model=self._served_model(model),
                           route_reason=r.reason, recalled=recalled, trace=trace)

    def respond_stream(self, messages, temperature: float = 0.7, force: str | None = None,
                       remember: bool = True):
        """Yield answer text in pieces. Only the plain-chat path streams natively; the tool
        and recursion paths compute fully (they need the whole answer to verify/act) then
        yield it once. Either way the caller gets a stream of str deltas."""
        sys_override, convo, last_user = self._split(messages)
        system, _ = self._system_with_memory(sys_override, last_user)
        convo = maybe_reduce(convo, self.llm, num_ctx=self.llm.num_ctx)
        r = _router.route(convo, llm=self.llm, force=force)
        convo, user_text = self._clean_convo(convo, r)
        model = config.MODELS.get(r.model_role, config.MODEL)

        collected = []
        if r.mode == "chat":
            for piece in self.llm.chat_stream(convo, system=system, temperature=temperature,
                                              model=model):
                collected.append(piece)
                yield piece
        else:
            answer, _ = self._dispatch(convo, system, r.mode, model, temperature)
            collected.append(answer)
            yield answer

        self._remember(remember, user_text, "".join(collected))
