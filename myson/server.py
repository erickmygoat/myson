"""OpenAI-compatible server for myson.

Any client — our own workspace UI, the OpenAI SDK, or another app — talks to
`/v1/chat/completions`. Internally the request flows through the brain
(memory -> persona -> router -> chat|tools|recursion -> memory) via `Brain`.

Bound to 127.0.0.1 on purpose (localhost only) — NOT 0.0.0.0. The old repo's
main.py exposed the agent to the whole LAN; we don't. No auth token, no upload
endpoint, no tunnel — a personal server on your own machine.
"""
import json
import os
import threading
import time
import uuid
from typing import Optional, List

from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel

from .core.llm import MindStoneMissing
from .brain import Brain
from .reasoning.recursive import reason_trace
from .config import MODEL, MODELS, MEMORY_DIR

app = FastAPI(title="myson", version="0.2")
_brain: Optional[Brain] = None
_seal = None
# One lock guards both lazy builds. Brain() construction loads the full model (via the
# embed probe), so without this the UI's concurrent first requests (/health, /myson/memory,
# /myson/seal all fire on page load) would each build a Brain and load the model two or
# three times at once -> OOM on 8 GB. Double-checked locking keeps the steady state lock-free.
_init_lock = threading.Lock()


def get_brain() -> Brain:
    """Lazily build the brain once (thread-safe; model loads on first build)."""
    global _brain
    if _brain is None:
        with _init_lock:
            if _brain is None:
                _brain = Brain()
    return _brain


def get_seal():
    """SEALEngine over the same memory dir the CLI uses (built lazily, thread-safe)."""
    global _seal
    if _seal is None:
        brain = get_brain()                 # build/reuse the single brain first
        with _init_lock:
            if _seal is None:
                from .learning.seal_safe import SEALEngine
                _seal = SEALEngine(
                    memory_path=os.path.join(MEMORY_DIR, "learned_rules.json"),
                    review_dir=os.path.join(MEMORY_DIR, "review_queue"),
                    snapshot_dir=os.path.join(MEMORY_DIR, "snapshots"),
                    llm=brain.llm,
                )
    return _seal


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    temperature: float = 0.7
    stream: bool = False
    force: Optional[str] = None   # myson extension: "chat" | "tools" | "recursion"


def _completion_envelope(content: str, model: str, extra: dict | None = None) -> dict:
    env = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0,
                     "message": {"role": "assistant", "content": content},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }
    if extra:
        env["myson"] = extra          # non-standard; standard clients ignore it
    return env


def _sse_chunk(delta: str, model: str, cid: str, finish: str | None = None) -> str:
    obj = {
        "id": cid, "object": "chat.completion.chunk", "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0,
                     "delta": ({"content": delta} if delta else {}),
                     "finish_reason": finish}],
    }
    return f"data: {json.dumps(obj)}\n\n"


@app.get("/v1/models")
def list_models():
    seen, data = set(), []
    for role, name in {"default": MODEL, **MODELS}.items():
        if name not in seen:
            seen.add(name)
            data.append({"id": name, "object": "model", "owned_by": "myson"})
    return {"object": "list", "data": data}


@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest):
    brain = get_brain()
    messages = [{"role": m.role, "content": m.content} for m in req.messages]

    if req.stream:
        cid = f"chatcmpl-{uuid.uuid4().hex[:24]}"

        def gen():
            try:
                for piece in brain.respond_stream(messages, temperature=req.temperature,
                                                   force=req.force):
                    yield _sse_chunk(piece, req.model or MODEL, cid)
            except MindStoneMissing as e:
                yield _sse_chunk(f"[no mind stone] {e}", req.model or MODEL, cid)
            yield _sse_chunk("", req.model or MODEL, cid, finish="stop")
            yield "data: [DONE]\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream")

    try:
        res = brain.respond(messages, temperature=req.temperature, force=req.force)
        extra = {"mode": res.mode, "model": res.model, "route_reason": res.route_reason,
                 "recalled": len(res.recalled)}
        return _completion_envelope(res.answer, req.model or res.model or MODEL, extra)
    except MindStoneMissing as e:
        return _completion_envelope(f"[no mind stone] {e}", req.model or MODEL)


@app.get("/health")
def health():
    brain = get_brain()
    roster = list(dict.fromkeys([MODEL, *MODELS.values()]))   # order-preserving dedup
    return {"ok": True, "model": MODEL, "roster": roster,
            "mind_stone_reachable": brain.llm.available(),
            "embeddings": brain.embed_fn is not None}


# ======================================================================================
# myson workspace API — the panels the UI needs (scratchpad / memory / SEAL queue).
# These are myson extensions, not part of the OpenAI surface.
# ======================================================================================

class ThinkRequest(BaseModel):
    messages: List[ChatMessage]
    max_steps: Optional[int] = None


def _trace_to_dict(t) -> dict:
    return {
        "answer": t.answer,
        "passes": t.passes,
        "halt_reason": t.halt_reason,
        "n_backtracks": t.n_backtracks,
        "llm_calls": t.llm_calls,
        "scratchpad": list(t.scratchpad),
        "steps": [{"pass_idx": s.pass_idx, "kind": s.kind, "best_score": s.best_score,
                   "scores": list(s.scores), "chosen": s.chosen} for s in t.steps],
        "votes": t.votes,
    }


@app.post("/myson/think")
def think(req: ThinkRequest):
    """Run the recursive loop and return the FULL trace (candidates, verifier scores,
    backtracks, halt reason) so the UI scratchpad can show myson thinking."""
    brain = get_brain()
    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    opts = {}
    if req.max_steps:
        opts["max_steps"] = req.max_steps
    try:
        t = reason_trace(messages, llm=brain.llm, embed=brain.embed_fn,
                         system=None, **opts)
        return _trace_to_dict(t)
    except MindStoneMissing as e:
        return JSONResponse({"error": str(e)}, status_code=503)


@app.get("/myson/memory")
def memory(q: Optional[str] = None, k: int = 10):
    brain = get_brain()
    try:
        if q:
            return {"query": q, "recalled": brain.mem.recall(q, k=k)}
        return {"recent": brain.mem.recent(k)}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/myson/seal")
def seal_list():
    try:
        return {"pending": get_seal().list_pending_reviews(),
                "versions": get_seal().list_versions()}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/myson/seal/{review_id}/{action}")
def seal_action(review_id: str, action: str):
    eng = get_seal()
    if action == "approve":
        return eng.approve_review(review_id, approver="workspace")
    if action == "reject":
        return eng.reject_review(review_id, note="rejected in workspace")
    if action == "apply":
        return eng.apply_approved(review_id)
    return JSONResponse({"error": f"unknown action {action}"}, status_code=400)


_UI_PATH = os.path.join(os.path.dirname(__file__), "ui", "index.html")


@app.get("/", response_class=HTMLResponse)
def workspace():
    try:
        with open(_UI_PATH, encoding="utf-8") as f:
            return HTMLResponse(f.read())
    except FileNotFoundError:
        return HTMLResponse("<h1>myson</h1><p>UI not found. API is at /v1/chat/completions.</p>")


def main(host: str = "127.0.0.1", port: int = 8080):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
