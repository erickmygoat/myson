"""OpenAI-compatible server for myson.

Any client — our own workspace UI, the OpenAI SDK, or another app — talks to
`/v1/chat/completions`. Internally the request flows through the brain
(memory -> persona -> router -> chat|tools|recursion -> memory) via `Brain`.

Bound to 127.0.0.1 on purpose (localhost only) — NOT 0.0.0.0. The old repo's
main.py exposed the agent to the whole LAN; we don't. No auth token, no upload
endpoint, no tunnel — a personal server on your own machine.
"""
import json
import time
import uuid
from typing import Optional, List

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .core.llm import MindStoneMissing
from .brain import Brain
from .config import MODEL, MODELS

app = FastAPI(title="myson", version="0.2")
_brain: Optional[Brain] = None


def get_brain() -> Brain:
    """Lazily build the brain once (so importing the module never needs Ollama)."""
    global _brain
    if _brain is None:
        _brain = Brain()
    return _brain


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


def main(host: str = "127.0.0.1", port: int = 8080):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
