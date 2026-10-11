"""OpenAI-compatible server for myson.

Any client — our own workspace UI, the OpenAI SDK, or another app — talks to
`/v1/chat/completions`. Internally we call Ollama (the Qwen mind stone) and, as the
build progresses, route through the brain layers (recursion -> tools -> memory ->
SEAL) via the `brain()` hook below.

Bound to 127.0.0.1 on purpose (localhost only) — NOT 0.0.0.0. The old repo's
main.py exposed the agent to the whole LAN; we don't.
"""
import time
import uuid
from typing import Optional, List

from fastapi import FastAPI
from pydantic import BaseModel

from .core.llm import LLMClient, MindStoneMissing
from .config import MODEL, SYSTEM

app = FastAPI(title="myson", version="0.1")
llm = LLMClient()


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    temperature: float = 0.7
    stream: bool = False  # streaming not implemented yet; returns a full response


def brain(messages: List[ChatMessage], temperature: float) -> str:
    """The single seam where the brain plugs in.

    Phases 2-4 insert recursion / tools / memory / SEAL here. For now it passes
    the conversation straight to the LLM. A client-supplied system message wins;
    otherwise we use the myson persona (config.SYSTEM).
    """
    sys_msg = next((m.content for m in messages if m.role == "system"), SYSTEM)
    convo = [{"role": m.role, "content": m.content} for m in messages if m.role != "system"]
    return llm.chat(convo, system=sys_msg, temperature=temperature).get("content", "")


@app.get("/v1/models")
def list_models():
    return {"object": "list",
            "data": [{"id": MODEL, "object": "model", "owned_by": "myson"}]}


@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest):
    try:
        content = brain(req.messages, req.temperature)
        finish = "stop"
    except MindStoneMissing as e:
        content = f"[no mind stone] {e}"
        finish = "stop"
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model or MODEL,
        "choices": [{"index": 0,
                     "message": {"role": "assistant", "content": content},
                     "finish_reason": finish}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


@app.get("/health")
def health():
    return {"ok": True, "model": MODEL, "mind_stone_reachable": llm.available()}


def main(host: str = "127.0.0.1", port: int = 8080):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
