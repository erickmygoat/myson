"""LLMClient — the socket the mind stone plugs into. Talks to local Ollama.

No cloud, no API keys. If Ollama isn't running or the model isn't pulled, it
says so plainly instead of silently failing.
"""
import json, urllib.request, urllib.error
from ..config import OLLAMA_URL, MODEL, SYSTEM, EMBED_MODEL, NUM_CTX


class MindStoneMissing(RuntimeError):
    pass


class LLMClient:
    def __init__(self, model: str = MODEL, url: str = OLLAMA_URL,
                 embed_model: str = EMBED_MODEL, num_ctx: int = NUM_CTX):
        self.model = model
        self.url = url.rstrip("/")
        self.embed_model = embed_model
        self.num_ctx = num_ctx

    def _post(self, path: str, payload: dict, timeout: int = 300) -> dict:
        req = urllib.request.Request(
            f"{self.url}{path}",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode())
        except urllib.error.URLError as e:
            raise MindStoneMissing(
                f"Ollama unreachable at {self.url} ({e}). Is it running? "
                f"Start Ollama and `ollama pull {self.model}`."
            )

    def _options(self, temperature: float) -> dict:
        return {"temperature": temperature, "num_ctx": self.num_ctx}

    def chat(self, messages: list[dict], system: str = SYSTEM,
             tools: list | None = None, temperature: float = 0.7,
             model: str | None = None) -> dict:
        """One chat turn. messages = [{role, content}]. Returns the assistant msg.

        `model` overrides the default mind stone for this call (roster routing).
        """
        msgs = ([{"role": "system", "content": system}] if system else []) + messages
        payload = {"model": model or self.model, "messages": msgs, "stream": False,
                   "options": self._options(temperature)}
        if tools:
            payload["tools"] = tools
        out = self._post("/api/chat", payload)
        if "error" in out:
            raise MindStoneMissing(out["error"])
        return out.get("message", {"role": "assistant", "content": ""})

    def ask(self, prompt: str, system: str = SYSTEM, **kw) -> str:
        return self.chat([{"role": "user", "content": prompt}], system=system, **kw).get("content", "")

    def chat_stream(self, messages: list[dict], system: str = SYSTEM,
                    temperature: float = 0.7, model: str | None = None):
        """Yield assistant content deltas (str) as Ollama streams them.

        Used by the server's SSE path. Raises MindStoneMissing if the model errors.
        """
        msgs = ([{"role": "system", "content": system}] if system else []) + messages
        payload = {"model": model or self.model, "messages": msgs, "stream": True,
                   "options": self._options(temperature)}
        req = urllib.request.Request(
            f"{self.url}/api/chat",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                for raw in r:
                    line = raw.decode().strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    if "error" in obj:
                        raise MindStoneMissing(obj["error"])
                    piece = (obj.get("message") or {}).get("content", "")
                    if piece:
                        yield piece
                    if obj.get("done"):
                        break
        except urllib.error.URLError as e:
            raise MindStoneMissing(
                f"Ollama unreachable at {self.url} ({e}). Is it running? "
                f"Start Ollama and `ollama pull {self.model}`."
            )

    def embed(self, text: str, model: str | None = None) -> list[float]:
        """Return an embedding vector for `text` via Ollama's /api/embed.

        Raises MindStoneMissing if Ollama is unreachable or the embed model errors, so
        callers (the verifier's Similarity, memory's hybrid retrieval) can catch it and
        fall back to the stdlib text-cosine path instead of crashing.
        """
        out = self._post("/api/embed", {"model": model or self.embed_model, "input": text},
                         timeout=60)
        if "error" in out:
            raise MindStoneMissing(out["error"])
        # /api/embed returns {"embeddings": [[...]]}; older /api/embeddings used "embedding".
        embs = out.get("embeddings")
        if embs:
            return list(embs[0])
        single = out.get("embedding")
        if single:
            return list(single)
        raise MindStoneMissing("embed: no vector returned")

    def available(self) -> bool:
        try:
            urllib.request.urlopen(f"{self.url}/api/tags", timeout=5)
            return True
        except Exception:
            return False
