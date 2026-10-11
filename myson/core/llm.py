"""LLMClient — the socket the mind stone plugs into. Talks to local Ollama.

No cloud, no API keys. If Ollama isn't running or the model isn't pulled, it
says so plainly instead of silently failing.
"""
import json, urllib.request, urllib.error
from ..config import OLLAMA_URL, MODEL, SYSTEM


class MindStoneMissing(RuntimeError):
    pass


class LLMClient:
    def __init__(self, model: str = MODEL, url: str = OLLAMA_URL):
        self.model = model
        self.url = url.rstrip("/")

    def _post(self, path: str, payload: dict) -> dict:
        req = urllib.request.Request(
            f"{self.url}{path}",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                return json.loads(r.read().decode())
        except urllib.error.URLError as e:
            raise MindStoneMissing(
                f"Ollama unreachable at {self.url} ({e}). Is it running? "
                f"Start Ollama and `ollama pull {self.model}`."
            )

    def chat(self, messages: list[dict], system: str = SYSTEM,
             tools: list | None = None, temperature: float = 0.7) -> dict:
        """One chat turn. messages = [{role, content}]. Returns the assistant msg."""
        msgs = ([{"role": "system", "content": system}] if system else []) + messages
        payload = {"model": self.model, "messages": msgs, "stream": False,
                   "options": {"temperature": temperature}}
        if tools:
            payload["tools"] = tools
        out = self._post("/api/chat", payload)
        if "error" in out:
            raise MindStoneMissing(out["error"])
        return out.get("message", {"role": "assistant", "content": ""})

    def ask(self, prompt: str, system: str = SYSTEM, **kw) -> str:
        return self.chat([{"role": "user", "content": prompt}], system=system, **kw).get("content", "")

    def available(self) -> bool:
        try:
            self._post("/api/tags", {}) if False else urllib.request.urlopen(
                f"{self.url}/api/tags", timeout=5)
            return True
        except Exception:
            return False
