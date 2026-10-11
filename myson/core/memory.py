"""Dead-simple persistent memory: append-only JSONL + keyword recall.
Swap for a vector store later; the interface stays the same.
"""
import json, os, time


class Memory:
    def __init__(self, path: str = "myson_memory.jsonl"):
        self.path = path

    def save(self, role: str, text: str):
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"t": time.time(), "role": role, "text": text}) + "\n")

    def recall(self, query: str, k: int = 5) -> list[str]:
        if not os.path.exists(self.path):
            return []
        q = set(query.lower().split())
        rows = []
        for line in open(self.path, encoding="utf-8"):
            try:
                r = json.loads(line)
            except Exception:
                continue
            score = len(q & set(r["text"].lower().split()))
            if score:
                rows.append((score, r["text"]))
        rows.sort(reverse=True)
        return [t for _, t in rows[:k]]
