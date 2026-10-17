"""myson 3-tier memory — working / episodic (JSONL+emb) / semantic (SQLite+emb).

Hybrid retrieval = a*cosine + b*recency + c*importance. Consolidation promotes
episodic -> semantic with a forgetting guard. Embeddings come from an injected
embed_fn (Ollama in production, a mock in tests) so there is no hard Ollama dep.
Upgrade path: swap the SQLite emb column for Chroma/FAISS/Qdrant past ~1e4-1e5 items.
"""
import os, json, time, sqlite3, math
from typing import Callable, Optional, List


def _cos(a, b) -> float:
    if not a or not b:
        return 0.0
    s = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)); nb = math.sqrt(sum(y * y for y in b))
    return s / (na * nb) if na and nb else 0.0


class Memory:
    def __init__(self, directory: str = ".myson_mem", embed_fn: Optional[Callable] = None):
        os.makedirs(directory, exist_ok=True)
        self.dir = directory
        self.epath = os.path.join(directory, "episodic.jsonl")
        self.dbpath = os.path.join(directory, "semantic.db")
        # default embed = bag-of-words hash vector (usable offline; inject Ollama for prod)
        self.embed = embed_fn or self._bow
        self.working: List[dict] = []
        self.db = sqlite3.connect(self.dbpath)
        self.db.execute("CREATE TABLE IF NOT EXISTS sem(id INTEGER PRIMARY KEY, text TEXT, "
                        "emb TEXT, importance REAL, ts REAL)")
        self.db.commit()

    @staticmethod
    def _bow(text: str, dim: int = 64) -> List[float]:
        v = [0.0] * dim
        for w in text.lower().split():
            v[hash(w) % dim] += 1.0
        return v

    # --- API compatible with the husk core/memory.py ---
    def save(self, role: str, text: str, importance: float = 1.0):
        rec = {"t": time.time(), "role": role, "text": text,
               "importance": importance, "emb": self.embed(text)}
        with open(self.epath, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        self.working.append({"role": role, "text": text})
        self.working = self.working[-20:]

    def _episodic(self) -> List[dict]:
        if not os.path.exists(self.epath):
            return []
        return [json.loads(l) for l in open(self.epath, encoding="utf-8") if l.strip()]

    def recall(self, query: str, k: int = 5, a: float = 1.0, b: float = 0.3, c: float = 0.2) -> List[str]:
        q = self.embed(query); now = time.time(); rows = []
        for r in self._episodic():
            rows.append((r["text"], r.get("emb"), r.get("importance", 1.0), r.get("t", now)))
        for text, emb, imp, ts in self.db.execute("SELECT text,emb,importance,ts FROM sem"):
            rows.append((text, json.loads(emb), imp, ts))
        scored = []
        for text, emb, imp, ts in rows:
            rec = math.exp(-(now - ts) / 86400.0)
            scored.append((a * _cos(q, emb) + b * rec + c * imp, text))
        scored.sort(reverse=True)
        seen, out = set(), []
        for _, t in scored:
            if t not in seen:
                seen.add(t); out.append(t)
            if len(out) >= k:
                break
        return out

    def consolidate(self, min_keep_ratio: float = 0.5) -> dict:
        """Promote important episodics -> semantic. Forgetting guard: additive only,
        and refuse any operation that would drop the store below min_keep_ratio."""
        eps = self._episodic()
        before = self.db.execute("SELECT COUNT(*) FROM sem").fetchone()[0]
        promoted = 0
        for r in eps:
            if r.get("importance", 1.0) >= 1.0:
                self.db.execute("INSERT INTO sem(text,emb,importance,ts) VALUES(?,?,?,?)",
                                (r["text"], json.dumps(r.get("emb")), r.get("importance", 1.0),
                                 r.get("t", time.time())))
                promoted += 1
        after = self.db.execute("SELECT COUNT(*) FROM sem").fetchone()[0]
        if after < before * min_keep_ratio:   # guard (can't trip on additive path; defensive)
            self.db.rollback()
            return {"promoted": 0, "aborted": "forgetting_guard"}
        self.db.commit()
        return {"promoted": promoted, "semantic_before": before, "semantic_after": after}
