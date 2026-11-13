"""Unit tests for the composed brain: router + Brain + OpenAI-compatible server.

Hermetic — no Ollama, no network. The LLM is a scripted mock with the same
dependency-injection shape the other suites use. Proves: the router picks the right
path from heuristics and /prefixes, Brain composes memory->persona->route->answer and
remembers across turns, and the server returns a well-formed OpenAI envelope for both
the JSON and the SSE streaming paths.

Run:  python myson/test_brain.py
 or:  python -m unittest myson.test_brain -v
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)                      # repo root (holds the myson/ package)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from myson import router
from myson.brain import Brain
from myson.memory.memory import Memory


class MockLLM:
    """Scripted stand-in for LLMClient — records calls, needs no model."""
    num_ctx = 8192
    model_id = "mock/model"          # so Brain._served_model reports a real value

    def __init__(self, judge="chat", embed_ok=False):
        self.calls = []
        self.judge = judge
        self.embed_ok = embed_ok
        self.last_messages = None    # capture what the model actually received

    def embed(self, text, model=None):
        if not self.embed_ok:
            raise RuntimeError("no embed in test")
        return [float(len(text) % 7), 1.0, 2.0]

    def available(self):
        return True

    def chat(self, messages, system=None, tools=None, temperature=0.7, model=None):
        self.calls.append(("chat", model, system))
        self.last_messages = messages
        return {"role": "assistant", "content": f"[mock {model}] {messages[-1]['content'][:20]}"}

    def chat_stream(self, messages, system=None, temperature=0.7, model=None):
        for w in ("piece-a ", "piece-b ", "piece-c"):
            yield w

    def ask(self, prompt, system=None, temperature=0.0, **kw):
        self.calls.append(("ask", None, None))
        return self.judge


# --------------------------------------------------------------------------------------
# Router
# --------------------------------------------------------------------------------------
class TestRouter(unittest.TestCase):
    def test_prefix_overrides_win(self):
        self.assertEqual(router.route("/think anything").mode, "recursion")
        self.assertEqual(router.route("/fast solve x^2-4=0 step by step").mode, "chat")
        self.assertEqual(router.route("/tools whatever").mode, "tools")

    def test_explicit_force_beats_everything(self):
        self.assertEqual(router.route("/think hard", force="chat").mode, "chat")

    def test_plain_chat_is_cheap(self):
        for q in ("hey what's up", "what is the capital of France", "thanks!"):
            self.assertEqual(router.route(q, llm=None).mode, "chat", q)

    def test_hard_reasoning_routes_to_recursion(self):
        r = router.route("explain step by step why mergesort is O(n log n) and prove it", llm=None)
        self.assertEqual(r.mode, "recursion")

    def test_explicit_tool_signal_routes_to_tools(self):
        r = router.route("write a python function and run it to check", llm=None)
        self.assertEqual(r.mode, "tools")

    def test_code_picks_coder_model_role(self):
        r = router.route("```py\ndef f():\n  pass\n```  refactor this", llm=None)
        self.assertEqual(r.model_role, "coder")

    def test_judge_only_fires_in_the_ambiguous_band(self):
        # a judge that always says 'recursion' must NOT upgrade a clearly-easy request
        llm = MockLLM(judge="recursion")
        self.assertEqual(router.route("hello", llm=llm).mode, "chat")
        self.assertNotIn(("ask", None, None), llm.calls)  # judge never consulted for easy text


# --------------------------------------------------------------------------------------
# Brain composition
# --------------------------------------------------------------------------------------
class TestBrain(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.mem = Memory(directory=self.d, embed_fn=None)
        self.llm = MockLLM()
        self.brain = Brain(llm=self.llm, memory=self.mem)

    def tearDown(self):
        self.mem.close()
        shutil.rmtree(self.d, ignore_errors=True)

    def test_chat_path_answers_and_reports_route(self):
        res = self.brain.respond([{"role": "user", "content": "hi myson"}])
        self.assertEqual(res.mode, "chat")
        self.assertIn("hi myson", res.answer)
        self.assertTrue(res.model)

    def test_memory_is_saved_and_recalled_across_turns(self):
        self.brain.respond([{"role": "user", "content": "remember I love the d4 opening"}])
        res = self.brain.respond([{"role": "user", "content": "what opening do I love"}])
        joined = " ".join(res.recalled).lower()
        self.assertIn("d4", joined)

    def test_client_system_message_overrides_persona(self):
        self.brain.respond([{"role": "system", "content": "SYS-OVERRIDE-MARKER"},
                            {"role": "user", "content": "hi"}])
        # the last chat call must have carried the override (possibly + recalled memory)
        last_system = self.llm.calls[-1][2]
        self.assertIn("SYS-OVERRIDE-MARKER", last_system)

    def test_stream_yields_pieces_and_persists(self):
        pieces = list(self.brain.respond_stream([{"role": "user", "content": "hey"}]))
        self.assertEqual("".join(pieces), "piece-a piece-b piece-c")
        # the user turn + streamed answer were remembered
        res = self.brain.respond([{"role": "user", "content": "did you hear me say hey"}], remember=False)
        self.assertTrue(any("hey" in r.lower() for r in res.recalled))

    def test_steering_prefix_is_stripped_before_the_model(self):
        # /fast forces chat; the model must receive "hello there", not "/fast hello there"
        self.brain.respond([{"role": "user", "content": "/fast hello there"}])
        self.assertEqual(self.llm.last_messages[-1]["content"], "hello there")
        # and memory stores the cleaned text, not the command
        res = self.brain.respond([{"role": "user", "content": "what did I say"}], remember=False)
        self.assertFalse(any("/fast" in r for r in res.recalled))

    def test_result_reports_the_model_that_actually_served(self):
        res = self.brain.respond([{"role": "user", "content": "hi"}])
        self.assertEqual(res.model, "mock/model")   # the loaded model, not a roster name


# --------------------------------------------------------------------------------------
# Server (OpenAI-compatible)
# --------------------------------------------------------------------------------------
class TestServer(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        import myson.server as server
        self.server = server
        self.mem = Memory(directory=self.d, embed_fn=None)
        server._brain = Brain(llm=MockLLM(), memory=self.mem)
        from fastapi.testclient import TestClient
        self.client = TestClient(server.app)

    def tearDown(self):
        self.mem.close()
        self.server._brain = None
        shutil.rmtree(self.d, ignore_errors=True)

    def test_models_endpoint_lists_roster_without_dupes(self):
        from myson import config
        data = self.client.get("/v1/models").json()["data"]
        ids = [m["id"] for m in data]
        self.assertEqual(len(ids), len(set(ids)))          # no duplicates
        self.assertIn(config.MODEL, ids)                   # active mind stone is listed

    def test_chat_completion_envelope_is_openai_shaped(self):
        r = self.client.post("/v1/chat/completions",
                             json={"messages": [{"role": "user", "content": "hello"}]})
        self.assertEqual(r.status_code, 200)
        j = r.json()
        for k in ("id", "object", "created", "model", "choices", "usage"):
            self.assertIn(k, j)
        self.assertEqual(j["choices"][0]["message"]["role"], "assistant")
        self.assertIn("mode", j["myson"])

    def test_streaming_is_sse_and_reassembles(self):
        r = self.client.post("/v1/chat/completions",
                             json={"messages": [{"role": "user", "content": "hello"}], "stream": True})
        self.assertEqual(r.status_code, 200)
        self.assertTrue(r.headers["content-type"].startswith("text/event-stream"))
        deltas = []
        for line in r.text.splitlines():
            if line.startswith("data: ") and "[DONE]" not in line:
                obj = json.loads(line[6:])
                deltas.append(obj["choices"][0]["delta"].get("content", ""))
        self.assertEqual("".join(deltas), "piece-a piece-b piece-c")
        self.assertIn("data: [DONE]", r.text)

    def test_force_field_selects_path(self):
        r = self.client.post("/v1/chat/completions",
                             json={"messages": [{"role": "user", "content": "x"}], "force": "tools"})
        self.assertEqual(r.json()["myson"]["mode"], "tools")


# --------------------------------------------------------------------------------------
# Backend factory (local-weights vs Ollama) — construction only, never loads a model
# --------------------------------------------------------------------------------------
class TestBackend(unittest.TestCase):
    @unittest.skipIf(os.environ.get("MYSON_BACKEND", "gguf").strip().lower() != "gguf",
                     "MYSON_BACKEND overrides the default backend in this environment")
    def test_gguf_backend_is_default_and_lazy(self):
        # "gguf" is the home path now; make_llm() with no backend must return it, unloaded.
        from myson.core.backend import make_llm
        from myson.core.gguf_llm import GGUFLLM
        llm = make_llm()                        # no backend arg -> config default (gguf)
        self.assertIsInstance(llm, GGUFLLM)
        self.assertIsNone(llm._llm)             # weights NOT loaded at construction
        for m in ("chat", "ask", "chat_stream", "embed", "available"):
            self.assertTrue(hasattr(llm, m), m)

    def test_local_backend_is_lazy(self):
        from myson.core.backend import make_llm
        from myson.core.local_llm import LocalLLM
        llm = make_llm(backend="local", model="some/repo-or-path")
        self.assertIsInstance(llm, LocalLLM)
        self.assertIsNone(llm._model)          # weights NOT loaded at construction
        for m in ("chat", "ask", "chat_stream", "embed", "available"):
            self.assertTrue(hasattr(llm, m), m)

    def test_both_backends_expose_the_full_client_interface(self):
        # interface-drift guard: the brain reads llm.num_ctx and calls chat/ask/chat_stream/
        # embed/available. If either backend drops one of these, the real path crashes even
        # though mock-based tests pass. (This is the guard the num_ctx bug slipped past.)
        from myson.core.backend import make_llm
        REQUIRED_ATTRS = ("num_ctx",)
        REQUIRED_METHODS = ("chat", "ask", "chat_stream", "embed", "available")
        for backend in ("local", "ollama", "gguf"):
            llm = make_llm(backend=backend, model="x")
            for a in REQUIRED_ATTRS:
                self.assertTrue(hasattr(llm, a), f"{backend} missing attr {a}")
                self.assertIsInstance(getattr(llm, a), int, f"{backend}.{a} not an int")
            for m in REQUIRED_METHODS:
                self.assertTrue(callable(getattr(llm, m, None)), f"{backend} missing {m}()")

    def test_ollama_backend_available_as_fallback(self):
        from myson.core.backend import make_llm
        from myson.core.llm import LLMClient
        self.assertIsInstance(make_llm(backend="ollama", model="qwen3.5:9b"), LLMClient)

    def test_local_missing_is_a_mind_stone_missing(self):
        # so every `except MindStoneMissing` handler catches a local-load failure too
        from myson.core.local_llm import LocalModelMissing
        from myson.core.llm import MindStoneMissing
        self.assertTrue(issubclass(LocalModelMissing, MindStoneMissing))

    def test_gguf_missing_is_a_mind_stone_missing(self):
        from myson.core.gguf_llm import GGUFModelMissing
        from myson.core.llm import MindStoneMissing
        self.assertTrue(issubclass(GGUFModelMissing, MindStoneMissing))

    def test_unknown_backend_is_an_error_not_a_silent_default(self):
        # a typo like MYSON_BACKEND=olama used to fall through to the default backend
        from myson.core.backend import make_llm
        from myson.core.gguf_llm import GGUFLLM
        with self.assertRaises(ValueError):
            make_llm(backend="olama", model="x")
        self.assertIsInstance(make_llm(backend=" GGUF ", model="x"), GGUFLLM)   # case/space tolerant


if __name__ == "__main__":
    unittest.main(verbosity=2)
