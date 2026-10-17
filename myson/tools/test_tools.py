"""Unit tests for the myson tool layer.

Run from the repo root:
    python -m unittest myson.tools.test_tools -v

No network and no running Ollama are required: the LLM is mocked, the sandbox
runs real child processes locally, and the web tools are exercised against a
patched transport so the offline paths are deterministic.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest

# Make the package importable whether run as a module or a file.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from myson.tools import sandbox, mathtool, web, agent_tools
from myson.tools.agent_tools import Tool, build_default_tools, run_with_tools


# --- a scriptable mock of core.llm.LLMClient -----------------------------------

class MockLLM:
    """Returns a scripted sequence of assistant messages and records every call."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []  # [{messages, system, tools, temperature}, ...]

    def chat(self, messages, system="", tools=None, temperature=0.7):
        self.calls.append({"messages": [dict(m) for m in messages],
                           "system": system, "tools": tools, "temperature": temperature})
        if self.responses:
            return self.responses.pop(0)
        return {"role": "assistant", "content": "(mock exhausted)"}

    def ask(self, prompt, system="", **kw):
        return self.chat([{"role": "user", "content": prompt}], system=system, **kw).get("content", "")


class MockLLMNoTools:
    """A client whose chat() does NOT accept a tools= kwarg (exercises the shim)."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.saw_tools_kwarg = False

    def chat(self, messages, system="", temperature=0.7):
        return self.responses.pop(0) if self.responses else {"content": "done"}


# --- sandbox (bounded code exec) -----------------------------------------------

class TestSandbox(unittest.TestCase):
    def setUp(self):
        # Jail under cwd (workspace), not the system temp dir: the child process
        # must be able to create files there, which some sandboxed temp dirs forbid.
        self.tmp = tempfile.mkdtemp(prefix=".myson_jail_", dir=os.getcwd())
        os.environ["MYSON_SANDBOX_DIR"] = self.tmp

    def tearDown(self):
        os.environ.pop("MYSON_SANDBOX_DIR", None)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_captures_stdout(self):
        self.assertEqual(sandbox.run_python("print('hello world')"), "hello world")

    def test_captures_error(self):
        out = sandbox.run_python("print(1); undefined_name")
        self.assertIn("1", out)
        self.assertIn("NameError", out)

    def test_timeout_kills(self):
        out = sandbox.run_python("import time; time.sleep(5)", timeout=1.0)
        self.assertIn("Timeout", out)

    def test_output_is_capped(self):
        out = sandbox.run_python("print('x' * 50000)", max_output=500)
        self.assertIn("truncated", out)
        self.assertLess(len(out), 1000)

    def test_cwd_is_jailed(self):
        out = sandbox.run_python("open('made.txt','w').write('hi'); print('ok')")
        self.assertEqual(out, "ok")
        self.assertTrue(os.path.exists(os.path.join(self.tmp, "made.txt")))

    def test_empty_code(self):
        self.assertIn("no code", sandbox.run_python("   "))

    def test_file_tools_roundtrip_and_jail(self):
        self.assertIn("wrote", sandbox.write_file("notes/a.txt", "data"))
        self.assertEqual(sandbox.read_file("notes/a.txt"), "data")
        self.assertIn("escapes the sandbox", sandbox.write_file("../escape.txt", "x"))
        self.assertIn("escapes the sandbox", sandbox.read_file("../../etc/passwd"))


# --- math tool -----------------------------------------------------------------

class TestMathSympy(unittest.TestCase):
    """The sympy tier (sympy is available in this environment)."""

    def setUp(self):
        if not mathtool.have_sympy():
            self.skipTest("sympy not installed; covered by TestMathNumeric")

    def test_arithmetic(self):
        self.assertEqual(mathtool.calc("2 + 2*3"), "8")

    def test_irrational_shows_exact_and_float(self):
        out = mathtool.calc("sqrt(8)")
        self.assertIn("sqrt(2)", out)
        self.assertIn("2.82842712475", out)

    def test_solve_quadratic(self):
        out = mathtool.calc("x**2 - 4 = 0")
        self.assertIn("-2", out)
        self.assertIn("2", out)

    def test_calculus(self):
        self.assertEqual(mathtool.calc("diff(sin(x), x)").replace(" ", ""), "cos(x)")

    def test_rejects_dunder(self):
        self.assertIn("disallowed", mathtool.calc("x.__class__"))


class TestMathNumeric(unittest.TestCase):
    """The no-sympy fallback (forced by patching the availability flag)."""

    def setUp(self):
        self._saved = mathtool._HAVE_SYMPY
        mathtool._HAVE_SYMPY = False

    def tearDown(self):
        mathtool._HAVE_SYMPY = self._saved

    def test_arithmetic(self):
        self.assertEqual(mathtool.calc("2 + 2*3"), "8")

    def test_math_functions(self):
        self.assertTrue(mathtool.calc("sqrt(8)").startswith("2.82842712"))

    def test_rejects_calls_outside_whitelist(self):
        self.assertIn("whitelisted", mathtool.calc("__import__('os')"))

    def test_rejects_unknown_name(self):
        self.assertIn("unknown name", mathtool.calc("x + 1"))

    def test_division_by_zero(self):
        self.assertIn("division by zero", mathtool.calc("1/0"))

    def test_equation_needs_sympy(self):
        self.assertIn("sympy", mathtool.calc("x**2 - 4 = 0"))


# --- web tool (offline-aware) --------------------------------------------------

class TestWeb(unittest.TestCase):
    def tearDown(self):
        os.environ.pop("MYSON_SEARCH_URL", None)

    def test_fetch_rejects_non_http(self):
        self.assertIn("only http", web.web_fetch("file:///etc/passwd"))

    def test_fetch_offline_message(self):
        orig = web._get
        web._get = lambda *a, **k: (_ for _ in ()).throw(web._Offline("name resolution failed"))
        try:
            out = web.web_fetch("http://example.invalid/page")
            self.assertIn("[offline]", out)
            self.assertIn("example.invalid", out)
        finally:
            web._get = orig

    def test_fetch_parses_html(self):
        orig = web._get
        web._get = lambda *a, **k: (200, b"<html><body><h1>Hi</h1><p>there</p></body></html>", "text/html")
        try:
            out = web.web_fetch("http://example.com/")
            self.assertIn("Hi", out)
            self.assertIn("there", out)
        finally:
            web._get = orig

    def test_search_stub_when_unconfigured(self):
        out = web.web_search("quantum computing")
        self.assertIn("stub", out)
        self.assertIn("quantum computing", out)

    def test_search_backend_offline(self):
        os.environ["MYSON_SEARCH_URL"] = "http://localhost:8888/search"
        orig = web._get
        web._get = lambda *a, **k: (_ for _ in ()).throw(web._Offline("refused"))
        try:
            self.assertIn("[offline]", web.web_search("anything"))
        finally:
            web._get = orig

    def test_search_backend_parses_results(self):
        os.environ["MYSON_SEARCH_URL"] = "http://localhost:8888/search"
        payload = b'{"results":[{"title":"T1","url":"http://a","content":"snip"}]}'
        orig = web._get
        web._get = lambda *a, **k: (200, payload, "application/json")
        try:
            out = web.web_search("q")
            self.assertIn("T1", out)
            self.assertIn("http://a", out)
        finally:
            web._get = orig


# --- the function-calling loop -------------------------------------------------

class TestRunWithTools(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix=".myson_jail_", dir=os.getcwd())
        os.environ["MYSON_SANDBOX_DIR"] = self.tmp
        self.tools = build_default_tools()

    def tearDown(self):
        os.environ.pop("MYSON_SANDBOX_DIR", None)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _tool_msgs(self, call):
        return [m for m in call["messages"] if m.get("role") == "tool"]

    def test_native_tool_call_then_answer(self):
        llm = MockLLM([
            {"role": "assistant", "content": "",
             "tool_calls": [{"function": {"name": "calc", "arguments": {"expression": "2+2"}}}]},
            {"role": "assistant", "content": "The answer is 4."},
        ])
        out = run_with_tools([{"role": "user", "content": "what is 2+2?"}],
                             llm, self.tools, max_steps=4)
        self.assertEqual(out, "The answer is 4.")
        # the tool actually ran and its result "4" was fed back to the 2nd call
        fed_back = [m["content"] for m in self._tool_msgs(llm.calls[1])]
        self.assertIn("4", fed_back)

    def test_native_arguments_as_json_string(self):
        llm = MockLLM([
            {"role": "assistant", "tool_calls": [
                {"function": {"name": "python", "arguments": '{"code": "print(2**10)"}'}}]},
            {"role": "assistant", "content": "It is 1024."},
        ])
        out = run_with_tools([{"role": "user", "content": "2^10?"}], llm, self.tools)
        self.assertEqual(out, "It is 1024.")
        self.assertIn("1024", [m["content"] for m in self._tool_msgs(llm.calls[1])])

    def test_text_json_fallback(self):
        llm = MockLLM([
            {"role": "assistant", "content": '{"tool": "calc", "args": {"expression": "6*7"}}'},
            {"role": "assistant", "content": "It is 42."},
        ])
        out = run_with_tools([{"role": "user", "content": "6*7?"}], llm, self.tools)
        self.assertEqual(out, "It is 42.")
        self.assertIn("42", [m["content"] for m in self._tool_msgs(llm.calls[1])])

    def test_text_json_in_code_fence(self):
        llm = MockLLM([
            {"role": "assistant", "content": '```json\n{"tool":"calc","args":{"expression":"10-3"}}\n```'},
            {"role": "assistant", "content": "7"},
        ])
        out = run_with_tools([{"role": "user", "content": "10-3?"}], llm, self.tools)
        self.assertEqual(out, "7")
        self.assertIn("7", [m["content"] for m in self._tool_msgs(llm.calls[1])])

    def test_plain_answer_no_tools(self):
        llm = MockLLM([{"role": "assistant", "content": "Just a direct answer."}])
        out = run_with_tools([{"role": "user", "content": "hi"}], llm, self.tools)
        self.assertEqual(out, "Just a direct answer.")

    def test_unknown_tool_recovers(self):
        llm = MockLLM([
            {"role": "assistant", "tool_calls": [{"function": {"name": "frobnicate", "arguments": {}}}]},
            {"role": "assistant", "content": "ok, recovered"},
        ])
        out = run_with_tools([{"role": "user", "content": "x"}], llm, self.tools)
        self.assertEqual(out, "ok, recovered")
        fed = [m["content"] for m in self._tool_msgs(llm.calls[1])]
        self.assertTrue(any("unknown tool" in c for c in fed))

    def test_step_budget_is_bounded(self):
        loop_call = {"role": "assistant",
                     "tool_calls": [{"function": {"name": "calc", "arguments": {"expression": "1+1"}}}]}
        llm = MockLLM([loop_call, loop_call, loop_call, loop_call, loop_call])
        out = run_with_tools([{"role": "user", "content": "spin"}], llm, self.tools, max_steps=3)
        self.assertIsInstance(out, str)
        # 3 in-loop steps + 1 forced final call
        self.assertEqual(len(llm.calls), 4)
        self.assertTrue(any("final answer" in m["content"].lower()
                            for m in llm.calls[-1]["messages"]))

    def test_client_without_tools_kwarg(self):
        llm = MockLLMNoTools([{"role": "assistant", "content": "answered without tools kwarg"}])
        out = run_with_tools([{"role": "user", "content": "hi"}], llm, self.tools)
        self.assertEqual(out, "answered without tools kwarg")

    def test_custom_tool_registration(self):
        hits = {"n": 0}

        def ping(x: str) -> str:
            hits["n"] += 1
            return f"pong:{x}"

        tools = {"ping": Tool("ping", ping, "ping back",
                              {"type": "object", "properties": {"x": {"type": "string"}},
                               "required": ["x"]})}
        llm = MockLLM([
            {"role": "assistant", "content": '{"tool":"ping","args":{"x":"hi"}}'},
            {"role": "assistant", "content": "got pong"},
        ])
        out = run_with_tools([{"role": "user", "content": "ping"}], llm, tools)
        self.assertEqual(out, "got pong")
        self.assertEqual(hits["n"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
