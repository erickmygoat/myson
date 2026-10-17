"""Tool registry + function-calling loop for the myson agent.

``run_with_tools(messages, llm, tools)`` runs the standard agentic loop: show the
model the tools, let it ask for a tool call, execute it locally, feed the result
back, and repeat until the model answers in plain text (or a step budget is hit).

It supports **two** tool-call protocols so it works with any local model:

  1. native  - the assistant message carries ``tool_calls`` (Ollama / OpenAI
               function-calling format);
  2. text    - the assistant replies with a JSON object such as
               ``{"tool": "calc", "args": {"expression": "2+2"}}`` (used by local
               models that don't emit native tool calls).

The loop is bounded (``max_steps``), every tool result is truncated before being
fed back, and unknown tools / bad arguments / tool exceptions are turned into
plain error strings the model can read and recover from — nothing here can crash
the caller.
"""
from __future__ import annotations

import inspect
import json
import re
from typing import Any, Callable

# --- config, resilient to standalone import in tests ---------------------------
try:
    from myson.config import SYSTEM as _SYSTEM
except Exception:  # pragma: no cover - import shim
    try:
        from ..config import SYSTEM as _SYSTEM
    except Exception:
        _SYSTEM = (
            "You are MY SON, a private local assistant running on the user's own "
            "machine. You are direct, capable, and candid."
        )

# --- the tool implementations, resilient to standalone import ------------------
try:
    from myson.tools import sandbox, mathtool, web
except Exception:  # pragma: no cover - import shim
    try:
        from . import sandbox, mathtool, web
    except Exception:
        import sandbox, mathtool, web  # type: ignore

MAX_TOOL_RESULT_CHARS = 6000


class Tool:
    """A callable the model can invoke, plus its JSON-schema description."""

    def __init__(self, name: str, func: Callable[..., Any], description: str,
                 parameters: dict | None = None):
        self.name = name
        self.func = func
        self.description = description
        self.parameters = parameters or {"type": "object", "properties": {}, "required": []}

    def spec(self) -> dict:
        """OpenAI / Ollama function-tool schema for this tool."""
        return {"type": "function",
                "function": {"name": self.name,
                             "description": self.description,
                             "parameters": self.parameters}}

    def call(self, args: dict | None) -> str:
        """Invoke the tool with model-supplied args; always return a string."""
        args = args or {}
        if not isinstance(args, dict):
            return f"error: arguments for {self.name!r} must be an object, got {type(args).__name__}"
        # Drop args the function doesn't accept (a common model mistake), unless
        # the function takes **kwargs.
        try:
            sig = inspect.signature(self.func)
            accepts_kwargs = any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())
            if not accepts_kwargs:
                allowed = set(sig.parameters)
                args = {k: v for k, v in args.items() if k in allowed}
        except (TypeError, ValueError):  # pragma: no cover - builtins w/o signature
            pass
        try:
            result = self.func(**args)
        except TypeError as e:
            return f"error: bad arguments for {self.name!r}: {e}"
        except Exception as e:
            return f"error: {self.name!r} raised {type(e).__name__}: {e}"
        text = result if isinstance(result, str) else json.dumps(result, default=str)
        if len(text) > MAX_TOOL_RESULT_CHARS:
            text = text[:MAX_TOOL_RESULT_CHARS] + "\n...[tool output truncated]"
        return text


def build_default_tools() -> dict[str, Tool]:
    """The default SAFE toolset: sandboxed python, math, web, jailed file I/O."""
    tools = [
        Tool("python", sandbox.run_python,
             "Run a short Python 3 snippet in a bounded sandbox (timeout + output "
             "cap, cwd jailed to a sandbox dir) and return its stdout/stderr. Use "
             "for computation, data wrangling, quick checks. Print what you want back.",
             {"type": "object",
              "properties": {"code": {"type": "string", "description": "Python source to run"}},
              "required": ["code"]}),
        Tool("calc", mathtool.calc,
             "Evaluate a math expression or solve an equation. Handles arithmetic, "
             "functions (sqrt, sin, log...), and — when sympy is installed — symbolic "
             "simplification, equation solving ('x**2-4=0'), and calculus (diff/integrate).",
             {"type": "object",
              "properties": {"expression": {"type": "string",
                                            "description": "e.g. '2+2*3', 'sqrt(8)', 'x**2-4=0'"}},
              "required": ["expression"]}),
        Tool("web_search", web.web_search,
             "Search the web for a query. Returns results if a search backend is "
             "configured, otherwise a clear offline/stub message. Do not rely on it "
             "when the machine is offline.",
             {"type": "object",
              "properties": {"query": {"type": "string", "description": "search query"}},
              "required": ["query"]}),
        Tool("web_fetch", web.web_fetch,
             "Fetch the visible text of an http/https URL. Returns a clear offline "
             "message if the network is unreachable.",
             {"type": "object",
              "properties": {"url": {"type": "string", "description": "http(s) URL to fetch"}},
              "required": ["url"]}),
        Tool("read_file", sandbox.read_file,
             "Read a UTF-8 text file from inside the sandbox jail (path is relative "
             "to the sandbox dir).",
             {"type": "object",
              "properties": {"path": {"type": "string", "description": "jail-relative path"}},
              "required": ["path"]}),
        Tool("write_file", sandbox.write_file,
             "Write a UTF-8 text file inside the sandbox jail (path is relative to "
             "the sandbox dir).",
             {"type": "object",
              "properties": {"path": {"type": "string"},
                             "content": {"type": "string"}},
              "required": ["path", "content"]}),
    ]
    return {t.name: t for t in tools}


def _normalize_tools(tools) -> dict[str, Tool]:
    if tools is None:
        return build_default_tools()
    if isinstance(tools, dict):
        return tools
    return {t.name: t for t in tools}


def _tool_protocol_prompt(tools: dict[str, Tool]) -> str:
    """Instructions appended to the system prompt so even models without native
    tool-calling can request tools via a JSON object."""
    lines = ["", "You have these tools available:"]
    for t in tools.values():
        props = ", ".join(t.parameters.get("properties", {}))
        lines.append(f"  - {t.name}({props}): {t.description}")
    lines += [
        "",
        "To use a tool, reply with ONLY a JSON object and nothing else:",
        '  {"tool": "<name>", "args": {<arguments>}}',
        "You will then be shown the tool's result and may call another tool or "
        "answer. When you have the final answer, reply in plain text with no JSON.",
    ]
    return "\n".join(lines)


def _as_message(msg) -> dict:
    """Coerce an LLM return value into a {role, content, ...} dict."""
    if isinstance(msg, dict):
        return msg
    if isinstance(msg, str):
        return {"role": "assistant", "content": msg}
    return {"role": "assistant", "content": str(msg)}


def _parse_text_tool_call(content: str, tools: dict[str, Tool]):
    """Return (name, args) if ``content`` is a JSON tool call for a known tool."""
    if not content or "{" not in content:
        return None
    s = content.strip()
    s = re.sub(r"^```(?:json)?", "", s).strip()
    s = re.sub(r"```$", "", s).strip()
    candidates = [s]
    start, end = s.find("{"), s.rfind("}")
    if start != -1 and end > start:
        candidates.append(s[start:end + 1])
    for cand in candidates:
        try:
            obj = json.loads(cand)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        name = obj.get("tool") or obj.get("name")
        if name in tools:
            args = obj.get("args")
            if args is None:
                args = obj.get("arguments", obj.get("parameters", {}))
            return name, args
    return None


def _extract_calls(msg: dict, tools: dict[str, Tool]):
    """Return a list of (name, args, tool_call_id) from an assistant message, via
    native tool_calls first, then the text-JSON fallback. Empty list = final answer."""
    calls = []
    native = msg.get("tool_calls")
    if native:
        for tc in native:
            fn = tc.get("function", tc) if isinstance(tc, dict) else {}
            name = fn.get("name")
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {"_raw": args}
            calls.append((name, args, tc.get("id") if isinstance(tc, dict) else None))
        return calls
    parsed = _parse_text_tool_call(msg.get("content", ""), tools)
    if parsed:
        calls.append((parsed[0], parsed[1], None))
    return calls


def run_with_tools(messages: list[dict], llm, tools=None, system: str = _SYSTEM,
                   max_steps: int = 6, temperature: float = 0.2) -> str:
    """Run the function-calling loop and return the model's final text answer.

    Parameters
    ----------
    messages : the conversation so far, ``[{"role","content"}, ...]``.
    llm      : an object with ``.chat(messages, system=, tools=, temperature=)``
               returning an assistant message (dict or str) — i.e. ``core.llm.LLMClient``.
    tools    : a ``{name: Tool}`` dict, a list of ``Tool``, or None for the defaults.
    """
    tools = _normalize_tools(tools)
    specs = [t.spec() for t in tools.values()]
    full_system = (system or "") + _tool_protocol_prompt(tools)
    convo = [dict(m) for m in messages]

    for _step in range(max_steps):
        try:
            raw = llm.chat(convo, system=full_system, tools=specs, temperature=temperature)
        except TypeError:
            # a client whose chat() doesn't accept a tools= kwarg
            raw = llm.chat(convo, system=full_system, temperature=temperature)
        msg = _as_message(raw)
        calls = _extract_calls(msg, tools)

        if not calls:
            return (msg.get("content") or "").strip()

        # Record the assistant's tool request, then run each call.
        convo.append({"role": "assistant",
                      "content": msg.get("content", ""),
                      **({"tool_calls": msg["tool_calls"]} if msg.get("tool_calls") else {})})
        for name, args, call_id in calls:
            if name not in tools:
                result = (f"error: unknown tool {name!r}. "
                          f"Available: {', '.join(tools)}")
            else:
                result = tools[name].call(args)
            tool_msg = {"role": "tool", "name": name, "content": result}
            if call_id:
                tool_msg["tool_call_id"] = call_id
            convo.append(tool_msg)

    # Step budget exhausted: ask once more, without tools, for a final answer.
    try:
        final = llm.chat(convo + [{"role": "user",
                                   "content": "Give your best final answer now, in plain text."}],
                         system=system, temperature=temperature)
    except Exception:
        return "[tool loop ended without a final answer: step budget exhausted]"
    return (_as_message(final).get("content") or
            "[tool loop ended without a final answer: step budget exhausted]").strip()
