"""Request router — decide how hard myson should think about each request.

Every request takes one of three paths:
  * "chat"      — straight to the model (fast; everyday talk).
  * "tools"     — the function-calling loop (run code, exact math, fetch the web).
  * "recursion" — the full sample -> verify -> backtrack -> redraft loop (slow; for
                  genuinely hard, multi-step reasoning).

The router is cheap on purpose: heuristics first, an LLM tiebreak only when the signal
is ambiguous AND a model is available. Hard caps elsewhere (config.RECURSION_MAX_STEPS,
RECURSION_WALL_SECONDS) keep the recursion path from hanging the UI even if routing is
over-eager.

Per-request overrides always win:
  * a `/think` prefix on the user's text forces recursion,
  * a `/fast` prefix forces plain chat,
  * a `/tools` prefix forces the tool loop,
  * an explicit `force=` argument beats everything.

----------------------------------------------------------------------------------------
YOUR CALL (the one function worth shaping yourself): `difficulty_score` + THRESHOLDS below.
The heuristics are a sensible default, but *which of YOUR questions deserve 30 seconds of
thinking* is a judgment only you can make. Tune the signals and the two thresholds to your
own taste; everything around this function already works.
----------------------------------------------------------------------------------------
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# recursion if score >= RECUR_AT ; tools if score >= TOOLS_AT (and not recursion) ; else chat
RECUR_AT = 0.60
TOOLS_AT = 0.35

_CODE_FENCE = re.compile(r"```|\bdef \b|\bclass \b|=>|\bimport \b|;\s*$", re.MULTILINE)
_MATH = re.compile(r"[0-9]\s*[-+*/^=]\s*[0-9]|\b(solve|integrate|derivative|factor|"
                   r"simplify|equation|prove|compute|calculate)\b|[∫∑√π]|\bx\^?2\b", re.I)
_MULTISTEP = re.compile(r"\b(step by step|first.*then|why|explain|design|plan|compare|"
                        r"trade-?offs?|debug|optimi[sz]e|refactor|analy[sz]e|strategy)\b", re.I)
_TOOLY = re.compile(r"\b(run|execute|fetch|scrape|download|search the web|look up|"
                    r"what('| i)?s the (current|latest))\b", re.I)


def difficulty_score(text: str) -> float:
    """Score how much thinking a request wants, in [0, 1]. ← tune this to your taste.

    Default signals (each nudges the score up):
      - code present / asked for      -> leans tools or recursion
      - math / symbolic work          -> leans tools (exact) or recursion (derivation)
      - multi-step / 'why' / 'design' -> leans recursion
      - sheer length                  -> mild nudge
    """
    t = text or ""
    score = 0.0
    if _CODE_FENCE.search(t):
        score += 0.35
    if _MATH.search(t):
        score += 0.35
    if _MULTISTEP.search(t):
        score += 0.30
    if _TOOLY.search(t):
        score += 0.35   # "run/execute/fetch/look up" is a strong, explicit tool signal
    # length: long asks tend to be harder; cap the nudge
    words = len(t.split())
    score += min(0.20, words / 600.0)
    return max(0.0, min(1.0, score))


@dataclass
class Route:
    mode: str            # "chat" | "tools" | "recursion"
    model_role: str = "default"   # a key in config.MODELS
    reason: str = ""
    cleaned_text: str = ""        # user text with any /prefix stripped


_PREFIX = {"/think": "recursion", "/fast": "chat", "/tools": "tools"}


def _last_user_text(messages) -> str:
    if isinstance(messages, str):
        return messages
    for m in reversed(messages):
        if (m.get("role") if isinstance(m, dict) else getattr(m, "role", None)) == "user":
            return (m.get("content") if isinstance(m, dict) else getattr(m, "content", "")) or ""
    return ""


def route(messages, llm=None, force: str | None = None, allow_judge: bool = True) -> Route:
    """Pick a path for this request. `messages` is OpenAI-style (or a bare string)."""
    text = _last_user_text(messages).strip()

    # 1) explicit force argument wins outright
    if force in ("chat", "tools", "recursion"):
        return Route(force, reason="forced")

    # 2) a /prefix on the user's text
    for pfx, mode in _PREFIX.items():
        if text.lower().startswith(pfx):
            return Route(mode, reason=f"prefix {pfx}", cleaned_text=text[len(pfx):].strip())

    # 3) heuristic score
    score = difficulty_score(text)
    tooly = bool(_TOOLY.search(text)) or bool(_MATH.search(text)) or bool(_CODE_FENCE.search(text))
    model_role = "coder" if _CODE_FENCE.search(text) else "default"

    if score >= RECUR_AT:
        return Route("recursion", model_role, reason=f"score {score:.2f} >= {RECUR_AT}", cleaned_text=text)
    if tooly and score >= TOOLS_AT:
        return Route("tools", model_role, reason=f"tooly, score {score:.2f} >= {TOOLS_AT}", cleaned_text=text)

    # 4) ambiguous band: optional one-shot LLM tiebreak, else chat
    if allow_judge and llm is not None and TOOLS_AT <= score < RECUR_AT:
        verdict = _judge(text, llm)
        if verdict in ("chat", "tools", "recursion"):
            return Route(verdict, model_role, reason=f"judge ({score:.2f})", cleaned_text=text)

    return Route("chat", model_role, reason=f"score {score:.2f} < {TOOLS_AT}", cleaned_text=text)


def _judge(text: str, llm) -> str | None:
    """One cheap classification call when the heuristic is on the fence."""
    prompt = (
        "Classify how to handle this user request. Reply with exactly ONE word:\n"
        "  chat      - ordinary conversation/knowledge, answer directly\n"
        "  tools     - needs running code, exact math, or a live web lookup\n"
        "  recursion - hard multi-step reasoning worth careful step-by-step work\n\n"
        f"REQUEST: {text[:800]}\n\nONE WORD:"
    )
    try:
        out = (llm.ask(prompt, system="You are a terse request classifier.",
                       temperature=0.0) or "").strip().lower()
    except Exception:
        return None
    for word in ("recursion", "tools", "chat"):
        if word in out:
            return word
    return None
