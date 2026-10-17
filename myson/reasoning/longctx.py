"""Recursive long-context handler — fit oversized input into the model's window.

Clean rebuild of the corpus's `recursive_context.py` idea: when a message is too big
for the context window, chunk it and map-reduce (summarize each chunk, then summarize
the summaries), bounded by a hard recursion guard so it always terminates.

Honest scope: this is a wrapper trick over any chat model (text in, text out) — it
preserves gist, not every detail. It is NOT a real long-context model.
"""
from __future__ import annotations

from typing import List

# ~4 chars/token is the usual rough rule; keep a safety margin for the reply + system.
_CHARS_PER_TOKEN = 4
_REPLY_HEADROOM_TOKENS = 1024
_MAX_DEPTH = 4


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // _CHARS_PER_TOKEN)


def _chunk(text: str, chunk_chars: int) -> List[str]:
    """Split on paragraph boundaries where possible, else hard-cut."""
    if len(text) <= chunk_chars:
        return [text]
    chunks, buf = [], []
    size = 0
    for para in text.split("\n\n"):
        p = para + "\n\n"
        if size + len(p) > chunk_chars and buf:
            chunks.append("".join(buf))
            buf, size = [], 0
        if len(p) > chunk_chars:  # a single giant paragraph: hard-cut it
            for i in range(0, len(p), chunk_chars):
                chunks.append(p[i:i + chunk_chars])
            continue
        buf.append(p)
        size += len(p)
    if buf:
        chunks.append("".join(buf))
    return chunks


def reduce_text(text: str, llm, budget_chars: int, depth: int = 0) -> str:
    """Summarize `text` down to <= budget_chars, recursively, with a depth guard."""
    if len(text) <= budget_chars or depth >= _MAX_DEPTH:
        return text[:budget_chars] if len(text) > budget_chars else text
    chunks = _chunk(text, budget_chars)
    if len(chunks) == 1:
        return chunks[0][:budget_chars]
    per_chunk_budget = max(200, budget_chars // max(1, len(chunks)))
    summaries = []
    for ch in chunks:
        try:
            s = llm.ask(
                "Summarize the following, preserving facts, names, numbers and intent. "
                f"Keep it under ~{per_chunk_budget // _CHARS_PER_TOKEN} tokens.\n\n" + ch,
                system="You compress text without losing load-bearing detail.",
                temperature=0.0,
            ) or ""
        except Exception:
            s = ch[:per_chunk_budget]
        summaries.append(s)
    combined = "\n\n".join(summaries)
    # recurse until it fits (or depth runs out)
    return reduce_text(combined, llm, budget_chars, depth + 1)


def maybe_reduce(messages: list, llm, num_ctx: int) -> list:
    """If the conversation overflows the context window, map-reduce the largest message.

    Returns a (possibly) shrunk copy of `messages`; leaves small conversations untouched.
    Never raises — on any failure it hard-truncates so the request still goes through.
    """
    if not messages:
        return messages
    budget_tokens = max(512, num_ctx - _REPLY_HEADROOM_TOKENS)
    total = sum(_approx_tokens(m.get("content", "")) for m in messages)
    if total <= budget_tokens:
        return messages

    out = [dict(m) for m in messages]
    # reduce from the largest message down until the whole convo fits
    budget_chars = budget_tokens * _CHARS_PER_TOKEN
    # give the single biggest message most of the budget, the rest a slim share
    idx_by_size = sorted(range(len(out)), key=lambda i: len(out[i].get("content", "")), reverse=True)
    for i in idx_by_size:
        if sum(_approx_tokens(m.get("content", "")) for m in out) <= budget_tokens:
            break
        content = out[i].get("content", "")
        share = max(400, budget_chars // len(out))
        if len(content) > share:
            out[i]["content"] = reduce_text(content, llm, share)
    return out
