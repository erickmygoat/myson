"""Web fetch / search tool for the myson agent.

A real HTTP client (``requests`` when present, else stdlib ``urllib`` so there is
no hard dependency) that is **offline-aware**: when the network cannot be reached
it returns a clear ``[offline]`` message instead of raising, so the agent loop
keeps working on a disconnected box.

  * ``web_fetch(url)``   - GET an http/https URL, strip HTML to visible text,
                           return it truncated. Clear offline message on failure.
  * ``web_search(query)``- Query a JSON search backend if one is configured via
                           ``MYSON_SEARCH_URL`` (e.g. a local SearXNG instance);
                           otherwise it is an honest **stub** that says no backend
                           is wired and returns an offline message.

Why a stub: there is no built-in, key-free web search. Rather than pretend, the
tool tells the model (and the user) exactly what to configure to make it live.
"""
from __future__ import annotations

import html
import json
import os
import re
import socket
import urllib.error
import urllib.parse
import urllib.request

USER_AGENT = "myson/0.1 (local personal assistant)"
DEFAULT_TIMEOUT = float(os.environ.get("MYSON_HTTP_TIMEOUT", "10"))
DEFAULT_MAX_CHARS = int(os.environ.get("MYSON_HTTP_MAX_CHARS", "8000"))
MAX_BYTES = int(os.environ.get("MYSON_HTTP_MAX_BYTES", str(2 * 1024 * 1024)))  # 2 MB


class _Offline(Exception):
    """Raised internally when the network is unreachable / blocked."""


def _is_offline_error(exc: Exception) -> bool:
    """Classify a transport exception as 'no network' vs a real HTTP response."""
    if isinstance(exc, (socket.timeout, ConnectionError, socket.gaierror)):
        return True
    if isinstance(exc, urllib.error.HTTPError):
        # A proxy/sandbox block often shows up as a 403 mentioning the policy.
        if exc.code in (403, 407, 502, 503):
            try:
                body = (exc.read(2048) or b"").decode("utf-8", "replace").lower()
            except Exception:
                body = ""
            if any(w in body for w in ("sandbox", "proxy", "blocked", "network policy")):
                return True
        return False  # a genuine HTTP status from a server we reached
    if isinstance(exc, urllib.error.URLError):
        return True
    return False


def _get(url: str, timeout: float) -> tuple[int, bytes, str]:
    """GET ``url``; return (status, body_bytes, content_type). Raise ``_Offline``
    when the network is unreachable, or re-raise HTTPError for real statuses."""
    try:  # prefer requests if it is installed
        import requests  # type: ignore
        try:
            resp = requests.get(url, timeout=timeout,
                                 headers={"User-Agent": USER_AGENT}, stream=True)
            body = resp.raw.read(MAX_BYTES, decode_content=True) or b""
            return resp.status_code, body, resp.headers.get("Content-Type", "")
        except requests.exceptions.RequestException as e:
            raise _Offline(str(e) or type(e).__name__)
    except ImportError:
        pass  # fall back to urllib

    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            ctype = r.headers.get("Content-Type", "")
            return r.status, r.read(MAX_BYTES), ctype
    except Exception as e:
        if _is_offline_error(e):
            raise _Offline(str(e) or type(e).__name__)
        raise


_TAG_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.I | re.S)
_ANYTAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"[ \t]+")
_NL_RE = re.compile(r"\n{3,}")


def _html_to_text(body: bytes, ctype: str) -> str:
    text = body.decode("utf-8", "replace")
    if "html" in ctype.lower() or "<html" in text[:2000].lower():
        text = _TAG_RE.sub(" ", text)
        text = _ANYTAG_RE.sub(" ", text)
        text = html.unescape(text)
    text = _WS_RE.sub(" ", text)
    text = _NL_RE.sub("\n\n", text)
    return text.strip()


def web_fetch(url: str, timeout: float = DEFAULT_TIMEOUT,
              max_chars: int = DEFAULT_MAX_CHARS) -> str:
    """Fetch an http/https URL and return its visible text (truncated).

    Returns a clear ``[offline]`` message if the network cannot be reached and an
    ``error:`` message for bad URLs or real HTTP error statuses.
    """
    if not isinstance(url, str) or not url.strip():
        return "error: no url provided"
    url = url.strip()
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return f"error: only http/https URLs are allowed (got {parsed.scheme or 'none'!r})"
    if not parsed.netloc:
        return "error: malformed url (no host)"
    try:
        status, body, ctype = _get(url, timeout)
    except _Offline as e:
        return (f"[offline] could not reach {parsed.netloc}: {e}. "
                f"myson is running without network access.")
    except urllib.error.HTTPError as e:
        return f"error: HTTP {e.code} from {parsed.netloc}"
    except Exception as e:
        return f"error: {type(e).__name__}: {e}"
    text = _html_to_text(body, ctype)
    if not text:
        return f"(fetched {parsed.netloc}, status {status}, but no readable text)"
    head = f"[{parsed.netloc}] status {status}\n"
    if len(text) > max_chars:
        return head + text[:max_chars] + f"\n...[truncated, {len(text) - max_chars} more chars]"
    return head + text


def _search_backend(query: str, timeout: float, max_results: int) -> str:
    """Query a configured JSON search endpoint (SearXNG-style ``?q=&format=json``)."""
    base = os.environ["MYSON_SEARCH_URL"]
    sep = "&" if "?" in base else "?"
    url = f"{base}{sep}q={urllib.parse.quote(query)}&format=json"
    status, body, _ = _get(url, timeout)
    try:
        data = json.loads(body.decode("utf-8", "replace"))
    except Exception:
        return f"error: search backend returned non-JSON (status {status})"
    results = data.get("results", data if isinstance(data, list) else [])
    if not results:
        return f"(no results for {query!r})"
    lines = [f"Results for {query!r}:"]
    for i, r in enumerate(results[:max_results], 1):
        title = r.get("title", "") if isinstance(r, dict) else str(r)
        link = r.get("url", r.get("href", "")) if isinstance(r, dict) else ""
        snippet = r.get("content", r.get("snippet", "")) if isinstance(r, dict) else ""
        lines.append(f"{i}. {title}\n   {link}\n   {snippet}".rstrip())
    return "\n".join(lines)


def web_search(query: str, max_results: int = 5,
               timeout: float = DEFAULT_TIMEOUT) -> str:
    """Search the web via a configured backend, or explain how to wire one.

    If ``MYSON_SEARCH_URL`` points at a JSON search endpoint it is queried for
    real; otherwise this is a clearly-labelled stub. Either way it returns a
    clear ``[offline]`` message when the network is unreachable.
    """
    if not isinstance(query, str) or not query.strip():
        return "error: no query provided"
    query = query.strip()
    if not os.environ.get("MYSON_SEARCH_URL"):
        return ("[web_search stub] No search backend is configured, so this is a "
                "stub. Set MYSON_SEARCH_URL to a JSON search endpoint (e.g. a local "
                "SearXNG instance) to make it live. "
                f"Offline: cannot return web results for {query!r}.")
    try:
        return _search_backend(query, timeout, max_results)
    except _Offline as e:
        return (f"[offline] search backend unreachable: {e}. "
                f"myson is running without network access.")
    except Exception as e:
        return f"error: {type(e).__name__}: {e}"
