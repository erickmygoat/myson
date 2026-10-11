"""A minimal SAFE tool set for the agent. Local, benign, auditable.
Deliberately excludes anything offensive (no scanning, spoofing, botnet, exploit)."""
import os, io, contextlib

WORKDIR = os.environ.get("MYSON_WORKDIR", os.getcwd())

def read_file(path: str) -> str:
    p = os.path.join(WORKDIR, path)
    return open(p, encoding="utf-8", errors="replace").read()[:20000]

def write_file(path: str, content: str) -> str:
    p = os.path.join(WORKDIR, path)
    os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
    open(p, "w", encoding="utf-8").write(content)
    return f"wrote {len(content)} chars -> {path}"

def run_python(code: str) -> str:
    """Run short python locally and capture stdout. Local personal use only."""
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            exec(code, {"__name__": "__main__"})
    except Exception as e:
        return f"error: {type(e).__name__}: {e}\n{buf.getvalue()}"
    return buf.getvalue()[:10000] or "(no output)"

def web_search(query: str) -> str:
    """Placeholder — wire to a search API or local browser when ready."""
    return f"[web_search not wired yet] would search: {query!r}"

TOOLS = {"read_file": read_file, "write_file": write_file,
         "run_python": run_python, "web_search": web_search}
