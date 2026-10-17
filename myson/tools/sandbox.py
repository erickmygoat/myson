"""Bounded code execution for the myson agent.

This hardens the husk's ``safe_tools.run_python`` (which did a bare ``exec`` in
this very interpreter) into a *bounded* subprocess runner:

  * timeout          - the code runs in a child process that is killed if it
                       exceeds ``timeout`` seconds (a thread cannot be killed in
                       CPython; a subprocess can);
  * workdir jail     - the child's cwd is a dedicated sandbox directory, so
                       relative file I/O stays contained there, and the file
                       tools below refuse paths that escape it;
  * captured output  - stdout *and* stderr are captured and returned, truncated
                       to ``max_output`` characters;
  * isolated interp  - the child runs ``python -I`` (isolated mode): it ignores
                       ``PYTHON*`` env vars, the user site-packages, and the
                       script directory, and gets a minimal environment;
  * POSIX rlimits    - on POSIX, best-effort CPU / address-space / file-size /
                       fork limits are applied in a ``preexec_fn``.

Honest boundary (read this before trusting it with hostile input)
----------------------------------------------------------------
This is a *resource- and scope-bounded* runner for the assistant's own helper
code, **not** an OS-level security sandbox. Code run with absolute paths can
still read/write outside the jail, and on Windows there are no rlimits (only the
timeout and the output cap apply). Do not use it as a containment boundary
against deliberately hostile code; for that you need an OS container / VM. It is
designed to keep *well-meaning but buggy* model-generated code from hanging the
assistant, flooding it with output, or scribbling over the project tree.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile

# --- config, resilient to being imported standalone in tests -------------------
try:                                            # installed as a package
    from myson.config import MODEL as _MODEL     # noqa: F401  (import-health check)
except Exception:                               # pragma: no cover - import shim
    pass

DEFAULT_TIMEOUT = float(os.environ.get("MYSON_EXEC_TIMEOUT", "10"))
DEFAULT_MAX_OUTPUT = int(os.environ.get("MYSON_EXEC_MAX_OUTPUT", "10000"))
# Soft memory cap for the child, in MB (POSIX only; ignored on Windows).
DEFAULT_MEM_MB = int(os.environ.get("MYSON_EXEC_MEM_MB", "512"))


def sandbox_dir() -> str:
    """The jail directory. Relative file I/O from executed code lands here.

    Defaults to ``<cwd>/.myson_sandbox`` so multi-step workflows can build on
    files they wrote earlier; override with ``MYSON_SANDBOX_DIR``.
    """
    d = os.environ.get("MYSON_SANDBOX_DIR") or os.path.join(os.getcwd(), ".myson_sandbox")
    os.makedirs(d, exist_ok=True)
    return os.path.abspath(d)


def _child_env(jail: str) -> dict:
    """A minimal environment for the child. ``-I`` already ignores PYTHON* vars;
    we keep only what the interpreter needs to start and point TEMP at the jail."""
    keep = ("PATH", "PYTHONPATH")  # PYTHONPATH kept so base site tooling resolves
    env = {k: os.environ[k] for k in keep if k in os.environ}
    if os.name == "nt":
        # Windows needs these to load the C runtime / resolve the interpreter.
        for k in ("SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "NUMBER_OF_PROCESSORS",
                  "PROCESSOR_ARCHITECTURE", "COMSPEC", "PATHEXT"):
            if k in os.environ:
                env[k] = os.environ[k]
    else:
        for k in ("LANG", "LC_ALL", "LC_CTYPE"):
            if k in os.environ:
                env[k] = os.environ[k]
    env["HOME"] = jail
    env["TEMP"] = jail
    env["TMP"] = jail
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def _posix_limits(mem_mb: int, cpu_s: float):
    """Return a ``preexec_fn`` that caps the child's resources, or ``None`` when
    unavailable (Windows, or ``resource`` missing)."""
    if os.name != "posix":
        return None
    try:
        import resource  # POSIX only
    except Exception:  # pragma: no cover - defensive
        return None

    def _apply():  # pragma: no cover - runs only in the POSIX child process
        try:
            cpu = max(1, int(cpu_s) + 1)
            resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
        except Exception:
            pass
        try:
            nbytes = max(64, mem_mb) * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (nbytes, nbytes))
        except Exception:
            pass
        try:
            fsize = 50 * 1024 * 1024  # 50 MB max single file written
            resource.setrlimit(resource.RLIMIT_FSIZE, (fsize, fsize))
        except Exception:
            pass
        try:
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        except Exception:
            pass

    return _apply


def _truncate(text: str, limit: int) -> str:
    if text is None:
        return ""
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[truncated, {len(text) - limit} more chars]"


def run_python(code: str, timeout: float = DEFAULT_TIMEOUT,
               max_output: int = DEFAULT_MAX_OUTPUT, mem_mb: int = DEFAULT_MEM_MB,
               workdir: str | None = None) -> str:
    """Run ``code`` in a bounded child process and return its combined output.

    The child runs ``python -I`` with cwd set to the jail, a minimal environment,
    and (on POSIX) resource limits. stdout and stderr are merged and truncated to
    ``max_output`` chars. On timeout the child is killed and a clear message is
    returned with any partial output.
    """
    if not isinstance(code, str) or not code.strip():
        return "error: no code provided"

    jail = os.path.abspath(workdir) if workdir else sandbox_dir()
    os.makedirs(jail, exist_ok=True)

    # Write the code to a temp file *inside the jail* so tracebacks have a path
    # and nothing is placed outside the sandbox.
    fd, script = tempfile.mkstemp(suffix=".py", prefix="snippet_", dir=jail)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(code)

        try:
            proc = subprocess.run(
                [sys.executable, "-I", script],
                cwd=jail,
                env=_child_env(jail),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                preexec_fn=_posix_limits(mem_mb, timeout),  # None on Windows
                check=False,
            )
        except subprocess.TimeoutExpired as e:
            partial = (e.stdout or "") + (e.stderr or "")
            msg = f"error: TimeoutError: execution exceeded {timeout:g}s and was killed"
            partial = partial.strip()
            return f"{msg}\n{_truncate(partial, max_output)}" if partial else msg
        except Exception as e:  # pragma: no cover - launch failures are rare
            return f"error: {type(e).__name__}: {e}"

        out = proc.stdout or ""
        err = (proc.stderr or "").strip()
        combined = out
        if err:
            combined += ("\n" if combined and not combined.endswith("\n") else "") + err
        if proc.returncode != 0 and "error:" not in combined.lower():
            combined += f"\n[exited with code {proc.returncode}]"
        combined = combined.strip()
        return _truncate(combined, max_output) if combined else "(no output)"
    finally:
        try:
            os.remove(script)
        except OSError:
            pass


# --- jailed file tools (replace the husk's unguarded read_file/write_file) -----

def _resolve_in_jail(path: str) -> str:
    """Resolve ``path`` against the jail and refuse anything that escapes it."""
    jail = sandbox_dir()
    full = os.path.abspath(os.path.join(jail, path))
    if os.path.commonpath([full, jail]) != jail:
        raise ValueError(f"path escapes the sandbox jail: {path!r}")
    return full


def read_file(path: str, max_chars: int = 20000) -> str:
    """Read a text file from inside the sandbox jail (paths are jail-relative)."""
    try:
        full = _resolve_in_jail(path)
    except ValueError as e:
        return f"error: {e}"
    if not os.path.exists(full):
        return f"error: FileNotFoundError: {path}"
    try:
        with open(full, encoding="utf-8", errors="replace") as f:
            return f.read()[:max_chars]
    except Exception as e:
        return f"error: {type(e).__name__}: {e}"


def write_file(path: str, content: str) -> str:
    """Write a text file inside the sandbox jail (paths are jail-relative)."""
    try:
        full = _resolve_in_jail(path)
    except ValueError as e:
        return f"error: {e}"
    try:
        os.makedirs(os.path.dirname(full) or ".", exist_ok=True)
        with open(full, "w", encoding="utf-8") as f:
            f.write(content)
        return f"wrote {len(content)} chars -> {path}"
    except Exception as e:
        return f"error: {type(e).__name__}: {e}"
