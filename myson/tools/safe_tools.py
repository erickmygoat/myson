"""A minimal SAFE tool set for the agent. Local, benign, auditable.
Deliberately excludes anything offensive (no scanning, spoofing, botnet, exploit).

The husk's original in-process ``exec``/open-anywhere implementations have been
hardened: ``run_python`` now runs in the bounded subprocess sandbox (timeout +
output cap + cwd jail), the file tools are jailed to the sandbox dir, and
``web_search`` is the real offline-aware tool. The function names and the
``TOOLS`` dict are unchanged, so anything importing this module keeps working but
is now safe by default.
"""
from .sandbox import run_python, read_file, write_file  # hardened, jailed, bounded
from .web import web_search, web_fetch                    # offline-aware

TOOLS = {"read_file": read_file, "write_file": write_file,
         "run_python": run_python, "web_search": web_search,
         "web_fetch": web_fetch}
