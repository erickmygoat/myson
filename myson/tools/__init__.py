"""SAFE tools only. No network-attack / offensive modules are ported here, ever.

Public surface:
  * low-level tools   - run_python, calc, web_search, web_fetch, read_file, write_file
  * agent layer       - Tool, build_default_tools, run_with_tools
  * legacy husk dict  - TOOLS (now backed by the hardened implementations)
"""
from .sandbox import run_python, read_file, write_file, sandbox_dir
from .mathtool import calc, have_sympy
from .web import web_search, web_fetch
from .agent_tools import Tool, build_default_tools, run_with_tools
from .safe_tools import TOOLS

__all__ = [
    "run_python", "read_file", "write_file", "sandbox_dir",
    "calc", "have_sympy", "web_search", "web_fetch",
    "Tool", "build_default_tools", "run_with_tools", "TOOLS",
]
