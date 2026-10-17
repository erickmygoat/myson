"""Math tool for the myson agent.

Two tiers, chosen automatically at runtime:

  * sympy present  -> full symbolic math: simplify, exact + float values, solve
                      equations, and calculus (``diff(x**2, x)``, ``integrate(...)``,
                      ``factor(...)``) — anything in sympy's namespace.
  * sympy absent   -> a strict AST-based numeric evaluator that understands
                      arithmetic and the ``math`` module's functions/constants and
                      nothing else (no names, attributes, imports, or calls outside
                      a fixed whitelist).

So the tool degrades gracefully: it keeps working for arithmetic even on a box
where sympy was never installed, and it never falls back to a bare ``eval``.
"""
from __future__ import annotations

import ast
import math
import operator
import re

try:
    import sympy as _sympy
    from sympy.parsing.sympy_parser import (
        parse_expr,
        standard_transformations,
        implicit_multiplication_application,
    )
    _HAVE_SYMPY = True
    _TRANSFORMS = standard_transformations + (implicit_multiplication_application,)
except Exception:  # pragma: no cover - exercised on boxes without sympy
    _sympy = None
    _HAVE_SYMPY = False

# A single '=' that is NOT part of ==, <=, >=, != -> an equation to solve.
_EQ_RE = re.compile(r"(?<![=<>!])=(?!=)")
# Cheap pre-screen against eval-escape vectors before sympy's parser sees it.
_UNSAFE_RE = re.compile(r"(__|\bimport\b|\blambda\b|;|`|\bexec\b|\beval\b)")


def have_sympy() -> bool:
    return _HAVE_SYMPY


# ======================================================================================
# sympy tier
# ======================================================================================

def _fmt_number(expr) -> str:
    """Show an exact form and, when it differs, a float approximation.

    Integers print as-is; irrational/transcendental values get both the exact
    form and a 12-digit float (``2*sqrt(2) = 2.82842712475``)."""
    try:
        if getattr(expr, "is_Integer", False):
            return str(expr)
        if getattr(expr, "is_number", False):
            approx = _sympy.N(expr, 12)
            exact = str(expr)
            if exact.replace(" ", "") == str(approx).replace(" ", ""):
                return exact
            return f"{exact} = {approx}"
    except Exception:
        pass
    return str(expr)


def _calc_sympy(expression: str) -> str:
    if _UNSAFE_RE.search(expression):
        return "error: expression contains a disallowed token"

    # Equation? -> solve it.
    if _EQ_RE.search(expression):
        lhs_s, rhs_s = _EQ_RE.split(expression, maxsplit=1)
        try:
            lhs = parse_expr(lhs_s, transformations=_TRANSFORMS)
            rhs = parse_expr(rhs_s, transformations=_TRANSFORMS)
        except Exception as e:
            return f"error: could not parse equation: {type(e).__name__}: {e}"
        eq = _sympy.Eq(lhs, rhs)
        syms = sorted(eq.free_symbols, key=lambda s: s.name)
        if not syms:
            return f"{bool(_sympy.simplify(lhs - rhs) == 0)} (identity check)"
        try:
            sols = _sympy.solve(eq, syms[0] if len(syms) == 1 else syms, dict=True)
        except Exception as e:
            return f"error: solve failed: {type(e).__name__}: {e}"
        if not sols:
            return "no solution"
        parts = []
        for sol in sols:
            parts.append(", ".join(f"{k} = {_fmt_number(v)}" for k, v in sol.items()))
        return "; ".join(parts)

    # Plain expression -> parse, simplify, evaluate.
    try:
        expr = parse_expr(expression, transformations=_TRANSFORMS)
    except Exception as e:
        return f"error: could not parse: {type(e).__name__}: {e}"
    try:
        expr = _sympy.simplify(expr)
    except Exception:
        pass  # simplify is best-effort; keep the parsed form
    if not expr.free_symbols:
        return _fmt_number(expr)
    return str(expr)


# ======================================================================================
# numeric tier (no sympy): strict AST evaluator over arithmetic + math.*
# ======================================================================================

_BIN_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}

# Whitelisted callables and constants only. No attribute access, no other names.
_ALLOWED_FUNCS = {
    name: getattr(math, name) for name in (
        "sqrt", "cbrt", "exp", "log", "log2", "log10", "pow", "fabs",
        "sin", "cos", "tan", "asin", "acos", "atan", "atan2",
        "sinh", "cosh", "tanh", "asinh", "acosh", "atanh",
        "floor", "ceil", "trunc", "factorial", "gcd", "lcm",
        "degrees", "radians", "hypot", "copysign", "fmod", "remainder",
        "comb", "perm", "isqrt", "dist",
    ) if hasattr(math, name)
}
_ALLOWED_FUNCS.update({"abs": abs, "round": round, "min": min, "max": max, "sum": sum})
_ALLOWED_NAMES = {
    "pi": math.pi, "e": math.e, "tau": math.tau,
    "inf": math.inf, "nan": math.nan,
}


def _eval_node(node):
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float, complex)):
            return node.value
        raise ValueError(f"disallowed constant: {node.value!r}")
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))
    if isinstance(node, ast.Name):
        if node.id in _ALLOWED_NAMES:
            return _ALLOWED_NAMES[node.id]
        raise ValueError(f"unknown name: {node.id!r}")
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_eval_node(e) for e in node.elts]
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_FUNCS:
            raise ValueError("only whitelisted math functions may be called")
        if node.keywords:
            raise ValueError("keyword arguments are not allowed")
        args = [_eval_node(a) for a in node.args]
        return _ALLOWED_FUNCS[node.func.id](*args)
    raise ValueError(f"disallowed expression element: {type(node).__name__}")


def _calc_numeric(expression: str) -> str:
    if _EQ_RE.search(expression):
        return ("error: equation solving needs sympy, which is not installed; "
                "this tool can only evaluate numeric expressions right now")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as e:
        return f"error: SyntaxError: {e.msg}"
    try:
        value = _eval_node(tree)
    except ValueError as e:
        return f"error: {e}"
    except ZeroDivisionError:
        return "error: division by zero"
    except Exception as e:
        return f"error: {type(e).__name__}: {e}"
    return repr(value)


# ======================================================================================
# public entry point
# ======================================================================================

def calc(expression: str) -> str:
    """Evaluate a math expression or solve an equation and return the result.

    Examples
    --------
    ``calc("2 + 2*3")``            -> ``"8"``
    ``calc("sqrt(8)")``           -> ``"2*sqrt(2) = 2.82842712475"`` (sympy)
    ``calc("x**2 - 4 = 0")``      -> ``"x = -2; x = 2"`` (sympy)
    ``calc("diff(sin(x), x)")``   -> ``"cos(x)"`` (sympy)
    """
    if not isinstance(expression, str) or not expression.strip():
        return "error: no expression provided"
    expression = expression.strip()
    return _calc_sympy(expression) if _HAVE_SYMPY else _calc_numeric(expression)
